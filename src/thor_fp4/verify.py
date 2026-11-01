# verify.py — dual-path verification of the native NVFP4 path.
#
# The native path (device quantize kernels + cuBLASLt block-scaled GEMM on the
# sm_110 tensor cores) is checked against an independent reference built from
# the SAME quantized codes: dequantize on the device (a separate kernel), then
# an FP32 matmul through torch/cuBLAS — a different library, kernel family,
# and accumulation path. Any element disagreeing beyond `bound` (relative) is
# reported. This is the check that catches a silently-wrong kernel or hardware
# path; it stays quiet when both paths agree (the residual is FP16-output
# rounding and accumulation-order differences, ~2e-4 relative for f16 output,
# reproduced in the unit tests). The default bound 1e-2 leaves headroom above
# that residual while still flagging real kernel errors, which produce O(1)
# deviations.

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from . import ops
from .linear import Linear

__all__ = ["VerifyReport", "verify", "verify_linear", "ReferenceLinear"]


@dataclass
class VerifyReport:
    n_elements: int = 0
    n_bad: int = 0
    worst_rel: float = 0.0
    rel_rms: float = 0.0
    worst_locations: list = field(default_factory=list)   # (flat index, native, ref)
    layers: dict = field(default_factory=dict)            # name -> VerifyReport

    @property
    def ok(self) -> bool:
        return self.n_bad == 0

    def merge(self, other: "VerifyReport"):
        self.n_elements += other.n_elements
        self.n_bad += other.n_bad
        self.worst_rel = max(self.worst_rel, other.worst_rel)
        self.rel_rms = max(self.rel_rms, other.rel_rms)
        self.worst_locations = (self.worst_locations + other.worst_locations)[:10]

    def __str__(self):
        head = (f"ok={self.ok} elements={self.n_elements} beyond_bound={self.n_bad} "
                f"({100.0 * self.n_bad / max(1, self.n_elements):.4g}%) "
                f"worst_rel={self.worst_rel:.3e} rel_rms={self.rel_rms:.3e}")
        for i, a, b in self.worst_locations[:5]:
            head += f"\n  idx={i}: native={a:.6g} ref={b:.6g}"
        for name, s in self.layers.items():
            head += (f"\n  {name}: beyond_bound={s.n_bad} "
                     f"worst_rel={s.worst_rel:.3e} rel_rms={s.rel_rms:.3e}")
        return head


class ReferenceLinear(nn.Module):
    """The independent execution path for one thor_fp4.Linear: quantize the
    input exactly as the native path does (same codes), dequantize both
    operands, matmul in torch/cuBLAS FP32, add bias."""

    def __init__(self, layer: Linear):
        super().__init__()
        self.layer = layer
        self.out_dtype = layer.out_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer = self.layer
        xc = x.detach().contiguous()
        if xc.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            xc = xc.float()
        kind = getattr(layer, "engine_kind", "lt")
        if kind == "mx8":
            # MX operands: the reference runs the torch fake-quant arithmetic
            # (e2m1/e4m3 codes with UE8M0 per-32 scales) and an FP32 matmul
            from .quant import mx_fake_quant
            xd = mx_fake_quant(xc.float(), "mxfp8")
            w = mx_fake_quant(layer.weight.detach().float(), "mxfp4")
        elif kind == "gv":
            # GEMV: activations are consumed exactly — only the weights are
            # quantized, so the reference dequantizes weights and matmuls the
            # raw activations
            if layer._weight_fp4 is None:
                layer.refresh_weight()
            w = layer._weight_fp4.dequantize()                    # f32 (N, K)
            xd = xc.float()
        else:
            if layer._weight_fp4 is None:
                layer.refresh_weight()
            w = layer._weight_fp4.dequantize()                    # f32 (N, K)
            packed, sblk = ops.quantize_nvfp4(xc.float(), layer.act_tensor_scale)
            xd = ops.dequantize_nvfp4(packed, sblk, layer.act_tensor_scale, *xc.shape)
        out = xd @ w.t()
        if layer.bias is not None:
            out = out + layer.bias.float()
        return out.to({"f16": torch.float16, "bf16": torch.bfloat16,
                       "f32": torch.float32}[self.out_dtype])


def _compare(native: torch.Tensor, ref: torch.Tensor, bound: float) -> VerifyReport:
    """bad = |native - ref| > bound * rms(ref): relative to the output's
    magnitude scale, so near-zero elements do not false-alarm (a per-element
    relative bound would flag every output that happens to round to ~0)."""
    rep = VerifyReport()
    n, r = native.float(), ref.float()
    scale = r.pow(2).mean().sqrt().clamp_min(1e-30)
    diff = (n - r).abs()
    bad = diff > bound * scale
    rep.n_elements = n.numel()
    rep.n_bad = int(bad.sum().item())
    rep.rel_rms = float((diff.pow(2).mean().sqrt() / scale).item())
    if rep.n_bad:
        rep.worst_rel = float((diff / scale).max().item())
        flat = bad.flatten().nonzero(as_tuple=False)[:10, 0]
        nf, rf = n.flatten(), r.flatten()
        for i in flat.tolist():
            rep.worst_locations.append((i, float(nf[i]), float(rf[i])))
    return rep


@torch.no_grad()
def verify_linear(layer: Linear, x: torch.Tensor, bound: float = 1e-2) -> VerifyReport:
    """Dual-path check of one thor_fp4.Linear on one input batch."""
    native = layer(x)
    ref = ReferenceLinear(layer)(x)
    return _compare(native, ref, bound)


def _swap(root: nn.Module, name: str, new: nn.Module):
    if "." in name:
        parent = root.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[1]
    else:
        parent, attr = root, name
    setattr(parent, attr, new)


@torch.no_grad()
def verify(model_or_layer, inputs, bound: float = 1e-2) -> VerifyReport:
    """Run `model_or_layer` on `inputs` through the native NVFP4 path and
    through the independent dequantize-then-matmul reference; report every
    element disagreeing beyond `bound` (relative).

    model_or_layer: a thor_fp4.Linear, or any nn.Module containing them.
    Each thor_fp4.Linear is checked per-layer on the model's real activations,
    and the model's end-to-end outputs are checked against a fully
    reference-swapped execution. inputs: a tensor or a list of tensors."""
    if isinstance(model_or_layer, Linear):
        rep = VerifyReport()
        for x in (inputs if isinstance(inputs, list) else [inputs]):
            rep.merge(verify_linear(model_or_layer, x, bound))
        return rep

    model, rep = model_or_layer, VerifyReport()
    assert isinstance(model, nn.Module)
    xs = inputs if isinstance(inputs, list) else [inputs]

    fp4 = {n: m for n, m in model.named_modules() if isinstance(m, Linear)}
    # pass 1 (native): capture each layer's real activations; check each
    # layer against its reference on those exact operands
    captured: dict[str, list] = {n: [] for n in fp4}
    hooks = []
    for name, m in fp4.items():
        def hook(mod, inp, out, _name=name):
            captured[_name].append(inp[0].detach())
        hooks.append(m.register_forward_hook(hook))
    outs_native = [model(x) for x in xs]
    for h in hooks:
        h.remove()
    for name, m in fp4.items():
        sub = VerifyReport()
        for xin in captured[name]:
            sub.merge(verify_linear(m, xin, bound))
        rep.layers[name] = sub
        rep.merge(sub)

    # pass 2 (end-to-end): every fp4 layer swapped for its reference. Each
    # reference layer is FED the native pass's captured activations, so both
    # executions see identical operands (f32 accumulation-order differences
    # between libraries flip sub-ulp fp16 bits in activations, which can flip
    # e2m1 codes downstream and masquerade as an error); non-fp4 glue (gelu,
    # norms) runs normally on the reference layers' outputs.
    if fp4:
        class FedReference(nn.Module):
            def __init__(self, layer, queue):
                super().__init__()
                self.layer, self.q = layer, queue

            def forward(self, x):
                real = self.q.pop(0) if self.q else x
                return ReferenceLinear(self.layer)(real)

        try:
            for name, m in fp4.items():
                _swap(model, name, FedReference(m, list(captured[name])))
            outs_ref = [model(x) for x in xs]
        finally:
            for name, m in fp4.items():
                _swap(model, name, m)
        # rounding compounds across layers; scale the bound with depth
        e2e_bound = bound * max(1, len(fp4)) ** 0.5
        sub = VerifyReport()
        for a, b in zip(outs_native, outs_ref):
            sub.merge(_compare(a, b, e2e_bound))
        rep.layers["<end-to-end>"] = sub
        rep.merge(sub)
    return rep

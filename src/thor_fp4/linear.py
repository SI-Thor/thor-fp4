# linear.py — thor_fp4.Linear, the nn.Linear drop-in.
#
# Weights are resident in NVFP4 (e2m1 codes, one UE4M3 scale per 16 elements,
# plus a per-tensor scale folded into the GEMM's alpha). Activations are
# quantized on the fly on the same stream. The GEMM is the cuBLASLt
# block-scaled path; see csrc/thor_fp4_core.cu for why the output comes out
# row-major with no transpose.
#
# Constraints from the hardware/library (measured, see README):
#   * in_features must be a multiple of 64 (the scale tiles are 4 sf wide);
#   * out_features must be >= 8 and a multiple of 8 (the GEMM's D-row count —
#     with the operand swap this is the weight side, not the batch);
#   * the batch M is unconstrained (any M >= 1 works, verified).
#
# Everything is capture-safe: after the first call for a given (M, dtype) the
# activation buffers are reused, so a captured region replays without
# allocating.

from __future__ import annotations

import torch
import torch.nn as nn

from . import ops
from .ops import NVFP4Tensor, MXTensor

__all__ = ["Linear"]

_DEFAULT_CUTLASS_VARIANTS = ["nvfp4_128x256x256_c2x1", "nvfp4_128x128x256_c1"]


def _resolve_engine(engine: str):
    """'cublaslt' | 'gemv' | 'cutlass' | 'cutlass:<variant-name>' -> (kind, variant_idx).

    kind: 'lt' (cuBLASLt NVFP4), 'gv' (register-dequant GEMV, M<=8),
    'nv' (CUTLASS NVFP4xNVFP4), 'mx8' (CUTLASS MXFP8 activations x MXFP4
    weights — the mix cuBLASLt refuses)."""
    if engine == "cublaslt":
        return "lt", -1
    if engine == "gemv":
        return "gv", -1
    from . import _ext
    L = _ext.cutlass()
    if L is None:
        raise RuntimeError("engine='cutlass*' needs build/libthorfp4_cutlass.so (make cutlass)")
    if engine == "cutlass":
        names = [bytes(L.tfc4_variant_name(i)).decode() for i in range(L.tfc4_n_variants())]
        name = next((c for c in _DEFAULT_CUTLASS_VARIANTS if c in names),
                    next(n for n in names if n.startswith("nvfp4_")))
    elif engine.startswith("cutlass:"):
        name = engine.split(":", 1)[1]
    else:
        raise ValueError(f"unknown engine {engine!r}")
    idx = L.tfc4_variant_index(name.encode())
    if idx < 0:
        raise ValueError(f"unknown cutlass variant {name!r}")
    kind = "mx8" if name.startswith("mxfp8") else "nv"
    return kind, idx


class Linear(nn.Module):
    """Applies a linear transform with NVFP4 weights:
    y = x_quant @ W_quant^T (+ bias).

    NOTE: the returned tensor is a reused per-(M, dtype) buffer (this is what
    makes repeated calls CUDA-graph-capture-safe); clone it if you need it
    past the next call with the same batch size.

    Args:
        in_features: K; must be a multiple of 64.
        out_features: N; must be >= 8 and a multiple of 8.
        bias: keep an additive bias (stored fp32, added in the output dtype).
        out_dtype: 'f16' (default, fastest) | 'bf16' | 'f32'.
        tensor_scale: None = two-level NVFP4 recipe (per-tensor scale picked
            from the weight amax, block scales near the top of UE4M3);
            1.0 = block scales only.
        act_tensor_scale: per-tensor scale for activations, 1.0 (default,
            dynamic block scales only) or a calibrated static value from
            thor_fp4.calibrate().
        engine: 'cublaslt' (default) | 'gemv' — the register-dequant GEMV,
            the bandwidth-bound path for batch M <= 8 (weights stream at FP4
            density; activations are NOT quantized) | 'cutlass[:variant]'.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, out_dtype: str = "f16",
                 tensor_scale: float | None = None, act_tensor_scale: float = 1.0,
                 engine: str = "cublaslt"):
        super().__init__()
        if in_features % 64 != 0:
            raise ValueError(
                f"in_features={in_features}: NVFP4 needs K % 64 == 0 "
                "(keep such layers in BF16 — see thor_fp4.convert)")
        if out_features < 8 or out_features % 8 != 0:
            raise ValueError(
                f"out_features={out_features}: NVFP4 needs N >= 8 and N % 8 == 0")
        self.in_features = in_features
        self.out_features = out_features
        self.engine_kind, self.engine_variant = _resolve_engine(engine)
        if self.engine_kind in ("nv", "mx8"):
            out_dtype = "bf16"          # the CUTLASS epilogues are BF16
        self.out_dtype = out_dtype
        self.act_tensor_scale = float(act_tensor_scale)
        self._weight_fp4: NVFP4Tensor | None = None
        _ = dtype
        self._tensor_scale_requested = tensor_scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self._act_bufs: dict[tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._gemv_outs: dict[tuple[int, torch.dtype], torch.Tensor] = {}

    def reset_parameters(self):
        # keep the fp32 master weight; NVFP4 copy is (re)built lazily so
        # QAT-style updates and .to() keep working
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self._weight_fp4 = None

    def _apply(self, fn, recurse=True):
        self._weight_fp4 = None
        self._act_bufs = {}
        self._gemv_outs = {}
        return super()._apply(fn, recurse)

    def refresh_weight(self):
        """Re-quantize the master weight (call after in-place weight edits)."""
        w = self.weight.detach().contiguous().float()
        if self.engine_kind == "mx8":
            self._weight_fp4 = MXTensor.from_float(w, "mxfp4")
        else:
            self._weight_fp4 = NVFP4Tensor.from_float(w, self._tensor_scale_requested)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # inference layer: the tensor-core path is not differentiable; run
        # under no_grad so fp32 master weights with requires_grad=True (the
        # convert() of a training model) do not leak a graph through the bias
        with torch.no_grad():
            if self._weight_fp4 is None:
                self.refresh_weight()
            w = self._weight_fp4
            M = x.shape[0]
            x = x.detach() if x.requires_grad else x
            if not x.is_contiguous():
                x = x.contiguous()
            dt = x.dtype
            if dt not in (torch.float32, torch.float16, torch.bfloat16):
                x = x.float()
                dt = torch.float32
            key = (M, dt)
            if self.engine_kind == "gv":
                # register-dequant GEMV: no activation quantize buffers; the
                # weights stream at FP4 density and the activations read exact
                if M > 8:
                    raise RuntimeError(
                        f"engine='gemv' serves M <= 8 (got {M}); use engine='cublaslt'")
                out = self._gemv_outs.get(key)
                if out is None:
                    out = torch.empty(M, self.out_features,
                                      dtype={"f16": torch.float16, "bf16": torch.bfloat16,
                                             "f32": torch.float32}[self.out_dtype],
                                      device=x.device)
                    self._gemv_outs[key] = out
                ops.gemv(x, w, self.out_dtype, out=out)
                if self.bias is not None:
                    out += self.bias.to(out.dtype)
                return out
            bufs = self._act_bufs.get(key)
            if bufs is None:
                if self.engine_kind == "mx8":
                    pb = ops.mx_packed_bytes(M, self.in_features, "mxfp8")
                    sb = ops.mx_scale_bytes(M, self.in_features)
                else:
                    pb = ops.packed_bytes(M, self.in_features)
                    sb = ops.scale_bytes(M, self.in_features)
                packed = torch.zeros(pb, dtype=torch.uint8, device=x.device)
                sblk = torch.zeros(sb, dtype=torch.uint8, device=x.device)
                out = torch.empty(M, self.out_features,
                                  dtype={"f16": torch.float16, "bf16": torch.bfloat16,
                                         "f32": torch.float32}[self.out_dtype],
                                  device=x.device)
                bufs = (packed, sblk, out)
                self._act_bufs[key] = bufs
            packed, sblk, out = bufs
            if self.engine_kind == "lt":
                ops.quantize_nvfp4(x, self.act_tensor_scale, packed, sblk)
                ops.gemm(packed, sblk, self.act_tensor_scale,
                         w.packed, w.sblk, w.tensor_scale,
                         out, M, self.out_features, self.in_features, self.out_dtype)
            elif self.engine_kind == "nv":
                # CUTLASS NVFP4: same operand encoding as the cuBLASLt path,
                # our kernel; alpha carries the two-level tensor scales
                from . import _ext
                ops.quantize_nvfp4(x, self.act_tensor_scale, packed, sblk)
                rc = _ext.cutlass().tfc4_gemm(
                    self.engine_variant, packed.data_ptr(), sblk.data_ptr(),
                    w.packed.data_ptr(), w.sblk.data_ptr(), out.data_ptr(),
                    M, self.out_features, self.in_features,
                    self.act_tensor_scale * w.tensor_scale, _ext.stream())
                assert rc == 0, f"tfc4_gemm rc={rc}"
            else:
                # MXFP8 activations x MXFP4 weights: the mix cuBLASLt refuses
                from . import _ext
                ops.quantize_mx(x, "mxfp8", packed, sblk)
                rc = _ext.cutlass().tfc4_gemm(
                    self.engine_variant, packed.data_ptr(), sblk.data_ptr(),
                    w.packed.data_ptr(), w.sblk.data_ptr(), out.data_ptr(),
                    M, self.out_features, self.in_features, 1.0, _ext.stream())
                assert rc == 0, f"tfc4_gemm rc={rc}"
            if self.bias is not None:
                out += self.bias.to(out.dtype)
            return out

    def extra_repr(self) -> str:
        ts = (self._weight_fp4.tensor_scale if self._weight_fp4 is not None
              else self._tensor_scale_requested)
        eng = {"lt": "cublaslt", "gv": "gemv"}.get(
            self.engine_kind, f"cutlass:{self.engine_variant}")
        return (f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, "
                f"out_dtype={self.out_dtype}, tensor_scale={ts}, engine={eng}")

    def fp4_weight_nbytes(self) -> int:
        """Bytes of the resident weight (codes + block scales)."""
        return self.in_features * self.out_features // 2 + self.out_features * (self.in_features // 16)

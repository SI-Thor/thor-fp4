# quant.py — the quantizer arithmetic, PTQ calibration, and QAT.
#
# nvfp4_fake_quant() mirrors the device kernels bit-for-bit on normal data:
# the block scale is the UE4M3 round-trip of amax/(6*ts) (encode -> decode,
# so quantization error of the scale itself is included), the codes are
# round-to-nearest e2m1 with ties up (matching the kernel's threshold
# compares), and the per-tensor scale ts multiplies back. QAT trains against
# exactly the arithmetic the tensor cores will commit.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "nvfp4_fake_quant", "mx_fake_quant", "STE", "qat", "QATLinear", "qat_linear",
    "QATSequential", "calibrate_linear", "two_level_scale",
]

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def two_level_scale(x: torch.Tensor) -> float:
    """The NVFP4 two-level per-tensor scale: amax/(6*448), which places the
    largest block scale at the top of the UE4M3 range and keeps small-amax
    blocks from underflowing into UE4M3 subnormals."""
    return max(x.abs().amax().item() / (6.0 * 448.0), 1e-30)


def nvfp4_fake_quant(x: torch.Tensor, tensor_scale: float = 1.0, blk: int = 16) -> torch.Tensor:
    """Round-to-nearest NVFP4 fake quantization, matching csrc/thor_fp4_core.cu:
    per-16 block scale UE4M3(amax/(6*ts)) (encode-decode round trip), e2m1
    codes with ties up, value = code * scale * ts."""
    shp = x.shape
    xv = x.reshape(-1, blk).float()
    amax = xv.abs().amax(dim=1, keepdim=True)
    # (amax * (1/6)) / ts, evaluated in this exact order: the device kernel
    # computes the same expression, and amax/(6*ts) differs by 1 ulp, enough
    # to flip e4m3 rounding at ties
    sf = (amax * (1.0 / 6.0) / tensor_scale).to(torch.float8_e4m3fn).to(torch.float32)
    ds = sf * tensor_scale
    safe = ds.clamp_min(1e-30)
    q = xv / safe
    sign = torch.where(q >= 0, 1.0, -1.0)
    a = q.abs().clamp_max(6.0)
    mid = _E2M1_MID.to(x.device)
    idx = torch.bucketize(a, mid, right=True)
    codes = _E2M1.to(x.device, torch.float32)[idx]
    out = codes * sign * ds
    # blocks whose encoded scale underflowed to 0 decode to 0 on the hardware
    return torch.where(ds > 0, out, torch.zeros_like(out)).reshape(shp)


def mx_fake_quant(x: torch.Tensor, kind: str = "mxfp4", blk: int = 32) -> torch.Tensor:
    """MX fake quant (e2m1 or e4m3 codes, UE8M0 = power-of-two scale per 32),
    matching tfp4_quant_mx: scale = 2^round(log2(amax/maxval))."""
    assert kind in ("mxfp4", "mxfp8")
    shp = x.shape
    xv = x.reshape(-1, blk).float()
    amax = xv.abs().amax(dim=1, keepdim=True)
    maxval = 6.0 if kind == "mxfp4" else 448.0
    e = torch.log2(amax / maxval).round().clamp(-127, 127)
    ds = torch.pow(2.0, e)
    safe = ds.clamp_min(1e-30)
    q = xv / safe
    if kind == "mxfp4":
        sign = torch.where(q >= 0, 1.0, -1.0)
        a = q.abs().clamp_max(6.0)
        idx = torch.bucketize(a, _E2M1_MID.to(x.device), right=True)
        out = _E2M1.to(x.device, torch.float32)[idx] * sign * ds
    else:
        out = q.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32) * ds
    return torch.where(ds > 0, out, torch.zeros_like(out)).reshape(shp)


class STE(torch.autograd.Function):
    """Straight-through estimator: identity gradient through fake quant."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        return g


def qat(x: torch.Tensor, tensor_scale: float = 1.0) -> torch.Tensor:
    """NVFP4 fake-quant with STE (identity backward)."""
    return STE.apply(nvfp4_fake_quant(x, tensor_scale))


class QATLinear(nn.Linear):
    """nn.Linear whose weight AND input activations are NVFP4-fake-quantized
    (STE); the bias stays fp32. The default tensor_scale=1 trains against
    block scales only; pass two_level=True to include the per-tensor scale
    (recomputed from the weight amax each forward)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 two_level: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.two_level = two_level

    def forward(self, x):
        wts = two_level_scale(self.weight) if self.two_level else 1.0
        return F.linear(qat(x), qat(self.weight, wts), self.bias)


def qat_linear(l: nn.Linear, two_level: bool = False) -> QATLinear:
    q = QATLinear(l.in_features, l.out_features, l.bias is not None, two_level)
    with torch.no_grad():
        q.weight.copy_(l.weight)
        if l.bias is not None:
            q.bias.copy_(l.bias)
    return q


class QATSequential(nn.Sequential):
    """An nn.Sequential whose Linears run QAT fake-quant (weights + inputs)."""

    def __init__(self, *layers: nn.Module, two_level: bool = False):
        mods = []
        for m in layers:
            mods.append(qat_linear(m, two_level) if isinstance(m, nn.Linear) else m)
        super().__init__(*mods)


@torch.no_grad()
def calibrate_linear(layer: nn.Linear, inputs: list[torch.Tensor] | torch.Tensor,
                     method: str = "amax") -> float:
    """Pick the activation per-tensor scale for a layer from calibration data
    (currently amax over the samples — the input to NVFP4Tensor/act_tensor_scale)."""
    xs = inputs if isinstance(inputs, list) else [inputs]
    amax = 0.0
    for x in xs:
        amax = max(amax, x.abs().amax().item())
    return amax / (6.0 * 448.0) if method == "amax" else 1.0

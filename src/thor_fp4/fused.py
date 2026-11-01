# fused.py — one-kernel-per-layer MLP chains (ROUND 2, DIRECTIONS.md §4).
#
# Round 1 measured that at small K the FP4 chain's cost is dominated by the
# standalone activation-quantize pass (a full read+write over the
# activations), not the GEMM. These layers remove that pass entirely: the
# GEMM's epilogue computes gelu(alpha*acc + bias) and writes D directly as
# MXFP8 codes plus one UE8M0 scale per 32 outputs — while the accumulators
# are still in registers — which is exactly the operand format the next
# layer's A side consumes. A chain of N layers therefore launches N kernels
# total: no quantize kernels, no gelu kernels, no bias adds.
#
# Numerics note (measured, BENCHMARKS.md): CUTLASS's SfDGen stores
# UE8M0(ceil-pow2)(amax/448) scales while our standalone quantize_mx rounds
# to the nearer power of two (ties up) — the fused path never saturates a
# block max, ours can clip by up to ~34% of 448 on ties that round down. The
# delivered operands differ slightly on those blocks; both paths are checked
# against the dequantize-then-matmul reference in the tests.
#
# Constraints (the fused epilogue's SF store): K % 128 == 0, N % 32 == 0;
# the bias pointer is required by the EVT (pass zeros for bias-free layers).
# Build: `make fused` -> build/libthorfp4_fused.so.

from __future__ import annotations

import torch
import torch.nn as nn

from . import _ext, ops
from .linear import Linear

__all__ = ["MX8Act", "FusedLinear", "mx8_chain"]


class MX8Act:
    """A row-major (M, K) tensor resident in MXFP8 (e4m3 codes + UE8M0
    per-32 scales in the tcgen05 tile layout) — what fused layers pass
    between each other."""

    def __init__(self, packed: torch.Tensor, sblk: torch.Tensor, shape):
        self.packed, self.sblk, self.shape = packed, sblk, tuple(shape)

    def nbytes(self) -> int:
        M, K = self.shape
        return M * (K + K // 32)


class FusedLinear(nn.Module):
    """y_codes, y_scales = quantize_mx8( gelu( x_codes @ W_mxfp4^T + bias ) )
    — the whole inter-layer glue of an MLP hidden layer in the GEMM's
    epilogue. Consumes and returns MX8Act (a plain tensor is quantized once
    on entry). The returned buffers are reused per batch shape (capture-safe
    like thor_fp4.Linear)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 variant: str | None = None, device=None):
        super().__init__()
        if in_features % 128 != 0:
            raise ValueError(f"in_features={in_features}: the fused epilogue needs K % 128 == 0")
        if out_features % 32 != 0:
            raise ValueError(f"out_features={out_features}: the SF store needs N % 32 == 0")
        L = _ext.fused()
        if L is None:
            raise RuntimeError("FusedLinear needs build/libthorfp4_fused.so (make fused)")
        if variant is None:
            names = [bytes(L.tfc4f_variant_name(i)).decode() for i in range(L.tfc4f_n_variants())]
            variant = next((n for n in names if "128x128x256" in n), names[0])
        self.variant_idx = L.tfc4f_variant_index(variant.encode())
        if self.variant_idx < 0:
            raise ValueError(f"unknown fused variant {variant!r}")
        self.variant = variant
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self._w: ops.MXTensor | None = None
        self._outs: dict[int, MX8Act] = {}

    def _apply(self, fn, recurse=True):
        self._w = None
        self._outs = {}
        return super()._apply(fn, recurse)

    def refresh_weight(self):
        w = self.weight.detach().contiguous().float().cuda()
        self._w = ops.MXTensor.from_float(w, "mxfp4")
        return self

    def forward(self, x):
        with torch.no_grad():
            if self._w is None:
                self.refresh_weight()
            if isinstance(x, MX8Act):
                M = x.shape[0]
                act = x
                assert x.shape[1] == self.in_features
            else:
                xc = x.detach().contiguous()
                if xc.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    xc = xc.float()
                M = xc.shape[0]
                packed = torch.zeros(ops.mx_packed_bytes(M, self.in_features, "mxfp8"),
                                      dtype=torch.uint8, device=xc.device)
                sblk = torch.zeros(ops.mx_scale_bytes(M, self.in_features),
                                   dtype=torch.uint8, device=xc.device)
                ops.quantize_mx(xc.float(), "mxfp8", packed, sblk)
                act = MX8Act(packed, sblk, (M, self.in_features))
            out = self._outs.get(M)
            if out is None:
                N = self.out_features
                dp = torch.zeros(M * N, dtype=torch.uint8, device=act.packed.device)
                sf = torch.zeros(int(_ext.fused().tfc4f_sfd_bytes(M, N)),
                                 dtype=torch.uint8, device=act.packed.device)
                out = MX8Act(dp, sf, (M, N))
                self._outs[M] = out
            bias = (self.bias if self.bias is not None
                    else torch.zeros(self.out_features, device=act.packed.device))
            rc = _ext.fused().tfc4f_gemm(
                self.variant_idx,
                act.packed.data_ptr(), act.sblk.data_ptr(),
                self._w.packed.data_ptr(), self._w.sblk.data_ptr(),
                bias.float().contiguous().data_ptr(),
                out.packed.data_ptr(), out.sblk.data_ptr(),
                M, self.out_features, self.in_features, 1.0, _ext.stream())
            assert rc == 0, f"tfc4f_gemm rc={rc}"
            return out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, "
                f"variant={self.variant}")


def mx8_chain(*dims: int, variant: str | None = None, device=None) -> nn.Sequential:
    """An MLP  dims[0] -> dims[1] -> ... -> dims[-1]  whose hidden layers are
    FusedLinear (one kernel per layer: GEMM + bias + GELU + MXFP8 quantize in
    the epilogue) and whose final layer is a thor_fp4.Linear on the CUTLASS
    mx8 engine (BF16 logits). Activations flow between layers as packed MXFP8
    — no separate quantize pass anywhere in the chain."""
    mods = []
    for i in range(len(dims) - 1):
        if i < len(dims) - 2:
            mods.append(FusedLinear(dims[i], dims[i + 1], bias=True,
                                    variant=variant, device=device))
        else:
            mods.append(Linear(dims[i], dims[i + 1], bias=True,
                               engine="cutlass:mx8").to(device))
    return nn.Sequential(*mods)

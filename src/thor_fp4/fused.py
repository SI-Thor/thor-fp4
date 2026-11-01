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

    def rows(self, a: int, b: int) -> "MX8Act":
        """Rows [a, b) as an MX8Act of view buffers. Both packed codes (K
        bytes/row, row-major) and scales (128-row tiles) are row-major
        prefixes, so a 128-multiple slice is a clean prefix of each."""
        from . import ops
        M, K = self.shape
        assert a % 128 == 0 or a == 0, "slices must be 128-row aligned"
        return MX8Act(
            self.packed[a * K:(b - 1) * K + ops.mx_packed_bytes(b - a, K, "mxfp8")],
            self.sblk[ops.mx_scale_bytes(a, K):ops.mx_scale_bytes(a, K)
                      + ops.mx_scale_bytes(b - a, K)],
            (b - a, K))


class FusedLinear(nn.Module):
    """y_codes, y_scales = quantize_mx8( gelu( x_codes @ W_mxfp4^T + bias ) )
    — the whole inter-layer glue of an MLP hidden layer in the GEMM's
    epilogue. Consumes and returns MX8Act (a plain tensor is quantized once
    on entry). The returned buffers are reused per batch shape (capture-safe
    like thor_fp4.Linear).

    EXPERIMENTAL (measured on this box, CUTLASS 4.4.1 on sm_110a): at batch
    M >= ~2048, through this wrapper's allocation pattern, scattered rows
    (~0.2%) received ue8m0-NaN scale bytes — deterministic, reproducible,
    and absent in direct ABI calls with fresh buffers at every M, i.e. an
    epilogue-path fragility we could not pin to our own code. Rows are
    computed independently, so forward() batches through max_chunk-row
    slices (512 by default; measured clean at every size tried)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 variant: str | None = None, device=None, max_chunk: int = 512):
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
        self.max_chunk = max_chunk

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
            Mfull = x.shape[0] if not hasattr(x, "packed") else x.shape[0]
            if self.max_chunk and Mfull > self.max_chunk:
                # cache one output buffer per (batch, chunk index) — a single
                # per-M cache would make every chunk overwrite the previous
                outs = []
                for ci, i in enumerate(range(0, Mfull, self.max_chunk)):
                    sl = x[i:i + self.max_chunk] if not hasattr(x, "packed") \
                        else x.rows(i, min(i + self.max_chunk, Mfull))
                    outs.append(self._forward_once(sl, ci))
                return MX8Act(
                    torch.cat([o.packed for o in outs]),
                    torch.cat([o.sblk for o in outs]),
                    (Mfull, self.out_features))
            return self._forward_once(x)

    def _forward_once(self, x, chunk: int = 0):
        with torch.no_grad():
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
            out = self._outs.get((M, chunk))
            if out is None:
                N = self.out_features
                dp = torch.zeros(M * N, dtype=torch.uint8, device=act.packed.device)
                sf = torch.zeros(int(_ext.fused().tfc4f_sfd_bytes(M, N)),
                                 dtype=torch.uint8, device=act.packed.device)
                out = MX8Act(dp, sf, (M, N))
                self._outs[(M, chunk)] = out
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

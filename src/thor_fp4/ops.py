# ops.py — functional NVFP4/MX ops on top of the core extension.
#
# NVFP4 = e2m1 codes (2/byte) + one UE4M3 scale per 16 elements in the tcgen05
# 128x4-tile layout, times an optional per-tensor scale (applied through the
# GEMM's alpha). MX = e2m1 or e4m3 codes + one UE8M0 scale per 32 (CUTLASS
# path). All buffers are torch-owned uint8 tensors, padded to 128-row tiles
# and zero-initialized once; the kernels never write rows >= R, so reuse for
# the same R is safe and CUDA-graph capture sees no allocation.

from __future__ import annotations

import torch

from . import _ext

__all__ = [
    "pad_rows", "packed_bytes", "scale_bytes", "mx_packed_bytes", "mx_scale_bytes",
    "quantize_nvfp4", "quantize_mx", "dequantize_nvfp4", "block_scales",
    "gemm", "gemv", "NVFP4Tensor", "MXTensor",
]


def pad_rows(R: int) -> int:
    return int(_ext.core().tfp4_pad_rows(R))


def packed_bytes(R: int, K: int) -> int:
    return int(_ext.core().tfp4_packed_bytes(R, K))


def scale_bytes(R: int, K: int) -> int:
    return int(_ext.core().tfp4_scale_bytes(R, K))


def mx_packed_bytes(R: int, K: int, kind: str) -> int:
    return int(_ext.core().tfp4_mx_packed_bytes(R, K, 0 if kind == "mxfp4" else 1))


def mx_scale_bytes(R: int, K: int) -> int:
    return int(_ext.core().tfp4_mx_scale_bytes(R, K))


def quantize_nvfp4(x: torch.Tensor, tensor_scale: float = 1.0,
                   packed: torch.Tensor | None = None,
                   sblk: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a row-major (R, K) cuda tensor (f32/f16/bf16) to NVFP4.

    Returns (packed, sblk) uint8 buffers; pass pre-allocated (zeroed, padded)
    buffers to reuse them, e.g. inside captured loops."""
    L = _ext.core()
    assert x.ndim == 2 and x.is_cuda and x.is_contiguous()
    R, K = x.shape
    assert K % 64 == 0, f"K={K} must be a multiple of 64 (cuBLASLt constraint)"
    if packed is None:
        packed = torch.zeros(packed_bytes(R, K), dtype=torch.uint8, device=x.device)
    if sblk is None:
        sblk = torch.zeros(scale_bytes(R, K), dtype=torch.uint8, device=x.device)
    rc = L.tfp4_quant(x.data_ptr(), R, K, _ext.IN_DTYPE[x.dtype], tensor_scale,
                      packed.data_ptr(), sblk.data_ptr(), _ext.stream())
    assert rc == 0, f"tfp4_quant rc={rc}"
    return packed, sblk


def quantize_mx(x: torch.Tensor, kind: str = "mxfp4",
                packed: torch.Tensor | None = None,
                sblk: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to an MX format: kind 'mxfp4' (e2m1 codes) or 'mxfp8' (e4m3
    codes), UE8M0 scale per 32 elements — the operand mix cuBLASLt refuses on
    sm_110 but the tensor cores (via CUTLASS) accept."""
    L = _ext.core()
    assert kind in ("mxfp4", "mxfp8")
    R, K = x.shape
    assert K % 64 == 0, f"K={K} must be a multiple of 64"
    if packed is None:
        packed = torch.zeros(mx_packed_bytes(R, K, kind), dtype=torch.uint8, device=x.device)
    if sblk is None:
        sblk = torch.zeros(mx_scale_bytes(R, K), dtype=torch.uint8, device=x.device)
    rc = L.tfp4_quant_mx(x.data_ptr(), R, K, _ext.IN_DTYPE[x.dtype],
                         0 if kind == "mxfp4" else 1,
                         packed.data_ptr(), sblk.data_ptr(), _ext.stream())
    assert rc == 0, f"tfp4_quant_mx rc={rc}"
    return packed, sblk


def dequantize_nvfp4(packed: torch.Tensor, sblk: torch.Tensor, tensor_scale: float,
                     R: int, K: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """NVFP4 buffers -> plain f32 (R, K). This is the operand builder for
    verify()'s independent reference path (the matmul itself then runs in
    torch/cuBLAS BF16, a different library and kernel family entirely)."""
    L = _ext.core()
    if out is None:
        out = torch.empty(R, K, dtype=torch.float32, device=packed.device)
    rc = L.tfp4_dequant_f32(packed.data_ptr(), sblk.data_ptr(), tensor_scale,
                            R, K, out.data_ptr(), _ext.stream())
    assert rc == 0, f"tfp4_dequant_f32 rc={rc}"
    return out


def block_scales(sblk: torch.Tensor, R: int, K: int,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """Blocked UE4M3 scales -> plain f32 (R, K/16): the decoded block-scale
    byte values, WITHOUT the per-tensor scale (multiply by tensor_scale for
    the scales the codes were divided by)."""
    L = _ext.core()
    if out is None:
        out = torch.empty(R, K // 16, dtype=torch.float32, device=sblk.device)
    rc = L.tfp4_scales_f32(sblk.data_ptr(), R, K, out.data_ptr(), _ext.stream())
    assert rc == 0, f"tfp4_scales_f32 rc={rc}"
    return out


def gemm(packedA: torch.Tensor, sblkA: torch.Tensor, tsA: float,
         packedB: torch.Tensor, sblkB: torch.Tensor, tsB: float,
         out: torch.Tensor, M: int, N: int, K: int, out_dtype: str = "f16") -> None:
    """out (M, N) row-major = A(M, K) @ B(N, K)^T, both NVFP4 buffers.
    out_dtype: 'f16' | 'bf16' | 'f32'. `out` must be contiguous with exactly
    M*N elements of that dtype."""
    L = _ext.core()
    assert out.is_contiguous() and out.numel() == M * N and str(out.dtype).endswith(
        {"f16": "float16", "bf16": "bfloat16", "f32": "float32"}[out_dtype])
    rc = L.tfp4_gemm(packedA.data_ptr(), sblkA.data_ptr(), tsA,
                     packedB.data_ptr(), sblkB.data_ptr(), tsB,
                     out.data_ptr(), M, N, K, _ext.OUT_DTYPE[out_dtype], _ext.stream())
    assert rc == 0, f"tfp4_gemm rc={rc}"


def gemv(x: torch.Tensor, w: "NVFP4Tensor", out_dtype: str = "f32",
         out: torch.Tensor | None = None) -> torch.Tensor:
    """out (M, N) = x (M, K) @ dequant(w)^T for M <= 8 — the register-dequant
    GEMV (DIRECTIONS.md §2). Bandwidth-bound small-batch path: the packed
    e2m1 weights are streamed once and dequantized in registers, so the layer
    costs the FP4 weight bytes, not the BF16 ones. Activations are read
    exactly (f32/f16/bf16 — never quantized); accumulation is FP32."""
    L = _ext.core()
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K
    assert x.ndim == 2 and x.is_cuda and x.is_contiguous()
    assert 1 <= M <= 8, f"gemv serves M <= 8 (got {M})"
    if out is None:
        out = torch.empty(M, N, dtype=_ext.OUT_TORCH[out_dtype], device=x.device)
    rc = L.tfp4_gemv(x.data_ptr(), M, K, _ext.IN_DTYPE[x.dtype],
                     w.packed.data_ptr(), w.gemv_scales().data_ptr(), w.tensor_scale,
                     out.data_ptr(), N, _ext.OUT_DTYPE[out_dtype], _ext.stream())
    assert rc == 0, f"tfp4_gemv rc={rc}"
    return out


class NVFP4Tensor:
    """A row-major (R, K) tensor resident in NVFP4 (codes + block scales +
    per-tensor scale), in torch-owned buffers."""

    def __init__(self, packed: torch.Tensor, sblk: torch.Tensor, tensor_scale: float,
                 shape: tuple[int, int]):
        self.packed, self.sblk, self.tensor_scale, self.shape = packed, sblk, float(tensor_scale), shape
        self._gemv_scales: torch.Tensor | None = None

    def gemv_scales(self) -> torch.Tensor:
        """The block scales un-scattered to plain (R, K/16) row-major bytes —
        the GEMV path's weight-side scale layout (coalesced loads). Cached."""
        if self._gemv_scales is None:
            R, K = self.shape
            from . import _ext
            out = torch.empty(R, K // 16, dtype=torch.uint8, device=self.packed.device)
            rc = _ext.core().tfp4_scales_gemv(
                self.sblk.data_ptr(), R, K, out.data_ptr(), _ext.stream())
            assert rc == 0, f"tfp4_scales_gemv rc={rc}"
            self._gemv_scales = out
        return self._gemv_scales

    @classmethod
    def from_float(cls, x: torch.Tensor, tensor_scale: float | None = None):
        """Quantize x (R, K). tensor_scale=None picks the two-level NVFP4
        recipe scale amax/(6*448) (which puts block scales at the top of the
        UE4M3 range); pass 1.0 for block scales only."""
        assert x.ndim == 2 and x.is_contiguous()
        x = x.float() if x.dtype != torch.float32 else x
        if tensor_scale is None:
            amax = x.abs().amax().item()
            tensor_scale = max(amax / (6.0 * 448.0), 1e-30)
        packed, sblk = quantize_nvfp4(x, tensor_scale)
        return cls(packed, sblk, tensor_scale, tuple(x.shape))

    def dequantize(self) -> torch.Tensor:
        R, K = self.shape
        return dequantize_nvfp4(self.packed, self.sblk, self.tensor_scale, R, K)

    def matmul_t(self, w: "NVFP4Tensor", out_dtype: str = "f16",
                 out: torch.Tensor | None = None) -> torch.Tensor:
        """(M, K) @ (N, K)^T -> (M, N); self is the activation-side operand."""
        M, K = self.shape
        N, K2 = w.shape
        assert K == K2
        if out is None:
            out = torch.empty(M, N, dtype=_ext.OUT_TORCH[out_dtype], device=self.packed.device)
        gemm(self.packed, self.sblk, self.tensor_scale,
             w.packed, w.sblk, w.tensor_scale, out, M, N, K, out_dtype)
        return out

    def nbytes(self) -> int:
        R, K = self.shape
        return R * K // 2 + R * (K // 16)


class MXTensor:
    """A row-major (R, K) tensor in an MX format ('mxfp4' or 'mxfp8')."""

    def __init__(self, packed: torch.Tensor, sblk: torch.Tensor, kind: str,
                 shape: tuple[int, int]):
        self.packed, self.sblk, self.kind, self.shape = packed, sblk, kind, tuple(shape)

    @classmethod
    def from_float(cls, x: torch.Tensor, kind: str = "mxfp4"):
        assert x.ndim == 2 and x.is_contiguous()
        packed, sblk = quantize_mx(x.float(), kind)
        return cls(packed, sblk, kind, x.shape)

    def nbytes(self) -> int:
        R, K = self.shape
        per = K // 2 if self.kind == "mxfp4" else K
        return R * per + R * (K // 32)

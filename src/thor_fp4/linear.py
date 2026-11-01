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
from .ops import NVFP4Tensor

__all__ = ["Linear"]


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
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, out_dtype: str = "f16",
                 tensor_scale: float | None = None, act_tensor_scale: float = 1.0):
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
        self.out_dtype = out_dtype
        self.act_tensor_scale = float(act_tensor_scale)
        self._weight_fp4: NVFP4Tensor | None = None
        self._tensor_scale_requested = tensor_scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self._act_bufs: dict[tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

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
        return super()._apply(fn, recurse)

    def refresh_weight(self):
        """Re-quantize the master weight (call after in-place weight edits)."""
        w = self.weight.detach().contiguous()
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
            bufs = self._act_bufs.get(key)
            if bufs is None:
                packed = torch.zeros(ops.packed_bytes(M, self.in_features),
                                     dtype=torch.uint8, device=x.device)
                sblk = torch.zeros(ops.scale_bytes(M, self.in_features),
                                   dtype=torch.uint8, device=x.device)
                out = torch.empty(M, self.out_features,
                                  dtype={"f16": torch.float16, "bf16": torch.bfloat16,
                                         "f32": torch.float32}[self.out_dtype],
                                  device=x.device)
                bufs = (packed, sblk, out)
                self._act_bufs[key] = bufs
            packed, sblk, out = bufs
            ops.quantize_nvfp4(x, self.act_tensor_scale, packed, sblk)
            ops.gemm(packed, sblk, self.act_tensor_scale,
                     w.packed, w.sblk, w.tensor_scale,
                     out, M, self.out_features, self.in_features, self.out_dtype)
            if self.bias is not None:
                out += self.bias.to(out.dtype)
            return out

    def extra_repr(self) -> str:
        ts = (self._weight_fp4.tensor_scale if self._weight_fp4 is not None
              else self._tensor_scale_requested)
        return (f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, "
                f"out_dtype={self.out_dtype}, tensor_scale={ts}")

    def fp4_weight_nbytes(self) -> int:
        """Bytes of the resident weight (codes + block scales)."""
        return self.in_features * self.out_features // 2 + self.out_features * (self.in_features // 16)

# thor-fp4: NVFP4 linear layers on the Jetson AGX Thor's sm_110 tensor cores.
#
# Quick start:
#     import thor_fp4, torch
#     lin = thor_fp4.Linear(4096, 4096).cuda()
#     y = lin(torch.randn(256, 4096, device="cuda"))          # real FP4 GEMM
#     model = thor_fp4.convert(model, policy="auto")          # shape-aware
#     rep = thor_fp4.verify(model, x)                         # dual-path check
#     g = thor_fp4.capture_module(model, x)                   # CUDA graph
#
# Numbers, mechanisms and constraints: README.md and BENCHMARKS.md.

__version__ = "0.1.0"

from . import ops
from ._ext import cublaslt_version
from .linear import Linear
from .ops import NVFP4Tensor, MXTensor, gemm, quantize_nvfp4, quantize_mx, dequantize_nvfp4
from .quant import (QATLinear, QATSequential, calibrate_linear, mx_fake_quant,
                    nvfp4_fake_quant, qat, two_level_scale)
from .convert import auto_decision, calibrate, convert, load_shape_table
from .verify import ReferenceLinear, VerifyReport, verify, verify_linear
from .graphs import Graphed, capture_module
from .cascade import Cascade, cascade_sweep, confidence_margin, confidence_softmax

__all__ = [
    "Linear", "NVFP4Tensor", "MXTensor", "ops",
    "quantize_nvfp4", "quantize_mx", "dequantize_nvfp4", "gemm",
    "nvfp4_fake_quant", "mx_fake_quant", "qat", "two_level_scale",
    "QATLinear", "QATSequential", "calibrate_linear",
    "convert", "calibrate", "auto_decision", "load_shape_table",
    "verify", "verify_linear", "VerifyReport", "ReferenceLinear",
    "Graphed", "capture_module",
    "Cascade", "cascade_sweep", "confidence_margin", "confidence_softmax",
    "cublaslt_version", "__version__",
]

# _ext.py — ctypes bindings to the thor-fp4 CUDA extension(s).
#
# build/libthorfp4_core.so  (cuBLASLt path, always present — `make core`)
# build/libthorfp4_cutlass.so (own sm_110a kernels, optional — `make cutlass`)
#
# The C ABI is documented at the top of csrc/thor_fp4_core.cu.

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD = _ROOT / "build"

c_void = ctypes.c_void_p
i64 = ctypes.c_int64


def _load(name: str, optional: bool = False):
    path = _BUILD / name
    if not path.exists():
        if optional:
            return None
        # friendlier than an import error: build on first use (~40 s once)
        print(f"thor_fp4: {path} missing, building it (nvcc)…", file=sys.stderr)
        subprocess.run(["make", "-C", str(_ROOT), "core"], check=True)
    lib = ctypes.CDLL(str(path))
    _decl_core(lib)
    return lib


def _decl_core(L):
    L.tfp4_init.argtypes = [ctypes.c_size_t]
    L.tfp4_init.restype = ctypes.c_int
    L.tfp4_cublaslt_version.argtypes = []
    L.tfp4_cublaslt_version.restype = ctypes.c_size_t
    L.tfp4_pad_rows.argtypes = [i64]
    L.tfp4_pad_rows.restype = i64
    L.tfp4_packed_bytes.argtypes = [i64, i64]
    L.tfp4_packed_bytes.restype = i64
    L.tfp4_scale_bytes.argtypes = [i64, i64]
    L.tfp4_scale_bytes.restype = i64
    L.tfp4_mx_packed_bytes.argtypes = [i64, i64, ctypes.c_int]
    L.tfp4_mx_packed_bytes.restype = i64
    L.tfp4_mx_scale_bytes.argtypes = [i64, i64]
    L.tfp4_mx_scale_bytes.restype = i64
    L.tfp4_quant.argtypes = [c_void, i64, i64, ctypes.c_int, ctypes.c_float, c_void, c_void, c_void]
    L.tfp4_quant.restype = ctypes.c_int
    L.tfp4_quant_mx.argtypes = [c_void, i64, i64, ctypes.c_int, ctypes.c_int, c_void, c_void, c_void]
    L.tfp4_quant_mx.restype = ctypes.c_int
    L.tfp4_gemm.argtypes = [c_void, c_void, ctypes.c_float, c_void, c_void, ctypes.c_float,
                            c_void, i64, i64, i64, ctypes.c_int, c_void]
    L.tfp4_gemm.restype = ctypes.c_int
    L.tfp4_dequant_f32.argtypes = [c_void, c_void, ctypes.c_float, i64, i64, c_void, c_void]
    L.tfp4_dequant_f32.restype = ctypes.c_int
    L.tfp4_scales_f32.argtypes = [c_void, i64, i64, c_void, c_void]
    L.tfp4_scales_f32.restype = ctypes.c_int


def _decl_cutlass(L):
    L.tfc4_gemm.argtypes = [ctypes.c_int, c_void, c_void, c_void, c_void, c_void,
                            i64, i64, i64, ctypes.c_float, c_void]
    L.tfc4_gemm.restype = ctypes.c_int
    L.tfc4_n_variants.argtypes = []
    L.tfc4_n_variants.restype = ctypes.c_int
    L.tfc4_variant_name.argtypes = [ctypes.c_int]
    L.tfc4_variant_name.restype = ctypes.c_char_p


_core = None
_cutlass = None
_ws_bytes = 32 << 20


def core():
    global _core
    if _core is None:
        _core = _load("libthorfp4_core.so")
        rc = _core.tfp4_init(_ws_bytes)
        assert rc == 0, f"tfp4_init rc={rc}"
    return _core


def cutlass():
    """The optional own-kernel library (sm_110a CUTLASS builds). None if absent."""
    global _cutlass
    if _cutlass is None:
        path = _BUILD / "libthorfp4_cutlass.so"
        if not path.exists():
            return None
        _cutlass = ctypes.CDLL(str(path))
        _decl_cutlass(_cutlass)
    return _cutlass


def cublaslt_version() -> int:
    return int(core().tfp4_cublaslt_version())


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


IN_DTYPE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
OUT_DTYPE = {"f16": 0, "bf16": 1, "f32": 2}
OUT_TORCH = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}

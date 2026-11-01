#!/usr/bin/env python
# repro_nan6.py — R3-0: is the bias-address failure ABSOLUTE or RELATIVE to the
# other operands?
#   A  keep the bad fl.bias pointer, but pass W/SB/A/SA from far-away copies
#   B  keep a clean slab bias, but move the operands adjacent to the slab
#   C  six more fresh-segment-start bias tensors (absolute-address census)

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import thor_fp4  # noqa: E402
from thor_fp4 import _ext, ops  # noqa: E402
from thor_fp4.fused import FusedLinear, MX8Act  # noqa: E402
import repro_nan as R  # noqa: E402
import repro_nan5 as R5  # noqa: E402


def call(fl, ap, asp, wp, wsp, bias_t, dp, sf):
    L = _ext.fused()
    dp.zero_(); sf.zero_()
    rc = L.tfc4f_gemm(fl.variant_idx, ap.data_ptr(), asp.data_ptr(),
                      wp.data_ptr(), wsp.data_ptr(), bias_t.data_ptr(),
                      dp.data_ptr(), sf.data_ptr(), R5.M, R5.N, R5.K, 1.0,
                      _ext.stream())
    assert rc == 0, rc
    torch.cuda.synchronize()
    return sorted(R.sfd_bad(sf, R5.M, R5.N))


def main():
    fl, act, bias0, dp, sf = R5.setup_failing()
    rows0 = call(fl, act.packed, act.sblk, fl._w.packed, fl._w.sblk, bias0, dp, sf)
    print(f"baseline: {'clean' if not rows0 else f'BAD {rows0}'}  bias=0x{bias0.data_ptr():x}"
          f"  W=0x{fl._w.packed.data_ptr():x} A=0x{act.packed.data_ptr():x}")

    # nullbias: pass bias_ptr = nullptr through tfc4f_gemm_mode (no -5 check
    # there). If Sm90RowBroadcast's nullptr path is taken, NO gmem bias load
    # and NO smem staging happen at all.
    L = _ext.fused()
    import ctypes
    decl = None
    L.tfc4f_gemm_mode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                  ctypes.c_float, ctypes.c_void_p]
    L.tfc4f_gemm_mode.restype = ctypes.c_int
    for i in range(2):
        dp.zero_(); sf.zero_()
        rc = L.tfc4f_gemm_mode(fl.variant_idx, 0,
                               act.packed.data_ptr(), act.sblk.data_ptr(),
                               fl._w.packed.data_ptr(), fl._w.sblk.data_ptr(),
                               None, dp.data_ptr(), sf.data_ptr(),
                               R5.M, R5.N, R5.K, 1.0, _ext.stream())
        torch.cuda.synchronize()
        rows = sorted(R.sfd_bad(sf, R5.M, R5.N))
        print(f"nullbias call{i}: rc={rc} {'clean' if not rows else f'BAD {rows}'}")

    # A: bad bias + far-away operand copies
    far = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")  # push far
    wp = torch.empty_like(fl._w.packed); wp.copy_(fl._w.packed)
    wsp = torch.empty_like(fl._w.sblk); wsp.copy_(fl._w.sblk)
    ap = torch.empty_like(act.packed); ap.copy_(act.packed)
    asp = torch.empty_like(act.sblk); asp.copy_(act.sblk)
    del far
    rowsA = call(fl, ap, asp, wp, wsp, bias0, dp, sf)
    print(f"A bad-bias + far operands: {'clean' if not rowsA else f'BAD {rowsA}'}"
          f"  (W'=0x{wp.data_ptr():x} A'=0x{ap.data_ptr():x})")
    del wp, wsp, ap, asp

    # B: clean slab bias + operands adjacent to the slab
    nfloat = 1 << 18
    slab = torch.zeros(nfloat, dtype=torch.float32, device="cuda")
    sbias = slab[:R5.N]
    wp = torch.empty_like(fl._w.packed); wp.copy_(fl._w.packed)
    rowsB = call(fl, act.packed, act.sblk, wp, fl._w.sblk, sbias, dp, sf)
    print(f"B slab-bias + W adjacent:  {'clean' if not rowsB else f'BAD {rowsB}'}"
          f"  (slab=0x{slab.data_ptr():x} W'=0x{wp.data_ptr():x})")
    del slab, wp

    # C: fresh-segment-start bias census
    print("C fresh-segment bias tensors (bias ptr -> result):")
    for i in range(6):
        torch.cuda.empty_cache()
        b = torch.zeros(R5.N, dtype=torch.float32, device="cuda")
        spacer = torch.empty(9 << 20, dtype=torch.uint8, device="cuda")
        rows = call(fl, act.packed, act.sblk, fl._w.packed, fl._w.sblk, b, dp, sf)
        print(f"  bias=0x{b.data_ptr():x} (W-bias gap = "
              f"0x{fl._w.packed.data_ptr() - b.data_ptr():x}) -> "
              f"{'clean' if not rows else f'BAD {rows}'}")
        del b, spacer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

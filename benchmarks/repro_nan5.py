#!/usr/bin/env python
# repro_nan5.py — R3-0: bias-pointer address sweep inside the exact failing
# configuration (repro_nan4 rung 8: FusedLinear, kaiming weights, seed-7 x,
# all other buffer addresses unchanged). The bias values are zeros in every
# case; only the pointer moves. For each candidate address we report the bad
# row count and the pointer's alignment properties, including its offset from
# the nearest 2 MiB boundary — the failing rung-8 bias sat exactly on one.

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import thor_fp4  # noqa: E402
from thor_fp4 import _ext, ops  # noqa: E402
from thor_fp4.fused import FusedLinear, MX8Act  # noqa: E402
import repro_nan as R  # noqa: E402

M, K, N = 2048, 512, 1024


def setup_failing():
    """verbatim rung 8 up to (but not including) the fused call"""
    torch.manual_seed(0)
    fl = FusedLinear(K, N, bias=True, max_chunk=0).cuda()
    x = R.x_of(M, K)
    with torch.no_grad():
        fl._w = ops.MXTensor.from_float(fl.weight.detach().contiguous().float().cuda(), "mxfp4")
        packed = torch.zeros(ops.mx_packed_bytes(M, K, "mxfp8"), dtype=torch.uint8, device="cuda")
        sblk = torch.zeros(ops.mx_scale_bytes(M, K), dtype=torch.uint8, device="cuda")
        ops.quantize_mx(x, "mxfp8", packed, sblk)
        act = MX8Act(packed, sblk, (M, K))
    bias0 = fl.bias.float().cuda().contiguous()      # the rung-8 pointer
    dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(int(_ext.fused().tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
    return fl, act, bias0, dp, sf


def call(fl, act, bias_t, dp, sf):
    L = _ext.fused()
    dp.zero_(); sf.zero_()
    rc = L.tfc4f_gemm(fl.variant_idx,
                      act.packed.data_ptr(), act.sblk.data_ptr(),
                      fl._w.packed.data_ptr(), fl._w.sblk.data_ptr(),
                      bias_t.data_ptr(), dp.data_ptr(), sf.data_ptr(),
                      M, N, K, 1.0, _ext.stream())
    assert rc == 0, rc
    torch.cuda.synchronize()
    return sorted(R.sfd_bad(sf, M, N)), sorted(R.codes_bad(dp, M, N))


def describe(p):
    m2 = p % (2 << 20)
    return (f"0x{p:012x} mod2M=0x{m2:x} mod32K=0x{p % 32768:x} "
            f"mod16B={'y' if p % 16 == 0 else 'N'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep", choices=["sweep", "values", "inputs-guard"])
    args = ap.parse_args()
    fl, act, bias0, dp, sf = setup_failing()
    rows0, cbad0 = call(fl, act, bias0, dp, sf)
    print(f"# repro_nan5 baseline (fl.bias at its rung-8 address): "
          f"{'clean' if not rows0 else f'BAD rows={rows0}'}; bias={describe(bias0.data_ptr())}")

    if args.mode == "sweep":
        nfloat = 1 << 20                       # 4 MiB slab of f32
        slab = torch.zeros(nfloat, dtype=torch.float32, device="cuda")
        base = slab.data_ptr()
        # offsets (in floats) to test: coarse 64K steps; dense 16-float steps in
        # [0, 4096); and the exact offsets of every 2 MiB / 32 KiB boundary the
        # slab spans
        offs = set(range(0, nfloat, 16384))     # 64 KiB steps
        offs.update(range(0, 1024))             # 4 KiB dense at 16B granularity
        for boundary in (2 << 20, 32 << 10, 4 << 10):
            first = (boundary - (base % boundary)) % boundary
            for o in range(first, nfloat * 4, boundary):
                if o % 4 == 0:
                    offs.add(o // 4)
        print(f"slab base {describe(base)}; {len(offs)} offsets")
        nbad = 0
        for o in sorted(offs):
            view = slab[o:o + N]
            rows, _ = call(fl, act, view, dp, sf)
            tag = "clean" if not rows else f"BAD n={len(rows)} rows={rows[:5]}"
            if rows:
                nbad += 1
                print(f"  bias=slab+{o*4:8d}B  {tag}   [{describe(view.data_ptr())}]")
        print(f"summary: {nbad}/{len(offs)} bias addresses BAD")
    elif args.mode == "values":
        # does the corruption depend on the bias VALUES at a bad address?
        # re-run the baseline bad address with nonzero bias content
        b = fl.bias
        with torch.no_grad():
            for fill, name in ((0.0, "zeros"), (1.0, "ones"), (float("nan"), "nans")):
                b.fill_(fill)
                rows, cbad = call(fl, act, b.float().cuda(), dp, sf)
                print(f"  bias={name:6s} at 0x{b.data_ptr():x}: "
                      f"{'clean' if not rows else f'BAD rows={rows[:5]}'}")
    elif args.mode == "inputs-guard":
        # in the failing config, are A/SA corrupted in place? guard them
        G = 8192
        big_p = torch.full((act.packed.numel() + 2 * G,), 0xA5, dtype=torch.uint8, device="cuda")
        big_s = torch.full((act.sblk.numel() + 2 * G,), 0xA5, dtype=torch.uint8, device="cuda")
        big_p[G:G + act.packed.numel()] = act.packed
        big_s[G:G + act.sblk.numel()] = act.sblk
        L = _ext.fused()
        for i in range(2):
            dp.zero_(); sf.zero_()
            rc = L.tfc4f_gemm(fl.variant_idx,
                              big_p[G:G + act.packed.numel()].data_ptr(),
                              big_s[G:G + act.sblk.numel()].data_ptr(),
                              fl._w.packed.data_ptr(), fl._w.sblk.data_ptr(),
                              fl.bias.data_ptr(), dp.data_ptr(), sf.data_ptr(),
                              M, N, K, 1.0, _ext.stream())
            assert rc == 0, rc
            torch.cuda.synchronize()
            rows = sorted(R.sfd_bad(sf, M, N))
            post_a = int(((big_p[G:G + act.packed.numel()] == 0x7F) |
                          (big_p[G:G + act.packed.numel()] == 0xFF)).sum())
            post_s = sorted(R.sfd_bad(big_s[G:G + act.sblk.numel()], M, K))
            gp = int((big_p[:G] == 0xA5).sum()) + int((big_p[-G:] == 0xA5).sum())
            gs = int((big_s[:G] == 0xA5).sum()) + int((big_s[-G:] == 0xA5).sum())
            print(f"  call{i}: out={rows[:5] if rows else 'clean'}  post-call A NaN-bytes={post_a}"
                  f"  post-call SA 0xFF rows={post_s[:5] if post_s else 'clean'}  guards A={gp}/{2*G} S={gs}/{2*G}")


if __name__ == "__main__":
    main()

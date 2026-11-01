#!/usr/bin/env python
# repro_nan4.py — R3-0: a step ladder from the clean repro_nan3-style ABI
# call to the failing repro_nan2 deep_wrapper, one added ingredient per
# rung (run each rung in its own process so allocator history is exactly
# that rung's):
#   1  base: plain tensors, ABI call (known clean)
#   2  + FusedLinear module constructed first (CPU param -> .cuda())
#   3  + W quantized from fl.weight (float copy) instead of a plain tensor
#   4  + deep's allocation ORDER: W quantize before x/packed/sblk
#   5  + .cpu() scan syncs between quantize and the call
#   6  + decode_mx + f32 gemm + gelu + blkmax before the call
#   7  + dp/sf allocated LAST (after all reference tensors) — == deep
# Run: for i in 1..7: python repro_nan4.py --rung $i

import argparse
import subprocess
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


def rung(n):
    torch.manual_seed(0)
    x = R.x_of(M, K)
    fl = None
    W = None
    if n >= 2:
        fl = FusedLinear(K, N, bias=True, max_chunk=0).cuda()
    if n in (2, 4):
        # module exists, but W still comes from a plain tensor
        w = torch.randn(N, K, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(3)) * (K ** -0.5)
        if n == 2:
            W = ops.MXTensor.from_float(w, "mxfp4")
    if n >= 3:
        src = fl.weight.detach().contiguous().float().cuda() if n >= 3 else w
        W = ops.MXTensor.from_float(src, "mxfp4")
    bias = torch.zeros(N, device="cuda")
    packed = torch.zeros(ops.mx_packed_bytes(M, K, "mxfp8"), dtype=torch.uint8, device="cuda")
    sblk = torch.zeros(ops.mx_scale_bytes(M, K), dtype=torch.uint8, device="cuda")
    ops.quantize_mx(x, "mxfp8", packed, sblk)
    act = MX8Act(packed, sblk, (M, K))
    if n == 1:
        w = torch.randn(N, K, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(3)) * (K ** -0.5)
        W = ops.MXTensor.from_float(w, "mxfp4")
    if n >= 5:
        bad = R.sfd_bad(act.sblk, M, K)
        int((act.packed[: M * K] == 0x7F).sum())  # the scans deep does
        sorted(bad)
    if n >= 6:
        import repro_nan2 as R2
        A = R2.decode_mx(act.packed, act.sblk, M, K, e4m3=True)
        Wd = R2.decode_mx(W.packed, W.sblk, N, K, e4m3=False)
        pre = torch.nn.functional.gelu(A @ Wd.T + (fl.bias.float().cuda() if fl else bias))
        pre.view(M, N // 32, 32).abs().amax(-1)
    dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(int(_ext.fused().tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
    L = _ext.fused()
    v = fl.variant_idx if fl else L.tfc4f_variant_index(b"mx8out_gelu_128x128x256_c1")
    bias_arg = bias.data_ptr()
    if n >= 8 and fl is not None:
        bias_arg = fl.bias.float().cuda().contiguous().data_ptr()  # deep's exact expression
    rc = L.tfc4f_gemm(v, act.packed.data_ptr(), act.sblk.data_ptr(),
                      W.packed.data_ptr(), W.sblk.data_ptr(), bias_arg,
                      dp.data_ptr(), sf.data_ptr(), M, N, K, 1.0, _ext.stream())
    assert rc == 0, rc
    torch.cuda.synchronize()
    rows = sorted(R.sfd_bad(sf, M, N))
    cbad = R.codes_bad(dp, M, N)
    print(f"rung {n}: {'clean' if not rows else f'BAD sfNaN rows={rows[:8]} (n={len(rows)})'}"
          f"{'  codeNaN n=' + str(len(cbad)) if cbad else ''}")
    print(f"  ptrs: A=0x{act.packed.data_ptr():x} SA=0x{act.sblk.data_ptr():x} "
          f"W=0x{W.packed.data_ptr():x} SB=0x{W.sblk.data_ptr():x} "
          f"D=0x{dp.data_ptr():x} SFD=0x{sf.data_ptr():x} bias=0x{bias_arg:x}")


def rung9():
    import repro_nan2 as R2
    R2.deep_wrapper(M, K, N, "kaiming")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=0)
    args = ap.parse_args()
    if args.rung:
        if args.rung == 9:
            rung9()
        else:
            rung(args.rung)
        return
    for i in range(1, 10):
        subprocess.run([sys.executable, __file__, "--rung", str(i)], check=True)


if __name__ == "__main__":
    main()

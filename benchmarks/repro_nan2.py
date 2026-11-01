#!/usr/bin/env python
# repro_nan2.py — R3-0 deep bisection of the ue8m0-NaN SFD rows.
#
# Given the repro (repro_nan.py: single FusedLinear, M>=2048, deterministic
# rows with ALL 32 scale bytes = 0xFF, a subset also with e4m3-NaN codes),
# this script answers, in one run:
#   1. are the kernel INPUTS clean?  (A e4m3 codes, SA ue8m0 bytes, W, SB)
#   2. is it weight-value dependent? (kaiming init vs the trained scorer's)
#   3. do the affected rows correlate with all-zero 32-blocks of the f32
#      reference  gelu(x_q @ W_q^T + b)  (computed from the same decoded
#      operands the mainloop consumed)?
#   4. does it reproduce through the raw ABI with persistent torch buffers
#      (abi-reuse), fresh torch buffers (abi-fresh), or raw cudaMalloc
#      buffers (abi-cuda)?  [abi paths reuse repro_nan.py's implementations]

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import thor_fp4  # noqa: E402
from thor_fp4 import _ext, ops  # noqa: E402
from thor_fp4.fused import FusedLinear, MX8Act  # noqa: E402
from scorer import boxstate, get_trained  # noqa: E402
import repro_nan as R  # noqa: E402


def decode_mx(packed: torch.Tensor, sblk: torch.Tensor, M: int, K: int,
              e4m3: bool) -> torch.Tensor:
    """(codes, ue8m0 scales in 128x4 atom) -> f32 (M, K); mirrors test_fused."""
    if e4m3:
        f8 = packed[: M * K].view(M, K).view(torch.float8_e4m3fn).float()
    else:
        # e2m1 packed 2/byte: element 2j = low nibble, 2j+1 = high nibble
        # (thor_fp4_core.cu quant_mx: dst[i] = lo | (hi << 4))
        nib = packed[: M * K // 2].view(torch.uint8).view(M, K // 2).int()
        lo = (nib & 0xF).long()
        hi = ((nib >> 4) & 0xF).long()
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                            -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                           device=packed.device)
        f8 = torch.empty(M, K, device=packed.device)
        f8[:, 0::2] = lut[lo]
        f8[:, 1::2] = lut[hi]
    S = K // 32
    assert S % 4 == 0
    sb = sblk
    idx = torch.arange(M, device=sb.device)
    a = idx // 128
    r = idx % 128
    b = r // 32
    c = r % 32
    sidx = torch.arange(S, device=sb.device)
    d = sidx // 4
    e = sidx % 4
    lin = (a[:, None] * (S // 4) + d[None, :]) * 512 + 16 * c[:, None] \
        + 4 * b[:, None] + e[None, :]
    sbf = sb.view(-1)[lin.reshape(-1)].view(M, S)
    sf = torch.ldexp(torch.ones_like(sbf, dtype=torch.float32),
                     (sbf.to(torch.int16) - 127))
    return (f8.view(M, S, 32) * sf.unsqueeze(-1)).reshape(M, K)


def scan_inputs(tag, act: MX8Act, M, K):
    """NaN-producing bytes on the A side? map bad SA bytes to rows."""
    a_nan = int(((act.packed[: M * K] == 0x7F) | (act.packed[: M * K] == 0xFF)).sum()) \
        if act.shape[1] == K and act.packed.numel() >= M * K else -1
    sbad = R.sfd_bad(act.sblk, M, K)  # same atom layout, S=K/32
    print(f"  {tag}: A-code NaN bytes={a_nan}  SA 0xFF rows={sorted(sbad)}")
    return sorted(sbad)


def deep_wrapper(M, K, N, weights: str):
    print(f"[wrapper-deep] M={M} K={K} N={N} weights={weights}")
    fl = FusedLinear(K, N, bias=True, max_chunk=0).cuda()
    if weights == "scorer":
        bf, _ = get_trained()
        l0 = [m for m in bf if isinstance(m, torch.nn.Linear)][0]
        assert l0.in_features == K and l0.out_features == N
        with torch.no_grad():
            fl.weight.copy_(l0.weight)
            fl.bias.copy_(l0.bias)
    x = R.x_of(M, K)
    with torch.no_grad():
        fl._w = ops.MXTensor.from_float(fl.weight.detach().contiguous().float().cuda(), "mxfp4")
        packed = torch.zeros(ops.mx_packed_bytes(M, K, "mxfp8"), dtype=torch.uint8, device="cuda")
        sblk = torch.zeros(ops.mx_scale_bytes(M, K), dtype=torch.uint8, device="cuda")
        ops.quantize_mx(x, "mxfp8", packed, sblk)
        act_in = MX8Act(packed, sblk, (M, K))
        # scan inputs + weights
        inrows = scan_inputs("input", act_in, M, K)
        scan_inputs("weight", fl._w, N, K)
        # f32 reference from the decoded operands the mainloop consumed
        A = decode_mx(act_in.packed, act_in.sblk, M, K, e4m3=True)
        W = decode_mx(fl._w.packed, fl._w.sblk, N, K, e4m3=False)
        pre = torch.nn.functional.gelu(A @ W.T + fl.bias.float().cuda())
        blkmax = pre.view(M, N // 32, 32).abs().amax(-1)      # (M, S)
        zero_blocks = (blkmax == 0).sum(1)                     # per row
        tiny_blocks = (blkmax < 2.0 ** -126).sum(1)
        # run the fused kernel on exactly these inputs (bypass quantize)
        dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
        sf = torch.zeros(int(_ext.fused().tfc4f_sfd_bytes(M, N)),
                         dtype=torch.uint8, device="cuda")
        L = _ext.fused()
        rc = L.tfc4f_gemm(fl.variant_idx,
                          act_in.packed.data_ptr(), act_in.sblk.data_ptr(),
                          fl._w.packed.data_ptr(), fl._w.sblk.data_ptr(),
                          fl.bias.float().cuda().contiguous().data_ptr(),
                          dp.data_ptr(), sf.data_ptr(),
                          M, N, K, 1.0, _ext.stream())
        assert rc == 0, rc
        torch.cuda.synchronize()
        bad = R.sfd_bad(sf, M, N)
        rows = sorted(bad)
        cbad = R.codes_bad(dp, M, N)
        print(f"  ptrs: A=0x{act_in.packed.data_ptr():012x} SA=0x{act_in.sblk.data_ptr():012x} "
              f"W=0x{fl._w.packed.data_ptr():012x} SB=0x{fl._w.sblk.data_ptr():012x} "
              f"D=0x{dp.data_ptr():012x} SFD=0x{sf.data_ptr():012x}")
        print(f"  result: sfNaN rows (n={len(rows)})={rows}")
        print(f"         codeNaN rows (n={len(cbad)})={cbad}")
        # what did the kernel actually write for these rows' codes?
        dmat = dp[: M * N].view(M, N).view(torch.float8_e4m3fn).float()
        for r in rows[:6]:
            amax_codes = dmat[r].view(N // 32, 32).abs().amax(-1)
            nz = int((dmat[r] != 0).sum())
            print(f"    row {r}: code nonzero={nz}/{N}  ref blkmax[min,max]="
                  f"[{blkmax[r].min().item():.3e},{blkmax[r].max().item():.3e}]"
                  f"  code blkmax[min,max]=[{amax_codes.min().item():.3e},"
                  f"{amax_codes.max().item():.3e}]")
        print(f"  input-side SA bad rows: {inrows}")
        zr = torch.nonzero(zero_blocks).flatten().tolist()
        tr = torch.nonzero(tiny_blocks).flatten().tolist()
        print(f"  reference rows with >=1 exact-zero 32-block (n={len(zr)}): {zr[:40]}")
        print(f"  reference rows with >=1 subnormal-max block (n={len(tr)}): {tr[:40]}")
        inter_sf_z = sorted(set(rows) & set(zr))
        inter_c_z = sorted(set(cbad) & set(zr))
        print(f"  |sfNaN ∩ zero-block| = {len(inter_sf_z)}/{len(rows)}   "
              f"|codeNaN ∩ zero-block| = {len(inter_c_z)}/{len(cbad)}")
        if rows:
            r = rows[0]
            print(f"  row {r}: ref blkmax[0:8]={blkmax[r, :8].tolist()} "
                  f"zero_blocks={zero_blocks[r].item()}")
    del fl
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--weights", nargs="+", default=["kaiming", "scorer"])
    args = ap.parse_args()
    print(f"# repro_nan2  {boxstate()}")
    torch.manual_seed(0)
    for w in args.weights:
        for M in args.M:
            deep_wrapper(M, args.K, args.N, w)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# bench_gemv.py — DIRECTIONS.md §2: small-batch (M <= 8) FP4-weight inference.
#
# Three paths, same weights, L2 defeated by rotating through NROT weight
# copies (cold numbers; the hot-L2 case flatters both sides equally):
#   bf16 mm     x.bfloat16() @ W.bfloat16().T          (cuBLAS, the baseline)
#   gemv        thor_fp4.ops.gemv                       (register dequant,
#               activations consumed exactly, no quantize pass)
#   fp4 e2e     quantize activation + cuBLASLt NVFP4 GEMM (the tensor-core
#               path at the same small M — the DIRECTIONS §2 bar applies to
#               "vs BF16 at M <= 8", whichever path delivers it)
#
# Usage: .venv/bin/python benchmarks/bench_gemv.py [--quick]

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from thor_fp4 import ops  # noqa: E402
from scorer import boxstate  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
NROT = 8
SHAPES = [
    # (N, K): the scorer's layers + d1-ish wide layers
    (1024, 512), (1024, 1024), (64, 1024),
    (2048, 2048), (4096, 4096), (8192, 8192), (4096, 8192),
]
MS = [1, 2, 4, 8]


def bench(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def main():
    torch.manual_seed(0)
    # warm the GPU clock with ~1.5 s of BF16 matmuls
    a = torch.randn(2048, 2048, device="cuda")
    for _ in range(400):
        a @ a
    del a

    idx = [0]

    def rot(t):
        idx[0] = (idx[0] + 1) % len(t)
        return t[idx[0]]

    lines = [f"thor-fp4 small-M bench (DIRECTIONS.md sec.2) | {boxstate()}",
             f"cold: {NROT} rotated weight copies; alternating order; median of 5x100",
             "",
             f"{'N':>6} {'K':>6} {'M':>2} | {'bf16 us':>9} {'gemv us':>9} {'fp4e2e us':>9}"
             f" | {'gemv/bf16':>9} {'fp4/bf16':>8} {'gemv GB/s':>9} {'fp4 GB/s':>8}"]

    for (N, K) in SHAPES:
        wts = [ops.NVFP4Tensor.from_float(torch.randn(N, K, device="cuda") * 0.05)
               for _ in range(NROT)]
        wbs = [w.dequantize().bfloat16() for w in wts]
        w0 = wts[0]
        for M in MS:
            xs = [torch.randn(M, K, device="cuda") for _ in range(NROT)]
            xb = xs[0].bfloat16()
            pb = torch.zeros(ops.packed_bytes(M, K), dtype=torch.uint8, device="cuda")
            sb = torch.zeros(ops.scale_bytes(M, K), dtype=torch.uint8, device="cuda")
            out = torch.empty(M, N, dtype=torch.float16, device="cuda")

            def bf():
                return rot(xs).bfloat16() @ rot(wbs).t()

            def gv():
                return ops.gemv(rot(xs), rot(wts), "f32")

            def e2e():
                xx = rot(xs)
                ops.quantize_nvfp4(xx, 1.0, pb, sb)
                ops.gemm(pb, sb, 1.0, w0.packed, w0.sblk, w0.tensor_scale,
                         out, M, N, K)

            # alternate orders across 5 rounds, take medians
            res = {n: [] for n in ("bf", "gv", "e2e")}
            for _ in range(5):
                res["bf"].append(bench(bf))
                res["gv"].append(bench(gv))
                res["e2e"].append(bench(e2e))
                res["gv"].append(bench(gv))   # re-interleave
                res["bf"].append(bench(bf))
            med = {"bf": sorted(res["bf"])[len(res["bf"])//2], "gv": sorted(res["gv"])[len(res["gv"])//2], "e2e": sorted(res["e2e"])[len(res["e2e"])//2]}
            wb = N * K * 0.5625 + N * (K // 16)
            lines.append(f"{N:>6} {K:>6} {M:>2} | {med['bf']:>9.1f} {med['gv']:>9.1f}"
                         f" {med['e2e']:>9.1f} | {med['bf']/med['gv']:>8.2f}x"
                         f" {med['bf']/med['e2e']:>7.2f}x {wb/med['gv']/1e3:>9.0f}"
                         f" {wb/med['e2e']/1e3:>8.0f}")
        lines.append("")
    lines.append(f"end | {boxstate()}")
    out = "\n".join(lines)
    print(out)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gemv.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()

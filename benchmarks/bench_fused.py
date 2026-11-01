#!/usr/bin/env python
# bench_fused.py — ROUND 2: the fused-epilogue chain vs the two-pass chain.
#
# The two-pass MXFP8 chain per layer: quantize_mx kernel + CUTLASS mx8 GEMM
# (bias add in torch, gelu in torch). The fused chain: ONE kernel per layer
# (GEMM + bias + GELU + MXFP8 quantize in the epilogue). This measures the
# per-layer and end-to-end difference at the scorer's shapes, eager and
# CUDA-graphed, plus the per-shape breakdown at M=4096.
#
# Run: .venv/bin/python benchmarks/bench_fused.py | tee benchmarks/results/fused.txt

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scorer import boxstate, get_trained, DIMS  # noqa: E402
import thor_fp4  # noqa: E402
from thor_fp4 import ops  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def bench(fn, iters=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters * 1e3)  # us
    return sorted(ts)[len(ts) // 2]


def main():
    lines = [f"# thor-fp4 bench_fused  {boxstate()}",
             "# fused: 1 kernel/layer (gemm+bias+gelu+mx8-quantize epilogue)",
             "# two-pass: quantize_mx kernel + mx8 gemm + bias/gelu in torch"]
    a = torch.randn(2048, 2048, device="cuda")
    for _ in range(400):
        a @ a

    # per-shape layer comparison at M=4096 (the shapes convert() reasons about)
    lines.append("")
    lines.append("== single hidden layer, M=4096 (quantize included both sides) ==")
    lines.append(f"{'N':>6} {'K':>6} | {'two-pass us':>11} {'fused us':>9} {'speed':>6}")
    for (N, K) in [(1024, 512), (1024, 1024), (2048, 2048), (5120, 1280), (1280, 5120)]:
        fl = thor_fp4.FusedLinear(K, N).cuda()
        x = torch.randn(4096, K, device="cuda")
        lin = thor_fp4.Linear(K, N, bias=True, engine="cutlass:mx8").cuda()
        with torch.no_grad():
            lin.weight.copy_(fl.weight); lin.bias.copy_(fl.bias)
            t_two = bench(lambda: lin(x))       # quantize + gemm + bias (+gelu separate below)
            gelu = torch.nn.functional.gelu
            def two():
                y = lin(x)
                return gelu(y.float())
            t_two_g = bench(two)
            t_fused = bench(lambda: fl(x))
        lines.append(f"{N:>6} {K:>6} | {t_two_g:>11.1f} {t_fused:>9.1f} {t_two_g/t_fused:>5.2f}x")
        del fl, lin
        torch.cuda.empty_cache()

    # end-to-end scorer chain, eager + graphed, batch sweep
    bf, _ = get_trained()
    lins = [m for m in bf if isinstance(m, nn.Linear)]
    fused = thor_fp4.mx8_chain(*DIMS).cuda()
    two = []
    for i, l in enumerate(lins):
        lin = thor_fp4.Linear(l.in_features, l.out_features, bias=True,
                              engine="cutlass:mx8").cuda()
        with torch.no_grad():
            lin.weight.copy_(l.weight); lin.bias.copy_(l.bias)
        two.append(lin)
        if i < len(lins) - 1:
            two.append(nn.GELU())
    two = nn.Sequential(*two)
    # copy weights by order (both chains are Sequential in layer order)
    with torch.no_grad():
        flist = [m for m in fused if isinstance(m, (thor_fp4.FusedLinear, thor_fp4.Linear))]
        for f, l in zip(flist, lins):
            f.weight.copy_(l.weight)
            f.bias.copy_(l.bias)

    lines.append("")
    lines.append("== scorer chain 512->1024->1024->64 (mx8 both) ==")
    lines.append(f"{'B':>4} | {'two eager':>10} {'two graph':>10} | {'fused eager':>11} {'fused graph':>11} | {'best speed':>10}")
    for B in (1, 32, 256, 4096):
        x = torch.randn(B, DIMS[0], device="cuda")
        with torch.no_grad():
            te = bench(lambda: two(x))
            fe = bench(lambda: fused(x))
        try:
            gt = thor_fp4.capture_module(two, x)
            gf = thor_fp4.capture_module(fused, x)
            with torch.no_grad():
                tg = bench(lambda: gt(x))
                fg = bench(lambda: gf(x))
        except Exception as e:
            print(f"# graph capture failed at B={B}: {e}")
            tg = fg = float("nan")
        best = min(tg, fg)
        speed = max(te, tg) / best
        lines.append(f"{B:>4} | {te:>9.1f}u {tg:>9.1f}u | {fe:>10.1f}u {fg:>10.1f}u | {speed:>9.2f}x")
    lines.append("")
    lines.append(f"# {boxstate()}")
    out = "\n".join(lines)
    print(out)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fused.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()

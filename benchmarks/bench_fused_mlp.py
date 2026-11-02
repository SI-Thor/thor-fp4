#!/usr/bin/env python
# bench_fused_mlp.py — R3-x: fused activation quantization, accounted.
#   A. per-shape GB/s saved: the bytes the two-pass chain moves that the
#      fused epilogue does not (bf16 D write + gelu pass + quantize pass),
#      over the measured time difference, M=4096 layer shapes;
#   B. end-to-end SwiGLU-class MLP block (the d1/Seed-OSS layer shape
#      2048 -> 8x8192 gate/up -> 2048): fused GatedMLP vs the two-pass
#      equivalent, eager and one CUDA graph, B in {256, 4096};
#   C. R3-z measurement on the same stage: the launch-overhead share of a
#      full fused step, eager vs one graph, B=1..8.
# Run: .venv/bin/python benchmarks/bench_fused_mlp.py 2>&1 | tee benchmarks/results/fused-mlp.txt

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scorer import boxstate  # noqa: E402
import thor_fp4  # noqa: E402
from thor_fp4.fused import GatedMLP  # noqa: E402

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


class TwoPassGated(nn.Module):
    """the same SwiGLU block with separate quantize/silu/mul passes"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = thor_fp4.Linear(d_model, d_ff, bias=True, engine="cutlass:mx8").cuda()
        self.up = thor_fp4.Linear(d_model, d_ff, bias=True, engine="cutlass:mx8").cuda()
        self.down = thor_fp4.Linear(d_ff, d_model, bias=True, engine="cutlass:mx8").cuda()

    def forward(self, x):
        h = F.silu(self.gate(x).float()) * self.up(x).float()
        return self.down(h)


def main():
    lines = [f"# thor-fp4 bench_fused_mlp (R3-x)  {boxstate()}"]
    a = torch.randn(2048, 2048, device="cuda")
    for _ in range(200):
        a @ a

    # ---- A. GB/s saved per layer shape (M=4096) --------------------------
    lines += ["", "== A. activation bytes the fused epilogue does not move, M=4096 ==",
              f"{'N':>6} {'K':>6} | {'two us':>8} {'fused us':>9} {'dt us':>7} "
              f"{'avoided MB':>10} {'GB/s':>7}"]
    M = 4096
    for (N, K) in [(1024, 512), (1024, 1024), (2048, 2048), (5120, 1280), (1280, 5120)]:
        fl = thor_fp4.FusedLinear(K, N).cuda()
        lin = thor_fp4.Linear(K, N, bias=True, engine="cutlass:mx8").cuda()
        with torch.no_grad():
            lin.weight.copy_(fl.weight)
            lin.bias.copy_(fl.bias)
        x = torch.randn(M, K, device="cuda")

        def two():
            y = lin(x)
            return thor_fp4.ops.quantize_mx(F.gelu(y.float()), "mxfp8")

        t_two = bench(two)
        t_fused = bench(lambda: fl(x))
        # bytes the two-pass moves that the fused path does not:
        #   bf16 D write 2MN + gelu read/write (2+4)MN
        #   + quantize read/write 4MN + MN + MN/32
        avoided = M * N * (2 + 6 + 4 + 1 + 1 / 32)
        gbs = avoided / (t_two - t_fused) / 1e3 if t_two > t_fused else float("nan")
        lines.append(f"{N:>6} {K:>6} | {t_two:>8.1f} {t_fused:>9.1f} "
                     f"{t_two - t_fused:>7.1f} {avoided / 2**20:>10.1f} {gbs:>7.0f}")
        del fl, lin, x
        torch.cuda.empty_cache()

    # ---- B. gated MLP block end-to-end ------------------------------------
    lines += ["", "== B. SwiGLU block 2048 -> 2x8192 -> 2048 (d1/Seed-OSS shape) ==",
              f"{'B':>5} | {'two eager':>10} {'two graph':>10} | {'fused eager':>11} "
              f"{'fused graph':>11} | eager/graph speed"]
    d_model, d_ff = 2048, 8192
    fused = GatedMLP(d_model, d_ff).cuda()
    two = TwoPassGated(d_model, d_ff)
    with torch.no_grad():
        two.gate.weight.copy_(fused.gate.weight); two.gate.bias.copy_(fused.gate.bias)
        two.up.weight.copy_(fused.up.weight); two.up.bias.copy_(fused.up.bias)
        two.down.weight.copy_(fused.down.weight); two.down.bias.copy_(fused.down.bias)
    for B in (1, 8, 256, 4096):
        x = torch.randn(B, d_model, device="cuda")
        with torch.no_grad():
            te = bench(lambda: two(x), iters=30 if B >= 4096 else 100)
            fe = bench(lambda: fused(x), iters=30 if B >= 4096 else 100)
            tg = fg = float("nan")
            try:
                gt = thor_fp4.capture_module(two, x)
                gf = thor_fp4.capture_module(fused, x)
                with torch.no_grad():
                    tg = bench(lambda: gt(x), iters=30 if B >= 4096 else 100)
                    fg = bench(lambda: gf(x), iters=30 if B >= 4096 else 100)
            except Exception as e:  # noqa: BLE001
                print(f"# graph capture failed at B={B}: {e}")
        lines.append(f"{B:>5} | {te:>9.1f}u {tg:>9.1f}u | {fe:>10.1f}u {fg:>10.1f}u |"
                     f" {te / fe:>5.2f}x {tg / fg if fg == fg else float('nan'):>5.2f}x")

    # ---- C. R3-z: launch-overhead share of one fused step ----------------
    lines += ["", "== C. launch overhead of one fused step (gated block), eager vs 1 graph ==",
              f"{'B':>5} | {'eager us':>9} {'graph us':>9} | overhead us (share)"]
    for B in (1, 2, 4, 8):
        x = torch.randn(B, d_model, device="cuda")
        with torch.no_grad():
            fe = bench(lambda: fused(x), iters=200)
            gf = thor_fp4.capture_module(fused, x)
            fg = bench(lambda: gf(x), iters=200)
        lines.append(f"{B:>5} | {fe:>9.1f} {fg:>9.1f} | {fe - fg:>7.1f} "
                     f"({(fe - fg) / fe * 100:>4.1f}%)")

    lines += ["", f"# {boxstate()}"]
    out = "\n".join(lines)
    print(out)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fused-mlp.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()

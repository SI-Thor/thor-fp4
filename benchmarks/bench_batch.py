# bench_batch.py — the batched-decision benchmark: tile-scorer latency and
# throughput at N = 1..256, BF16 vs NVFP4, eager vs CUDA-graphed.
#
# Run: .venv/bin/python benchmarks/bench_batch.py 2>&1 | tee benchmarks/results/batch.txt

import statistics
import sys
import time

import torch

from scorer import boxstate, get_trained, test_batch, to_bf16, to_real_fp4

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
import thor_fp4  # noqa: E402


def warm_clock(sec=1.6):
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        x @ x
    torch.cuda.synchronize()


def timeit(fn, iters=30, rounds=5):
    gms = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        gms.append(e0.elapsed_time(e1) / iters)
    return statistics.median(gms)


bf, _ = get_trained()
x_full = test_batch(4096)
bf16_m = to_bf16(bf)
fp4_m = to_real_fp4(bf)


def fp_chain(t):
    outs = t
    for mod in fp4_m:
        outs = mod(outs) if isinstance(mod, thor_fp4.Linear) else mod(outs.float())
    return outs


print(f"# thor-fp4 bench_batch  cublasLt={thor_fp4.cublaslt_version()}  {boxstate()}")
print(f"# {'B':>4} | {'bf16 eager':>11} {'bf16 graph':>11} | {'fp4 eager':>11} {'fp4 graph':>11} | {'best dec/s':>10}")
for B in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    xb = x_full[:B].contiguous()
    xh = xb.half()

    def bf_eager():
        with torch.no_grad():
            return bf16_m(xb.bfloat16())

    def fp_eager():
        with torch.no_grad():
            return fp_chain(xh)

    with torch.no_grad():
        g_bf = thor_fp4.capture_module(bf16_m, xb.bfloat16())
        g_fp = thor_fp4.Graphed(fp_chain, xh)

    warm_clock(0.5)
    be = timeit(bf_eager)
    bg = timeit(lambda: g_bf(xb.bfloat16()))
    fe = timeit(fp_eager)
    fg = timeit(lambda: g_fp(xh))
    best = max(B / (m * 1e-3) for m in (be, bg, fe, fg))
    print(f"{B:>4} | {be*1e3:>8.1f} us {bg*1e3:>8.1f} us | "
          f"{fe*1e3:>8.1f} us {fg*1e3:>8.1f} us | {best:>10.0f}")
print(f"# {boxstate()}")

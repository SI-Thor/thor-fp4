# bench_gemm.py — GEMM-level measurements that feed BENCHMARKS.md and the
# shape-aware auto policy (writes src/thor_fp4/shape_table.json).
#
# Sections:
#   A. big-GEMM throughput: nvfp4 (library path) vs bf16 (torch mm) at the
#      cube/SDXL/decision shapes, bare GEMM (pre-quantized operands).
#   B. per-shape end-to-end: quantize(f16 act) + GEMM vs bf16 mm — the honest
#      per-layer comparison the auto policy reads (what a converted layer
#      actually costs, including its activation quantization).
#
# Box-state discipline: clock warmed before every timed section, loadavg and
# mem beside every table, median of rounds.
#
# Run: .venv/bin/python benchmarks/bench_gemm.py 2>&1 | tee benchmarks/results/gemm.txt

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import thor_fp4  # noqa: E402
from thor_fp4 import ops  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def boxstate() -> str:
    load = open("/proc/loadavg").read().split()[0]
    avail = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 2**20
    try:
        clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip().replace("\n", " ")
    except Exception:
        clk = "?"
    return f"load={load} mem_avail={avail:.0f}GiB sm={clk}"


def warm_clock(sec=1.6):
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        x @ x
    torch.cuda.synchronize()


def timeit(fn, iters=20, rounds=5):
    """median gpu_ms per call"""
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


def timeit_pair(fa, fb, iters=20, rounds=5):
    """time two variants in alternating order within each round, so slow box
    drift between seats averages out of the ratio (the lab's sweeps did the
    same by interleaving variants inside every round)"""
    a, b = [], []
    for r in range(rounds):
        first, second = (fa, fb) if r % 2 == 0 else (fb, fa)
        fa_list, fb_list = (a, b) if r % 2 == 0 else (b, a)
        for fn, acc in ((first, fa_list), (second, fb_list)):
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(iters):
                fn()
            e1.record()
            torch.cuda.synchronize()
            acc.append(e0.elapsed_time(e1) / iters)
    return statistics.median(a), statistics.median(b)


print(f"# thor-fp4 bench_gemm  torch={torch.__version__} cublasLt={thor_fp4.cublaslt_version()}")
print(f"# {boxstate()}")

# ---------------------------------------------------------------------------
# A. big-GEMM throughput (bare GEMM, pre-quantized operands)
# ---------------------------------------------------------------------------
print("\n== A. bare GEMM: nvfp4 (library) vs bf16 (torch mm), ms / TFLOPS ==")
print(f"# {'M':>6} {'N':>6} {'K':>6} | {'bf16 ms':>9} {'TFLOPS':>7} | {'fp4 ms':>9} {'TFLOPS':>7} | {'fp4/bf16':>8}   {boxstate()}")
A_SHAPES = [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
            (4096, 5120, 5120), (1024, 8192, 8192), (256, 8192, 8192)]
for M, N, K in A_SHAPES:
    xb = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    wb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    xa = thor_fp4.NVFP4Tensor.from_float(torch.randn(M, K, device="cuda"))
    wa = thor_fp4.NVFP4Tensor.from_float(torch.randn(N, K, device="cuda"))
    outf = torch.empty(M, N, dtype=torch.float16, device="cuda")
    warm_clock(1.0)
    bf_ms, fp_ms = timeit_pair(lambda: xb @ wb.t(),
                               lambda: xa.matmul_t(wa, out_dtype="f16", out=outf))
    fl = 2 * M * N * K
    print(f"{M:>6} {N:>6} {K:>6} | {bf_ms:9.3f} {fl/bf_ms/1e9:7.0f} | {fp_ms:9.3f} {fl/fp_ms/1e9:7.0f} | {bf_ms/fp_ms:8.2f}x")
    del xb, wb, xa, wa, outf
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# B. per-shape end-to-end (what a converted layer costs): quantize(f16)+gemm
#    vs bf16 mm. This is the table the auto policy reads.
# ---------------------------------------------------------------------------
SDXL = [(960, 320), (320, 960), (2560, 320), (320, 2560),
        (1920, 640), (640, 1920), (5120, 1280), (1280, 5120),
        (10240, 5120), (5120, 8192), (8192, 8192), (4096, 2048),
        (512, 2048), (1024, 2048), (2048, 2048), (256, 512), (64, 5120)]
MS = [4096, 256]

rows = {}
for M in MS:
    print(f"\n== B. end-to-end layer cost at M={M}: fp4(quant f16 + gemm) vs bf16 mm ==")
    print(f"# {'N':>6} {'K':>6} | {'bf16 ms':>9} | {'fp4 ms':>9} {'quant':>8} {'gemm':>8} | {'fp4/bf16':>8}   {boxstate()}")
    for N, K in SDXL:
        xb = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        x16 = xb.half()
        wb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        wa = thor_fp4.NVFP4Tensor.from_float(torch.randn(N, K, device="cuda"))
        packed = torch.zeros(ops.packed_bytes(M, K), dtype=torch.uint8, device="cuda")
        sblk = torch.zeros(ops.scale_bytes(M, K), dtype=torch.uint8, device="cuda")
        out = torch.empty(M, N, dtype=torch.float16, device="cuda")

        def fp4_layer():
            ops.quantize_nvfp4(x16, 1.0, packed, sblk)
            ops.gemm(packed, sblk, 1.0, wa.packed, wa.sblk, wa.tensor_scale,
                     out, M, N, K, "f16")

        warm_clock(0.6)
        bf_ms, fp_ms = timeit_pair(lambda: xb @ wb.t(), fp4_layer)
        q_ms = timeit(lambda: ops.quantize_nvfp4(x16, 1.0, packed, sblk))
        g_ms = fp_ms - q_ms
        wins = fp_ms < bf_ms
        rows.setdefault(f"{N}x{K}", {})[str(M)] = {
            "bf16_ms": round(bf_ms, 5), "fp4_ms": round(fp_ms, 5),
            "quant_ms": round(q_ms, 5), "fp4_wins": bool(wins)}
        print(f"{N:>6} {K:>6} | {bf_ms:9.3f} | {fp_ms:9.3f} {q_ms:8.3f} {max(g_ms,0):8.3f} | "
              f"{(bf_ms/fp_ms):8.2f}x {'' if wins else '(bf16 wins)'}")
        del xb, x16, wb, wa, packed, sblk, out
        torch.cuda.empty_cache()

# derive the crossover rule from the M=4096 measurements (the
# serving-heavy case): the (min_k, min_n) corner that separates every fp4
# winner from every bf16 winner among the measured shapes
def verdict(row):
    return row["fp4_wins"]
m4096 = [(int(k.split("x")[1]), int(k.split("x")[0]), v["4096"])
         for k, v in rows.items() if "4096" in v]
winners = [(K, N) for K, N, v in m4096 if verdict(v)]
losers = [(K, N) for K, N, v in m4096 if not verdict(v)]
min_n = min(N for _, N in winners) if winners else 8192
cands = sorted({K for K, N in winners if N >= min_n})
min_k = next((k for k in cands
              if not any(Lk >= k and Ln >= min_n for Lk, Ln in losers)), 4096)
table = {"generated": time.strftime("%Y-%m-%d %H:%M"), "box": boxstate(),
         "crossover": {"min_k": int(min_k), "min_n": int(min_n)}, "shapes": rows}
outp = ROOT / "src/thor_fp4/shape_table.json"
outp.write_text(json.dumps(table, indent=1))
print(f"\n# shape table -> {outp}  crossover: min_k={min_k} min_n={min_n}")
print(f"# winners(K,N)={sorted(winners)}")
print(f"# {boxstate()}")

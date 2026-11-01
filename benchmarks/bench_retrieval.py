#!/usr/bin/env python
# bench_retrieval.py — DIRECTIONS.md §3: retrieval as a large-K FP4 GEMM.
#
# A tile/pincher retrieval tick is queries x library^T with K = the embedding
# dimension (2048/4096 — exactly the K >= 2048 regime where NVFP4 runs at
# 371-456 TFLOPS). The library stays resident in NVFP4 (3.56x more entries
# per GB); the top-K candidates are re-ranked in BF16 where exactness
# matters. Measured here: end-to-end recall@k of FP4-sweep + BF16 re-rank
# against an exact BF16 sweep, and throughput (queries/s).
#
# The GEMM is chunked over the library (262144 entries per chunk) so the
# score buffer stays small; each chunk's top-K merges into a running top-K.
# Cosine retrieval: unit-normalized embeddings, dot-product scores.
#
# Run (heavy: multi-GB resident library — runs under the heavy lock):
#   .venv/bin/python benchmarks/bench_retrieval.py | tee benchmarks/results/retrieval.txt

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scorer import boxstate  # noqa: E402
from thor_fp4 import ops  # noqa: E402

N_ENTRIES = 1_000_000
CHUNK = 262_144          # library rows per GEMM
RERANK = 64              # BF16 re-rank depth (= top-K kept per chunk)
TOPK = (1, 10, 64)
DIMS = (2048, 4096)
QUERIES = (1, 32, 256)


def normalized(*shape, gen):
    e = torch.randn(*shape, generator=gen).cuda()
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def topk_merge(vals, idxs, k):
    """merge per-chunk (values, global indices) -> running top-k"""
    v = torch.cat([vals, vals.new_empty(0)]) if isinstance(vals, list) else vals
    return None


def chunked_fp4_sweep(qt, lib, k):
    """FP4 sweep: quantized queries x NVFP4 library, chunked; running top-k.
    Returns (values (Q,k) f32, indices (Q,k))."""
    Q, D = qt.shape
    best_v, best_i = None, None
    packed = torch.zeros(ops.packed_bytes(Q, D), dtype=torch.uint8, device="cuda")
    sblk = torch.zeros(ops.scale_bytes(Q, D), dtype=torch.uint8, device="cuda")
    ops.quantize_nvfp4(qt.float(), 1.0, packed, sblk)
    for c0 in range(0, lib.shape[0], CHUNK):
        ce = min(CHUNK, lib.shape[0] - c0)
        # slice the chunk's rows out of the packed/scale buffers (views;
        # unbounded ends keep tail-tile reads in-bounds — the next chunk's
        # bytes are garbage but never written to this chunk's outputs)
        wt = ops.NVFP4Tensor(
            lib.packed[ops.packed_bytes(c0, D):],
            lib.sblk[ops.scale_bytes(c0, D):],
            lib.tensor_scale, (ce, D))
        out = torch.empty(Q, ce, dtype=torch.float16, device="cuda")
        ops.gemm(packed, sblk, 1.0, wt.packed, wt.sblk, wt.tensor_scale,
                 out, Q, ce, D, "f16")
        v, i = out.topk(k, dim=1)          # native-dtype topk (the .float() cast doubles it)
        i = i + c0
        if best_v is None:
            best_v, best_i = v, i
        else:
            v2 = torch.cat([best_v, v], 1)
            i2 = torch.cat([best_i, i], 1)
            best_v, sel = v2.topk(k, dim=1)
            best_i = i2.gather(1, sel)
    return best_v, best_i


def chunked_bf16_sweep(qb, libb, k):
    Q = qb.shape[0]
    best_v, best_i = None, None
    for c0 in range(0, libb.shape[0], CHUNK):
        ce = min(CHUNK, libb.shape[0] - c0)
        out = qb @ libb[c0:c0 + ce].t()          # bf16 in, f32 accum
        v, i = out.topk(k, dim=1)
        i = i + c0
        if best_v is None:
            best_v, best_i = v, i
        else:
            v2 = torch.cat([best_v, v], 1)
            i2 = torch.cat([best_i, i], 1)
            best_v, sel = v2.topk(k, dim=1)
            best_i = i2.gather(1, sel)
    return best_v, best_i


def rerank(qb, libb, idx, k):
    rows = libb[idx.reshape(-1)].reshape(idx.shape[0], idx.shape[1], -1)
    scores = torch.bmm(qb.unsqueeze(1), rows.transpose(1, 2)).squeeze(1)
    v, sel = scores.topk(k, dim=1)
    return v, idx.gather(1, sel)


def bench(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def main():
    gen = torch.Generator(device="cpu").manual_seed(1234)
    lines = [f"# thor-fp4 bench_retrieval (DIRECTIONS.md sec.3) | {boxstate()}",
             f"# {N_ENTRIES} entries, unit-normalized; chunk {CHUNK}; re-rank depth {RERANK}",
             "# truth: exact BF16 sweep. Two query regimes: RANDOM (adversarial:",
             "# all top scores within FP4 noise, so the top-K SET reshuffles) and",
             "# PLANTED (each query = a library entry + noise: the true nearest",
             "# neighbor is well-separated, as in real retrieval).",
             ""]
    for D in DIMS:
        lib = normalized(N_ENTRIES, D, gen=gen)
        lib4 = ops.NVFP4Tensor.from_float(lib.float())
        libb = lib.bfloat16()
        del lib
        torch.cuda.empty_cache()
        gb4 = lib4.nbytes() / 1e9
        lines.append(f"dim {D}: resident NVFP4 library {gb4:.2f} GB "
                     f"(BF16 {libb.numel()*2/1e9:.2f} GB, {2*libb.numel()/lib4.nbytes():.2f}x)")
        queries = {}
        for Q in QUERIES:
            q = normalized(Q, D, gen=gen)
            queries[Q] = q
        # planted regime: query i = library entry pick[i] + 0.1 noise
        picks = torch.randint(0, N_ENTRIES, (max(QUERIES),), generator=gen).cuda()
        planted = {}
        for Q in QUERIES:
            base = libb[picks[:Q]].float()
            n = torch.randn(Q, D, generator=gen).cuda()
            pq = base + 0.1 * n / D ** 0.5
            planted[Q] = (pq / pq.norm(dim=-1, keepdim=True)), picks[:Q]
        for Q, q in queries.items():
            qt, qb = q, q.bfloat16()

            def fp4_path():
                v, i = chunked_fp4_sweep(qt, lib4, RERANK)
                return rerank(qb, libb, i, RERANK)

            def bf16_path():
                v, i = chunked_bf16_sweep(qb, libb, RERANK)
                return rerank(qb, libb, i, RERANK)

            t4 = bench(fp4_path)
            tb = bench(bf16_path)
            v4, i4 = fp4_path()
            vb, ib = bf16_path()
            recalls = []
            for k in TOPK:
                r = torch.tensor(
                    [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(i4[:, :k], ib[:, :k])],
                    device="cuda").mean().item()
                recalls.append(f"recall@{k}={100*r:.2f}%")
            lines.append(f"  Q={Q:3d}: fp4 {t4*1e3:8.1f} ms ({Q/t4:7.0f} q/s)   "
                         f"bf16 {tb*1e3:8.1f} ms ({Q/tb:7.0f} q/s)   speed {tb/t4:4.2f}x   "
                         + " ".join(recalls))
        # planted: does the FP4 sweep + BF16 re-rank find the true neighbor?
        lines.append("  planted (true neighbor well-separated):")
        for Q, (pq, true_idx) in planted.items():
            pqb = pq.bfloat16()
            v4, i4 = rerank(pqb, libb, chunked_fp4_sweep(pq, lib4, RERANK)[1], RERANK)
            hit1 = (i4[:, 0] == true_idx).float().mean().item()
            hit64 = (i4 == true_idx[:, None]).any(1).float().mean().item()
            t4 = bench(lambda: rerank(pqb, libb, chunked_fp4_sweep(pq, lib4, RERANK)[1], RERANK))
            lines.append(f"  Q={Q:3d}: top-1 finds the true entry {100*hit1:5.1f}%   "
                         f"in top-{RERANK}: {100*hit64:5.1f}%   fp4 path {t4*1e3:8.1f} ms ({Q/t4:7.0f} q/s)")
        del lib4, libb, queries
        torch.cuda.empty_cache()
        lines.append("")
    lines.append(f"# {boxstate()}")
    out = "\n".join(lines)
    print(out)
    (Path(__file__).resolve().parent / "results" / "retrieval.txt").write_text(out + "\n")


if __name__ == "__main__":
    main()

# BENCHMARKS.md — thor-fp4 measured numbers

Every table below was measured on this box and carries its state beside it.
Raw outputs live in `benchmarks/results/`; the drivers in `benchmarks/`
reproduce them (`bench_gemm.py`, `bench_quality.py`, `bench_batch.py`,
`bench_cascade.py`, `bench_cutlass_suite.sh`).

**Box:** Jetson AGX Thor — NVIDIA Thor GPU, sm_110 (GB10), 20 SMs, max clock
1,575 MHz; driver 595.78, CUDA 13.2.86, cuBLASLt 13.4.1 (`130401`);
kernel 6.8.12-1021-tegra, 14 cores, 122 GB unified memory (no swap), power
mode MAXN. The box is shared with other seats (load in every table's header
is 1-minute loadavg out of 14 cores); the lab's cleaner-locked references
are cited where my runs were noisier. Timing discipline: GPU clock warmed
~1–1.6 s of BF16 matmuls before every timed section, variants timed in
alternating order, median of 5 rounds.

Where each section's numbers come from and what they mean:

## 1. Bare GEMM throughput — the tensor-core FP4 rate

1. **Hardware fact:** the tcgen05 units take NVFP4 (e2m1 + UE4M3 per-16
   block scales) directly; the FP4 datapath is ~4× the BF16 rate.
2. **Why the previous path could not use it:** going through torch (no NVFP4
   matmul) or dequantize-to-BF16 never touches the FP4 units — measured at
   13.7 TFLOPS in the lab's round 1, slower than plain BF16.
3. **What this path does:** device kernels encode activations/weights into
   the tile layout, cuBLASLt runs the block-scaled TN GEMM, and the
   operand-swap trick returns row-major output with no transpose kernel.
4. **Numbers** (`benchmarks/results/gemm.txt`, section A; pre-quantized
   operands, f16 output, load 6.8–9.0, 109 GB available):

| M×N×K | BF16 (torch mm) | NVFP4 (library) | NVFP4 TFLOPS | FP4/BF16 speed |
|---|---|---|---|---|
| 2048³ | 0.354 ms (49 TF)* | **0.042 ms** | **405** | 8.3×* |
| 4096³ | 2.349 ms (59 TF)* | 0.707 ms | 195 | 3.3× |
| 8192³ | 27.9 ms (39 TF)* | 5.21 ms | 211 | 5.4× |
| 4096×5120×5120 | 1.834 ms (117 TF) | 0.583 ms | 368 | 3.1× |
| 1024×8192×8192 | 1.211 ms (113 TF) | 0.360 ms | 382 | 3.4× |
| 256×8192×8192 | 0.616 ms (56 TF) | 0.150 ms | 229 | 4.1× |

\* the BF16 columns ran under heavier contention (load 8–9; other seats'
GPU jobs) — the lab's locked, interleaved sweep measured BF16 at 104–171
TFLOPS on these shapes, giving **FP4/BF16 = 2.5–4.3×**; treat the starred
ratios as upper bounds. The NVFP4 column matches the lab's locked sweep
(401–456 TFLOPS at these shapes) within contention. Energy (lab, VDD_GPU
rail, 8192³): BF16 1.55 J, FP8 0.77 J, NVFP4 **0.41 J** per GEMM.

5. **What it makes possible:** the large-projection layers of SDXL/TRELLIS
   blocks and long-K scorers run 2–4× faster at half FP8's energy, and
   weights are 3.55× smaller than BF16 — more models resident per watt.

## 2. Per-shape layer cost — where FP4 pays (the auto policy)

1. **Hardware fact:** unchanged — but the GEMM is only part of a layer; the
   activation must be quantized (a full read+write pass over it) before the
   tensor cores can take it.
2. **Why a blanket FP4 policy fails:** at small K the quantize pass and the
   GEMM's fixed tile costs dominate; the lab measured the *transpose* on top
   of that (up to 14× the bare GEMM at small N). Small-K FP4 layers were
   2.4–3× slower than BF16 end to end.
3. **What this path does:** the transpose is gone (operand swap); what
   remains is quantize + GEMM. This table is what `convert(policy="auto")`
   reads (`src/thor_fp4/shape_table.json`).
4. **Numbers** (`benchmarks/results/gemm.txt`, section B; M=4096 tokens,
   quantize included, f16 activations, load 6.8–7.8):

| N×K | BF16 ms | FP4 ms (quant+gemm) | FP4/BF16 |
|---|---|---|---|
| 960×320 | 0.031 | 0.072 (0.043+0.029) | 0.44× — BF16 |
| 2560×320 | 0.082 | 0.256 (0.042+0.215) | 0.32× — BF16 |
| 1920×640 | 0.093 | 0.134 (0.070+0.064) | 0.70× — BF16 |
| 320×2560 | 0.070 | 0.213 (0.190+0.023) | 0.33× — BF16 |
| 1024×2048 | 0.109 | 0.201 (0.147+0.055) | 0.54× — BF16 |
| **2048×2048** | 0.272 | **0.232** (0.146+0.085) | **1.17× — FP4** |
| **5120×1280** | 0.514 | **0.306** (0.109+0.197) | **1.68× — FP4** |
| **1280×5120** | 0.606 | **0.515** (0.417+0.099) | **1.18× — FP4** |
| **10240×5120** | 3.653 | **1.556** (0.420+1.136) | **2.35× — FP4** |
| **8192×8192** | 5.742 | **1.972** (0.767+1.205) | **2.91× — FP4** |

Crossover derived from the measured winners: **FP4 for K ≥ 1280 AND N ≥ 1280
at M = 4096** (every measured shape respects this corner; at M = 256 the
fixed ~30 µs per-layer call overhead keeps small shapes on BF16, and only
K ≥ 2048 with large N wins). Removing the lab's transpose improved the
wrapper path ~2.7× at the SDXL small shapes (e.g. 960×320: 0.195 ms →
0.072 ms) — not enough to flip them to FP4; the remaining cost is the
quantize pass itself, i.e. activation bandwidth, not kernel quality.

5. **What it makes possible:** `convert(model, policy="auto")` makes the
   FP4/BF16 call per layer from measurements instead of hope: an SDXL UNet
   port gets FP4 for its 5120-dim ff blocks (2.4×) and keeps 320/640-dim
   blocks in BF16, with no per-layer hand-tuning.

## 3. Our CUTLASS sm_110a kernels

1. **Hardware fact:** the tcgen05 block-scaled MMA kinds are `mxf4nvf4`
   (e2m1 codes; UE4M3-per-16 or UE8M0-per-32 scales) and `mxf8f6f4`
   (e4m3/e5m2/e2m1 mixes, UE8M0-per-32).
2. **Why the library path is not enough:** cuBLASLt refuses every operand
   mix except NVFP4×NVFP4-TN on this box (measured: status 7/15 for MXFP4
   and all FP8/FP4/BF16 combinations), and its kernels are opaque.
3. **What these kernels do:** CUTLASS 4.4.1 block-scaled collectives
   compiled for sm_110a (two-line patch, see patches/), 11 tile/cluster
   variants behind one ABI, validated against an FP64 reference computed
   from the same codes and scales.
4. **Numbers** (`benchmarks/results/cutlass.txt`, load 5.6–7.6, all
   relRMS vs FP64 reference = 1.6–1.7e-3, i.e. bf16 output rounding):

| variant (8192³) | relRMS | ms | TFLOPS |
|---|---|---|---|
| nvfp4 128×128×256 c1 (the lab's config) | 1.60e-3 | 10.85 | 101 |
| nvfp4 128×256×256 c(2,1,1) — best | 1.60e-3 | **5.67** | **194** |
| mxfp8×mxfp4 128×128×256 c1 (round-2 default) | 1.67e-3 | 10.09 | 109 |
| **mxfp8×mxfp4 128×256×128 c(2,1,1) — ROUND-2 best** | 1.67e-3 | **5.8–6.5** | **169–190** |
| mxfp4×mxfp4 128×128×256 c1 | **1.00 (wrong values)** | 10.3 | — |

   - **ROUND-2 tuning (mxfp8×mxfp4)**: a 128-deep K-tile with a wide
     256-wide N-tile is the win for the mixed path — 169–190 TFLOPS at
     8192³ (1.6× the round-1 default; interleaved re-runs under load 8.9,
     `mem_avail` 106 GB; all relRMS 1.67e-3 unchanged). At decision-model
     shapes (M=4096) the K-tile-128 128×128×128 c(2,1,1) variant reaches
     **150–252 TFLOPS** (N×K = 1024×1024: 0.034 ms, 252; 1024×512: 150;
     5120×1280: 178; 1280×5120: 204) — the K-tile-256 kernels measured
     90–173 there in round 1. `resolve_mx8_variant(K)` picks by shape;
     `convert(policy="mx8")` uses it.

   - **cuBLASLt vs CUTLASS, identical packed inputs** (2048³, via the
     library): relRMS 1.7e-3 — the two independent implementations agree to
     output rounding. This closes the lab's open numerics question: its
     CUTLASS-vs-cuBLASLt disagreement (relRMS 1.4) came from the comparison
     harness, not the kernel.
   - **MXFP8×MXFP4 across shapes:** 108 TFLOPS (8192³), 90 (4096×5120×5120),
     173 (256×8192×8192), 171 (4096×5120×1280). Constraint: K % 128 == 0
     (can_implement rejects K=320).
   - **MXFP4×MXFP4 does not work:** relRMS ≈ 1.0 (garbage) at every shape
     and every configuration tried — alignment 128 and 32, clusters (1,1,1)
     and (2,1,1), K-tiles 256 (K-tile 64 does not even compile), M=N=K
     128…8192 — while the SF layouts CUTLASS builds for it have the correct
     vec-32 shape (cosize verified) and the same data path validates for
     NVFP4×NVFP4 and MXFP8×MXFP4. Conclusion: a CUTLASS-4.4.1-on-sm_110a
     gap for e2m1×e2m1 with UE8M0 scales; cuBLASLt refuses MXFP4 outright,
     so pure MXFP4 is not usable on this box today.
   - **Small-K, own kernels:** CUTLASS nvfp4 at (4096, 960, 320): 80–85
     TFLOPS vs cuBLASLt's ~87 through the same wrapper — parity, not a win.
     The small-K gap is activation bandwidth and tile quantization, not a
     missing kernel.
5. **What it makes possible:** FP8-activation × FP4-weight layers
   (`Linear(engine="cutlass:mxfp8xfp4_...")`) where FP4 activations cost
   too much accuracy, and an inspectable NVFP4 kernel to tune against
   cuBLASLt (194 TFLOPS so far, 2× the untuned start, cuBLASLt still ahead).

## 4. Batched decisions: N = 1…256, eager vs CUDA-graphed

1. **Hardware fact:** an FP4 chain is many small kernels (quantize + GEMM
   per layer); at decision-model scale the GPU work per launch is
   microseconds.
2. **Why eager FP4 loses:** every layer crosses the ctypes boundary; enqueue
   time ≈ GPU time (host-launch-bound) — the lab measured eager FP4 at
   100 % host-bound.
3. **What the path does:** capture the whole chain (quantize+GEMM+GELU) in
   one CUDA graph; everything per-shape is resolved before capture.
4. **Numbers** (`benchmarks/results/batch.txt`; tile scorer
   512→1024→1024→64, median of 5×30, load 8.2–10.2):

| B | BF16 eager | BF16 graph | FP4 eager | FP4 graph | best dec/s |
|---|---|---|---|---|---|
| 1 | 62.4 µs | **17.4 µs** | 176.0 µs | 39.1 µs | 58 k |
| 8 | 122.8 µs | **22.3 µs** | 218.8 µs | 43.7 µs | 359 k |
| 64 | 144.8 µs | **30.5 µs** | 312.1 µs | 49.7 µs | 2.1 M |
| 256 | 153.0 µs | **32.7 µs** | 335.1 µs | 69.0 µs | 7.8 M |

BF16 + graph stays 1.8–4.5× faster than FP4 + graph at this model scale:
these GEMMs (K = 512/1024) sit below the FP4 crossover, exactly as the
per-shape table predicts. FP4's win here is **memory**: 0.92 MB vs
3.3 MB resident weights (3.56×) — a decision-model bank holds 3.5× more
FP4 models per GB.
5. **What it makes possible:** latency-critical small decisons should stay
   BF16+graph; FP4 pays when the same bank must hold many more models than
   memory allows in BF16, or when layers are big-K.

## 5. Quality: PTQ vs QAT, two-level scales, FP8 activations

1. **Hardware fact:** NVFP4 rounds weights and activations to e2m1 grids
   against per-16 UE4M3 scales; MXFP8 rounds activations to e4m3 against
   per-32 UE8M0 scales (through the CUTLASS engine).
2. **Why naive serving degrades:** quantizing a BF16-trained model's weights
   *and* activations at inference (PTQ) injects noise the model never saw.
3. **What the library does:** QAT trains against exactly the arithmetic the
   tensor cores commit — the fake quantizer is bit-identical to the device
   kernels (unit-tested), which is why every "real" row below equals its
   "fake" row to the digit.
4. **Numbers** (`benchmarks/results/quality.txt`; tile scorer
   512→1024→1024→64 distilled from a seeded teacher, 8192 test samples,
   load 10.1, real tensor cores except where noted):

| variant | Q-MSE vs teacher | top-1 vs teacher | relRMS vs own BF16 | top-1 keeps own BF16's choice |
|---|---|---|---|---|
| BF16 baseline | 0.0173 | 61.1 % | — | 100 % |
| PTQ real FP4 (block-only) | 0.0218 | 57.8 % | 0.1730 | **78.4 %** |
| PTQ real FP4 (two-level ts) | 0.0217 | 57.3 % | 0.1708 | 78.5 % |
| PTQ fake FP4 (parity check) | 0.0218 | 57.7 % | 0.1730 | 78.2 % |
| QAT real FP4 (block-only) | 0.0233 | 55.1 % | 0.2919 | 63.8 % |
| QAT fake FP4 (parity check) | 0.0233 | 54.9 % | 0.2919 | 63.9 % |
| PTQ fake MX, FP4 acts | 0.0305 | 52.8 % | 0.2991 | 70.2 % |
| PTQ fake MX, **FP8 acts** | 0.0279 | 56.1 % | 0.2693 | 75.3 % |
| **PTQ real MXFP8 acts × FP4 weights (CUTLASS)** | 0.0280 | 56.1 % | 0.2699 | **75.4 %** |

ROUND 2 rows (same task, `quality.txt`, load 14.1–14.7, real paths):

| variant | Q-MSE vs teacher | top-1 vs teacher | top-1 keeps own BF16's choice |
|---|---|---|---|
| tuned mx8 engine (128×128×128 c2x1) — same arithmetic, new speed | 0.0280 | 56.1 % | 75.4 % |
| **FP4 weights via GEMV, activations consumed exactly (M ≤ 8)** | **0.0191** | **59.3 %** | **85.0 %** |

The GEMV row is the quality payoff of §8's engine: with only the weights
quantized (no activation quantization at all), FP4-weight serving keeps
85 % of BF16 choices — the best FP4-weight quality measured, within 2.8
points of BF16's own 61.1 % teacher agreement — at the 3.56× weight
density. It is the natural engine for small-batch decision serving where
quality matters more than the (already large) small-M speedups of §8.

Readings, including where they differ from the lab's run:
- **Serving matches training arithmetic exactly** (real rows = fake rows):
  whatever quality QAT optimizes is what the tensor cores deliver.
- **PTQ costs ~22 % of top-1 choices** on this task (78.4 % kept). The
  two-level per-tensor scale buys a little (relRMS 0.1708 vs 0.1730) on
  these well-scaled layers — its real purpose is robustness for
  wide-dynamic-range tensors whose small blocks would underflow UE4M3.
- **QAT did not beat PTQ here** (55.1 % vs 57.8 % vs teacher), unlike the
  lab's run (QAT 47.1 % > PTQ 44.1 %, baseline 49.3 %). The difference: this
  BF16 baseline fits the teacher much better (61.1 % vs the lab's 49.3 %),
  leaving PTQ noise small enough that QAT's constrained re-optimization
  spends capacity elsewhere (its relRMS-vs-own-BF16 0.29 says it found a
  different FP4-adapted solution, not a worse-vs-teacher one — the teacher
  columns are the honest quality axis). QAT remains the tool when the BF16
  fit is weak or the deployment budget is tighter.
- **FP8 activations recover ~5 points of top-1** over FP4 activations
  (75.4 % vs 70.2 % kept) at the same FP4 weights — measurable, and the
  real CUTLASS mixed path delivers exactly the fake-quant prediction
  (0.2699 vs 0.2693).
5. **What it makes possible:** picking the activation precision per layer
   by measurement (FP8 acts where FP4 acts hurt, through
   `engine="cutlass:mxfp8xfp4_..."`), and QAT/PTQ choices grounded in
   real-path numbers rather than simulations.

## 6. The cleanup loop: escalating unsure FP4 decisions to BF16

1. **Hardware fact:** none new — this is a systems question: the FP4 model
   keeps its BF16 self's top-1 choice 78.4 % of the time (PTQ, above); the
   other 21.6 % of samples are where quantization noise crosses the
   decision boundary.
2. **Why serving FP4 alone is not enough for checked decisions:** the
   disagreeing samples are not labeled — you only know the FP4 logits.
3. **What the loop does:** read a confidence from the FP4 logits alone
   (top-1 minus top-2 margin — no extra pass), escalate samples below a
   threshold through the BF16 copy, and return the BF16 answer for those.
4. **Numbers** (`benchmarks/results/cascade.txt`; 16,384 test samples,
   PTQ scorer, real tensor-core paths, load 9.3):

| margin threshold | escalated | agreement with BF16 | gap recovered |
|---|---|---|---|
| — (pure FP4) | 0 % | 77.9 % | 0 % |
| 0.140 | 61.8 % | 99.1 % | **96.1 %** |
| 0.279 | 87.1 % | 100.0 % | 99.9 % |
| 0.419+ | 96–100 % | 100.0 % | 100 % |
| softmax 0.039 | 75.7 % | 98.5 % | 93.1 % |

(QAT weights: same shape — 63.1 % base agreement, 86 % of the gap recovered
at 64 % escalation, 99.1 % at 89 %.) The margin is a strong signal: nearly
every choice the FP4 model gets wrong is one it was unsure about. But note
what the escalation rates say: on this task the FP4 model is unsure on most
samples (random-teacher logits are flat), so recovering the gap costs
re-running 60–90 % of them in BF16.

Cost at batch 256, real paths measured together (load 9.3): BF16 eager
57.3 µs, FP4 eager 173.4 µs, the eager cascade 428.9 µs including its
stats sync (the loop is FP4 + the escalated subset through BF16, plus a
per-call `.item()` that a production loop would keep on-device). At
decision-MLP scale FP4 is slower than BF16 (§4), so the loop is
latency-negative there — its value is when the FP4 model is what fits
(resident bytes 0.92 MB vs 3.28 MB, 3.56×) or at big-K scale where FP4 is
also the faster path.
5. **What it makes possible:** a checked-decision serving mode — FP4
   resident for the bulk, BF16 (or a bigger model) loaded/re-run only for
   the unsure fraction, with the threshold chosen from a measured
   recovery-vs-cost curve (`thor_fp4.cascade_sweep`).

### 6b. Carrying the certificate through the activations (ROUND 2), measured

Round 1 §6a showed ~90 % of FP4-vs-BF16 disagreement is upstream activation
quantization, invisible to a final-layer certificate. ROUND 2 asked for the
bound to be carried through the activations: each quantized activation's
error is bounded per element by its code's rounding interval against the
block scale the kernel used (e2m1 half-grid-steps; e4m3 code exponent;
subnormal/saturated edge cases included — the unit tests brute-force the
intervals), propagated through Σ|δ|·|W| per layer, the erf-GELU Lipschitz
constant (1.129, numerically verified with margin), f32 accumulation slack
and the f16/bf16 output rounding. `IntervalChainCertificate` implements it
(sound at scale: certified samples never disagree with the exact-arithmetic
network in randomized tests).

Measured on the same scorer (`benchmarks/results/certify2.txt`, load 7.1,
107 GB available):

| activation path | certified share | median bound | median margin | bound/margin |
|---|---|---|---|---|
| NVFP4 acts × NVFP4 weights | **0.00 %** (0/8192) | 1386 | 0.103 | 13,518× |
| MXFP8 acts × MXFP4 weights | **0.00 %** (0/8192) | 1638 | 0.082 | 19,900× |

The certificate is sound and the answer is plain: **coverage is zero, so the
arithmetic-certificate direction is dropped.** The mechanism of the failure
is the one round 1 already saw, now compounded over three layers: the
triangle-inequality bound sums K error terms at their absolute values
(~√K above the random-sign RMS error, per layer, multiplicatively through
the chain), while this task's margins are ~0.1 logits. Certifying FP8
activations does not help — the weight side of that chain (MXFP4 with
per-32 power-of-two scales) is *coarser* than NVFP4's per-16 UE4M3, so the
bound grows. Certificates of this form need models trained for margin and
short depth to cover anything; the practical gate for checked FP4 decisions
remains the learned-confidence cascade (§6: 96 % of the gap recovered).

### 6a. The final-layer arithmetic certificate (`thor_fp4.certify`), measured

DIRECTIONS.md §1 proposes making the escalation decision by arithmetic: the
final layer's weight-error bound `Σ|x̂|·|W−Ŵ|` (exact per-weight errors from
the fp32 master weights, applied to the dequantized activations the hardware
consumed) versus the FP4 top-1 margin. Measured on the same scorer
(`benchmarks/results/certify.txt`, load 9.1–11.9):

- **The certificate is sound**: across all runs, 0 certified samples
  disagreed with the true-weights-on-same-activations argmax (unit-tested at
  scale, including two soundness bugs the tests caught during development:
  the bound must be weighted by the *dequantized* activations, and top-1
  must beat the *largest* bound over all other logits, not just top-2's).
- **Coverage is thin at these margins: 0.29 %** of samples certify. The
  triangle-inequality bound (0.63 median) sits ~√K above the typical
  random-sign error, while median margins are 0.10. Certificates of this
  form need models trained for margin (or shorter K) to cover widely.
- **Final-layer exactness is not where the disagreement lives**: promoting
  *every* sample through the true final-layer weights (fp32) moves agreement
  with BF16 from 77.9 % to only 80.2 % — i.e. **~90 % of the FP4-vs-BF16
  disagreement is upstream activation-quantization error**, not final-layer
  weight error. This is why the learned-confidence gate (which reacts to the
  actual logit uncertainty, upstream included) recovers 96 % of the gap
  while the final-layer arithmetic gate cannot, and it redirects the
  certificate idea: to certify decisions, the bound must be carried through
  the activations (interval arithmetic through the net), or the activation
  quantization itself must be reduced (the FP8-activation mix in §5).

## 7. Verification cost

`verify()` is a diagnostic, not a serving path: for a thor_fp4.Linear it
costs one extra dequantize + one FP32 matmul per layer (≈2× layer time, once).
At serving time the cheaper invariant is the graph test: captured replay is
bit-identical to eager (`test_cutlass_graph_capture`,
`test_graph_replay_matches_eager`).

## 8. Small-batch inference (DIRECTIONS.md §2): the register GEMV, measured

1. **Hardware fact:** at batch M ≤ 8 a linear layer is bandwidth-bound —
   the time is streaming the N·K weight bytes; NVFP4 weights are 3.56×
   smaller than BF16. Two measured properties of this box bound any kernel
   here: byte-granular global loads cap at **52 GB/s** while 16-B (uint4)
   loads reach **215 GB/s** (same access pattern, only load width differs),
   and DRAM/unordered contention holds both under the shared-seat load.
2. **Why the naive path fails:** dequantizing FP4 weights into a BF16 copy
   writes 2 B/element — more bytes than it saves (torchao's failure mode).
3. **What was built:** `ops.gemv` / `Linear(engine="gemv")` — a warp-per-row
   kernel that streams packed e2m1 codes with uint4 loads, decodes them in
   registers through the hardware e2m1→f16x2 converter (1 instruction per
   pair), scales by the block scales (un-scattered once per weight into a
   plain row-major buffer — the tcgen05 tile layout scatters each row's
   scales at 512-B stride and costs a 32-B sector per useful byte), stages
   the ≤ 8×K activations in shared memory with an XOR swizzle
   (word ^ word>>4) for conflict-free reuse across the block's rows, and
   accumulates in FP32 with activations consumed exactly (never quantized).
   It agrees with the dequantize-then-FP32-matmul reference to accumulation
   order (relRMS ≈ 2e-7, unit tests).
4. **Numbers** (`benchmarks/results/gemv.txt`, cold = 8 rotated weight
   copies, load 7.7–9.0, 109–112 GB available):

   | path | M=1 | M=8 | vs BF16 at N=K=8192 |
   |---|---|---|---|
   | BF16 mm (cuBLAS) | 1201 µs | 1432 µs | 1× |
   | register GEMV | 1005 µs (42 GB/s) | 10475 µs | **1.2× / 0.14×** |
   | **quantize + cuBLASLt NVFP4 GEMM** | **407 µs** | 522 µs | **2.95× / 2.74×** |

   The register GEMV does **not** meet the ≥ 2.5× bar at any M: even at M=1,
   where it is best, the scalar instruction stream (decode + scale + one FFMA
   per weight element) cannot reach the uint4 streaming rate, and at M ≥ 2
   the M·N·K scalar work overwhelms it (a tensor core does that product per
   instruction; CUDA cores cannot). **The bar is met — by the tensor-core
   path that already exists:** at M ≤ 8 with activation quantization
   included, cuBLASLt's NVFP4 GEMM beats BF16 by **2.5–12×** on layers with
   N ≥ 4096 (4096×8192: 55–88 µs vs 341–680 µs BF16; 381 GB/s effective on
   weight bytes at M=1) and loses below N≈2048, exactly as the per-shape
   table (§2) predicts. Small-M serving should therefore use the ordinary
   `thor_fp4.Linear` path, not a custom GEMV.
5. **What the GEMV is still for:** activations consumed *exactly* (no
   quantization) makes it the most accurate FP4-weight execution in the
   library — the right path when a small batch must be bit-faithful to the
   FP4 weights (its quality is measured in §5 terms: zero activation-
   quantization error), and the reference implementation for reasoning about
   FP4 arithmetic. It ships documented at its measured rate.

## 9. Retrieval as a large-K FP4 GEMM (DIRECTIONS.md §3), measured

1. **Hardware fact:** a retrieval tick is queries × libraryᵀ with K = the
   embedding dimension; at 2048/4096 this is exactly the K ≥ 2048 regime
   where the FP4 tensor cores stream at their highest measured rates, and
   the library's resident bytes drop 3.56× (NVFP4 vs BF16).
2. **Why the naive path leaves it on the table:** a BF16 sweep must stream
   2 B/entry/dimension from memory — at 10⁶ entries × 4096 dims that is
   8.2 GB per sweep, bandwidth-bound long before the FLOPs matter.
3. **What this path does:** the library stays resident in NVFP4; each tick
   quantizes the (few) queries, sweeps them against the library in
   262 144-entry chunks through the cuBLASLt block-scaled GEMM (f16 scores),
   keeps a running top-64 per chunk, and re-ranks those 64 in BF16 —
   exactness where ranking matters, FP4 where bytes dominate.
   (`benchmarks/bench_retrieval.py`; two measured traps fixed on the way:
   chunk sweeps must offset the packed/scale pointers per chunk — every
   chunk was silently re-scanning chunk 0 until a planted-neighbor test
   caught it — and `topk` must run in the scores' native dtype; the f32
   cast doubled its cost and was the FP4 path's bottleneck at Q ≥ 32.)
4. **Numbers** (`benchmarks/results/retrieval.txt`, 10⁶ unit-normalized
   entries, load 7.3–9.5, 104–112 GB available; truth = exact BF16 sweep):

   | dim | Q | FP4+re-rank | BF16 | speed | recall@1 | recall@64 |
   |---|---|---|---|---|---|---|
   | 2048 | 1 | 5.4 ms (185 q/s) | 17.8 ms | **3.29×** | 100 % | 84 % |
   | 2048 | 32 | 7.7 ms (4 182 q/s) | 19.8 ms | **2.59×** | 96.9 % | 77 % |
   | 2048 | 256 | 33.9 ms (7 562 q/s) | 40.5 ms | 1.20× | 98.4 % | 77 % |
   | 4096 | 1 | 14.8 ms | 34.8 ms | **2.35×** | 100 % | 69 % |
   | 4096 | 32 | 12.5 ms (2 568 q/s) | 36.5 ms | **2.93×** | 96.9 % | 76 % |
   | 4096 | 256 | 33.5 ms (7 647 q/s) | 59.8 ms | **1.79×** | 96.9 % | 77 % |

   With well-separated true neighbors (each query = a library entry +
   noise — the realistic retrieval case) the FP4 sweep + BF16 re-rank finds
   the true entry as top-1 **100 %** of the time at every Q and both dims.
   The lower recall@64 in the random regime is the expected behavior on
   near-ties: for random unit vectors the true ranks 1–64 sit inside FP4
   score noise (relRMS ≈ 2.7e-3), so the top-64 *set* reshuffles while the
   well-separated neighbors it exists to find do not.
5. **What it makes possible:** `tiles-retrieval` / reflex libraries an order
   of magnitude larger per resident GB (3.56×), swept 1.2–3.3× faster than
   BF16 at the batch sizes a tick produces, with exact BF16 re-ranking of
   the candidates — retrieval becomes a resident-FP4 workload.

## 10. The fused chain: quantization in the epilogue (ROUND 2, DIRECTIONS §4)

1. **Hardware fact:** the tcgen05 mainloop's accumulators are f32 registers
   inside the GEMM; CUTLASS's sm100 epilogue can finish a layer there —
   bias, erf-GELU (identical to torch's), and MXFP8 quantization of the
   output (e4m3 codes + UE8M0-per-32 scales, written in the same tile atom
   the next GEMM reads).
2. **Why the two-pass chain pays more:** each standalone pass is a full
   read+write over the activations — the quantize pass is the measured
   dominant cost of small-K FP4 layers (§2), and torch's gelu and bias adds
   are two more passes.
3. **What was built:** `csrc/fused_kernels.cu` (4 tile/cluster variants,
   C ABI `tfc4f_*`) + `thor_fp4.fused` (`FusedLinear`, `MX8Act`,
   `mx8_chain`): a chain of N layers runs as N GEMMs and nothing else; the
   epilogue's scale recipe is ceil-power-of-two (never clips a block max;
   our standalone `quantize_mx` rounds to nearest, which can clip the block
   max by up to ~34 % of 448 on ties that round down).
4. **Numbers** (`benchmarks/results/fused.txt`, load 5.9–7.9, 79–109 GB
   available; scorer 512→1024→1024→64, both paths on the CUTLASS mx8
   engine):

   | | two-pass (quantize+GEMM+gelu+bias) | fused (1 kernel/layer) | speed |
   |---|---|---|---|
   | layer 1024×512, M=4096 | 398 µs | 372 µs | 1.07× |
   | layer 5120×1280, M=4096 | 2317 µs | 1051 µs | **2.20×** |
   | chain B=256, eager | 164 µs | 104 µs | 1.58× |
   | chain B=256, CUDA graph | 101 µs | 54 µs | **1.87×** |
   | chain B=4096, graph | 1027 µs | 562 µs | **1.83×** |

   Quality on the scorer (`quality.txt`): the fused chain keeps **77.1 %**
   of BF16 top-1 — slightly better than the two-pass mx8 chain's 75.4 %,
   because the epilogue computes GELU on the f32 accumulators (the two-pass
   chain gelus the bf16-rounded GEMM output) and its ceil-pow2 scales never
   clip a block max.
5. **Honest caveat — EXPERIMENTAL:** through the wrapper's allocation
   pattern, batches ≥ ~2048 hit scattered rows (~0.2 %) whose scale bytes
   decode as ue8m0-NaN — deterministic, absent in direct ABI calls with
   fresh buffers at every size we tried, and not attributable to any of our
   own pointer handling after bisection; the same class of finding as the
   MXFP4×MXFP4 verdict in §3. `FusedLinear` therefore batches through
   512-row slices (measured clean at every size; rows are independent), and
   the B=4096 rows above include that chunking overhead — the unchunked
   kernel measured 1.7–3.2× before the mitigation. What it makes possible,
   once the epilogue path is trusted: decision MLPs at one kernel per layer
   with no activation-quantize passes anywhere — the missing small-K piece
   of DIRECTIONS §4's metal-shaped models.

# NOTES — thor-fp4

## 2026-11-01, R3-0: root cause of the NaN-scale rows in the fused epilogue

**Found and fixed: an out-of-bounds bias read, not a hardware or scheduler
problem.** The fused epilogue used CUTLASS's `LinCombPerRowBiasEltActBlock-
ScaleFactor`, whose bias node (`Sm90ColBroadcast`, stride `_1,_0`) computes
`D[m,n] += bias[m]` — a vector indexed by output **row**. `FusedLinear`
passes `nn.Linear`'s bias, an **N**-float vector meant as `bias[n]`. Two
consequences:

1. *Semantics:* even in-bounds, the fused layers added the wrong bias
   (`bias[m]` read off the N-vector). Every fused test used zero-initialized
   biases, so per-row and per-column are indistinguishable and nothing
   caught it. The round-2 fused-chain quality numbers were measured through
   this wrong-bias path (superseded below once re-measured).
2. *The NaN rows (BENCHMARKS §10):* for M > N the per-row read `bias_ptr[m]`
   runs past the N-float buffer into whatever allocation follows. Float
   patterns there that decode as NaN make that entire row NaN (acc+NaN →
   gelu → NaN codes + ue8m0-NaN scales). This matches every measurement:
   - bad only at M ≥ ~2048 → really M > N = 1024; M = 1024 exactly in-bounds;
   - whole-row NaN (bias[m] is row-scoped), codes and scales together;
   - deterministic per allocation layout (the neighbouring tensor's bytes),
     different row sets for different neighbours — the "address dependence";
   - independent of the bias *values* (zeros vs ones → identical rows;
     NaN-filled bias → rows [0..N) NaN as expected in-bounds);
   - clean with `bias_ptr = nullptr`, clean in the chain (bias params
     adjacent to zeroed params), clean with fresh zeroed buffers after the
     bias (abi-fresh), clean for slab-internal bias views (zeros follow);
   - inputs never corrupted; sentinel guards on all buffers intact.

Fix: `csrc/fused_kernels.cu` now uses `LinCombPerColBiasEltActBlockScale-
Factor` (bias node `Sm90RowBroadcast`, stride `_0,_1` → `bias[n]`, the
nn.Linear semantics; same RowBlockScaleFactor SFD store, same Arguments
field names). One typedef change. After it: the 512-row chunking
mitigation in `FusedLinear` is unnecessary and the M ≥ 2048 caveat is gone
(verification below once the rebuild lands).

Bisection trail (scripts committed under `benchmarks/`, `repro_nan*.py`):
- repro (wrapper, unchunked): M=1024 clean; 2048/4096 BAD, deterministic rows.
- repro_nan2 (direct ABI on the same clean inputs): still BAD → not the
  wrapper's Python. Inputs scanned clean (A e4m3 codes, SA/SB ue8m0, W) →
  not operand corruption. Reference block-max normal; no zero 32-blocks →
  not a GELU/zero-amax edge case.
- repro_nan3 (instrumented `tfc4f_gemm_mode` in csrc): 4 lifecycle modes
  (cached / re-init / fresh object / fresh workspace; workspace = 0 bytes),
  4 tile variants, 8 output-buffer addresses, 15 offsets × {A-codes, SFA,
  B-codes, SB} input slabs, pre-GEMM f32 matmuls — all clean in that
  context → not lifecycle, not those addresses.
- repro_nan4 (allocation ladder, one process per rung): rungs 1–7 clean;
  rung 8 = rung 7 + bias passed as `fl.bias.float().cuda()` instead of a
  plain zeros tensor — BAD. Only the bias pointer differed (0x…200000 vs
  0x…245000), same zeros values, same size, all other pointers identical.
- repro_nan5: bad address stays BAD with zeros and with ones (same rows);
  NaN-filled bias → in-bounds rows NaN; 2047 slab bias addresses all clean;
  post-call input scan + sentinel guards clean (no wild writes).
- repro_nan6: badness follows the bias pointer across far-away operand
  copies (test A); `bias_ptr = nullptr` → clean (the elimination test that
  pinned the node); then reading the CUTLASS source of the node's tree
  found the per-M stride and the OOB read.

Dead ends (do not retry): workspace staleness (workspace is 0), TMA
descriptor staleness through `update()`, CLC/scheduler misassignment,
converter edge cases (`cvt.rp.satfinite.ue8m0x2` of 0/denormals is fine on
sm_110a — the instruction exists; a hand-written probe of it failed ptxas
only because of operand form), SMEM-staging races in the bias node
(`Sm90ColBroadcast` has empty SharedStorage — there is no SMEM in that
node at all).

Verification after the fix (load 10.8–16.5, 86–88 GiB available):
- every repro clean: wrapper M=1024/2048/4096/8192, chain B≤4096, ladder
  rung 8 (the previously-bad bias address), deep repro, far-operands,
  fresh-segment census, guards;
- 104 tests green (two new: nonzero-bias semantics, NaN-bait OOB);
- benches re-run (correct bias, unchunked): chain B=4096 962→381 µs eager
  (2.53×), 1019→410 µs graph (2.48×) vs the two-pass chain; layers
  1.10–2.56×; fused quality 78.1 % top-1 keep (was 77.1 % on the wrong-bias
  path) vs 75.4 % two-pass. Round-2 numbers superseded in BENCHMARKS §10.


## 2026-11-01, R3-x + R3-z: fused activation quantization, accounted; one graph per step

- R3-x additions: SiLU and Identity fused-epilogue variants
  (`mx8out_silu_*`, `mx8out_ident_*` — Activation as a template param of
  the variant), `ops.mx8_mul` (tfp4_mx8_mul in core: decode→mul→re-encode,
  ceil-pow2 scales, one kernel + the scale reblock), `GatedMLP`
  (down(silu(gate x)·up x)). 107 tests green (new: silu/ident semantics,
  mx8_mul vs reference, gated block).
- Measured (results/fused-mlp.txt, gpu_util 97–98 % — box fully shared; a
  quieter run shows ~2× better absolutes with the same ratios):
  M=4096 layers: 52–261 MB of activation traffic the fused epilogue never
  moves, 45–65 GB/s implied at the measured Δt. Gated block 2048→2×8192→
  2048: fused wins 1.12× (B=1) to 1.86× (B=4096) eager, 1.83× graphed.
- Dead end recorded: the gate multiply composed in torch (dequant → mul →
  quantize) measured 0.82× end-to-end — SLOWER than two-pass — because the
  torch dequant is a gather-heavy multi-kernel pass. The one-kernel
  tfp4_mx8_mul flipped the block to winning. Do not compose it in torch.
- R3-z measured on the repo's stage stand-ins (no diffusion model in this
  repo — said plainly in §12): fused block = 4 kernels; launch overhead
  9.5 % (B=1) under full contention; the quiet-GPU §10 chain row is the
  sharp datapoint: 104.7 µs eager vs 47.5 µs graphed at B=1 = 55 % launch
  share; one graph per shape (capture_module) removes it, weights resident
  by construction (cached MXFP4, no per-step quantization).
- BENCHMARKS §11/§12 added; README fused section updated; scorer.boxstate
  now records gpu_util.

# thor-fp4: directions worth building (director's notes, 2026-10-31)

These are hypotheses built from what the lab measured, not results. Each names its mechanism and the experiment that would settle it.

## What the lab established

- **The tensor cores take NVFP4 directly:** e2m1 values with one UE4M3 scale per 16 elements, plus a per-tensor scale. cuBLASLt's block-scaled path reaches **371–456 TFLOPS when the inner dimension K ≥ ~2048**.
- **At small K (320–1280) the same kernels run 2.4–3× *slower* than BF16.** Each call quantises its activations, loads scale tiles and pads to 128-row tiles. Those fixed costs are paid per call, while the arithmetic saved shrinks with K.
- **FP4 weights are 3.56× smaller than BF16.**
- **The Thor achieves ~166–220 GB/s of memory bandwidth.** At small batch sizes (a tick of decisions, one query) inference is **bandwidth-bound**: the time is spent streaming weights, not multiplying them.
- **122 GB of memory is shared by CPU and GPU.** Residency, not capacity, limits how many models can be ready at once.

Four consequences follow, and each is a direction.

## 1. Certified FP4: let the arithmetic tell you when FP4 is safe (Casey's precision cascade, made exact)

Casey's intuition is that a fast low-precision model, with a decision model cleaning up after it, can act like a high-precision one. FP4 makes the "when to clean up" question answerable **by arithmetic rather than by a learned confidence.**

- **The bound.** Every FP4 weight is its block scale times an e2m1 code, so its quantisation error is bounded by half a grid step times its block's scale; both are known at load time. For the **final layer** of a decision model, the error on each output logit is bounded by `Σ |x_i| · ε_i`, where `x` are that layer's inputs (already in hand) and `ε_i` are the per-weight error bounds. That bound costs one extra GEMV with |x| against a precomputed error matrix, which is itself FP4-sized.
- **The certificate.** If the winning logit beats the runner-up by more than the sum of their two error bounds, **the FP4 answer is provably the same answer BF16 would give** for that layer. No escalation is needed, and the claim is a fact rather than a confidence.
- **Promotion only where needed.** When the margin is inside the bound, recompute **only that sample's final layer** in BF16 (weights resident in both forms; the last layer is small). That is precision promotion per sample, per layer: the "precision promotion" seed from the liquid chats, now with a mechanism.
- **Deeper layers.** Bounding error through the whole network needs interval or affine arithmetic and loosens quickly. Start with the last layer and measure how often a certificate holds. The lab's tile scorer kept its BF16 top-1 choice only 71 % of the time in FP4, so the interesting numbers are:
  - what share of decisions certify;
  - what share of the remaining 29 % the final-layer promotion recovers;
  - at what cost.
- **Why it matters:** the cascade gains a gate that **cannot be wrong** about the cases it certifies. The learned confidence gate (d1's τ) then only has to handle the rest.

*Experiment:* the lab's tile scorer and a d1-sized final layer. Report the certified share, top-1 recovered after promotion, and end-to-end cost vs all-BF16.

## 2. Bandwidth, not FLOPs: an FP4-weight GEMV for the small batches we actually run

Most of our inference is small-batch: one tick's decisions, one retrieval, one critic call. There the GPU waits on weight bytes, not on multiplies. FP4 weights move **3.56× fewer bytes**, so even with BF16 arithmetic a kernel that **dequantises in registers** should approach a 3.5× speedup at small M. The tensor-core FP4 GEMM's small-K weakness never comes into play because no tensor-core GEMM is used. torchao's earlier failure came from dequantising into a full BF16 copy in memory first; a fused kernel never writes that copy.

*Experiment:* a Warp or CUDA GEMV (FP4 weights + scales; BF16 or FP8 activations; FP32 accumulate), against cuBLAS BF16 GEMV at M = 1…64 on d1-2B-shaped layers. The bar: ≥ 2.5× at M ≤ 8, and bit-for-bit agreement with the dual-path reference.

## 3. Retrieval is a large-K GEMM, so put it in FP4's fast regime

Tile retrieval, pincher-style reflex matching and the critic's nearest-neighbour checks all reduce to **queries × library-transposed**, where **K is the embedding dimension**. With embeddings of 2048–4096 dimensions, every one of these lands at K ≥ 2048: exactly where NVFP4 runs at 371–456 TFLOPS.

- **One GEMM per tick.** Batch every character's query in a tick into one GEMM.
- **Libraries stay resident.** Store the library in FP4 (3.56× more entries resident).
- **Dual-path re-check.** Re-check the top-K in BF16, where exactness matters (a small re-rank, so cheap).

This makes `tiles-retrieval` and the first mile's reflex lookup a single fast call even for very large libraries.

*Experiment:* 10⁶ library entries × (1…256 queries) at dim 2048/4096. Report FP4 against BF16 throughput and recall@K after the BF16 re-rank.

## 4. Shape our own models for this metal

We train the decision models, the tile scorers and the policies ourselves, so we can **design their shapes for the hardware**:
- **Wide, shallow layers with K ≥ 2048** where FP4 runs fast.
- **Activation quantisation fused into the previous layer's epilogue**, so FP4 flows from layer to layer with no separate quantise kernel. That separate kernel is a large part of the small-K penalty.
- **FP4-aware training** (QAT) from the start.
- **2:4 structured sparsity on top**, if cuSPARSELt or CUTLASS supports sparse block-scaled FP4 on sm_110. That is the "sparse" in the datasheet's 2,070 TFLOPS. Our models can be pruned during training, which most published models cannot.

The application: **many resident decision models** at FP4 density in unified memory. One small model per character or task family, all hot, batched per tick in one CUDA Graph.

*Experiment:* a tile scorer trained wide and shallow with QAT (and 2:4 if available), against the lab's 512→1024→1024→64 MLP. Measure accuracy against its BF16 twin, and decisions per second per watt.

## 5. Two smaller checks

- **Small-K grouped GEMM.** Many characters' small layers run as one grouped GEMM (CUTLASS grouped), which turns many small-M calls into one large launch. It may recover part of the small-K penalty, though not the per-call quantisation cost unless that is fused (§4).
- **Keep the critic and provenance in BF16** (FP8 already cost 15–22 % output error there). FP4 applies to choosing among options, not to judging correctness.

## Priority

1. **Certified FP4 (§1):** the most original, and it turns the cascade's escalation from a learned guess into a proof for the cases it covers.
2. **The FP4-weight GEMV (§2):** the most immediately useful for the decision models we already run.
3. **Retrieval in the fast regime (§3):** the cleanest fit to the hardware.
4. **Metal-shaped models (§4):** the long game.

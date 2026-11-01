# thor-fp4

NVFP4 linear layers on the Jetson AGX Thor's tensor cores, for PyTorch.

`thor_fp4.Linear` is an `nn.Linear` drop-in whose weights live in NVFP4
(e2m1 codes with one UE4M3 scale per 16 elements plus a per-tensor scale) and
whose activations are quantized on the fly on the GPU; the matmul runs on the
sm_110 block-scaled tensor cores through cuBLASLt, with BF16/FP32 output and
transposes handled inside. Around it: a calibrating PTQ quantizer, QAT with a
straight-through estimator, shape-aware model conversion (`convert(model,
policy="auto")`), CUDA-graph capture, dual-path verification, and our own
CUTLASS sm_110a kernels (including the MXFP8-activation × MXFP4-weight mix
that cuBLASLt refuses on this silicon).

Measured numbers, with the box state beside them, live in
[BENCHMARKS.md](BENCHMARKS.md). This README explains the mechanisms.

## The 10-minute path (on a Thor)

```bash
git clone <this repo> && cd thor-fp4
make venv          # local .venv: torch, numpy, pytest (from the wheel cache)
make core          # build/libthorfp4_core.so — the cuBLASLt path (~40 s)
make test          # 63 tests: bit-exactness, tolerances, verify, capture

.venv/bin/python - <<'EOF'
import torch, thor_fp4

lin = thor_fp4.Linear(8192, 8192).cuda()          # weights resident in NVFP4
y = lin(torch.randn(2048, 8192, device="cuda"))   # real tensor-core FP4 GEMM
print(y.shape, y.dtype, lin.fp4_weight_nbytes() / 1e6, "MB weights")

model = thor_fp4.convert(model, policy="auto")    # FP4 where it pays (see below)
rep = thor_fp4.verify(model, x, bound=1e-2)       # dual-path check (see below)
g = thor_fp4.capture_module(model, x)             # CUDA graph of the FP4 chain
EOF

make cutlass       # optional: our own sm_110a kernels (~20 min, ~5 GB RAM;
                   # needs CUTLASS 4.4.1 + patches/cutlass-4.4.1-sm110.patch)
```

Requirements: Jetson AGX Thor (sm_110), CUDA 13.2+ with cuBLASLt ≥ 13.4,
PyTorch ≥ 2.4 (built and measured with torch 2.15.0+cu132). Everything builds
locally (`.venv`, `build/`); nothing is installed system-wide.

## How the hardware wants its operands

1. **The hardware fact.** The Thor's tcgen05 tensor cores take block-scaled
   FP4 directly: e2m1 codes (2 per byte) with one UE4M3 scale byte per 16
   elements, in a 128-row × 4-scale-factor tile layout — byte index
   `(a·(S/4)+d)·512 + 16·c + 4·b + e` for scale (row r, sf column s) with
   a=r/128, b=(r%128)/32, c=r%32, d=s/4, e=s%4. This is NVIDIA's NVFP4
   format and CUTLASS's `SfKMajorAtom`.

2. **Why the obvious path could not use it.** `torch` has no NVFP4 tensor
   core matmul on this build, and the usual route people try — quantize,
   dequantize, multiply in BF16 — never touches the FP4 units (the lab's
   round-1 measurement of that route: 13.7 TFLOPS, slower than BF16 itself).
   cuBLASLt *does* have the kernels but only for one operand order (TN), and
   only when the caller supplies the block-scale tensors in the tile layout
   above, with scale pointers set even to query an algorithm. Several
   plausible calls return algorithms that compute wrong values (row-major
   output) or none at all (MXFP4, any mixed FP4/FP8 operand pair).

3. **What this library does differently.**
   - Device kernels quantize activations to the tile layout on the fly
     (f32/f16/bf16 inputs), and weights once at conversion; buffers are
     padded to 128-row tiles and zero-initialized, because the kernels read
     whole tail tiles.
   - The GEMM is issued as `out = act @ Wᵀ` computed via
     `(W @ actᵀ)ᵀ`: cuBLASLt's column-major D buffer *is* a row-major
     `(W @ actᵀ)ᵀ`, so the library gets torch's row-major layout with **zero
     transpose kernels** (the lab paid up to 5–14× the GEMM's time
     transposing; see BENCHMARKS.md). A side effect: the batch dimension is
     unconstrained — any M ≥ 1 works, including M = 1, which computes
     garbage on the naive ordering.
   - The per-tensor scale of the two-level NVFP4 recipe rides the GEMM's
     `alpha`, so it costs nothing at run time.
   - Everything per-shape (cuBLASLt descriptors, algorithm, workspaces,
     activation buffers) is resolved on first use and reused — repeated
     calls, and whole FP4 chains, capture into CUDA graphs.

4. **Measured numbers.** [BENCHMARKS.md](BENCHMARKS.md) — the library path
   at 405 TFLOPS bare (2048³; the lab's locked sweep measured the same
   shapes at 2.5–4.3× BF16 and ~half FP8's joules), per-shape end-to-end
   costs, quality tables, and the cascade measurement, each with the box
   state beside it.

5. **What it makes possible.** 3.55× smaller resident weights than BF16
   (0.5625 B/parameter) at 2–3× the BF16 GEMM rate on large-K layers — so
   more decision models resident on the device and faster large projections
   (SDXL-class ff blocks at K ≥ 5120), while small-K layers stay BF16 via the
   shape-aware policy. FP8-activation × FP4-weight layers (through our
   CUTLASS engine) halve the activation-quantization error where FP4
   activations hurt.

## The API in one screen

```python
import thor_fp4, torch

# 1. drop-in layer
lin = thor_fp4.Linear(4096, 5120, bias=True, out_dtype="f16").cuda()
y = lin(x)                                # x: (M, 4096) f32/f16/bf16, any M

# 2. convert a model, shape-aware by default (FP4 where measured to win)
cal = thor_fp4.calibrate(model, sample_batch)      # optional static act scales
model4 = thor_fp4.convert(model, policy="auto", calibration=cal)
model4.thor_fp4_policy                            # per-layer decisions + reasons
# or explicit: convert(model, policy={"critic.*": "bf16", "*": "auto"})

# 3. independent check of the native path (XOR dual-path verification)
rep = thor_fp4.verify(model4, x, bound=1e-2)       # per-layer + end-to-end
assert rep.ok

# 4. graphs and batching
g = thor_fp4.capture_module(model4, x)             # replay == eager
out = g(x)

# 5. QAT (trains against exactly the arithmetic the tensor cores commit)
qat_model = thor_fp4.QATSequential(layer1, act, layer2, ...)

# 6. the cleanup loop: FP4 first, escalate to BF16 when unsure
cas = thor_fp4.Cascade(model_fp4, model_bf16, threshold=0.35)
logits, escalated, confidence = cas(x)
rows = thor_fp4.cascade_sweep(model_fp4, model_bf16, x)   # threshold curve

# 6b. or escalate by arithmetic, not confidence (DIRECTIONS.md §1): the
# final layer's weight-error certificate; uncertified samples get promoted
logits, certified = thor_fp4.certified_forward(model4, x)

# 7. our own kernels (make cutlass): engine="cutlass" or a variant name
lin = thor_fp4.Linear(8192, 8192, engine="cutlass:mxfp8xfp4_128x128x256_c1").cuda()
```

## Shape-aware conversion: why "auto" is not "fp4 everywhere"

The FP4 tensor cores are ~4× the BF16 rate at big K, but today's kernels are
tuned for compute-bound GEMMs. At UNet-ish K (320–1280) the FP4 path *loses*
to BF16 end to end: quantizing the activation costs a full pass over it, and
small-K GEMMs don't amortize that. `convert(policy="auto")` reads a measured
per-shape table (`src/thor_fp4/shape_table.json`, regenerated by
`benchmarks/bench_gemm.py`) and picks FP4 only where FP4 actually won on this
box — currently K ≥ 1280 **and** N ≥ 1280 at M = 4096 — plus the hard
constraints (K % 64 == 0, N ≥ 8, N % 8 == 0). Layers that fail any of these
stay in their original precision (the critic stays BF16 under
`policy={"critic.*": "bf16", "*": "auto"}`).

## Dual-path verification (`verify`)

A new kernel path can be silently wrong — this library's own development hit
row-major-D computing garbage, M = 1 computing garbage, and a scale layout
that decoded to noise. `thor_fp4.verify(model_or_layer, inputs, bound)` runs
the native path and an independent reference (dequantize the *same* codes on
the device, matmul through torch/cuBLAS in FP32 — a different library, kernel
family, and accumulation order) and reports every element beyond `bound`
(relative to the output's RMS scale, so near-zero elements don't false-alarm).
The unit tests inject a broken kernel and a scrambled scale layout and
require `verify` to flag them, and require it to stay quiet on the clean path
(residual ≈ 2e-4 for f16 output). One scope note: it verifies *execution*, so
corrupting operand bytes before both paths read them changes both sides
equally and is (documentedly) not its job.

## Our own kernels (CUTLASS, sm_110a)

`make cutlass` builds `libthorfp4_cutlass.so` from
[csrc/cutlass_kernels.cu](csrc/cutlass_kernels.cu): 11 GEMM variants (tile
shape × cluster × operand format) behind one C ABI, usable as
`Linear(engine="cutlass:<variant>")` and self-benchmarked by
`build/bench_cutlass`. What they are for:

- **NVFP4 × NVFP4, open-source**: validates against an FP64 reference from
  the same codes (relRMS 1.6e-3, bf16 rounding) and agrees with cuBLASLt on
  identical packed inputs (1.7e-3). Best tuned config so far: 128×256×256,
  cluster (2,1,1) at 194 TFLOPS (8192³) vs 101 untuned — cuBLASLt (371–456)
  remains the default engine; our kernel is the inspectable, tunable one.
  This also resolves the lab's open question: its CUTLASS-vs-cuBLASLt
  disagreement was an artifact of its comparison harness, not the kernel.
- **MXFP8 × MXFP4**: FP8 activations × FP4 weights — rejected by cuBLASLt on
  sm_110 in every combination (measured), valid and correct here at
  108–173 TFLOPS. Halves activation-quantization error vs FP4 activations
  (see the quality table in BENCHMARKS.md).
- **MXFP4 × MXFP4**: *does not work* — the kernel runs but computes wrong
  values (relRMS ≈ 1.0) in every configuration tried, while the same data
  path validates for the two formats above; the built scale layouts are the
  correct vec-32 shape, so this is a CUTLASS-4.4.1-on-sm_110a gap. cuBLASLt
  refuses MXFP4 outright. Pure MXFP4 is not usable on this box today.

CUTLASS 4.4.1 (BSD-3) is not vendored; apply
[patches/cutlass-4.4.1-sm110.patch](patches/cutlass-4.4.1-sm110.patch) (an
`arch::Sm110` tag + a builder gate) and point `CUTLASS_ROOT` at it —
`make cutlass` explains, and patches/README.md has the exact commands.

## Known constraints (all measured; see BENCHMARKS.md)

- K must be a multiple of 64; N (= out_features) ≥ 8 and a multiple of 8.
  M (batch) is unconstrained (1, 3, 17, 255 tested).
- Activation and scale buffers are padded to 128-row tiles (the kernels read
  tail tiles); the library handles this — only raw `ops` users need care.
- Small-K layers: keep them in BF16 (`policy="auto"` does).
- `Linear.forward` returns a reused per-(M, dtype) buffer — `.clone()` it if
  you need it past the next call. (This is what makes chains graph-capturable.)
- FP4 layer chains capture into CUDA graphs; eager FP4 through ctypes is
  host-launch-bound (measured in BENCHMARKS.md §batched decisions).

## Repository layout

```
csrc/thor_fp4_core.cu      quantize (f32/f16/bf16), cuBLASLt GEMM (swap trick),
                           dequantize, MX quantize — the core extension
csrc/cutlass_kernels.cu    our sm_110a GEMM variants (C ABI tfc4_*)
csrc/bench_main.cu         standalone validation+timing for those variants
src/thor_fp4/              the Python package (linear, quant, convert, verify,
                           graphs, cascade, ops, _ext)
tests/                     63 tests: bit-exactness, tolerances, pitfalls,
                           verify injection, convert policies, QAT, capture
benchmarks/                bench_gemm / quality / batch / cascade drivers +
                           bench_cutlass_suite.sh; results in benchmarks/results
BENCHMARKS.md              every measured number with the box state beside it
patches/                   the two-line CUTLASS 4.4.1 sm_110 enablement
```

## Origin

Built from lab-fp4's findings (`~/lm-wt/lab-fp4`, exp-fp4-path.md and
exp-fp4-models.md, 2026-10-31): the cuBLASLt NVFP4 path at 371–456 TFLOPS,
the recovered block-scale layout, the measured layout/M/K pitfalls, and the
CUTLASS sm_110a enablement. MIT license (see LICENSE); CUTLASS is BSD-3 and
optional.

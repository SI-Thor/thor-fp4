#!/usr/bin/env python
# bench_certify2.py — ROUND 2: the certificate carried through the
# activations (interval arithmetic), measured on the tile scorer.
#
# Round 1 measured that the final-layer certificate covers 0.29% and that
# ~90% of FP4-vs-BF16 disagreement is upstream activation-quantization
# error. This run bounds that error too: per-code rounding intervals from
# the block scales the kernels actually used, propagated through |W|, the
# GELU Lipschitz constant, and the next quantization. Both activation
# precisions are measured: NVFP4 acts (cuBLASLt) and MXFP8 acts x MXFP4
# weights (our CUTLASS engine).
#
# Run: .venv/bin/python benchmarks/bench_certify2.py | tee benchmarks/results/certify2.txt

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scorer import boxstate, get_trained, test_batch, to_bf16, to_real_fp4  # noqa: E402

import thor_fp4  # noqa: E402

bf, _ = get_trained()
x = test_batch(8192)
bf16_m = to_bf16(bf)


def chain(m, t):
    outs = t
    for mod in m:
        outs = mod(outs) if isinstance(mod, thor_fp4.Linear) else mod(outs.float())
    return outs


def exact_chain(m, t):
    h = t.double()
    for mod in m:
        if isinstance(mod, thor_fp4.Linear):
            h = h @ mod.weight.detach().double().t() + mod.bias.detach().double()
        else:
            h = torch.nn.functional.gelu(h)
    return h


print(f"# thor-fp4 bench_certify2 (interval certificate)  cublasLt={thor_fp4.cublaslt_version()}  {boxstate()}")

engines = [("cublaslt", "NVFP4 acts x NVFP4 weights (cuBLASLt)")]
if thor_fp4._ext.cutlass() is not None:
    engines.append(("cutlass:mxfp8xfp4_128x128x256_c1",
                    "MXFP8 acts x MXFP4 weights (CUTLASS)"))

for engine, label in engines:
    fp4_m = to_real_fp4(bf, engine=engine)
    cert = thor_fp4.IntervalChainCertificate(fp4_m)
    with torch.no_grad():
        ok, bound, logits = cert.check(x)
        exact_choice = exact_chain(fp4_m, x).argmax(1)
        native_choice = chain(fp4_m, x).argmax(1)
        slow_choice = bf16_m(x.bfloat16()).argmax(1)
        n = len(x)
        ncert = int(ok.sum())
        sound = 100.0 if ncert == 0 else \
            100.0 * (native_choice[ok] == exact_choice[ok]).float().mean().item()
        t2 = logits.topk(2, 1).values
        margin = (t2[:, 0] - t2[:, 1])
        print(f"\n== {label}")
        print(f"certified share:            {100*ncert/n:6.2f}%  ({ncert}/{n})")
        print(f"sound (certified == exact): {sound:.1f}%   (must be 100)")
        print(f"fp4-vs-bf16 agreement:      {100*(native_choice == slow_choice).float().mean().item():6.1f}%")
        print(f"median interval bound {bound.median().item():.3f} vs median margin {margin.median().item():.3f}"
              f"  (bound/margin median {(bound/margin.clamp_min(1e-9)).median().item():.1f})")
        per = cert.last_layer_stats
        print(f"per-layer median bound:     " + " ".join(f"{v:.3f}" for v in per))
print(f"\n# {boxstate()}")

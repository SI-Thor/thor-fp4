# bench_quality.py — PTQ vs QAT quality on the tile-scorer task, through the
# REAL tensor cores, including the two-level per-tensor scale and the
# MXFP8-activation mix (fake quant, and real via the CUTLASS engine if built).
#
# Run: .venv/bin/python benchmarks/bench_quality.py 2>&1 | tee benchmarks/results/quality.txt

import sys
import time

import torch

from scorer import (boxstate, fp4_forward, get_trained, make_teacher, rrms,
                    test_batch, DIMS)

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
import thor_fp4  # noqa: E402

torch.manual_seed(0)
print(f"# thor-fp4 bench_quality  cublasLt={thor_fp4.cublaslt_version()}  {boxstate()}")
t0 = time.time()
bf, qa = get_trained()
print(f"# models trained/loaded in {time.time()-t0:.0f}s  {boxstate()}")

teacher = make_teacher()
x = test_batch()
with torch.no_grad():
    y_teacher = teacher(x)
    y_bf = bf(x)                       # the model's own BF16 behavior
gelu = torch.nn.functional.gelu


def fake_forward(m, x, act_ts=1.0, w_two_level=True):
    outs = x
    lins = [mod for mod in m if isinstance(mod, torch.nn.Linear)]
    for i, l in enumerate(lins):
        ts = thor_fp4.two_level_scale(l.weight) if w_two_level else 1.0
        outs = thor_fp4.nvfp4_fake_quant(outs, act_ts) @ thor_fp4.nvfp4_fake_quant(l.weight, ts).t() + l.bias
        if i < len(lins) - 1:
            outs = gelu(outs)
    return outs


def agree(a, b):
    return (a.argmax(1) == b.argmax(1)).float().mean().item()


rows = []


def report(name, yh, y_bf_ref=None):
    mse = torch.nn.functional.mse_loss(yh, y_teacher).item()
    keep = agree(yh, y_bf_ref if y_bf_ref is not None else y_bf)
    r = rrms(yh, y_bf)
    rows.append((name, mse, 100 * agree(yh, y_teacher), r, 100 * keep))
    print(f"  {name:<34} Q-MSE {mse:.4f}  top1-teacher {100*agree(yh, y_teacher):5.1f}%  "
          f"relRMS-vs-BF16 {r:.4f}  top1-keep {keep*100:5.1f}%")


print(f"\n== tile scorer {DIMS} (real tensor cores unless noted)  {boxstate()} ==")
with torch.no_grad():
    report("BF16 baseline", y_bf)
    # PTQ block-only, real
    report("PTQ real fp4 (block-only)", fp4_forward(bf, x, tensor_scale=1.0))
    # PTQ two-level, real
    report("PTQ real fp4 (two-level ts)", fp4_forward(bf, x))
    # PTQ two-level, fake (arithmetic parity)
    report("PTQ fake fp4 (two-level ts)", fake_forward(bf, x))
    # PTQ block-only, fake
    report("PTQ fake fp4 (block-only)", fake_forward(bf, x, w_two_level=False))
    # QAT, real + fake
    report("QAT real fp4 (block-only)", fp4_forward(qa, x, tensor_scale=1.0))
    report("QAT fake fp4 (block-only)", fake_forward(qa, x, w_two_level=False))

# activation-precision comparison: FP4 acts vs FP8 acts (weights FP4 both).
# FP8 acts through the fake quantizers (isolates the quantization effect);
# the real mixed path runs below if the CUTLASS engine is built.
print(f"\n== activation precision (weights MXFP4; fake quant)  {boxstate()} ==")
with torch.no_grad():
    for act_kind in ("mxfp4", "mxfp8"):
        outs = x
        lins = [mod for mod in bf if isinstance(mod, torch.nn.Linear)]
        for i, l in enumerate(lins):
            w = thor_fp4.mx_fake_quant(l.weight, "mxfp4")
            outs = thor_fp4.mx_fake_quant(outs, act_kind) @ w.t() + l.bias
            if i < len(lins) - 1:
                outs = gelu(outs)
        report(f"PTQ fake mx4w/{act_kind}a", outs)

try:
    l = thor_fp4._ext.cutlass()
    assert l is not None
    with torch.no_grad():
        y = fp4_forward(bf, x, engine="cutlass:mxfp8xfp4_128x128x256_c1")
        report("PTQ real mx8a x fp4w (CUTLASS)", y)
except Exception as e:
    print(f"# (mixed real path unavailable: {type(e).__name__}: {e})")

print(f"\n# done  {boxstate()}")

# ---- ROUND 2 additions ----
print(f"\n== ROUND 2: tuned mx8 engine, GEMV activations (exact), fused chain  {boxstate()} ==")
with torch.no_grad():
    try:
        y = fp4_forward(bf, x, engine="cutlass:mx8")
        report("PTQ real mx8a x fp4w (tuned)", y)
    except Exception as e:
        print(f"# (tuned mx8 unavailable: {type(e).__name__}: {e})")
    # the GEMV path consumes activations EXACTLY (only weights quantized):
    # its quality is the upper bound for any FP4-weight path. Runs in chunks
    # of 8 (the engine's batch limit).
    try:
        outs = []
        for i in range(0, len(x), 8):
            outs.append(fp4_forward(bf, x[i:i+8], engine="gemv"))
        report("PTQ real fp4w, GEMV (exact acts)", torch.cat(outs))
    except Exception as e:
        print(f"# (gemv quality unavailable: {type(e).__name__}: {e})")
    # the fused chain (one kernel per layer) if the fused lib is built
    try:
        chain = thor_fp4.mx8_chain(*DIMS).cuda()
        lins = [mod for mod in bf if isinstance(mod, torch.nn.Linear)]
        fl = [m for m in chain if isinstance(m, thor_fp4.FusedLinear)]
        flin = [m for m in chain if isinstance(m, thor_fp4.Linear)]
        with torch.no_grad():
            for f, l in zip(fl + flin, lins):
                f.weight.copy_(l.weight); f.bias.copy_(l.bias)
        report("PTQ real fused mx8 chain", chain(x).float())
    except Exception as e:
        print(f"# (fused chain unavailable: {type(e).__name__}: {e})")
print(f"\n# done  {boxstate()}")

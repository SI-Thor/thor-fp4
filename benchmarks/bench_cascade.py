# bench_cascade.py — the cleanup-loop measurement (BRIEF §5): an FP4
# tile scorer keeps its BF16 self's top-1 choice ~71% of the time; how much
# of that gap does an escalate-when-unsure loop recover, and at what cost?
#
# Confidence is read from the FP4 logits alone (no extra pass). Cost is
# reported three ways:
#   * escalation rate (the fraction of samples re-run in BF16),
#   * measured latency per batch on the real paths (eager + graphed),
#   * model memory (the FP4 model is what stays resident; the BF16 copy is
#     the escalation target — at this model scale FP4's win is residency,
#     not speed, per lab-fp4 exp-fp4-models.md §D).
#
# Run: .venv/bin/python benchmarks/bench_cascade.py 2>&1 | tee benchmarks/results/cascade.txt

import statistics
import sys
import time

import torch

from scorer import boxstate, get_trained, test_batch, to_bf16, to_real_fp4

sys.path.insert(0, __file__.rsplit("/", 1)[0])
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


bf, qa = get_trained()
x = test_batch(16384)
bf16_m = to_bf16(bf)

print(f"# thor-fp4 bench_cascade  cublasLt={thor_fp4.cublaslt_version()}  {boxstate()}")
for weights_name, weights in (("PTQ (bf16-trained)", bf), ("QAT", qa)):
    fp4_m = to_real_fp4(weights)

    def fp4_chain(t):
        outs = t
        for mod in fp4_m:
            outs = mod(outs) if isinstance(mod, thor_fp4.Linear) else mod(outs.float())
        return outs

    with torch.no_grad():
        slow_choice = bf16_m(x.bfloat16()).float().argmax(1)
        fast_logits = fp4_chain(x.half()).float()
        fast_choice = fast_logits.argmax(1)
        base_agree = (fast_choice == slow_choice).float().mean().item()
        gap = 1.0 - base_agree
        print(f"\n== {weights_name}: fp4-vs-bf16 top-1 agreement {base_agree*100:.1f}% "
              f"(gap to close: {gap*100:.1f}% of samples)  {boxstate()}")

        rows = thor_fp4.cascade_sweep(lambda t: fp4_chain(t.half()), lambda t: bf16_m(t.bfloat16()),
                                      x.half(), y_bf16_choice=slow_choice,
                                      confidences=(thor_fp4.confidence_margin,
                                                   thor_fp4.confidence_softmax))
        print(f"# {'confidence':>18} {'threshold':>10} {'escalated':>10} {'agreement':>10} {'gap recovered':>13}")
        # show a compact sweep: the margin confidence, 8 points
        for r in [r for r in rows if r["confidence"] == "confidence_margin"]:
            print(f"{'':>18} {r['threshold']:>10.3f} {100*r['escalated']:>9.1f}% "
                  f"{100*r['agreement']:>9.1f}% {100*r['recovered_gap']:>12.1f}%")
        best_softmax = [r for r in rows if r["confidence"] == "confidence_softmax"]
        if best_softmax:
            r = min(best_softmax, key=lambda r: abs(r["escalated"] - 0.5))
            print(f"{'softmax (mid)':>18} {r['threshold']:>10.3f} {100*r['escalated']:>9.1f}% "
                  f"{100*r['agreement']:>9.1f}% {100*r['recovered_gap']:>12.1f}%")

    # latency of the loop at a mid threshold on B=256 (eager, real paths)
    thr_row = min([r for r in rows if r["confidence"] == "confidence_margin"],
                  key=lambda r: abs(r["escalated"] - 0.3))
    cas = thor_fp4.Cascade(fp4_m, bf16_m, thr_row["threshold"])
    xb = x[:256]

    def cascade_eager():
        return cas(xb.half())

    warm_clock(0.5)
    with torch.no_grad():
        t_cas = timeit(cascade_eager)
        t_fp4 = timeit(lambda: fp4_chain(xb.half()))
        t_bf = timeit(lambda: bf16_m(xb.bfloat16()))
    print(f"# latency B=256 eager: cascade {t_cas*1e3:.1f} us  (fp4 {t_fp4*1e3:.1f} us, "
          f"bf16 {t_bf*1e3:.1f} us; escalation rate on this batch "
          f"{cas(xb.half())[1].float().mean().item()*100:.0f}%)  {boxstate()}")

    wbytes_bf = sum(p.numel() * 2 for p in bf16_m.parameters())
    wbytes_fp4 = sum(m.fp4_weight_nbytes() for m in fp4_m if isinstance(m, thor_fp4.Linear))
    print(f"# resident bytes: bf16 {wbytes_bf/1e6:.2f} MB  fp4 {wbytes_fp4/1e6:.2f} MB "
          f"({wbytes_bf/wbytes_fp4:.2f}x)")
print(f"\n# done  {boxstate()}")

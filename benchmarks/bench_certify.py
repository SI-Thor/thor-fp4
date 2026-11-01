# bench_certify.py — the arithmetic certificate (DIRECTIONS.md §1) measured
# on the tile scorer: what share of decisions certify, is the certificate
# sound, how much does final-layer promotion recover, and how that compares
# with the learned-confidence cascade (which escalates the whole model).
#
# Run: .venv/bin/python benchmarks/bench_certify.py 2>&1 | tee benchmarks/results/certify.txt

import sys

import torch

from scorer import boxstate, get_trained, test_batch, to_bf16, to_real_fp4

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import thor_fp4  # noqa: E402

bf, _ = get_trained()
x = test_batch(16384)
fp4_m = to_real_fp4(bf)
bf16_m = to_bf16(bf)


def chain(m, t):
    outs = t
    for mod in m:
        outs = mod(outs) if isinstance(mod, thor_fp4.Linear) else mod(outs.float())
    return outs


print(f"# thor-fp4 bench_certify  cublasLt={thor_fp4.cublaslt_version()}  {boxstate()}")
cert = thor_fp4.FinalLayerCertificate(fp4_m[4])
with torch.no_grad():
    logits = chain(fp4_m, x.half()).float()
    a = x.half()
    for mod in fp4_m[:4]:
        a = mod(a) if isinstance(mod, thor_fp4.Linear) else mod(a.float())
    certified, bound = cert.check(a.float(), logits)
    true_final = a.float() @ fp4_m[4].weight.detach().float().t() + fp4_m[4].bias.float()
    slow_choice = bf16_m(x.bfloat16()).float().argmax(1)
    fp4_choice = logits.argmax(1)
    true_choice = true_final.argmax(1)

    n_cert = int(certified.sum())
    print(f"certified share:                {100*n_cert/len(x):6.2f}%   ({n_cert}/{len(x)})")
    print(f"certificate sound (vs true-weights-on-same-acts): "
          f"{100*(fp4_choice[certified] == true_choice[certified]).float().mean().item():.1f}%  (must be 100)")
    print(f"fp4 agreement with bf16:        {100*(fp4_choice == slow_choice).float().mean().item():6.1f}%")
    promoted = fp4_choice.clone()
    promoted[~certified] = true_choice[~certified]
    print(f"after final-layer promotion:    {100*(promoted == slow_choice).float().mean().item():6.1f}%  "
          f"(final-layer exactness alone recovers "
          f"{100*((promoted == slow_choice).float().mean()-(fp4_choice == slow_choice).float().mean()).item():.1f} pts)")
    # how much of the gap is final-layer weight error at all? promote EVERYTHING
    allpromo = true_choice
    print(f"promote ALL (upper bound):      {100*(allpromo == slow_choice).float().mean().item():6.1f}%")
    print(f"# (gap closed by final-layer exactness: "
          f"{100*((allpromo == slow_choice).float().mean()-(fp4_choice == slow_choice).float().mean()).item():.1f} pts of "
          f"{100*(1-(fp4_choice == slow_choice).float().mean().item()):.1f} — the rest is upstream activation error)")
    m = (bound.median().item())
    mg = (logits.topk(2, 1).values[:, 0] - logits.topk(2, 1).values[:, 1]).median().item()
    print(f"# median logit bound {m:.3f} vs median margin {mg:.3f} — the triangle-inequality "
          f"bound is ~sqrt(K)x the typical random-sign error, so coverage is thin at K=1024")
print(f"# {boxstate()}")

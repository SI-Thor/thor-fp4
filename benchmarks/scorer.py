# scorer.py — the tile-scorer task from lab-fp4 (exp-fp4-models.md §3):
# a decision MLP 512->1024->1024->64 over 64 candidate tiles, distilled from
# a fixed seeded teacher. Shared by bench_quality / bench_batch / bench_cascade.

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import thor_fp4  # noqa: E402

DIMS = [512, 1024, 1024, 64]
RESULTS = Path(__file__).resolve().parent / "results"


def boxstate() -> str:
    load = open("/proc/loadavg").read().split()[0]
    avail = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 2**20
    return f"load={load} mem_avail={avail:.0f}GiB"


def make_teacher(seed=123):
    g = torch.Generator(device="cpu").manual_seed(seed)
    layers = []
    for i in range(len(DIMS) - 1):
        W = (torch.randn((DIMS[i + 1], DIMS[i]), generator=g) * (1.0 / DIMS[i] ** 0.5)).cuda()
        b = torch.zeros(DIMS[i + 1]).cuda()
        layers.append((W, b))

    def teacher(x):
        for W, b in layers[:-1]:
            x = torch.nn.functional.gelu(x @ W.T + b)
        W, b = layers[-1]
        return x @ W.T + b
    return teacher


def make_mlp():
    return nn.Sequential(nn.Linear(DIMS[0], DIMS[1]), nn.GELU(),
                         nn.Linear(DIMS[1], DIMS[2]), nn.GELU(),
                         nn.Linear(DIMS[2], DIMS[3]))


def train(qat_mode: str = "bf16", steps=2500, bs=2048, seed=0, lr=2e-3):
    """qat_mode: 'bf16' | 'qat' (STE fake-quant weights+activations)."""
    torch.manual_seed(seed)
    teacher = make_teacher()
    m = make_mlp().cuda()
    if qat_mode == "qat":
        m = nn.Sequential(thor_fp4.QATLinear(DIMS[0], DIMS[1]), nn.GELU(),
                          thor_fp4.QATLinear(DIMS[1], DIMS[2]), nn.GELU(),
                          thor_fp4.QATLinear(DIMS[2], DIMS[3])).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device="cpu").manual_seed(999)
    for step in range(steps):
        x = torch.randn((bs, DIMS[0]), generator=g).cuda()
        with torch.no_grad():
            y = teacher(x)
        loss = (m(x) - y).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return m, loss.item()


def test_batch(n=8192, seed=4242):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((n, DIMS[0]), generator=g).cuda()


def to_real_fp4(m, engine="cublaslt", **kw):
    """an nn.Sequential of thor_fp4.Linear + GELU from trained weights."""
    lins = [mod for mod in m if isinstance(mod, nn.Linear)]
    mods = []
    for i, l in enumerate(lins):
        lin = thor_fp4.Linear(l.in_features, l.out_features, bias=True,
                              engine=engine, **kw).cuda()
        with torch.no_grad():
            lin.weight.copy_(l.weight); lin.bias.copy_(l.bias)
        mods.append(lin)
        if i < len(lins) - 1:
            mods.append(nn.GELU())
    return nn.Sequential(*mods)


def to_bf16(m):
    lins = [mod for mod in m if isinstance(mod, nn.Linear)]
    mods = []
    for i, l in enumerate(lins):
        lin = nn.Linear(l.in_features, l.out_features).cuda().bfloat16()
        with torch.no_grad():
            lin.weight.copy_(l.weight.bfloat16()); lin.bias.copy_(l.bias.bfloat16())
        mods.append(lin)
        if i < len(lins) - 1:
            mods.append(nn.GELU())
    return nn.Sequential(*mods)


def fp4_forward(m, x, engine="cublaslt", **kw):
    """run an mlp through the real FP4 tensor cores using trained weights."""
    real = to_real_fp4(m, engine=engine, **kw)
    outs = x.float()
    for mod in real:
        if isinstance(mod, thor_fp4.Linear):
            outs = mod(outs)
        else:
            outs = mod(outs.float())   # gelu in f32 between fp4 layers
    return outs.float()


def rrms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def get_trained(cache=True):
    """(bf16-trained model, qat-trained model), cached on disk."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "scorer.pt"
    if cache and path.exists():
        d = torch.load(path, weights_only=False)
        bf, qa = make_mlp(), make_mlp()
        bf.load_state_dict(d["bf16"]); qa.load_state_dict(d["qat"])
        return bf.cuda(), qa.cuda()
    bf, _ = train("bf16")
    qa, _ = train("qat")
    torch.save({"bf16": bf.state_dict(), "qat": qa.state_dict()}, path)
    return bf.cuda(), qa.cuda()

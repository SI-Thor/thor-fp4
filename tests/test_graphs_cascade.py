# CUDA-graph capture and the cascade (FP4-first, escalate-when-unsure).

import pytest
import torch
import torch.nn as nn

import thor_fp4
from conftest import requires_cuda, rrms


class MLP(nn.Module):
    def __init__(self, dims=(512, 1024, 64)):
        super().__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x))).float()


@requires_cuda
def test_graph_replay_matches_eager():
    m = thor_fp4.convert(MLP().cuda(), policy="fp4")
    x = torch.randn(64, 512, device="cuda")
    g = thor_fp4.capture_module(m, x)
    eager = m(x).clone()
    out1 = g(x).clone()
    out2 = g(x).clone()
    assert torch.equal(out1, out2)
    assert rrms(out1, eager) < 1e-3


@requires_cuda
def test_graphed_static_output_semantics():
    m = thor_fp4.convert(MLP().cuda(), policy="fp4")
    x = torch.randn(64, 512, device="cuda")
    g = thor_fp4.capture_module(m, x)
    a = g(x)
    keep = a.clone()
    g(x + 1)
    assert torch.equal(a, keep) is False   # same buffer, new contents


@requires_cuda
def test_cascade_extremes():
    torch.manual_seed(0)
    bf = MLP().cuda()
    fp4 = thor_fp4.convert(bf, policy="fp4")
    x = torch.randn(256, 512, device="cuda")
    # threshold above every confidence -> everything escalates -> slow model
    c_all = thor_fp4.Cascade(fp4, bf, threshold=float("inf"))
    out, esc, conf = c_all(x)
    assert esc.all() and torch.equal(out, bf(x).float())
    # threshold below every confidence -> pure FP4
    c_none = thor_fp4.Cascade(fp4, bf, threshold=-1.0)
    out2, esc2, _ = c_none(x)
    assert esc2.sum() == 0 and torch.equal(out2, fp4(x).float())


@requires_cuda
def test_cascade_sweep_recovers():
    torch.manual_seed(0)
    bf = MLP().cuda()
    # a deliberately noisy fast model: quantized copy of bf
    fp4 = thor_fp4.convert(bf, policy="fp4")
    x = torch.randn(512, 512, device="cuda")
    rows = thor_fp4.cascade_sweep(fp4, bf, x)
    assert len(rows) > 5
    # at ~100% escalation the agreement must reach 1.0
    assert rows[-1]["agreement"] == pytest.approx(1.0)
    assert rows[-1]["escalated"] == pytest.approx(0.0, abs=0.05) or \
           rows[-1]["agreement"] > rows[0]["agreement"]
    # monotone-ish: agreement never drops far below the no-escalation point
    base = rows[0]["agreement"]
    assert all(r["agreement"] >= base - 0.02 for r in rows)

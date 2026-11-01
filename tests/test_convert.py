# convert(): per-layer precision routing. FP4 only where the hardware takes
# it and the measured shape table says it pays; BF16 elsewhere (the critic
# case); "auto" reads the shipped table.

import pytest
import torch
import torch.nn as nn

import thor_fp4
from conftest import requires_cuda


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.small = nn.Linear(300, 128)      # K not a multiple of 64
        self.head = nn.Linear(128, 64)
        self.critic = nn.Linear(512, 256)

    def forward(self, x):
        h = self.head(self.small(x))
        c = self.critic(torch.nn.functional.pad(x, (0, 512 - x.shape[1])))
        return h + c[:, :64]


@requires_cuda
def test_convert_eligibility():
    m = thor_fp4.convert(Toy().cuda(), policy="fp4")
    assert isinstance(m.head, thor_fp4.Linear)
    assert not isinstance(m.small, thor_fp4.Linear)      # K=300 % 64 != 0
    assert m.thor_fp4_policy["small"].startswith("bf16")
    assert m.thor_fp4_policy["head"] == "fp4"


@requires_cuda
def test_convert_per_layer_policy():
    m = thor_fp4.convert(Toy().cuda(), policy={"critic": "bf16", "head": "fp4"})
    assert m.thor_fp4_policy["critic"] == "bf16"
    assert isinstance(m.head, thor_fp4.Linear)
    assert not isinstance(m.critic, thor_fp4.Linear)


@requires_cuda
def test_convert_pattern_policy():
    m = thor_fp4.convert(Toy().cuda(), policy={"crit*": "bf16", "*": "auto"})
    assert m.thor_fp4_policy["critic"] == "bf16"


@requires_cuda
def test_auto_reads_measured_table():
    table_small = {"crossover": {"min_k": 4096, "min_n": 4096}, "shapes": {}}
    d = thor_fp4.auto_decision(1024, 512, table_small)
    assert d.spec == "bf16" and "4096" in d.reason
    table_measured = {"crossover": {"min_k": 2048, "min_n": 512},
                      "shapes": {"128x320": {"fp4_ms": 0.02, "bf16_ms": 0.05,
                                             "fp4_wins": True, "m": 4096}}}
    d2 = thor_fp4.auto_decision(320, 128, table_measured)
    assert d2.spec == "fp4" and "measured" in d2.reason
    d3 = thor_fp4.auto_decision(64, 64, table_measured)
    assert d3.spec == "bf16"


@requires_cuda
def test_convert_output_matches_direct_fp4():
    m = Toy().cuda()
    x = torch.randn(16, 320, device="cuda")
    mc = thor_fp4.convert(m, policy={"small": "bf16", "head": "fp4", "critic": "fp4"})
    # the fp4 head must reproduce a directly-built Linear on the same weights
    direct = thor_fp4.Linear(128, 64).cuda()
    with torch.no_grad():
        direct.weight.copy_(m.head.weight)
        direct.bias.copy_(m.head.bias)
    torch.manual_seed(1)
    h = torch.randn(16, 128, device="cuda")
    assert torch.equal(mc.head(h), direct(h))


@requires_cuda
def test_convert_calibration_plumbed():
    m = Toy().cuda()
    cal = thor_fp4.calibrate(m, torch.randn(32, 300, device="cuda"))
    assert set(cal) == {"small", "head", "critic"}
    mc = thor_fp4.convert(m, policy="fp4", calibration=cal)
    assert mc.head.act_tensor_scale == cal["head"]


@requires_cuda
def test_convert_deep_copy_default():
    m = Toy().cuda()
    mc = thor_fp4.convert(m, policy="fp4")
    assert mc is not m
    with torch.no_grad():
        m.head.weight.zero_()
    assert mc.head.weight.abs().max().item() > 0

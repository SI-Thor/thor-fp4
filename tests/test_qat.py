# QAT: the STE fake-quant path must train, and a QAT-trained model served
# through the real tensor cores must match its training-time arithmetic to
# output rounding (that is the point of mirroring the kernels exactly).

import pytest
import torch
import torch.nn as nn

import thor_fp4
from conftest import requires_cuda, rrms


@requires_cuda
def test_ste_gradient_is_identity():
    # the STE must pass gradients unchanged: a broken version that builds the
    # graph through bucketize kills them (found by bench_quality, kept as a
    # regression test)
    x = torch.randn(64, 128, device="cuda", requires_grad=True)
    y = thor_fp4.qat(x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@requires_cuda
def test_qat_linear_trains():
    torch.manual_seed(0)
    teacher = nn.Linear(256, 128).cuda()
    q = thor_fp4.QATLinear(256, 128).cuda()
    opt = torch.optim.Adam(q.parameters(), lr=1e-2)
    x = torch.randn(512, 256, device="cuda")
    with torch.no_grad():
        y = teacher(x)
    first = None
    for i in range(30):
        loss = (q(x) - y).pow(2).mean()
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 4-bit weights + 4-bit activations put a high floor on matching a random
    # teacher; the check is that QAT moves clearly toward it
    assert loss.item() < first * 0.6


@requires_cuda
@pytest.mark.parametrize("two_level", [False, True])
def test_qat_model_serves_through_real_tensor_cores(two_level):
    # train a tiny 2-layer QAT MLP, then run the same weights as a real FP4
    # model: outputs must agree with the training-time fake quant to f16
    # output rounding
    torch.manual_seed(0)
    dims = [256, 512, 128]
    qat = nn.Sequential(
        thor_fp4.QATLinear(dims[0], dims[1], two_level=two_level),
        nn.GELU(),
        thor_fp4.QATLinear(dims[1], dims[2], two_level=two_level),
    ).cuda()
    opt = torch.optim.Adam(qat.parameters(), lr=1e-3)
    x = torch.randn(1024, dims[0], device="cuda")
    target = torch.randn(1024, dims[2], device="cuda")
    for _ in range(20):
        loss = (qat(x) - target).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def real(model, inp):
        with torch.no_grad():
            outs = inp.float()
            for i in range(0, len(model), 2):
                l = model[i]
                ts = thor_fp4.two_level_scale(l.weight) if l.two_level else 1.0
                outs = thor_fp4.nvfp4_fake_quant(outs) @ thor_fp4.nvfp4_fake_quant(l.weight.detach(), ts).t() + l.bias
                if i + 1 < len(model):
                    outs = torch.nn.functional.gelu(outs)
            return outs

    with torch.no_grad():
        fake = qat(x)
        real_out = real(qat, x)
    assert rrms(real_out, fake) < 2e-3, rrms(real_out, fake)

    # and through the actual library
    lin1 = thor_fp4.Linear(dims[0], dims[1], tensor_scale=None if two_level else 1.0).cuda()
    lin2 = thor_fp4.Linear(dims[1], dims[2], tensor_scale=None if two_level else 1.0).cuda()
    with torch.no_grad():
        lin1.weight.copy_(qat[0].weight); lin1.bias.copy_(qat[0].bias)
        lin2.weight.copy_(qat[2].weight); lin2.bias.copy_(qat[2].bias)
    y = lin2(torch.nn.functional.gelu(lin1(x).float()))
    assert rrms(y.float(), fake) < 2e-2, rrms(y.float(), fake)


@requires_cuda
def test_calibrate_linear_amax():
    l = nn.Linear(64, 64).cuda()
    x = torch.randn(128, 64, device="cuda") * 3
    s = thor_fp4.calibrate_linear(l, x)
    assert s == pytest.approx(x.abs().max().item() / (6.0 * 448.0))

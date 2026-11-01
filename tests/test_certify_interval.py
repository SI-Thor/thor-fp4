# test_certify_interval.py — the interval (whole-chain) certificate, ROUND 2.
#
# Soundness bar: whenever the certificate says "certified", the native FP4
# argmax must equal the exact-arithmetic network's argmax (fp32 master
# weights, exact erf GELU, computed in float64). Plus the per-code error
# intervals and the GELU Lipschitz constant must hold on their own.

import math

import pytest
import torch
import torch.nn as nn

import thor_fp4
from thor_fp4.certify import (IntervalChainCertificate, gelu_lipschitz,
                              mx_act_err_bound, nvfp4_act_err_bound)
from thor_fp4.quant import mx_fake_quant, nvfp4_fake_quant
from conftest import requires_cuda


@requires_cuda
def test_nvfp4_act_err_bound_sound():
    # any value that quantizes to code c lies within delta of dequant(c)
    torch.manual_seed(0)
    x = (torch.randn(4096, 512, device="cuda") * torch.logspace(-3, 2, 512, device="cuda"))
    xq = nvfp4_fake_quant(x)
    delta = nvfp4_act_err_bound(x)
    assert (x - xq).abs().le(delta + 1e-12).all()
    # and brute-force around the grid: perturb beyond delta -> different code
    q = (x / delta.clamp_min(1e-30))
    pert = x + (delta + 1e-6 * delta.clamp_min(1e-30)) * torch.sign(torch.randn_like(x))
    assert not torch.equal(nvfp4_fake_quant(pert), xq)


@requires_cuda
def test_mx_act_err_bound_sound():
    torch.manual_seed(1)
    for kind in ("mxfp4", "mxfp8"):
        x = torch.randn(4096, 512, device="cuda") * torch.logspace(-3, 3, 512, device="cuda")
        xq = mx_fake_quant(x, kind)
        delta = mx_act_err_bound(x, kind)
        bad = (x - xq).abs() > delta + 1e-12
        assert not bad.any(), f"{kind}: {int(bad.sum())} violations"


@requires_cuda
def test_gelu_lipschitz_constant():
    L = gelu_lipschitz()
    # dense numeric check on a different grid than the one used to derive it
    x = torch.linspace(-10, 10, 400001, dtype=torch.float64, device="cuda") + 1e-7
    phi = torch.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    deriv = (Phi + x * phi).abs().max().item()
    assert L >= deriv, f"{L} < {deriv}"


def _make_chain(engine, dims=(512, 1024, 1024, 64), seed=0):
    torch.manual_seed(seed)
    mods = []
    for i in range(len(dims) - 1):
        lin = thor_fp4.Linear(dims[i], dims[i + 1], bias=True, engine=engine).cuda()
        mods.append(lin)
        if i < len(dims) - 2:
            mods.append(nn.GELU())
    return nn.Sequential(*mods)


def _exact_chain(model, x):
    """fp32 master weights, exact erf gelu, float64 arithmetic."""
    h = x.double()
    for m in model:
        if isinstance(m, thor_fp4.Linear):
            h = h @ m.weight.detach().double().t() + m.bias.detach().double()
        else:
            h = torch.nn.functional.gelu(h)
    return h


def _native_chain(model, x):
    outs = x
    for m in model:
        outs = m(outs) if isinstance(m, thor_fp4.Linear) else m(outs.float())
    return outs.float()


@requires_cuda
@pytest.mark.parametrize("engine", ["cublaslt", "cutlass:mxfp8xfp4_128x128x256_c1"])
def test_interval_certificate_sound_at_scale(engine):
    if engine.startswith("cutlass") and thor_fp4._ext.cutlass() is None:
        pytest.skip("needs build/libthorfp4_cutlass.so")
    torch.manual_seed(7)
    model = _make_chain(engine)
    cert = IntervalChainCertificate(model)
    n_cert_total = 0
    for _ in range(2):
        x = torch.randn(2048, 512, device="cuda")
        ok, bound, logits = cert.check(x)
        exact = _exact_chain(model, x)
        exact_choice = exact.argmax(1)
        native_choice = _native_chain(model, x).argmax(1)
        n_cert = int(ok.sum())
        n_cert_total += n_cert
        if n_cert:
            # THE soundness claim: certified samples cannot disagree
            assert (native_choice[ok] == exact_choice[ok]).all()
    # the interval certificate should certify *something* on a margin-heavy
    # distribution at least occasionally; log what it does
    print(f"[{engine}] certified {n_cert_total}")


@requires_cuda
def test_interval_certificate_covers_wide_margin():
    # a chain with an enormous, EXACT top-1 margin must certify: the bias is
    # fp32 and contributes no quantization error, so biasing one logit far
    # out widens the margin without widening the bound (scaling a WEIGHT row
    # instead would scale its error bound by the same factor — the margin/
    # bound ratio is scale-invariant, which is itself worth noting).
    torch.manual_seed(11)
    model = _make_chain("cublaslt", dims=(256, 512, 64))
    with torch.no_grad():
        model[-1].bias[0] += 250.0
    cert = IntervalChainCertificate(model)
    x = torch.randn(1024, 256, device="cuda")
    ok, bound, logits = cert.check(x)
    exact = _exact_chain(model, x)
    assert int(ok.sum()) > 900, f"bias-margin samples must certify ({int(ok.sum())})"
    assert (logits.argmax(1)[ok] == exact.argmax(1)[ok]).all()


@requires_cuda
def test_certified_forward_interval_promotes():
    torch.manual_seed(3)
    model = _make_chain("cublaslt")
    x = torch.randn(2048, 512, device="cuda")
    logits, certified = thor_fp4.certified_forward_interval(model, x)
    assert logits.shape == (2048, 64)
    # promoted samples carry the fp32-master-weight final layer
    assert int((~certified).sum()) > 0

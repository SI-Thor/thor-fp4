# The final-layer certificate: sound by construction — every certified
# sample's argmax must match the true-weights argmax on the SAME (quantized)
# activations. Coverage is a separate question: the triangle-inequality bound
# is ~sqrt(K) looser than typical random-sign error, so on flat-margin models
# it covers little (measured 0.29% on the trained tile scorer — see
# benchmarks/results/certify.txt). The tests exercise the certified path on a
# layer whose weights are exactly NVFP4-representable (zero weight error ->
# wide coverage) and check soundness at natural margins elsewhere.
# (An earlier version "exercised" the path by amplifying the logits — that is
# unsound by construction, scaling the margin without scaling the error
# bounds, and the soundness assertion caught it. Kept as a note.)

import pytest
import torch
import torch.nn as nn

import thor_fp4
from thor_fp4 import ops
from conftest import requires_cuda


class MLP(nn.Module):
    def __init__(self, dims=(512, 1024, 64)):
        super().__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x))).float()


def _run_and_capture(m, x):
    acts = {}
    h = m.fc2.register_forward_hook(lambda mod, i, o: acts.__setitem__("x", i[0]))
    logits = m(x).clone()          # fc2's output buffer is reused per shape
    h.remove()
    return logits, acts["x"].float()


def _true_out_on_consumed_acts(layer, x):
    """true weights applied to the activations the hardware actually
    consumed (dequantized codes) — the certificate's stated scope"""
    p, s = ops.quantize_nvfp4(x, layer.act_tensor_scale)
    xhat = ops.dequantize_nvfp4(p, s, layer.act_tensor_scale, *x.shape)
    return xhat @ layer.weight.detach().float().t() + layer.bias.float()


@requires_cuda
def test_certificate_sound_and_covers_clean_weights():
    torch.manual_seed(0)
    m = thor_fp4.convert(MLP().cuda(), policy="fp4")
    # make the last layer's weights exactly representable in NVFP4: the
    # exact error matrix is then 0 and most samples certify — this exercises
    # the certified path with the real bound machinery
    with torch.no_grad():
        m.fc2.refresh_weight()
        m.fc2.weight.copy_(m.fc2._weight_fp4.dequantize())
        m.fc2.refresh_weight()
    cert = thor_fp4.FinalLayerCertificate(m.fc2)
    x = torch.randn(2048, 512, device="cuda")
    with torch.no_grad():
        logits, acts = _run_and_capture(m, x)
        true_choice = _true_out_on_consumed_acts(m.fc2, acts).argmax(1)
        certified, _ = cert.check(acts, logits)
        assert certified.float().mean().item() > 0.5
        assert (logits[certified].argmax(1) == true_choice[certified]).all(), \
            "a certified sample disagreed — the certificate is UNSOUND"


@requires_cuda
def test_certificate_sound_with_quantization_error():
    torch.manual_seed(0)
    m = thor_fp4.convert(MLP().cuda(), policy="fp4")
    cert = thor_fp4.FinalLayerCertificate(m.fc2)
    for trial in range(3):
        x = torch.randn(2048, 512, device="cuda") * (1 + trial)
        with torch.no_grad():
            logits, acts = _run_and_capture(m, x)
            true_choice = _true_out_on_consumed_acts(m.fc2, acts).argmax(1)
            certified, _ = cert.check(acts, logits)
            # natural coverage on a random-init model is thin; soundness is
            # what must hold wherever it certifies
            if certified.any():
                assert (logits[certified].argmax(1) == true_choice[certified]).all(), \
                    "a certified sample disagreed — the certificate is UNSOUND"


@requires_cuda
def test_certified_forward_promotion_is_exact():
    torch.manual_seed(0)
    m = thor_fp4.convert(MLP().cuda(), policy="fp4")
    x = torch.randn(1024, 512, device="cuda")
    with torch.no_grad():
        logits, certified = thor_fp4.certified_forward(m, x)
        logits = logits.clone()
        _, acts = _run_and_capture(m, x)
        # promotion recomputes uncertified samples with the true weights on
        # the raw activations (the best available recompute short of
        # re-running the network)
        true_out = acts @ m.fc2.weight.detach().float().t() + m.fc2.bias.float()
        unc = ~certified
        assert unc.any()
        assert torch.allclose(logits[unc], true_out[unc], atol=2e-2)

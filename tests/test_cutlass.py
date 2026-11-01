# The CUTLASS sm_110a engines (skipped when the optional .so is not built):
# NVFP4xNVFP4 must match the cuBLASLt path to output rounding; the mixed
# MXFP8-activation x MXFP4-weight engine must match its torch fake-quant
# reference (the device MX encoders and the fake quantizer use independent
# UE8M0 rounding, so the tolerance is looser than the NVFP4 bit-exactness).

import pytest
import torch

import thor_fp4
from conftest import requires_cuda, rrms

has_cutlass = thor_fp4._ext.cutlass() is not None
cutlass = pytest.mark.skipif(not has_cutlass, reason="make cutlass")


@cutlass
@requires_cuda
@pytest.mark.parametrize("M", [1, 8, 17, 128])
def test_cutlass_nvfp4_matches_cublaslt(M):
    lin = thor_fp4.Linear(1024, 512, engine="cutlass").cuda()   # default = an nvfp4 variant
    lt = thor_fp4.Linear(1024, 512, engine="cublaslt").cuda()
    with torch.no_grad():
        lt.weight.copy_(lin.weight); lt.bias.copy_(lin.bias)
    x = torch.randn(M, 1024, device="cuda")
    a = lin(x).float()         # bf16 out
    b = lt(x).float()          # f16 out
    assert rrms(a, b) < 8e-3, rrms(a, b)     # bf16 + f16 output rounding


@cutlass
@requires_cuda
def test_cutlass_nvfp4_dual_path_verify():
    lin = thor_fp4.Linear(1024, 512, engine="cutlass").cuda()
    rep = thor_fp4.verify(lin, torch.randn(64, 1024, device="cuda"), bound=3e-2)
    assert rep.ok, str(rep)


@cutlass
@requires_cuda
def test_cutlass_mixed_f8_acts():
    # FP8 activations x FP4 weights: the operand mix cuBLASLt refuses
    lin = thor_fp4.Linear(1024, 512, engine="cutlass:mxfp8xfp4_128x128x256_c1").cuda()
    assert lin.engine_kind == "mx8"
    x = torch.randn(64, 1024, device="cuda")
    y = lin(x)
    assert y.dtype == torch.bfloat16
    w = thor_fp4.mx_fake_quant(lin.weight.detach(), "mxfp4")
    ref = thor_fp4.mx_fake_quant(x, "mxfp8") @ w.t() + lin.bias.float()
    assert rrms(y.float(), ref) < 2e-2, rrms(y.float(), ref)


@cutlass
@requires_cuda
def test_cutlass_graph_capture():
    lin = thor_fp4.Linear(512, 256, engine="cutlass").cuda()
    x = torch.randn(32, 512, device="cuda")
    g = thor_fp4.Graphed(lambda t: lin(t), x)
    assert torch.equal(g(x).clone(), g(x).clone())
    assert rrms(g(x).float(), lin(x).float()) < 1e-3

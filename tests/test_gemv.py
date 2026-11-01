# test_gemv.py — the register-dequant NVFP4 GEMV (DIRECTIONS.md §2).
#
# Correctness bar: the GEMV must agree with the independent reference
# (dequantize the SAME codes on the device, matmul in torch/cuBLAS FP32) to
# accumulation-order noise — activations are consumed exactly (no
# quantization), so the only differences are f32 summation order.

import pytest
import torch

import thor_fp4
from thor_fp4 import ops
from conftest import requires_cuda, rrms


@requires_cuda
@pytest.mark.parametrize("M", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("N,K", [(512, 1024), (1024, 64), (64, 1024), (320, 640)])
def test_gemv_matches_reference(M, N, K):
    w = torch.randn(N, K, device="cuda") * 0.05
    wt = ops.NVFP4Tensor.from_float(w)
    x = torch.randn(M, K, device="cuda")
    y = ops.gemv(x, wt, "f32")
    ref = x @ wt.dequantize().t()
    assert y.shape == (M, N)
    assert rrms(y, ref) < 1e-5
    # worst element, relative to the output scale
    scale = ref.abs().max()
    assert ((y - ref).abs().max() / scale).item() < 1e-4


@requires_cuda
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_gemv_16bit_inputs(dt):
    w = torch.randn(768, 512, device="cuda") * 0.05
    wt = ops.NVFP4Tensor.from_float(w)
    x = torch.randn(4, 512, device="cuda", dtype=dt)
    y = ops.gemv(x, wt, "f32")
    ref = x.float() @ wt.dequantize().t()
    assert rrms(y, ref) < 1e-5


@requires_cuda
def test_gemv_output_dtypes():
    w = torch.randn(256, 512, device="cuda")
    wt = ops.NVFP4Tensor.from_float(w)
    x = torch.randn(2, 512, device="cuda")
    ref = x @ wt.dequantize().t()
    for od in ("f16", "bf16", "f32"):
        y = ops.gemv(x, wt, od)
        assert y.dtype == {"f16": torch.float16, "bf16": torch.bfloat16,
                           "f32": torch.float32}[od]
    assert rrms(ops.gemv(x, wt, "f16").float(), ref) < 1e-3   # f16 rounding


@requires_cuda
def test_gemv_rejects_large_M():
    wt = ops.NVFP4Tensor.from_float(torch.randn(64, 128, device="cuda"))
    x = torch.randn(9, 128, device="cuda")
    with pytest.raises(AssertionError):
        ops.gemv(x, wt, "f32")


@requires_cuda
def test_gemv_scales_layout_matches_blocked():
    # the un-scattered (plain) scale bytes must decode to the same values as
    # the 128x4-tile layout's block_scales()
    torch.manual_seed(3)
    R, K = 300, 1024          # crosses 128-row tiles, exercises the tail
    w = torch.randn(R, K, device="cuda")
    wt = ops.NVFP4Tensor.from_float(w)
    plain = wt.gemv_scales()
    blocked = ops.block_scales(wt.sblk, R, K)
    pf = plain.view(torch.float8_e4m3fn).float().reshape(R, K // 16)
    assert torch.equal(pf, blocked)
    # cached
    assert wt.gemv_scales() is plain


@requires_cuda
def test_linear_gemv_engine():
    lin = thor_fp4.Linear(1024, 512, bias=True, engine="gemv", out_dtype="f32").cuda()
    x = torch.randn(8, 1024, device="cuda")
    y = lin(x)
    assert y.shape == (8, 512)
    ref = x @ lin._weight_fp4.dequantize().t() + lin.bias.float()
    assert rrms(y.float(), ref) < 1e-5
    with pytest.raises(RuntimeError):
        lin(torch.randn(9, 1024, device="cuda"))
    # the default f16 output dtype carries its rounding, nothing worse
    lin16 = thor_fp4.Linear(1024, 512, bias=True, engine="gemv").cuda()
    with torch.no_grad():
        lin16.weight.copy_(lin.weight); lin16.bias.copy_(lin.bias)
    y16 = lin16(x)
    assert rrms(y16.float(), ref) < 2e-3


@requires_cuda
def test_verify_gemv_quiet_and_flags_broken_kernel():
    lin = thor_fp4.Linear(512, 256, engine="gemv").cuda()
    x = torch.randn(4, 512, device="cuda")
    rep = thor_fp4.verify(lin, x)
    assert rep.ok, str(rep)
    assert rep.rel_rms < 1e-5
    # simulate a silently-wrong kernel: perturb the native path's output
    from thor_fp4 import linear as lin_mod
    real = lin_mod.ops.gemv

    def broken(*args, **kw):
        real(*args, **kw)
        kw["out"].add_(0.25)       # the out buffer (Linear passes it by keyword)

    lin_mod.ops.gemv = broken
    try:
        rep2 = thor_fp4.verify(lin, x)
    finally:
        lin_mod.ops.gemv = real
    assert not rep2.ok
    assert rep2.n_bad > 0 and rep2.worst_locations

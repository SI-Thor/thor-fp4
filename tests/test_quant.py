# Quantizer: the device kernels must be bit-exact against the torch fake
# quantizer (QAT trains against exactly the arithmetic the tensor cores
# commit), and reject shapes the hardware cannot take.

import pytest
import torch

import thor_fp4
from thor_fp4 import ops
from conftest import requires_cuda, rrms


@requires_cuda
@pytest.mark.parametrize("K", [320, 1024, 2048])
@pytest.mark.parametrize("ts_mode", ["block_only", "two_level"])
def test_fake_quant_bit_exact(K, ts_mode):
    x = torch.randn(64, K, device="cuda") * torch.linspace(0.001, 3, K, device="cuda")
    ts = 1.0 if ts_mode == "block_only" else thor_fp4.two_level_scale(x)
    packed, sblk = thor_fp4.quantize_nvfp4(x, ts)
    dev = thor_fp4.dequantize_nvfp4(packed, sblk, ts, 64, K)
    fake = thor_fp4.nvfp4_fake_quant(x, ts)
    assert torch.equal(dev, fake), f"maxdiff {(dev - fake).abs().max().item():.3e}"


@requires_cuda
def test_fake_quant_zero_blocks():
    x = torch.zeros(8, 128, device="cuda")
    x[:, :64] = torch.randn(8, 64, device="cuda")
    packed, sblk = thor_fp4.quantize_nvfp4(x)
    dev = thor_fp4.dequantize_nvfp4(packed, sblk, 1.0, 8, 128)
    assert torch.equal(dev, thor_fp4.nvfp4_fake_quant(x))
    assert dev[:, 64:].abs().max().item() == 0.0


@requires_cuda
@pytest.mark.parametrize("dt", [torch.float32, torch.float16, torch.bfloat16])
def test_quant_input_dtypes(dt):
    x = torch.randn(32, 256, device="cuda", dtype=dt)
    packed, sblk = thor_fp4.quantize_nvfp4(x)
    dev = thor_fp4.dequantize_nvfp4(packed, sblk, 1.0, 32, 256)
    fake = thor_fp4.nvfp4_fake_quant(x.float())
    assert torch.equal(dev, fake)


@requires_cuda
def test_quant_rejects_bad_K():
    with pytest.raises(AssertionError, match="multiple of 64"):
        thor_fp4.quantize_nvfp4(torch.randn(8, 48, device="cuda"))


@requires_cuda
def test_e2m1_code_packing():
    # a block of exactly-known values (K=64, first block): ties round up
    vals = torch.zeros(1, 64, device="cuda")
    vals[0, :16] = torch.tensor([0.25, 0.6, 2.6, -6.0] + [0.0] * 12, device="cuda")
    packed, sblk = thor_fp4.quantize_nvfp4(vals)
    d = thor_fp4.dequantize_nvfp4(packed, sblk, 1.0, 1, 64)
    assert d[0, 0].item() == pytest.approx(0.5)   # tie rounds up
    assert d[0, 1].item() == pytest.approx(0.5)
    assert d[0, 2].item() == pytest.approx(3.0)
    assert d[0, 3].item() == pytest.approx(-6.0)


@requires_cuda
def test_nvfp4tensor_roundtrip():
    x = torch.randn(128, 512, device="cuda")
    t = thor_fp4.NVFP4Tensor.from_float(x)
    assert t.shape == (128, 512)
    assert t.nbytes() == 128 * 512 // 2 + 128 * 32
    d = t.dequantize()
    assert rrms(d, x) < 0.15   # single-layer e2m1 noise on randn inputs (~0.09)
    # two-level uses the full UE4M3 range for block scales
    t2 = thor_fp4.NVFP4Tensor.from_float(x, tensor_scale=None)
    sf = ops.block_scales(t2.sblk, 128, 512)
    assert sf.max().item() <= 448.0 and sf.max().item() > 200.0


@requires_cuda
def test_mx_quant_bytes():
    x = torch.randn(64, 512, device="cuda")
    for kind in ("mxfp4", "mxfp8"):
        t = thor_fp4.MXTensor.from_float(x, kind)
        per = 256 if kind == "mxfp4" else 512
        assert t.nbytes() == 64 * per + 64 * 16


@requires_cuda
def test_gemm_vs_dequant_matmul():
    # the library's stated tolerance: native GEMM vs dequantize-then-matmul
    # agrees to FP16-output rounding (~2e-4 relRMS)
    M, N, K = 256, 512, 1024
    x = torch.randn(M, K, device="cuda") * 0.5
    w = torch.randn(N, K, device="cuda") * 0.3
    a = thor_fp4.NVFP4Tensor.from_float(x, 1.0)
    b = thor_fp4.NVFP4Tensor.from_float(w, 1.0)
    for od in ("f16", "bf16", "f32"):
        out = a.matmul_t(b, out_dtype=od)
        ref = a.dequantize() @ b.dequantize().t()
        assert rrms(out, ref) < 2e-3, f"{od}: {rrms(out, ref):.3e}"

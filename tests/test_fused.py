# test_fused.py — the fused-epilogue chain (ROUND 2 / DIRECTIONS sec.4):
# one kernel per layer (GEMM + bias + GELU + MXFP8 quantize in the epilogue).
#
# Correctness bars:
#  * the written (codes, scales) must decode to gelu(x_q @ W_q^T + b) within
#    MXFP8 quantization error of the value the epilogue saw (checked against
#    a fake-quant reference built from the same operands);
#  * a whole fused chain must agree with the two-pass chain (quantize_mx +
#    CUTLASS mx8 GEMM + torch gelu) within the documented scale-recipe
#    difference (ceil-pow2 in the epilogue vs round-to-nearest in quantize_mx);
#  * captured CUDA-graph replay == eager.

import pytest
import torch
import torch.nn as nn

import thor_fp4
from thor_fp4 import ops
from thor_fp4.fused import FusedLinear, MX8Act, mx8_chain
from thor_fp4.quant import mx_fake_quant
from conftest import requires_cuda, rrms


def _fused_lib():
    return thor_fp4._ext.fused()


def _decode_mx8(act: MX8Act):
    """codes+scales -> f32 (M, K): mirror of the tile layout + e4m3 + UE8M0"""
    from thor_fp4 import _ext
    M, K = act.shape
    # decode via the dequant kernel is nvfp4-only; do it in torch
    pk = act.packed[: M * K].view(M, K, 1).to(torch.uint8)
    f8 = pk.view(torch.float8_e4m3fn).float()
    # scales: un-scatter the tile layout (same atom as quantize_mx writes)
    S = K // 32
    nceil = (M + 127) // 128 * 128
    Sc = (S + 3) // 4 * 4
    sb = act.sblk
    idx = torch.arange(M, device=sb.device)
    a = (idx // 128); r = idx % 128
    b = r // 32; c = r % 32
    sidx = torch.arange(S, device=sb.device)
    d = sidx // 4; e = sidx % 4
    # byte index (a*(S/4)+d)*512 + 16*c + 4*b + e  (Sc == S required)
    assert Sc == S, "test assumes S % 4 == 0"
    lin = (a[:, None] * (S // 4) + d[None, :]) * 512 + 16 * c[:, None] + 4 * b[:, None] + e[None, :]
    sbf = sb.view(-1)[lin.reshape(-1)].view(torch.uint8).view(M, S)
    sf = torch.ldexp(torch.ones_like(sbf, dtype=torch.float32),
                     (sbf.to(torch.int16) - 127))
    return (f8.view(M, K // 32, 32) * sf.unsqueeze(-1)).reshape(M, K)


@requires_cuda
def test_fused_layer_matches_reference():
    if _fused_lib() is None:
        pytest.skip("needs build/libthorfp4_fused.so")
    torch.manual_seed(0)
    M, K, N = 256, 512, 1024
    fl = FusedLinear(K, N, bias=True).cuda()
    x = torch.randn(M, K, device="cuda")
    out = fl(x)
    assert isinstance(out, MX8Act) and out.shape == (M, N)
    got = _decode_mx8(out)
    # reference: the value the epilogue quantized — gelu of the f32 GEMM on
    # the SAME operands the mainloop consumed
    xd = mx_fake_quant(x, "mxfp8")
    wd = mx_fake_quant(fl.weight.detach().float(), "mxfp4")
    ref_pre = torch.nn.functional.gelu(xd @ wd.t() + fl.bias.float())
    ref = mx_fake_quant(ref_pre, "mxfp8")
    # ceil-vs-RN scale recipe: agreement at the level of quantization noise
    assert rrms(got, ref) < 0.02, rrms(got, ref)
    # and against the pre-quantization value: MXFP8 quantization error only
    assert rrms(got, ref_pre) < 0.02


@requires_cuda
def test_fused_chain_end_to_end():
    if _fused_lib() is None:
        pytest.skip("needs build/libthorfp4_fused.so")
    torch.manual_seed(1)
    dims = (512, 1024, 1024, 64)
    fused = mx8_chain(*dims).cuda()
    # same weights through the two-pass chain (Linear(engine=cutlass:mx8))
    two = nn.Sequential()
    with torch.no_grad():
        mods = []
        fused_layers = [m for m in fused if isinstance(m, FusedLinear)]
        for i, m in enumerate(fused):
            if isinstance(m, FusedLinear):
                lin = thor_fp4.Linear(m.in_features, m.out_features, bias=True,
                                      engine="cutlass:mx8").cuda()
                lin.weight.copy_(m.weight); lin.bias.copy_(m.bias)
                mods.append(lin); mods.append(nn.GELU())
            elif isinstance(m, thor_fp4.Linear):
                lin = thor_fp4.Linear(m.in_features, m.out_features, bias=True,
                                      engine="cutlass:mx8").cuda()
                lin.weight.copy_(m.weight); lin.bias.copy_(m.bias)
                mods.append(lin)
        two = nn.Sequential(*mods)
    x = torch.randn(512, dims[0], device="cuda")
    yf = fused(x).float()
    yt = two(x).float()
    assert yf.shape == yt.shape
    # both chains are MXFP8-quantized activations; they differ only by the
    # epilogue's ceil-pow2 scale recipe vs quantize_mx's RN
    assert rrms(yf, yt) < 0.05, rrms(yf, yt)
    top_agree = (yf.argmax(1) == yt.argmax(1)).float().mean().item()
    assert top_agree > 0.9, top_agree


@requires_cuda
def test_fused_constraints():
    if _fused_lib() is None:
        pytest.skip("needs build/libthorfp4_fused.so")
    with pytest.raises(ValueError):
        FusedLinear(320, 1024)
    with pytest.raises(ValueError):
        FusedLinear(512, 100)


@requires_cuda
def test_fused_graph_capture():
    if _fused_lib() is None:
        pytest.skip("needs build/libthorfp4_fused.so")
    torch.manual_seed(2)
    fused = mx8_chain(512, 1024, 64).cuda()
    x = torch.randn(128, 512, device="cuda")
    y1 = fused(x).float().clone()
    g = thor_fp4.capture_module(fused, x)
    y2 = g(x).float()
    assert torch.equal(y1, y2)

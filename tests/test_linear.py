# thor_fp4.Linear: the drop-in nn.Linear. Output must match the fake-quant
# arithmetic to output-dtype rounding for every M (the hardware constraints
# on the batch disappeared with the operand swap), and the layer must survive
# the nn.Module lifecycle (.cuda(), in-place weight edits, capture).

import pytest
import torch

import thor_fp4
from conftest import requires_cuda, rrms


@requires_cuda
@pytest.mark.parametrize("M", [1, 3, 5, 8, 17, 128, 255])
def test_linear_matches_fake_quant(M):
    lin = thor_fp4.Linear(1024, 512, bias=True).cuda()
    x = torch.randn(M, 1024, device="cuda")
    y = lin(x)
    assert y.shape == (M, 512) and y.dtype == torch.float16
    ts = lin._weight_fp4.tensor_scale
    ref = (thor_fp4.nvfp4_fake_quant(x, 1.0)
           @ thor_fp4.nvfp4_fake_quant(lin.weight.detach(), ts).t() + lin.bias.float())
    assert rrms(y.float(), ref) < 1e-3, rrms(y.float(), ref)


@requires_cuda
@pytest.mark.parametrize("od,dt", [("f16", torch.float16), ("bf16", torch.bfloat16),
                                   ("f32", torch.float32)])
def test_linear_output_dtypes(od, dt):
    lin = thor_fp4.Linear(256, 128, out_dtype=od).cuda()
    assert lin(torch.randn(16, 256, device="cuda")).dtype == dt


@requires_cuda
def test_linear_input_layouts_and_dtypes():
    lin = thor_fp4.Linear(256, 128).cuda()
    base = torch.randn(16, 256, device="cuda")
    ref = lin(base).float()
    # non-contiguous input: same logical (M, K) values, strided storage
    buf = torch.empty(16, 512, device="cuda")
    xt = buf[:, :256]
    xt.copy_(base)
    assert not xt.is_contiguous()
    assert rrms(lin(xt).float(), ref) < 1e-3
    # half / bfloat inputs (fp16 arithmetic is not identical to fp32, so
    # compare against a fake-quant of the same dtype)
    for dt in (torch.float16, torch.bfloat16, torch.float64):
        out = lin(base.to(dt))
        assert out.shape == (16, 128)


@requires_cuda
def test_linear_constraints():
    with pytest.raises(ValueError, match="% 64"):
        thor_fp4.Linear(300, 128)
    with pytest.raises(ValueError, match="N >= 8"):
        thor_fp4.Linear(128, 4)
    with pytest.raises(ValueError, match="N % 8"):
        thor_fp4.Linear(128, 12)


@requires_cuda
def test_linear_weight_lifecycle():
    lin = thor_fp4.Linear(256, 128).cuda()
    x = torch.randn(8, 256, device="cuda")
    y1 = lin(x).clone()                         # returned buffer is reused
    with torch.no_grad():                      # in-place edit then refresh
        lin.weight.mul_(2.0)
    lin.refresh_weight()
    y2 = lin(x)
    # e4m3 doubling is exact, so the quantized output doubles to f16 rounding
    assert torch.allclose(y2.float(), y1.float() * 2, rtol=0.02, atol=1e-3)
    lin2 = thor_fp4.Linear(256, 128).cuda()
    lin2.load_state_dict(lin.state_dict())      # params copy -> requantized
    assert torch.equal(lin2(x), y2)


@requires_cuda
def test_linear_no_grad_input():
    lin = thor_fp4.Linear(256, 128).cuda()
    x = torch.randn(8, 256, device="cuda", requires_grad=True)
    out = lin(x)
    assert out.requires_grad is False   # inference layer


@requires_cuda
def test_linear_repeat_calls_deterministic():
    lin = thor_fp4.Linear(512, 256).cuda()
    x = torch.randn(33, 512, device="cuda")
    a = lin(x).clone()
    b = lin(x).clone()
    assert torch.equal(a, b)


@requires_cuda
def test_linear_fp4_nbytes():
    lin = thor_fp4.Linear(2048, 512).cuda()
    lin(torch.randn(4, 2048, device="cuda"))
    assert lin.fp4_weight_nbytes() == 2048 * 512 // 2 + 512 * 128
    assert lin.weight.numel() * 4 / lin.fp4_weight_nbytes() > 7   # ~7.1x vs fp32 master

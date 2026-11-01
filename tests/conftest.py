import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def rrms(a, b):
    return ((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt()).item()

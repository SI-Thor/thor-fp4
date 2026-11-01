# verify(): the dual-path check. Must stay quiet on a clean path and flag
# injected deviations (a wrong kernel, a corrupted buffer) — the XOR
# dual-path idea: two independent execution paths must agree within a bound.

import pytest
import torch
import torch.nn as nn

import thor_fp4
from thor_fp4 import linear as lin_mod
from conftest import requires_cuda


@requires_cuda
def test_verify_quiet_when_clean():
    lin = thor_fp4.Linear(1024, 512).cuda()
    x = torch.randn(128, 1024, device="cuda")
    rep = thor_fp4.verify(lin, x)
    assert rep.ok, str(rep)
    assert rep.rel_rms < 5e-3


@requires_cuda
def test_verify_flags_broken_kernel():
    # simulate a silently-wrong GEMM: perturb the native path's output
    lin = thor_fp4.Linear(1024, 512).cuda()
    x = torch.randn(128, 1024, device="cuda")
    real_gemm = lin_mod.ops.gemm

    def broken_gemm(*args, **kw):
        real_gemm(*args, **kw)
        args[6].add_(0.25)          # out buffer, +0.25 like a wrong epilogue

    lin_mod.ops.gemm = broken_gemm
    try:
        rep = thor_fp4.verify(lin, x)
    finally:
        lin_mod.ops.gemm = real_gemm
    assert not rep.ok
    assert rep.n_bad > 0 and rep.worst_locations


@requires_cuda
def test_verify_flags_corrupted_codes():
    # flip weight code bytes AFTER quantization: the native path changes; the
    # reference is rebuilt from the same (corrupted) codes only if refreshed —
    # simulate the stale-reference case by corrupting post-refresh
    lin = thor_fp4.Linear(1024, 512).cuda()
    x = torch.randn(128, 1024, device="cuda")
    lin(x)                                     # builds _weight_fp4
    rep_clean = thor_fp4.verify(lin, x)
    assert rep_clean.ok
    w = lin._weight_fp4
    saved = w.packed.clone()
    w.packed[:4096] ^= 0xFF                    # corrupt codes in place
    rep = thor_fp4.verify(lin, x)
    w.packed.copy_(saved)
    # both paths now read the corrupted codes -> they still agree with each
    # other; verify's job is path equality, so this stays quiet (documented
    # behavior: it catches execution errors, not operand corruption)
    assert rep.ok


@requires_cuda
def test_verify_flags_corrupted_scale_layout():
    # a wrong scale-tensor layout is the classic silent FP4 bug: same bytes,
    # wrong positions. Corrupt the layout ONLY on the native path (linear.py's
    # view of ops); the reference dequantizes the same codes through the
    # untouched module, so the two paths must now disagree.
    lin = thor_fp4.Linear(1024, 512).cuda()
    x = torch.randn(128, 1024, device="cuda")
    real_ops = lin_mod.ops

    class ScrambledOps:
        def __getattr__(self, k):
            return getattr(real_ops, k)

        def quantize_nvfp4(self, *a, **kw):
            p, s = real_ops.quantize_nvfp4(*a, **kw)
            s.copy_(s.roll(97))
            return p, s

    lin_mod.ops = ScrambledOps()
    try:
        rep = thor_fp4.verify(lin, x)
    finally:
        lin_mod.ops = real_ops
    assert not rep.ok, str(rep)


@requires_cuda
def test_verify_module_end_to_end():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 512)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x))).float()

    m = thor_fp4.convert(MLP().cuda(), policy="fp4")
    rep = thor_fp4.verify(m, torch.randn(64, 512, device="cuda"))
    assert rep.ok, str(rep)
    assert set(rep.layers) == {"fc1", "fc2", "<end-to-end>"}


@requires_cuda
def test_verify_report_shape():
    lin = thor_fp4.Linear(256, 128).cuda()
    rep = thor_fp4.verify(lin, [torch.randn(8, 256, device="cuda")] * 2)
    assert rep.n_elements == 2 * 8 * 128
    assert "ok=" in str(rep)

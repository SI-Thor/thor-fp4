#!/usr/bin/env python
# repro_nan.py — R3-0: reproduce and bisect the ue8m0-NaN SFD scale rows the
# FusedLinear wrapper showed at M >= ~2048 (BENCHMARKS.md sec.10 caveat).
#
# Paths compared (one kernel variant throughout: mx8out_gelu_128x128x256_c1):
#   wrapper      FusedLinear unchunked (max_chunk=0): quantize_mx input,
#                cached D/SFD buffers, ensure()->update() per call  [the
#                original failure context]
#   wrapper-chain mx8_chain(512,1024,1024,64), max_chunk=0, scorer weights,
#                B=M — checks each layer's SFD separately (propagation test)
#   abi-reuse    direct tfc4f_gemm, persistent torch buffers — isolates the
#                Python wrapper from the kernel+cache path
#   abi-fresh    direct tfc4f_gemm, fresh torch buffers per call (the
#                configuration the previous agent could NOT reproduce in)
#   abi-cuda     direct call, raw cudaMalloc'd buffers (no torch caching
#                allocator anywhere in the loop)
#
# Detection: SFD byte == 0xFF (ue8m0 NaN), or with --sentinel a surviving
# 0xA5 pre-fill (byte never written). D code bytes 0x7F/0xFF are e4m3 NaNs.
# Every bad SFD byte is mapped back to (row, scale-col) via the 128x4 atom.

import argparse
import ctypes
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import thor_fp4  # noqa: E402
from thor_fp4 import _ext, ops  # noqa: E402
from thor_fp4.fused import FusedLinear, MX8Act, mx8_chain  # noqa: E402
from scorer import boxstate, get_trained  # noqa: E402

NAN_SF = 0xFF
SENT = 0xA5
_cudart = None


def sfd_bad(sf: torch.Tensor, M: int, N: int, mark=(NAN_SF,)):
    """map every marked SFD byte back to (row, scale-col); -> dict row->cols"""
    S = N // 32
    assert S % 4 == 0, "decoder assumes S % 4 == 0"
    sb = sf.view(-1)
    m = torch.zeros_like(sb, dtype=torch.bool)
    for v in mark:
        m |= sb == v
    bad = {}
    n_atoms = S // 4
    for i in torch.nonzero(m).flatten().tolist():
        atom, off = divmod(i, 512)
        e = off % 4
        c, b = divmod(off // 4, 4)
        row = 128 * (atom // n_atoms) + 32 * b + c
        col = 4 * (atom % n_atoms) + e
        bad.setdefault(row, []).append(col)
    return bad


def codes_bad(dp: torch.Tensor, M: int, N: int):
    d = dp[: M * N].view(M, N)
    isnan = (d == 0x7F) | (d == 0xFF)
    return torch.nonzero(isnan.any(1)).flatten().tolist()


def report(tag, act, M, N, check_sent=False):
    marks = (NAN_SF, SENT) if check_sent else (NAN_SF,)
    bad = sfd_bad(act.sblk, M, N, marks)
    rows = sorted(bad)
    cbad = codes_bad(act.packed, M, N)
    if not rows and not cbad:
        print(f"  {tag}: clean")
        return True
    det = []
    if rows:
        det.append(f"sfNaN rows={rows[:10]}{'…' if len(rows) > 10 else ''} "
                   f"(n={len(rows)}, cols/row={sorted(len(bad[r]) for r in rows)[:4]})")
    if cbad:
        det.append(f"codeNaN rows={cbad[:10]}{'…' if len(cbad) > 10 else ''} (n={len(cbad)})")
    print(f"  {tag}: BAD  " + "  ".join(det))
    return False


def x_of(M, K, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(M, K, device="cuda", generator=g)


def path_wrapper(M, K, N, calls, sent):
    print(f"[wrapper] M={M} K={K} N={N}")
    fl = FusedLinear(K, N, bias=True, max_chunk=0).cuda()
    x = x_of(M, K)
    for i in range(calls):
        if sent and (M, 0) in fl._outs:
            fl._outs[(M, 0)].sblk.fill_(SENT)
        y = fl(x)
        torch.cuda.synchronize()
        report(f"call{i}", y, M, N, check_sent=sent)
    del fl
    torch.cuda.empty_cache()


def path_wrapper_chain(M, calls, sent):
    print(f"[wrapper-chain] scorer dims 512-1024-1024-64, B={M}")
    chain = mx8_chain(512, 1024, 1024, 64).cuda()
    try:
        bf, _ = get_trained()
        lins = [m for m in bf if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            fmods = [m for m in chain if isinstance(m, (FusedLinear, thor_fp4.Linear))]
            for m, l in zip(fmods, lins):
                m.weight.copy_(l.weight)
                m.bias.copy_(l.bias)
        src = "scorer.pt (bf16-trained)"
    except Exception as e:  # noqa: BLE001
        src = f"random init ({e.__class__.__name__})"
    for m in chain:
        if isinstance(m, FusedLinear):
            m.max_chunk = 0
    x = x_of(M, 512)
    for i in range(calls):
        with torch.no_grad():
            a = x
            for li, m in enumerate(chain):
                a = m(a)
                torch.cuda.synchronize()
                if isinstance(a, MX8Act):
                    report(f"call{i} L{li}", a, M, a.shape[1], check_sent=sent)
        # re-fill sentinels for the next call
        if sent:
            for m in chain:
                if isinstance(m, FusedLinear):
                    for o in m._outs.values():
                        o.sblk.fill_(SENT)
    print(f"  (weights: {src})")
    del chain
    torch.cuda.empty_cache()


def _abi_setup(M, K, N):
    L = _ext.fused()
    v = L.tfc4f_variant_index(b"mx8out_gelu_128x128x256_c1")
    w = torch.randn(N, K, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(3)) * (K ** -0.5)
    W = ops.MXTensor.from_float(w, "mxfp4")
    bias = torch.zeros(N, device="cuda")
    packed = torch.zeros(ops.mx_packed_bytes(M, K, "mxfp8"), dtype=torch.uint8, device="cuda")
    sblk = torch.zeros(ops.mx_scale_bytes(M, K), dtype=torch.uint8, device="cuda")
    ops.quantize_mx(x_of(M, K), "mxfp8", packed, sblk)
    act = MX8Act(packed, sblk, (M, K))
    return L, v, act, W, bias


def path_abi_reuse(M, K, N, calls, sent):
    print(f"[abi-reuse] M={M} K={K} N={N}")
    L, v, act, W, bias = _abi_setup(M, K, N)
    dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(int(L.tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
    for i in range(calls):
        if sent and i > 0:
            sf.fill_(SENT)
        rc = L.tfc4f_gemm(v, act.packed.data_ptr(), act.sblk.data_ptr(),
                          W.packed.data_ptr(), W.sblk.data_ptr(), bias.data_ptr(),
                          dp.data_ptr(), sf.data_ptr(), M, N, K, 1.0, _ext.stream())
        assert rc == 0, rc
        torch.cuda.synchronize()
        report(f"call{i}", MX8Act(dp, sf, (M, N)), M, N, check_sent=sent)
    del dp, sf, act, W
    torch.cuda.empty_cache()


def path_abi_fresh(M, K, N, calls, sent):
    print(f"[abi-fresh] M={M} K={K} N={N}")
    L, v, act, W, bias = _abi_setup(M, K, N)
    for i in range(calls):
        dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
        sf = torch.zeros(int(L.tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
        rc = L.tfc4f_gemm(v, act.packed.data_ptr(), act.sblk.data_ptr(),
                          W.packed.data_ptr(), W.sblk.data_ptr(), bias.data_ptr(),
                          dp.data_ptr(), sf.data_ptr(), M, N, K, 1.0, _ext.stream())
        assert rc == 0, rc
        torch.cuda.synchronize()
        report(f"call{i}", MX8Act(dp, sf, (M, N)), M, N)
        del dp, sf
    del act, W
    torch.cuda.empty_cache()


def _cuda_bytes(n, fill=0):
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so")
    p = ctypes.c_void_p()
    assert _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) == 0
    if fill:
        assert _cudart.cudaMemset(p, ctypes.c_int(fill), ctypes.c_size_t(n)) == 0
    return p


def path_abi_cuda(M, K, N, calls, sent):
    print(f"[abi-cuda] M={M} K={K} N={N}")
    L, v, act, W, bias = _abi_setup(M, K, N)
    sfd_n = int(L.tfc4f_sfd_bytes(M, N))
    dev = [ _cuda_bytes(M * N, 0), _cuda_bytes(sfd_n, 0) ]
    host_sf = torch.empty(sfd_n, dtype=torch.uint8, device="cpu")
    host_dp = torch.empty(M * N, dtype=torch.uint8, device="cpu")
    for i in range(calls):
        if sent and i > 0:
            assert _cudart.cudaMemset(dev[1], ctypes.c_int(SENT), ctypes.c_size_t(sfd_n)) == 0
        rc = L.tfc4f_gemm(v, act.packed.data_ptr(), act.sblk.data_ptr(),
                          W.packed.data_ptr(), W.sblk.data_ptr(), bias.data_ptr(),
                          dev[0].value, dev[1].value, M, N, K, 1.0, _ext.stream())
        assert rc == 0, rc
        torch.cuda.synchronize()
        assert _cudart.cudaMemcpy(host_sf.data_ptr(), dev[1], ctypes.c_size_t(sfd_n), 2) == 0  # D2H
        assert _cudart.cudaMemcpy(host_dp.data_ptr(), dev[0], ctypes.c_size_t(M * N), 2) == 0
        report(f"call{i}", MX8Act(host_dp, host_sf, (M, N)), M, N, check_sent=sent)
    for p in dev:
        _cudart.cudaFree(p)
    del act, W
    torch.cuda.empty_cache()


PATHS = {
    "wrapper": lambda a: path_wrapper(a.M, a.K, a.N, a.calls, a.sentinel),
    "wrapper-chain": lambda a: path_wrapper_chain(a.M, a.calls, a.sentinel),
    "abi-reuse": lambda a: path_abi_reuse(a.M, a.K, a.N, a.calls, a.sentinel),
    "abi-fresh": lambda a: path_abi_fresh(a.M, a.K, a.N, a.calls, a.sentinel),
    "abi-cuda": lambda a: path_abi_cuda(a.M, a.K, a.N, a.calls, a.sentinel),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--paths", nargs="+", default=list(PATHS))
    ap.add_argument("--sentinel", action="store_true")
    ap.add_argument("--warmup", action="store_true",
                    help="mimic bench_fused's 400x 2048^3 matmul warmup first")
    args = ap.parse_args()
    print(f"# repro_nan  {boxstate()}  K={args.K} N={args.N} calls={args.calls} "
          f"sentinel={args.sentinel} warmup={args.warmup}")
    if args.warmup:
        a = torch.randn(2048, 2048, device="cuda")
        for _ in range(400):
            a @ a
        del a
        torch.cuda.empty_cache()
    torch.manual_seed(0)
    Ms = args.M
    for p in args.paths:
        for M in Ms:
            import copy
            a1 = copy.copy(args)
            a1.M = M
            PATHS[p](a1)


if __name__ == "__main__":
    main()

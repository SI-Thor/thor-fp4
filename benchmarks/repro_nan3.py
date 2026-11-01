#!/usr/bin/env python
# repro_nan3.py — R3-0 lifecycle bisection of the ue8m0-NaN rows.
#
# repro_nan/repro_nan2 established: whole-row NaN (codes + all 32 scales),
# inputs clean, reference values normal, allocation-history dependent, and
# absent with fresh per-call output buffers. This script bisects the KERNEL
# SIDE lifecycle through the new tfc4f_gemm_mode ABI:
#   mode 0  cached ensure() (library path)
#   mode 1  cached object, initialize() re-run before every launch
#   mode 2  fresh Gemm object + fresh workspace every call
#   mode 3  cached object, fresh cudaMalloc'd workspace + initialize every call
# plus: all 4 tile variants; a multi-address sweep (8 different output buffer
# allocations, same inputs); and the workspace size for the shape.

import argparse
import ctypes
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import thor_fp4  # noqa: E402
from thor_fp4 import _ext, ops  # noqa: E402
from thor_fp4.fused import MX8Act  # noqa: E402
from scorer import boxstate  # noqa: E402
import repro_nan as R  # noqa: E402

MODES = {0: "cached", 1: "cached+reinit", 2: "fresh-object", 3: "fresh-ws"}


def decl(L):
    L.tfc4f_gemm_mode.argtypes = [ctypes.c_int, ctypes.c_int, c := ctypes.c_void_p,
                                  c, c, c, c, c, c, ctypes.c_int64,
                                  ctypes.c_int64, ctypes.c_int64,
                                  ctypes.c_float, c]
    L.tfc4f_gemm_mode.restype = ctypes.c_int
    L.tfc4f_ws_bytes.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                                 ctypes.c_int64]
    L.tfc4f_ws_bytes.restype = ctypes.c_int64


def setup(M, K, N, seed=7):
    L = _ext.fused()
    decl(L)
    w = torch.randn(N, K, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(3)) * (K ** -0.5)
    W = ops.MXTensor.from_float(w, "mxfp4")
    bias = torch.zeros(N, device="cuda")
    packed = torch.zeros(ops.mx_packed_bytes(M, K, "mxfp8"), dtype=torch.uint8, device="cuda")
    sblk = torch.zeros(ops.mx_scale_bytes(M, K), dtype=torch.uint8, device="cuda")
    ops.quantize_mx(R.x_of(M, K), "mxfp8", packed, sblk)
    return L, MX8Act(packed, sblk, (M, K)), W, bias


def call_mode(L, v, mode, act, W, bias, dp, sf, M, K, N):
    rc = L.tfc4f_gemm_mode(v, mode, act.packed.data_ptr(), act.sblk.data_ptr(),
                           W.packed.data_ptr(), W.sblk.data_ptr(), bias.data_ptr(),
                           dp.data_ptr(), sf.data_ptr(), M, N, K, 1.0, _ext.stream())
    assert rc == 0, f"mode {mode} rc={rc}"


def sweep_modes(M, K, N, calls=3):
    print(f"[modes] M={M} K={K} N={N}")
    L, act, W, bias = setup(M, K, N)
    ws = L.tfc4f_ws_bytes(0, M, N, K)
    print(f"  workspace bytes for shape: {ws}")
    nv = L.tfc4f_n_variants()
    names = [bytes(L.tfc4f_variant_name(i)).decode() for i in range(nv)]
    dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(int(L.tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
    for v, name in enumerate(names):
        for mode, mname in MODES.items():
            bad = []
            for i in range(calls):
                dp.zero_(); sf.zero_()
                call_mode(L, v, mode, act, W, bias, dp, sf, M, K, N)
                torch.cuda.synchronize()
                rows = sorted(R.sfd_bad(sf, M, N))
                if rows:
                    bad.append((i, len(rows), rows[:6]))
            print(f"  {name:32s} mode={mode} ({mname:14s}): "
                  f"{'clean' if not bad else 'BAD ' + str(bad)}")
    del dp, sf, act, W
    torch.cuda.empty_cache()


def sweep_addresses(M, K, N, npairs=8):
    print(f"[addresses] M={M} K={K} N={N}: {npairs} output-buffer allocations, mode 0")
    L, act, W, bias = setup(M, K, N)
    v = 0
    # interleave some unrelated allocations so the pairs land at different
    # addresses, and remember (dp.data_ptr(), sf.data_ptr())
    pairs = []
    for i in range(npairs):
        dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
        sf = torch.zeros(int(L.tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
        pad = torch.empty(4096 * (i + 1), dtype=torch.uint8, device="cuda")
        pairs.append((dp, sf, pad))
    for i, (dp, sf, _) in enumerate(pairs):
        dp.zero_(); sf.zero_()
        call_mode(L, v, 0, act, W, bias, dp, sf, M, K, N)
        torch.cuda.synchronize()
        rows = sorted(R.sfd_bad(sf, M, N))
        print(f"  pair{i}: dp=0x{dp.data_ptr():012x} sf=0x{sf.data_ptr():012x} "
              f"-> {'clean' if not rows else f'BAD rows(n={len(rows)})={rows[:6]}'}")
    del pairs, act, W
    torch.cuda.empty_cache()


def guarded(M, K, N, guard=8192):
    """Inputs and outputs with sentinel padding on both sides; the call gets
    the inner pointer, the guards must survive."""
    L, act, W, bias = setup(M, K, N)
    v = 0

    def mk(n):
        t = torch.full((n + 2 * guard,), 0xA5, dtype=torch.uint8, device="cuda")
        return t, t[guard:guard + n]

    in_p, in_pv = mk(act.packed.numel())
    in_s, in_sv = mk(act.sblk.numel())
    in_p[guard:guard + act.packed.numel()] = act.packed
    in_s[guard:guard + act.sblk.numel()] = act.sblk
    w_p, w_pv = mk(W.packed.numel())
    w_s, w_sv = mk(W.sblk.numel())
    w_p[guard:guard + W.packed.numel()] = W.packed
    w_s[guard:guard + W.sblk.numel()] = W.sblk
    out_p, out_pv = mk(M * N)
    out_s, out_sv = mk(int(L.tfc4f_sfd_bytes(M, N)))
    return dict(L=L, v=v, bias=bias, M=M, K=K, N=N,
                tensors=[in_p, in_s, w_p, w_s, out_p, out_s],
                views=[in_pv, in_sv, w_pv, w_sv, out_pv, out_sv],
                act_shape=act.shape, w_shape=W.shape)


def check_guards(g):
    ok = True
    for name, t in zip(["in_p", "in_s", "w_p", "w_s", "out_p", "out_s"], g["tensors"]):
        head = t[:8192]
        tail = t[t.numel() - 8192:]
        hb = int((head == 0xA5).sum())
        tb = int((tail == 0xA5).sum())
        if hb != 8192 or tb != 8192:
            ok = False
            print(f"    GUARD SMASHED {name}: head {hb}/8192 tail {tb}/8192 A5-bytes")
    return ok


def sweep_guards(M, K, N, calls=3):
    print(f"[guards] M={M} K={K} N={N}: sentinel-padded in/out buffers, mode 0")
    g = guarded(M, K, N)
    L, v, bias = g["L"], g["v"], g["bias"]
    in_pv, in_sv, w_pv, w_sv, out_pv, out_sv = g["views"]
    for i in range(calls):
        out_pv.zero_()
        out_sv.zero_()
        rc = L.tfc4f_gemm_mode(v, 0, in_pv.data_ptr(), in_sv.data_ptr(),
                               w_pv.data_ptr(), w_sv.data_ptr(), bias.data_ptr(),
                               out_pv.data_ptr(), out_sv.data_ptr(),
                               M, N, K, 1.0, _ext.stream())
        assert rc == 0, rc
        torch.cuda.synchronize()
        rows = sorted(R.sfd_bad(out_sv, M, N))
        inrows = sorted(R.sfd_bad(in_sv, M, K))
        print(f"  call{i}: out sfNaN rows={rows[:8] if rows else 'clean'}"
              f"   post-call INPUT SA bad rows={inrows[:8] if inrows else 'clean'}")
        check_guards(g)
    del g
    torch.cuda.empty_cache()


def sweep_inputs(M, K, N, span=2 << 20):
    """Place the A-side operand buffers at many offsets inside a big slab and
    call the kernel once per offset — isolates input-address dependence for
    the e4m3 codes and the ue8m0 scales independently."""
    print(f"[inputs] M={M} K={K} N={N}: A-codes and SFA scales swept over "
          f"offsets in a {span>>20}MiB slab (mode 0)")
    L, act, W, bias = setup(M, K, N)
    v = 0
    dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(int(L.tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
    nb_p, nb_s = act.packed.numel(), act.sblk.numel()
    base_p = act.packed.clone()
    base_s = act.sblk.clone()

    def run_with(p_ptr, s_ptr, tag):
        dp.zero_(); sf.zero_()
        rc = L.tfc4f_gemm_mode(v, 0, p_ptr, s_ptr, W.packed.data_ptr(),
                               W.sblk.data_ptr(), bias.data_ptr(),
                               dp.data_ptr(), sf.data_ptr(), M, N, K, 1.0,
                               _ext.stream())
        assert rc == 0, rc
        torch.cuda.synchronize()
        rows = sorted(R.sfd_bad(sf, M, N))
        print(f"    {tag:34s} -> {'clean' if not rows else f'BAD n={len(rows)} rows={rows[:6]}'}")

    offs = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144, 524288, 1048576, 2097152]
    for which in ("codes", "scales", "both"):
        if which in ("codes", "both"):
            slab = torch.full((nb_p + span + 4096,), 0x5A, dtype=torch.uint8, device="cuda")
            for o in offs:
                slab[o:o + nb_p] = base_p
                run_with(slab[o:o + nb_p].data_ptr(), act.sblk.data_ptr(),
                         f"A-codes at slab+{o} (ptr 0x{slab[o:o+nb_p].data_ptr():012x})")
                slab[o:o + nb_p] = 0x5A
            del slab
        if which in ("scales", "both"):
            slab = torch.full((nb_s + span + 4096,), 0x5A, dtype=torch.uint8, device="cuda")
            for o in offs:
                slab[o:o + nb_s] = base_s
                run_with(act.packed.data_ptr(), slab[o:o + nb_s].data_ptr(),
                         f"SFA scales at slab+{o} (ptr 0x{slab[o:o+nb_s].data_ptr():012x})")
                slab[o:o + nb_s] = 0x5A
            del slab
        if which == "both":
            # both slabs alive so addresses interact; sweep codes with scales
            # parked at a few distinct offsets
            slab_p = torch.full((nb_p + span + 4096,), 0x5A, dtype=torch.uint8, device="cuda")
            slab_s = torch.full((nb_s + span + 4096,), 0x5A, dtype=torch.uint8, device="cuda")
            for o in offs[:8]:
                slab_p[o:o + nb_p] = base_p
                for so in (0, 65536, 262144):
                    slab_s[so:so + nb_s] = base_s
                    run_with(slab_p[o:o + nb_p].data_ptr(),
                             slab_s[so:so + nb_s].data_ptr(),
                             f"both: codes+{o} scales+{so}")
                slab_p[o:o + nb_p] = 0x5A
            del slab_p, slab_s
    del dp, sf, act, W
    torch.cuda.empty_cache()


def sweep_weights(M, K, N, span=2 << 20):
    """Same as sweep_inputs but for the B-side (mxfp4 weight codes + SB)."""
    print(f"[weights] M={M} K={K} N={N}: B-codes and SB swept over slab offsets")
    L, act, W, bias = setup(M, K, N)
    v = 0
    dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(int(L.tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
    nb_p, nb_s = W.packed.numel(), W.sblk.numel()
    base_p, base_s = W.packed.clone(), W.sblk.clone()
    offs = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144, 524288, 1048576, 2097152]

    def run_with(w_ptr, s_ptr, tag):
        dp.zero_(); sf.zero_()
        rc = L.tfc4f_gemm_mode(v, 0, act.packed.data_ptr(), act.sblk.data_ptr(),
                               w_ptr, s_ptr, bias.data_ptr(),
                               dp.data_ptr(), sf.data_ptr(), M, N, K, 1.0,
                               _ext.stream())
        assert rc == 0, rc
        torch.cuda.synchronize()
        rows = sorted(R.sfd_bad(sf, M, N))
        print(f"    {tag:34s} -> {'clean' if not rows else f'BAD n={len(rows)} rows={rows[:6]}'}")

    for which in ("codes", "scales"):
        slab = torch.full((nb_p + span + 4096 if which == "codes" else nb_s + span + 4096,),
                          0x5A, dtype=torch.uint8, device="cuda")
        nb = nb_p if which == "codes" else nb_s
        for o in offs:
            slab[o:o + nb] = base_p if which == "codes" else base_s
            if which == "codes":
                run_with(slab[o:o + nb].data_ptr(), W.sblk.data_ptr(),
                         f"B-codes at slab+{o} (0x{slab[o:o+nb].data_ptr():012x})")
            else:
                run_with(W.packed.data_ptr(), slab[o:o + nb].data_ptr(),
                         f"SB scales at slab+{o} (0x{slab[o:o+nb].data_ptr():012x})")
            slab[o:o + nb] = 0x5A
        del slab
    del dp, sf, act, W
    torch.cuda.empty_cache()


def sweep_pregemm(M, K, N):
    """repro_nan3-context call, but with the f32 reference GEMM (torch sgemm
    of the decoded operands) run immediately before the fused kernel — the
    one structural thing every failing setup has and these clean sweeps lack."""
    print(f"[pregemm] M={M} K={K} N={N}: decoded-operand f32 matmul before the call")
    L, act, W, bias = setup(M, K, N)
    v = 0
    dp = torch.zeros(M * N, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(int(L.tfc4f_sfd_bytes(M, N)), dtype=torch.uint8, device="cuda")
    import repro_nan2 as R2
    for variant, pre in (
            ("none", None),
            ("f32 gemm A_dec @ W_dec.T", "gemm"),
            ("gemm + gelu", "gelu"),
            ("gemm + gelu + blkmax", "blkmax")):
        if pre is not None:
            A = R2.decode_mx(act.packed, act.sblk, M, K, e4m3=True)
            Wd = R2.decode_mx(W.packed, W.sblk, N, K, e4m3=False)
            y = A @ Wd.T + bias
            if pre in ("gelu", "blkmax"):
                y = torch.nn.functional.gelu(y)
            if pre == "blkmax":
                bm = y.view(M, N // 32, 32).abs().amax(-1)
                torch.cuda.synchronize()
        dp.zero_(); sf.zero_()
        rc = L.tfc4f_gemm_mode(v, 0, act.packed.data_ptr(), act.sblk.data_ptr(),
                               W.packed.data_ptr(), W.sblk.data_ptr(), bias.data_ptr(),
                               dp.data_ptr(), sf.data_ptr(), M, N, K, 1.0, _ext.stream())
        assert rc == 0, rc
        torch.cuda.synchronize()
        rows = sorted(R.sfd_bad(sf, M, N))
        print(f"    pre={variant:28s} -> {'clean' if not rows else f'BAD n={len(rows)} rows={rows[:6]}'}")
    del dp, sf, act, W
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[2048])
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--sections", nargs="+", default=["modes", "addresses", "guards"])
    args = ap.parse_args()
    print(f"# repro_nan3  {boxstate()}")
    torch.manual_seed(0)
    for M in args.M:
        if "modes" in args.sections:
            sweep_modes(M, args.K, args.N)
        if "addresses" in args.sections:
            sweep_addresses(M, args.K, args.N)
        if "inputs" in args.sections:
            sweep_inputs(M, args.K, args.N)
        if "weights" in args.sections:
            sweep_weights(M, args.K, args.N)
        if "pregemm" in args.sections:
            sweep_pregemm(M, args.K, args.N)
        if "guards" in args.sections:
            sweep_guards(M, args.K, args.N)


if __name__ == "__main__":
    main()

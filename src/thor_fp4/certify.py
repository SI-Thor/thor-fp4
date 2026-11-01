# certify.py — an arithmetic certificate for FP4 decisions (DIRECTIONS.md §1).
#
# The learned-confidence cascade escalates when the FP4 logits *look* unsure.
# This module makes the escalation decision by arithmetic instead, for the
# final layer of a decision model:
#
#   * At load time, the layer still has its fp32 master weights, so the exact
#     quantization error of every weight is known: E = |W - dequant(quant(W))|
#     (at most half an e2m1 grid step times the block scale — computing it
#     exactly is tighter than the bound and costs nothing at run time).
#   * At run time, for the delivered (already-quantized) activations x̂, the
#     final layer's logits satisfy
#         |out_fp4 - out_true| <= |x̂| @ E^T + rounding_slack
#     per logit, where out_true uses the true BF16 weights on the SAME
#     activations. The bound is one GEMV-sized matmul against an E-sized matrix.
#   * If the FP4 top-1 margin beats the sum of the two logit bounds, the FP4
#     argmax is provably the argmax the true weights would give on those
#     activations — a certificate, not a confidence. Uncertified samples get
#     their final layer recomputed with the BF16 weights (promotion).
#
# Scope (stated plainly): the certificate covers the final layer's WEIGHT
# error only. The activation-quantization error that produced x̂ is upstream
# and not bounded here; carrying error bounds through the network needs
# interval arithmetic and loosens quickly. So "certified" means: same choice
# as BF16 weights on the same FP4 activations — which is exactly the
# promotion question, and is checked for soundness in the tests.

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .linear import Linear
from .quant import mx_fake_quant, nvfp4_fake_quant

__all__ = ["FinalLayerCertificate", "certified_forward",
           "nvfp4_act_err_bound", "mx_act_err_bound", "gelu_lipschitz",
           "IntervalChainCertificate", "certified_forward_interval"]


class FinalLayerCertificate:
    """Weight-error certificate for the final layer of a thor_fp4.Linear.

        cert = FinalLayerCertificate(last_fp4_layer)
        certified, bound = cert.check(x, logits)     # x = that layer's input

    check() returns a bool mask of samples whose FP4 argmax provably matches
    the true-weights argmax on the same activations, plus the per-sample
    bound used. E is stored fp16 (2 bytes/weight — the master weights stay
    fp32 in the layer; drop the attribute to free it)."""

    def __init__(self, layer: Linear, slack: float = 1.02):
        self.layer = layer
        self.slack = slack
        if layer._weight_fp4 is None:
            layer.refresh_weight()
        w_q = layer._weight_fp4.dequantize()                # f32 (N, K)
        err = (layer.weight.detach().float() - w_q).abs()
        self.err = err.half()

    def check(self, x: torch.Tensor, logits: torch.Tensor):
        """x: (M, K) the activations the layer consumed (pre-quantization);
        logits: (M, N) its FP4 output. Returns (certified (M,) bool, bound (M,)).

        The hardware computed sum(dequant(q(x)) * W_hat) — the weight-error
        bound must therefore be weighted by |x_hat| (the dequantized
        activations), NOT by |x|: using |x| would silently fold the
        activation-quantization error into a certificate that claims not to
        cover it (this exact unsoundness was caught by the unit test)."""
        from . import ops
        x = x.detach().contiguous()
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.float()
        p, s = ops.quantize_nvfp4(x.float(), self.layer.act_tensor_scale)
        xhat = ops.dequantize_nvfp4(p, s, self.layer.act_tensor_scale, *x.shape)
        bound = xhat.abs() @ self.err.float().t()                    # (M, N)
        top = logits.float().topk(2, dim=-1)
        margin = top.values[:, 0] - top.values[:, 1]
        # argmax is preserved only if top-1 beats EVERY other logit after
        # error: bound against the largest bound among all other logits
        # (top-2 alone is not enough — a third logit with a large bound can
        # overtake; caught by the soundness test)
        b_top1 = bound.gather(1, top.indices[:, :1]).squeeze(1)
        need = b_top1 + bound.max(dim=1).values
        # output rounding: the logits carry f16 (or bf16) output rounding on
        # top of the exact sum; allow one ulp of each of the top-2 values
        eps_out = 2.0 ** -11 * (top.values.abs().sum(1) + 1e-30)
        ok = margin > self.slack * need + eps_out
        return ok, need + eps_out


@torch.no_grad()
def certified_forward(model: nn.Module, x: torch.Tensor):
    """Run `model` (any nn.Module containing thor_fp4.Linear layers) and
    apply the final-layer certificate: certified samples return the FP4
    logits; uncertified samples get the last FP4 layer recomputed with its
    fp32 master weights on the same activations (promotion). Returns
    (logits, certified_mask).

    This is the arithmetic escalation gate: every certified sample's argmax
    is provably what the true final-layer weights would produce on the same
    activations; the promoted samples get exactly those weights. The model
    itself is run as-is (hooks capture the last FP4 layer's input/output),
    so activation glue in forward() is honored; the certificate assumes the
    last FP4 layer is the decision head."""
    lins = [m for m in model.modules() if isinstance(m, Linear)]
    last = lins[-1]
    cert = FinalLayerCertificate(last)
    cap = {}
    h = last.register_forward_hook(
        lambda mod, i, o: cap.update(xin=i[0], xout=o))
    out = model(x)
    h.remove()
    logits = cap["xout"].float().clone()
    certified, _ = cert.check(cap["xin"].float(), logits)
    if int((~certified).sum()):
        w = last.weight.detach().float()
        promo = cap["xin"].float() @ w.t()
        if last.bias is not None:
            promo = promo + last.bias.float()
        logits[~certified] = promo[~certified]
    return logits, certified


# ---------------------------------------------------------------------------
# ROUND 2: the certificate carried through the ACTIVATIONS (interval
# arithmetic). Round 1 measured that ~90% of FP4-vs-BF16 disagreement is
# upstream activation-quantization error, invisible to a final-layer
# certificate. The block scales used at runtime are known, so the
# activation-quantization error is BOUNDED per element: an e2m1/e4m3 code c
# against its block's decoded scale s commits any value in the code's
# rounding interval, i.e. |v - dequant(c)| <= halfstep(c) * s (saturated
# codes: up to the amax the RN power-of-two scale permits). Those intervals
# propagate through the chain: error in = delta_in, weight error E, then
# per layer
#     |z_exact - z_fp4| <= delta_in @ |W|^T + |x_hat| @ E^T + delta_in @ E^T
#                        + f32-accumulation slack + output rounding,
# through GELU by its Lipschitz constant, and the next quantization adds its
# own interval. The final margin test is the same as round 1's.
# ---------------------------------------------------------------------------

_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
# half-width of each e2m1 code's rounding interval, in code space: codes
# 0..3 sit on a 0.5-spaced grid, 4-5 on 1.0, 6 on 2.0; code 7 (value 6)
# additionally absorbs the clip of values up to 6*(1+2^-4) (the UE4M3 block
# scale rounds, so the code-space amax can exceed 6 by at most one UE4M3
# half-ulp) — 1.0 covers both.
_E2M1_HALF = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0])
_E4M3_MIN_SUB = 2.0 ** -9


def nvfp4_act_err_bound(x: torch.Tensor, tensor_scale: float = 1.0,
                        blk: int = 16) -> torch.Tensor:
    """Per-element error bound of tfp4_quant(x) (NVFP4 acts): the rounding
    interval of the code the kernel emits, times the decoded block scale.
    Bit-mirrors the device kernel's arithmetic (nvfp4_fake_quant)."""
    shp = x.shape
    xv = x.detach().reshape(-1, blk).float()
    amax = xv.abs().amax(dim=1, keepdim=True)
    sf = (amax * (1.0 / 6.0) / tensor_scale).to(torch.float8_e4m3fn).to(torch.float32)
    ds = sf * tensor_scale
    safe = ds.clamp_min(1e-30)
    q = xv / safe
    idx = torch.bucketize(q.abs().clamp_max(6.0), _E2M1_MID.to(x.device), right=True)
    half = _E2M1_HALF.to(x.device)[idx]
    # code 7 (value 6) also absorbs the clip of values above 6 in code space.
    # How far above depends on the SCALE's own rounding: a normal UE4M3 scale
    # rounds by at most one half-ulp (2^-4 relative -> values <= 6.4 -> 1.0
    # covers it), but a SUBNORMAL scale (sf < 2^-6, tiny-amax blocks) has an
    # absolute step, so sf_true/sf_dec can reach 1.5 and values reach 9 in
    # code space (error 3*ds).
    wide = (idx == 7) & (sf < 2.0 ** -6)
    half = torch.where(wide, torch.full_like(half, 3.0), half)
    delta = half * ds
    # blocks whose encoded scale underflowed to 0 decode to 0; their values
    # are bounded by the e4m3 subnormal floor the encoder compared against
    under = (ds <= 0).expand_as(delta)
    return torch.where(under, torch.full_like(delta, 6.0 * max(tensor_scale, 1e-30)
                                              * _E4M3_MIN_SUB * 1.001), delta).reshape(shp)


def mx_act_err_bound(x: torch.Tensor, kind: str = "mxfp8",
                     blk: int = 32) -> torch.Tensor:
    """Per-element error bound of tfp4_quant_mx(x): e4m3 (mxfp8) or e2m1
    (mxfp4) rounding intervals against the decoded UE8M0 scale. Saturated
    e4m3 codes (|code| = 448) take the amax the round-to-nearest-power-of-two
    scale permits: ds >= (amax/448)*2^-0.5, so amax <= 448*sqrt(2)*ds and the
    interval half-width is (633.4 - 448)*ds."""
    assert kind in ("mxfp4", "mxfp8")
    shp = x.shape
    xv = x.detach().reshape(-1, blk).float()
    amax = xv.abs().amax(dim=1, keepdim=True)
    maxval = 6.0 if kind == "mxfp4" else 448.0
    v = amax / maxval
    _, e = torch.frexp(v)
    lo = torch.ldexp(torch.ones_like(v), e - 1)
    hi = lo * 2
    code = torch.where(v - lo < hi - v, e - 1, e).clamp(-127, 127)
    ds = torch.ldexp(torch.ones_like(v), code).clamp_min(1e-30)
    q = xv / ds
    if kind == "mxfp4":
        idx = torch.bucketize(q.abs().clamp_max(6.0), _E2M1_MID.to(x.device), right=True)
        half = _E2M1_HALF.to(x.device)[idx]
        # UE8M0 rounds to the NEARER power of two (ties up): a block whose
        # amax/6 rounds DOWN has amax/ds < 1.5, so code 7's pre-image reaches
        # 9 in code space (error 3*ds)
        half = torch.where(idx == 7, torch.full_like(half, 3.0), half)
        delta = half * ds
    else:
        # the value the code decodes to (RN e4m3), whose interval we bound
        c = q.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32).abs()
        expo = torch.floor(torch.log2(c.clamp_min(1e-30)))
        half = torch.ldexp(torch.ones_like(c),
                           (expo - 4).clamp_min(-10).to(torch.int32))
        # saturated codes: the UE8M0 scale can round down, so the code-space
        # value reaches < 448*1.5; the interval reaches (448*1.5 - 448 + step/2)
        half = torch.where(c >= 448.0, torch.full_like(c, 448.0 * 0.5 + 16.0), half)
        delta = half * ds
    return delta.reshape(shp)


def gelu_lipschitz(x_sample=None) -> float:
    """sup|gelu'| for the exact (erf) GELU, computed on a dense grid with a
    0.1% safety margin — the multiplier used to carry intervals through GELU
    (GELU is not monotone, so an interval cannot be propagated exactly)."""
    if x_sample is None:
        x_sample = torch.linspace(-8, 8, 1_600_001, dtype=torch.float64)
    x = x_sample.double()
    phi = torch.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return float((Phi + x * phi).abs_().max().item()) * 1.001


class IntervalChainCertificate:
    """Carries per-element error intervals through a chain of thor_fp4.Linear
    layers (any mix of engines) with erf-GELU between them, and certifies the
    final argmax against the exact-arithmetic network (fp32 weights, no
    quantization, exact gelu). The reference for "true" is the ORIGINAL fp32
    master weights — the delivered logits are the model's own native output.

        cert = IntervalChainCertificate(model)          # Sequential(Linear,GELU,...)
        certified, bound, logits = cert.check(x)        # runs the model

    What is bounded, per layer: weight quantization error (exact, from the
    master weights), activation quantization (per-code intervals from the
    block scales actually used), f32 accumulation-order slack (gamma * |x|@|W|,
    covering both the tensor cores and the torch recomputation), output
    rounding (f16/bf16 half-ulp), and GELU by its Lipschitz constant. The
    certificate is sound but not tight: the triangle inequality over K terms
    sits ~sqrt(K) above the typical random-sign error."""

    def __init__(self, model: nn.Module, slack: float = 1.02):
        self.model = model
        self.slack = slack
        self.lip = gelu_lipschitz()
        self.layers = [m for m in model.modules() if isinstance(m, Linear)]
        if not self.layers:
            raise ValueError("no thor_fp4.Linear layers found")
        # per-layer weight error and |W| (fp32; the master weights are fp32)
        self.W, self.What, self.E = [], [], []
        for m in self.layers:
            w = m.weight.detach().float()
            if m.engine_kind == "mx8":
                wq = mx_fake_quant(w, "mxfp4")
            else:
                ts = None
                if m._weight_fp4 is None:
                    m.refresh_weight()
                wq = m._weight_fp4.dequantize()
            self.W.append(w)
            self.What.append(wq)
            self.E.append((w - wq).abs())

    def _fake_act(self, layer: Linear, x: torch.Tensor) -> torch.Tensor:
        """The delivered activation codes' values (bit-identical to the
        device kernels; see tests) for the layer's engine."""
        if layer.engine_kind == "mx8":
            return mx_fake_quant(x, "mxfp8")
        return nvfp4_fake_quant(x, layer.act_tensor_scale)

    def _act_err(self, layer: Linear, x: torch.Tensor) -> torch.Tensor:
        if layer.engine_kind == "mx8":
            return mx_act_err_bound(x, "mxfp8")
        return nvfp4_act_err_bound(x, layer.act_tensor_scale)

    @torch.no_grad()
    def check(self, x: torch.Tensor):
        """Run the model natively and return (certified (M,), bound (M,),
        logits (M, N) f32)."""
        # capture each layer's input through a native forward (the anchor:
        # bounds are computed on the activations the hardware really got)
        cap: list[torch.Tensor] = []
        hooks = [m.register_forward_hook(
            lambda mod, i, o: cap.append(i[0].detach().clone()))
            for m in self.layers]
        logits = self.model(x)
        for h in hooks:
            h.remove()
        logits = logits.float().clone()

        M = x.shape[0]
        delta = torch.zeros_like(x, dtype=torch.float32)      # |h_exact - x_hat| per element
        gamma = 2.0 * math.ceil(math.log2(x.shape[1])) * 2.0 ** -24
        layer_stats = []
        for li, layer in enumerate(self.layers):
            xin = cap[li].float()                              # native input
            xd = self._fake_act(layer, xin)                    # delivered codes' values
            W, What, E = self.W[li], self.What[li], self.E[li]
            od = {"f16": 2.0 ** -11, "bf16": 2.0 ** -9, "f32": 0.0}[layer.out_dtype]
            # delivered pre-activation, as the hardware computed it (codes x
            # codes, f32 accumulate, output rounding), then + bias in od
            zhat = xd @ What.t()
            if layer.bias is not None:
                zhat = zhat + layer.bias.float()
            bpre = (delta @ W.abs().t()                        # exact-input, exact-weights term
                    + xd.abs() @ E.t()                         # delivered acts, weight error
                    + delta @ E.t())                           # cross term
            bpre = self.slack * (bpre + gamma * (xd.abs() @ W.abs().t())) + od * zhat.abs()
            layer_stats.append(float(bpre.median().item()))
            if li == len(self.layers) - 1:
                top = logits.topk(2, dim=-1)
                margin = top.values[:, 0] - top.values[:, 1]
                b_top1 = bpre.gather(1, top.indices[:, :1]).squeeze(1)
                need = b_top1 + bpre.max(dim=1).values
                # final rounding of the delivered logits (they include the
                # f16/bf16 bias add): one half-ulp of each top-2 value
                eps_out = od * (top.values.abs().sum(1) + 1e-30)
                ok = margin > self.slack * need + eps_out
                self.last_layer_stats = layer_stats
                return ok, need + eps_out, logits
            # hidden layer: output rounding -> GELU (Lipschitz) -> next layer's
            # activation quantization adds its own interval. Mirror the native
            # layer exactly: the GEMM output rounds to out_dtype, then the bias
            # is added IN that dtype.
            dt16 = {"f16": torch.float16, "bf16": torch.bfloat16,
                    "f32": torch.float32}[layer.out_dtype]
            zb = xd @ What.t()
            z16 = zb.to(dt16)
            if layer.bias is not None:
                z16 = z16 + layer.bias.float().to(dt16)        # add in out_dtype, like the layer
            g = torch.nn.functional.gelu(z16.float())
            nxt = self.layers[li + 1]
            delta = self.lip * bpre + self._act_err(nxt, g)
        raise RuntimeError("unreachable")


@torch.no_grad()
def certified_forward_interval(model: nn.Module, x: torch.Tensor):
    """Interval-certified forward: certified samples return the native FP4
    logits (their argmax provably matches the exact-arithmetic network);
    uncertified samples get the final layer recomputed with the fp32 master
    weights on the same activations. Returns (logits, certified_mask).

    The interval bound covers the whole chain — weight quantization,
    activation quantization (from the runtime block scales), GELU and output
    rounding — unlike FinalLayerCertificate, which covers only the final
    layer's weight error."""
    cert = IntervalChainCertificate(model)
    certified, _, logits = cert.check(x)
    lins = cert.layers
    last = lins[-1]
    cap = {}
    h = last.register_forward_hook(
        lambda mod, i, o: cap.update(xin=i[0]))
    _ = model(x)
    h.remove()
    if int((~certified).sum()):
        w = last.weight.detach().float()
        promo = cap["xin"].float() @ w.t()
        if last.bias is not None:
            promo = promo + last.bias.float()
        logits[~certified] = promo[~certified]
    return logits, certified

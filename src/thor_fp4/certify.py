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

import torch
import torch.nn as nn

from .linear import Linear

__all__ = ["FinalLayerCertificate", "certified_forward"]


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

# cascade.py — the cleanup-loop hook: run FP4, escalate to BF16 when unsure.
#
# A quantized decision model keeps its BF16 self's top-1 choice only some
# fraction of the time (measured on the tile scorer: 71% for PTQ). This
# runner executes the cheap FP4 model first, reads a confidence from its
# logits, and re-runs only the low-confidence samples through the BF16 copy.
# The caller picks the threshold; thor_fp4.cascade_sweep() maps thresholds to
# (agreement recovered, escalation rate) so the cost/quality point is a
# measured decision, not a guess.

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Cascade", "confidence_margin", "confidence_softmax", "cascade_sweep"]


def confidence_margin(logits: torch.Tensor) -> torch.Tensor:
    """top1 - top2 of the raw logits, per row."""
    top2 = logits.topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def confidence_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1).max(dim=-1).values


class Cascade(nn.Module):
    """fast = FP4 model, slow = BF16 copy, threshold in confidence units.

    forward(x) returns (logits, escalated_mask, confidence): samples whose
    confidence is below the threshold get the slow model's logits."""

    def __init__(self, fast: nn.Module, slow: nn.Module, threshold: float,
                 confidence=confidence_margin):
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.threshold = float(threshold)
        self.confidence = confidence
        self.last_escalated: int = 0

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.fast(x).float()
            conf = self.confidence(out)
            esc = conf < self.threshold
            self.last_escalated = int(esc.sum().item())
            if self.last_escalated:
                out[esc] = self.slow(x[esc]).float()
            return out, esc, conf


@torch.no_grad()
def cascade_sweep(fast: nn.Module, slow: nn.Module, x: torch.Tensor,
                  y_bf16_choice: torch.Tensor | None = None,
                  confidences=(confidence_margin,), n_thresholds: int = 9):
    """Sweep thresholds for a fast/slow pair on x. Returns a list of rows:
    (threshold, escalated_frac, agreement_with_slow, recovered_gap).

    agreement_with_slow: fraction of samples whose final choice equals the
    slow (BF16) model's choice — the escalation target. recovered_gap: the
    fraction of the fast model's wrong-vs-slow choices that the cascade
    fixed."""
    slow_out = slow(x).float()
    slow_choice = slow_out.argmax(-1) if y_bf16_choice is None else y_bf16_choice
    fast_out = fast(x).float()
    fast_choice = fast_out.argmax(-1)
    wrong = fast_choice != slow_choice
    total_wrong = int(wrong.sum().item())
    rows = []
    for conf_fn in confidences:
        conf = conf_fn(fast_out)
        lo, hi = conf.min().item(), conf.max().item()
        for i in range(n_thresholds):
            t = lo + (hi - lo) * i / max(1, n_thresholds - 1)
            esc = conf < t
            final = fast_choice.clone()
            final[esc] = slow_choice[esc]
            agree = (final == slow_choice).float().mean().item()
            recovered = (final[wrong] == slow_choice[wrong]).sum().item() / max(1, total_wrong)
            rows.append({"confidence": conf_fn.__name__, "threshold": t,
                         "escalated": esc.float().mean().item(),
                         "agreement": agree, "recovered_gap": recovered})
    return rows

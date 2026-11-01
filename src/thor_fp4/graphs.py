# graphs.py — CUDA-graph capture for FP4 models.
#
# Why graphs matter here: an FP4 chain is many small kernels (quantize + GEMM
# per layer); eager through ctypes is host-launch-bound (measured: enqueue
# time == GPU time at 100%). A captured graph replays the whole chain as one
# launch. The core extension resolves everything shape-dependent on the first
# call and never allocates inside a steady-state call, so capture works — the
# capture below first runs the function eagerly on the static buffers
# (resolving cuBLASLt algorithms and warming the per-shape buffers), then
# captures.

from __future__ import annotations

import torch

__all__ = ["Graphed"]


class Graphed:
    """Capture fn(*inputs) as a CUDA graph with static input/output buffers.

        g = Graphed(model, example_x)          # model: any callable of tensors
        out = g(x)                             # copy-in, replay, return the static out

    The inputs are copied into static buffers of the example shapes on every
    call; the returned tensor is the static output buffer (valid until the
    next call — .clone() it if you keep it). Only tensors flow through; the
    function must be capture-safe (no .item(), no syncs, fixed shapes)."""

    def __init__(self, fn, *example_inputs, warmup: int = 3, pool=None):
        self.fn = fn
        self.static_in = [t.clone() for t in example_inputs]
        for _ in range(warmup):
            out = fn(*self.static_in)
        torch.cuda.synchronize()
        self.static_out = out
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.static_out = fn(*self.static_in)

    def __call__(self, *inputs):
        for s, t in zip(self.static_in, inputs):
            s.copy_(t)
        self.graph.replay()
        return self.static_out


def capture_module(model: torch.nn.Module, *example_inputs, warmup: int = 3) -> Graphed:
    """Graphed(model.forward-ish) convenience: capture a whole module."""
    return Graphed(lambda *xs: model(*xs), *example_inputs, warmup=warmup)

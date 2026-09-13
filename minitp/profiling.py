"""Optional logical profiler scopes (v0.4).

``set_trace_labels(True)`` wraps model sub-steps in
``torch.profiler.record_function`` so a torch.profiler trace (Chrome /
Perfetto JSON) shows logical names: mini_tp.layer_N.attention, .qkv_gemm,
.rope, .sdpa, .o_proj_allreduce, .mlp, .lm_head, .distributed_argmax.

Off by default: a single boolean check per forward. Use with::

    from minitp import profiling
    profiling.set_trace_labels(True)
    with profile(activities=[CPU, CUDA]) as prof:
        ...generate...
    prof.export_chrome_trace("trace.json")
"""

from __future__ import annotations

from contextlib import contextmanager

_TRACE_LABELS = False


def set_trace_labels(enabled: bool) -> None:
    global _TRACE_LABELS
    _TRACE_LABELS = enabled


def trace_labels_enabled() -> bool:
    return _TRACE_LABELS


@contextmanager
def scope(name: str):
    """record_function scope when labels are enabled; no-op otherwise."""
    if not _TRACE_LABELS:
        yield
        return
    from torch.profiler import record_function

    with record_function(name):
        yield


def record_comm_available() -> bool:
    """torch.distributed.record_comm names collectives in traces (newer
    torch). Feature-detected so older versions silently keep plain names."""
    import torch.distributed as dist

    return hasattr(dist, "record_comm")

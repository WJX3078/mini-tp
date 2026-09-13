"""Thin, instrumented wrappers around torch.distributed collectives.

Not a reimplementation of torch.distributed. Three jobs:

1. One auditable call site for every collective in the runtime.
2. Optional, *semantically correct* communication profiling. CUDA/NCCL
   execution is asynchronous: wall-clock around the enqueue measures host
   launch cost, not GPU execution. On CUDA tensors we record paired CUDA
   events around the collective and resolve device elapsed time lazily in
   ``drain_comm_stats()`` — a single synchronize at readout, never inside the
   hot path. On CPU/Gloo the collective blocks, so wall time *is* execution
   time and ``gpu_elapsed_ms`` is None.
3. A no-op path when TP=1 (no process group) and near-zero overhead when
   profiling is disabled (a single boolean check before the real op).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

_PROFILING = False
_RECORDS: list[dict] = []
_PENDING: list[tuple[torch.Tensor, torch.Tensor, dict]] = []  # (start_ev, end_ev, meta)


@dataclass
class CommStats:
    """Aggregated view used by tests/benchmarks."""

    calls: int
    total_bytes: int
    host_launch_ms: float
    gpu_elapsed_ms: float | None  # None on CPU/Gloo or if not drained


def set_profiling(enabled: bool) -> None:
    global _PROFILING
    _PROFILING = enabled


def reset_comm_stats() -> None:
    _RECORDS.clear()
    _PENDING.clear()


def drain_comm_stats() -> None:
    """Resolve pending CUDA-event pairs (one synchronize total, not per op)."""
    if not _PENDING:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for start_ev, end_ev, meta in _PENDING:
        meta["gpu_elapsed_ms"] = start_ev.elapsed_time(end_ev)
    _PENDING.clear()


def get_comm_stats() -> list[dict]:
    drain_comm_stats()
    return list(_RECORDS)


def comm_summary(window: list[dict] | None = None) -> CommStats:
    stats = get_comm_stats() if window is None else window
    gpu = [s["gpu_elapsed_ms"] for s in stats if s["gpu_elapsed_ms"] is not None]
    return CommStats(
        calls=len(stats),
        total_bytes=sum(s["bytes"] for s in stats),
        host_launch_ms=sum(s["host_launch_ms"] for s in stats) * 1e3,
        gpu_elapsed_ms=sum(gpu) if gpu else None,
    )


def _meta(op: str, tensor: torch.Tensor, group) -> dict:
    return {
        "op": op,
        "bytes": tensor.numel() * tensor.element_size(),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "world_size": dist.get_world_size(group) if group is not None else 1,
        "host_launch_ms": None,
        "gpu_elapsed_ms": None,
    }


def _group(pg):
    return pg if pg is not None else dist.group.WORLD if dist.is_initialized() else None


def _profiled(op: str, tensors: list[torch.Tensor], group, run):
    """Record one profiled collective. The op is enqueued exactly once, bracketed
    by CUDA events (device span, resolved at drain) and perf_counter (host launch)."""
    meta = _meta(op, tensors[0], group)
    is_cuda = tensors[0].is_cuda and torch.cuda.is_available()
    start_ev = end_ev = None
    t0 = time.perf_counter()
    if is_cuda:
        start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)
        start_ev.record()
    result = run()
    if is_cuda:
        end_ev.record()
        _PENDING.append((start_ev, end_ev, meta))
    meta["host_launch_ms"] = (time.perf_counter() - t0) * 1e3
    _RECORDS.append(meta)
    return result


def all_reduce(t: torch.Tensor, pg=None, op=dist.ReduceOp.SUM) -> torch.Tensor:
    group = _group(pg)
    if group is None:
        return t
    if not _PROFILING:
        dist.all_reduce(t, op=op, group=group)
        return t
    return _profiled(
        "all_reduce", [t], group, lambda: dist.all_reduce(t, op=op, group=group)
    )


def all_gather(out: torch.Tensor, t: torch.Tensor, pg=None) -> torch.Tensor:
    """``out`` has the full (concatenated along last dim) shape; ``t`` the local shard."""
    group = _group(pg)
    if group is None:
        out.copy_(t)
        return out
    t = t.contiguous()
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    run = lambda: dist.all_gather(parts, t, group=group)  # noqa: E731

    def finish():
        run()
        out.copy_(torch.cat(parts, dim=-1))
        return out

    if not _PROFILING:
        return finish()
    return _profiled("all_gather", [out], group, finish)


def reduce_scatter(out: torch.Tensor, t: torch.Tensor, pg=None) -> torch.Tensor:
    """``t`` full tensor (dim 0 = world * local), ``out`` local shard; sums then scatters."""
    group = _group(pg)
    if group is None:
        out.copy_(t)
        return out

    def run():
        dist.reduce_scatter_tensor(out, t.contiguous(), op=dist.ReduceOp.SUM, group=group)
        return out

    if not _PROFILING:
        return run()
    return _profiled("reduce_scatter", [t], group, run)


def broadcast(t: torch.Tensor, src: int, pg=None) -> torch.Tensor:
    group = _group(pg)
    if group is None:
        return t
    if not _PROFILING:
        dist.broadcast(t, src=src, group=group)
        return t
    return _profiled("broadcast", [t], group, lambda: dist.broadcast(t, src=src, group=group))

"""Thin, instrumented wrappers around torch.distributed collectives.

Not a reimplementation of torch.distributed: the point is (a) one call site so
every collective in the runtime is auditable, (b) optional byte/time/call
accounting for the communication profiler (off by default), and (c) a no-op
path when TP=1 so the same layer code runs single-GPU.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

_PROFILING = False
_STATS: list[dict] = []


def set_profiling(enabled: bool) -> None:
    global _PROFILING
    _PROFILING = enabled


def reset_comm_stats() -> None:
    _STATS.clear()


def get_comm_stats() -> list[dict]:
    return list(_STATS)


def _record(op: str, tensors: list[torch.Tensor], group, elapsed: float) -> None:
    if group is None:
        return
    _STATS.append(
        {
            "op": op,
            "bytes": sum(t.numel() * t.element_size() for t in tensors),
            "dtype": str(tensors[0].dtype) if tensors else None,
            "world_size": dist.get_world_size(group),
            "elapsed_ms": elapsed * 1e3,
        }
    )


def _group(pg):
    return pg if pg is not None else dist.group.WORLD if dist.is_initialized() else None


def all_reduce(t: torch.Tensor, pg=None, op=dist.ReduceOp.SUM) -> torch.Tensor:
    group = _group(pg)
    if group is None:
        return t
    if _PROFILING:
        t0 = time.perf_counter()
        dist.all_reduce(t, op=op, group=group)
        _record("all_reduce", [t], group, time.perf_counter() - t0)
    else:
        dist.all_reduce(t, op=op, group=group)
    return t


def all_gather(out: torch.Tensor, t: torch.Tensor, pg=None) -> torch.Tensor:
    """``out`` has the full (concatenated along last dim) shape; ``t`` the local shard."""
    group = _group(pg)
    if group is None:
        out.copy_(t)
        return out
    t = t.contiguous()
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    if _PROFILING:
        t0 = time.perf_counter()
        dist.all_gather(parts, t, group=group)
        _record("all_gather", [out], group, time.perf_counter() - t0)
    else:
        dist.all_gather(parts, t, group=group)
    out.copy_(torch.cat(parts, dim=-1))
    return out


def reduce_scatter(out: torch.Tensor, t: torch.Tensor, pg=None) -> torch.Tensor:
    """``t`` full tensor (dim 0 = world * local), ``out`` local shard; sums then scatters."""
    group = _group(pg)
    if group is None:
        out.copy_(t)
        return out
    if _PROFILING:
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(out, t.contiguous(), op=dist.ReduceOp.SUM, group=group)
        _record("reduce_scatter", [t], group, time.perf_counter() - t0)
    else:
        dist.reduce_scatter_tensor(out, t.contiguous(), op=dist.ReduceOp.SUM, group=group)
    return out


def broadcast(t: torch.Tensor, src: int, pg=None) -> torch.Tensor:
    group = _group(pg)
    if group is None:
        return t
    dist.broadcast(t, src=src, group=group)
    return t

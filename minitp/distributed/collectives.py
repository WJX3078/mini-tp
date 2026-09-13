"""Thin, instrumented wrappers around torch.distributed collectives.

Contract (v0.3, regression-tested):

- ``all_reduce(t)``   -> ``t``      (in-place, identity return)
- ``broadcast(t)``    -> ``t``
- ``reduce_scatter(out, t)`` -> ``out``
- ``all_gather(out, t)``     -> ``out``

...on every path: profiling on/off, CPU/Gloo, CUDA/NCCL, TP=1 (no group).
v0.2 violated this when profiling: ``dist.all_reduce`` returns ``None`` and the
wrapper passed it through, crashing ``RowParallelLinear`` under TP>1 +
profiling (docs/V03_AUDIT.md B1).

Profiling semantics (v0.3): CUDA/NCCL execution is asynchronous, so wall-clock
around the enqueue is *host launch* time, not GPU time. On CUDA tensors we
record paired CUDA events around **only the distributed op** (never the
post-processing such as ``torch.cat``/copies — B3), and resolve device time
lazily in ``drain_comm_stats()``: one synchronize at readout, never in the hot
path. Record fields and units:

- ``host_launch_ms``   — whole wrapper host time incl. postprocess (ms)
- ``collective_gpu_ms``— CUDA-event device span of the distributed op (ms,
  None on CPU/Gloo where the op blocks and host time *is* execution time)
- ``postprocess_host_ms`` — host time of non-collective glue (ms, 0 if none)
- ``bytes`` — logical payload: input tensor for all_reduce/reduce_scatter,
  output tensor for all_gather (definition documented in
  docs/COMMUNICATION_PROFILING.md)
- ``dtype`` / ``shape`` / ``world_size``

When profiling is disabled the wrapper is one boolean check around the raw
``torch.distributed`` call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

_PROFILING = False
_RECORDS: list[dict] = []
_PENDING: list[tuple[torch.Tensor, torch.Tensor, dict]] = []  # (start_ev, end_ev, meta)
_ALLGATHER_BASE_OK: bool | None = None  # feature-detected once per process


@dataclass
class CommStats:
    """Aggregated view used by tests/benchmarks. All *_ms fields are milliseconds."""

    calls: int
    total_bytes: int
    host_launch_ms: float
    collective_gpu_ms: float | None  # None on CPU/Gloo
    postprocess_host_ms: float


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
        meta["collective_gpu_ms"] = start_ev.elapsed_time(end_ev)
    _PENDING.clear()


def get_comm_stats() -> list[dict]:
    drain_comm_stats()
    return list(_RECORDS)


def comm_summary(window: list[dict] | None = None) -> CommStats:
    stats = get_comm_stats() if window is None else window
    gpu = [s["collective_gpu_ms"] for s in stats if s["collective_gpu_ms"] is not None]
    return CommStats(
        calls=len(stats),
        total_bytes=sum(s["bytes"] for s in stats),
        host_launch_ms=sum(s["host_launch_ms"] for s in stats),
        collective_gpu_ms=sum(gpu) if gpu else None,
        postprocess_host_ms=sum(s["postprocess_host_ms"] for s in stats),
    )


def _meta(op: str, tensor: torch.Tensor, group) -> dict:
    return {
        "op": op,
        "bytes": tensor.numel() * tensor.element_size(),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "world_size": dist.get_world_size(group) if group is not None else 1,
        "host_launch_ms": 0.0,       # ms, whole wrapper
        "collective_gpu_ms": None,   # ms, CUDA events; None on CPU/Gloo
        "postprocess_host_ms": 0.0,  # ms, host glue (cat/copy) only
    }


def _group(pg):
    return pg if pg is not None else dist.group.WORLD if dist.is_initialized() else None


def _profiled(
    op: str,
    tensors: list[torch.Tensor],
    group,
    collective_fn,
    postprocess_fn=None,
):
    """Run one profiled collective. The op is enqueued exactly once; CUDA
    events bracket ONLY the distributed call (B3), post-processing is timed
    separately on the host. Units: milliseconds, converted exactly once."""
    meta = _meta(op, tensors[0], group)
    is_cuda = tensors[0].is_cuda and torch.cuda.is_available()
    t0 = time.perf_counter()
    if is_cuda:
        start_ev, end_ev = torch.cuda.Event(True), torch.cuda.Event(True)
        start_ev.record()
        collective_fn()
        end_ev.record()
        _PENDING.append((start_ev, end_ev, meta))
    else:
        collective_fn()
    if postprocess_fn is not None:
        tp0 = time.perf_counter()
        postprocess_fn()
        meta["postprocess_host_ms"] = (time.perf_counter() - tp0) * 1e3
    meta["host_launch_ms"] = (time.perf_counter() - t0) * 1e3
    _RECORDS.append(meta)
    # collective_fn/postprocess_fn close over the output tensors; the wrapper
    # return contract is enforced by each public function below (B1).
    return tensors[1] if len(tensors) > 1 else tensors[0]


def all_reduce(t: torch.Tensor, pg=None, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """In-place sum across the TP group. Always returns ``t``."""
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
    """Gather equal-shaped shards and concatenate along the last dim into
    ``out``. Always returns ``out``. The CUDA-event window covers only the
    distributed op; cat/copy is post-processing (B3)."""
    group = _group(pg)
    if group is None:
        out.copy_(t)
        return out
    t = t.contiguous()
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]

    def collective():
        dist.all_gather(parts, t, group=group)

    def postprocess():
        out.copy_(torch.cat(parts, dim=-1))

    if not _PROFILING:
        collective()
        postprocess()
        return out
    _profiled("all_gather", [out], group, collective, postprocess)
    return out


def all_gather_dim0(out: torch.Tensor, t: torch.Tensor, pg=None) -> torch.Tensor:
    """Fast path: gather rank blocks along **dim 0** into ``out[world, *t.shape]``.

    Uses ``all_gather_into_tensor`` **only on NCCL** — measured on torch
    2.6.0+cu124, the Gloo backend neither validates nor correctly executes
    this op for flat payloads (silently returns garbage / rejects valid
    shapes), so CPU/Gloo deterministically takes the list path. All ranks
    share the backend, so the gate cannot diverge across ranks.
    """
    group = _group(pg)
    if group is None:
        out.copy_(t.unsqueeze(0))
        return out
    t = t.contiguous()
    world = dist.get_world_size(group)
    global _ALLGATHER_BASE_OK
    if _ALLGATHER_BASE_OK is None:
        _ALLGATHER_BASE_OK = dist.get_backend(group) == "nccl"

    def collective():
        if _ALLGATHER_BASE_OK:
            dist.all_gather_into_tensor(out.reshape(world * t.numel()), t.reshape(-1), group=group)
        else:
            dist.all_gather([out[i] for i in range(world)], t, group=group)

    if not _PROFILING:
        collective()
        return out
    _profiled("all_gather_dim0", [out], group, collective)
    return out


def reduce_scatter(out: torch.Tensor, t: torch.Tensor, pg=None) -> torch.Tensor:
    """Sum then scatter along dim 0: ``t`` full [world*local, ...], ``out`` local.
    Always returns ``out``."""
    group = _group(pg)
    if group is None:
        out.copy_(t)
        return out

    def run():
        dist.reduce_scatter_tensor(out, t.contiguous(), op=dist.ReduceOp.SUM, group=group)

    if not _PROFILING:
        run()
        return out
    _profiled("reduce_scatter", [t], group, run)
    return out


def broadcast(t: torch.Tensor, src: int, pg=None) -> torch.Tensor:
    """In-place broadcast from ``src``. Always returns ``t``."""
    group = _group(pg)
    if group is None:
        return t
    if not _PROFILING:
        dist.broadcast(t, src=src, group=group)
        return t
    return _profiled(
        "broadcast", [t], group, lambda: dist.broadcast(t, src=src, group=group)
    )

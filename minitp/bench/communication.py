"""Communication accounting helpers on top of the collectives instrumentation."""

from __future__ import annotations

from minitp.distributed import get_comm_stats, reset_comm_stats, set_profiling


def summary(window: list[dict] | None = None) -> dict:
    stats = get_comm_stats() if window is None else window
    by_op: dict[str, dict] = {}
    for s in stats:
        agg = by_op.setdefault(s["op"], {"calls": 0, "bytes": 0, "ms": 0.0})
        agg["calls"] += 1
        agg["bytes"] += s["bytes"]
        agg["ms"] += s["elapsed_ms"]
    return {
        "calls": len(stats),
        "total_bytes": sum(s["bytes"] for s in stats),
        "total_ms": round(sum(s["elapsed_ms"] for s in stats), 2),
        "by_op": by_op,
    }


def record_window(fn):
    """Profile a single callable: returns (result, comm_summary)."""
    reset_comm_stats()
    set_profiling(True)
    try:
        result = fn()
        return result, summary()
    finally:
        set_profiling(False)

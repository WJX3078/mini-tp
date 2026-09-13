"""Communication profiler regression tests (v0.3): return contract, units,
profiling-boundary semantics. Gloo TP=2 where collectives are involved."""

import os

import pytest
import torch
import torch.distributed as dist

from minitp.distributed import (
    all_gather,
    all_gather_dim0,
    all_reduce,
    broadcast,
    comm_summary,
    drain_comm_stats,
    get_comm_stats,
    reduce_scatter,
    reset_comm_stats,
    set_profiling,
)
from minitp.distributed.context import ParallelContext


def test_profiling_off_is_passthrough():
    set_profiling(False)
    reset_comm_stats()
    all_reduce(torch.tensor([1.0]))  # pg=None -> no-op, must not record
    assert get_comm_stats() == []


def _pg_ctx(rank: int, tp: int) -> ParallelContext:
    return ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)


def _tp2_contract_and_units(rank, tp):
    """B1/B2 regressions: identity return contract under BOTH profiling states,
    and comm_summary units exactly equal the sum of record values (ms)."""
    ctx = _pg_ctx(rank, tp)
    out = {}

    for profiling in (False, True):
        set_profiling(profiling)
        reset_comm_stats()
        t = torch.ones(4, 8) * (rank + 1)
        out[f"allreduce_{profiling}"] = all_reduce(t, ctx.process_group) is t
        assert torch.equal(t, torch.full((4, 8), 3.0))

        tb = torch.ones(4) if rank == 0 else torch.zeros(4)
        out[f"broadcast_{profiling}"] = broadcast(tb, src=0, pg=ctx.process_group) is tb
        assert torch.equal(tb, torch.ones(4))

        full = torch.ones(8) * (rank + 1)
        shard = torch.empty(4)
        out[f"reduce_scatter_{profiling}"] = reduce_scatter(shard, full, ctx.process_group) is shard

        g_out = torch.empty(4, 16)
        g_in = torch.full((4, 8), float(rank + 1))
        out[f"all_gather_{profiling}"] = all_gather(g_out, g_in, ctx.process_group) is g_out
        assert torch.equal(g_out[:, :8], torch.full((4, 8), 1.0))
        assert torch.equal(g_out[:, 8:], torch.full((4, 8), 2.0))

        # dim-0 fast path (used by distributed argmax)
        pair = torch.tensor([[float(rank), float(rank * 10)]])
        d0 = torch.empty(tp, 1, 2)
        out[f"all_gather_dim0_{profiling}"] = all_gather_dim0(d0, pair, ctx.process_group) is d0
        assert torch.equal(d0[0], torch.tensor([[0.0, 0.0]]))
        assert torch.equal(d0[1], torch.tensor([[1.0, 10.0]]))

    # units: summary must be the plain sum of record values (no double scaling)
    set_profiling(True)
    reset_comm_stats()
    for _ in range(3):
        all_reduce(torch.ones(4), ctx.process_group)
    stats = get_comm_stats()
    summary = comm_summary(stats)
    out["units_ok"] = abs(summary.host_launch_ms - sum(s["host_launch_ms"] for s in stats)) < 1e-9
    out["record_fields"] = sorted(stats[0].keys())
    assert out["units_ok"], (summary.host_launch_ms, [s["host_launch_ms"] for s in stats])
    # collective_gpu_ms is None on CPU/Gloo (blocking op)
    assert all(s["collective_gpu_ms"] is None for s in stats)
    set_profiling(False)
    reset_comm_stats()
    return out


@pytest.mark.distributed
def test_tp2_contract_units_gloo():
    if os.environ.get("SKIP_DISTRIBUTED_TESTS"):
        pytest.skip("distributed tests disabled")
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_contract_and_units)
    first = r[0]
    for key in ("allreduce_True", "broadcast_True", "reduce_scatter_True",
                "all_gather_True", "all_gather_dim0_True",
                "allreduce_False", "broadcast_False", "reduce_scatter_False",
                "all_gather_False", "all_gather_dim0_False", "units_ok"):
        assert first[key] is True, (key, first)


def test_drain_is_idempotent_and_clearable():
    reset_comm_stats()
    drain_comm_stats()
    assert get_comm_stats() == []


def test_mock_timer_units():
    """Deterministic units check with synthetic records (no timer)."""
    from minitp.distributed.collectives import _RECORDS, comm_summary

    _RECORDS.clear()
    for ms in (0.5, 1.0, 2.5):
        _RECORDS.append({
            "op": "all_reduce", "bytes": 16, "dtype": "torch.float32", "shape": [4],
            "world_size": 2, "host_launch_ms": ms, "collective_gpu_ms": None,
            "postprocess_host_ms": 0.0,
        })
    summary = comm_summary()
    assert abs(summary.host_launch_ms - 4.0) < 1e-9  # ms in == ms out, no 1e3
    assert summary.calls == 3 and summary.total_bytes == 48
    _RECORDS.clear()

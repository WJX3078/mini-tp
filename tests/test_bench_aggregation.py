"""Benchmark aggregation + CLI legality tests."""

import pytest
import torch
import torch.distributed as dist

from minitp.bench.benchmark import _aggregate_across_ranks, _percentiles
from minitp.distributed.context import ParallelContext


def test_percentiles_empty_is_none():
    assert _percentiles([]) is None
    p = _percentiles([5.0, 1.0, 3.0])
    assert p["mean"] == 3.0 and p["p50"] == 3.0 and p["max"] == 5.0


def test_aggregation_tp1_noop():
    ctx = ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None)
    assert _aggregate_across_ranks(ctx, {"prefill_s": 1.0}) == {}


def _tp2_aggregation(rank, tp):
    ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)
    values = {
        "prefill_s": 1.0 + rank,
        "tpot_ms": 10.0 - rank,
        "peak_mem_gib": 2.0,
    }
    agg = _aggregate_across_ranks(ctx, values)
    ok = (
        agg["prefill_s"]["max"] == 2.0 and agg["prefill_s"]["min"] == 1.0
        and agg["prefill_s"]["mean"] == 1.5 and abs(agg["prefill_s"]["skew"] - 1.0) < 1e-9
        and agg["tpot_ms"]["max"] == 10.0 and agg["tpot_ms"]["min"] == 9.0
        and agg["peak_mem_gib"]["max"] == 2.0
    )
    return ok


@pytest.mark.distributed
def test_tp2_aggregation_gloo():
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_aggregation)
    assert all(r.values()), r

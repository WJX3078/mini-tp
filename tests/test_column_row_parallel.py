"""Column/RowParallelLinear correctness: TP=1 (single process) and TP=2 (Gloo)."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from minitp.distributed.context import ParallelContext
from minitp.parallel.linear import ColumnParallelLinear, RowParallelLinear, shard_range

torch.manual_seed(0)


def _pg_ctx(rank: int, tp: int) -> ParallelContext:
    return ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)


def _make_ctx() -> ParallelContext:
    return ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None)


def _tp2_column(rank, tp):
    torch.manual_seed(42)
    W = torch.randn(32, 16)
    b = torch.randn(32)
    X = torch.randn(4, 16)
    ref = F.linear(X, W, b)
    ctx = _pg_ctx(rank, tp)

    layer = ColumnParallelLinear(16, 32, ctx, bias=True, gather_output=True)
    with torch.no_grad():
        layer.weight.copy_(W[rank * 16 : (rank + 1) * 16])
        layer.bias.copy_(b[rank * 16 : (rank + 1) * 16])
    err_gather = (layer(X) - ref).abs().max().item()

    layer2 = ColumnParallelLinear(16, 32, ctx, bias=True, gather_output=False)
    with torch.no_grad():
        layer2.weight.copy_(W[rank * 16 : (rank + 1) * 16])
        layer2.bias.copy_(b[rank * 16 : (rank + 1) * 16])
    local = layer2(X)
    parts = [torch.empty_like(local) for _ in range(tp)]
    dist.all_gather(parts, local, group=ctx.process_group)
    err_manual = (torch.cat(parts, dim=-1) - ref).abs().max().item()
    return max(err_gather, err_manual)


def _tp2_row(rank, tp):
    torch.manual_seed(7)
    W = torch.randn(20, 16)
    b = torch.randn(20)
    X = torch.randn(4, 16)
    ref = F.linear(X, W, b)
    ctx = _pg_ctx(rank, tp)

    layer = RowParallelLinear(16, 20, ctx, bias=True, input_is_parallel=False)
    with torch.no_grad():
        layer.weight.copy_(W[:, rank * 8 : (rank + 1) * 8])
        layer.bias.copy_(b)
    err_full_input = (layer(X) - ref).abs().max().item()

    layer2 = RowParallelLinear(16, 20, ctx, bias=True, input_is_parallel=True)
    with torch.no_grad():
        layer2.weight.copy_(W[:, rank * 8 : (rank + 1) * 8])
        layer2.bias.copy_(b)
    out2 = layer2(X[:, rank * 8 : (rank + 1) * 8].contiguous())
    err_parallel_input = (out2 - ref).abs().max().item()
    return max(err_full_input, err_parallel_input)


def _tp2_row_bias_not_doubled(rank, tp):
    """Regression: bias must be added once (after AllReduce), not per rank."""
    ctx = _pg_ctx(rank, tp)
    layer = RowParallelLinear(12, 8, ctx, bias=True, input_is_parallel=False)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(1.0)  # if reduced across ranks, output would be 2.0
    X = torch.zeros(2, 12)
    return layer(X).max().item()


def _tp2_column_uneven(rank, tp):
    torch.manual_seed(3)
    out_size = 10
    W = torch.randn(out_size, 8)
    X = torch.randn(2, 8)
    ref = F.linear(X, W, None)
    ctx = _pg_ctx(rank, tp)
    layer = ColumnParallelLinear(8, out_size, ctx, bias=False, gather_output=True)
    with torch.no_grad():
        s, e = shard_range(out_size, rank, tp)
        layer.weight.copy_(W[s:e])
    return (layer(X) - ref).abs().max().item()


@pytest.mark.distributed
def test_tp2_column_and_row_gloo():
    if os.environ.get("SKIP_DISTRIBUTED_TESTS"):
        pytest.skip("distributed tests disabled")
    from tests.distributed.harness import run_tp2

    results = run_tp2(_tp2_column)
    assert max(results.values()) < 1e-5, results

    results = run_tp2(_tp2_row)
    assert max(results.values()) < 1e-5, results

    results = run_tp2(_tp2_row_bias_not_doubled)
    assert max(results.values()) == 1.0, results

    results = run_tp2(_tp2_column_uneven)
    assert max(results.values()) < 1e-5, results


def test_column_parallel_tp1():
    layer = ColumnParallelLinear(16, 32, _make_ctx(), bias=True, gather_output=True)
    W = torch.randn(32, 16)
    b = torch.randn(32)
    X = torch.randn(4, 16)
    with torch.no_grad():
        layer.weight.copy_(W)
        layer.bias.copy_(b)
    torch.testing.assert_close(layer(X), F.linear(X, W, b), atol=1e-5, rtol=1e-5)


def test_row_parallel_tp1():
    layer = RowParallelLinear(16, 20, _make_ctx(), bias=True, input_is_parallel=False)
    W = torch.randn(20, 16)
    b = torch.randn(20)
    X = torch.randn(4, 16)
    with torch.no_grad():
        layer.weight.copy_(W)
        layer.bias.copy_(b)
    torch.testing.assert_close(layer(X), F.linear(X, W, b), atol=1e-5, rtol=1e-5)


def test_uneven_shard_range():
    assert shard_range(10, 0, 3) == (0, 4)
    assert shard_range(10, 1, 3) == (4, 7)
    assert shard_range(10, 2, 3) == (7, 10)


def test_divisibility_errors():
    ctx = ParallelContext(0, 0, 2, 0, 2, torch.device("cpu"), None)
    with pytest.raises(ValueError, match="divisible"):
        ColumnParallelLinear(8, 15, ctx)
    with pytest.raises(ValueError, match="divisible"):
        RowParallelLinear(15, 8, ctx)

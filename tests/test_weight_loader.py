"""Weight loader shard-content tests (TP=2 via Gloo, plus TP=1)."""

import pytest
import torch

from minitp.weight_loader import _shard, _shard_heads


def test_shard_dim0_tp1():
    W = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    assert torch.equal(_shard(W, 0, 0, 1), W)


def _tp2_shards(rank, tp):
    W = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    out = _shard(W, 0, rank, tp)  # column parallel: rows split
    out1 = _shard(W, 1, rank, tp)  # row parallel: cols split
    ok0 = torch.equal(out, W[rank * 3 : (rank + 1) * 3])
    ok1 = torch.equal(out1, W[:, rank * 4 : (rank + 1) * 4])
    return ok0 and ok1


def _tp2_kv_heads(rank, tp):
    # 2 kv heads x head_dim 4: rank r must receive exactly head r
    t = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    got = _shard_heads(t, rank, tp, num_kv_heads=2, head_dim=4)
    return torch.equal(got, t[rank * 4 : (rank + 1) * 4])


def _tp2_kv_replication(rank, tp):
    # 1 kv head, tp=2: both ranks replicate head 0
    t = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    got = _shard_heads(t, rank, tp, num_kv_heads=1, head_dim=4)
    return torch.equal(got, t)


@pytest.mark.distributed
def test_tp2_shard_contents_gloo():
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_shards)
    assert all(r.values()), r
    r = run_tp2(_tp2_kv_heads)
    assert all(r.values()), r
    r = run_tp2(_tp2_kv_replication)
    assert all(r.values()), r

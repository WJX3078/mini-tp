"""VocabParallelEmbedding / VocabParallelLMHead / distributed argmax correctness."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from minitp.distributed.context import ParallelContext
from minitp.parallel.embedding import VocabParallelEmbedding, VocabParallelLMHead
from minitp.parallel.linear import shard_range

torch.manual_seed(0)


def _pg_ctx(rank: int, tp: int) -> ParallelContext:
    return ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)


def _make_ctx() -> ParallelContext:
    return ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None)


def _tp2_embedding(rank, tp):
    torch.manual_seed(5)
    vocab, hidden = 10, 8  # uneven: 10 % 2 == 0 here; uneven covered below
    table = torch.randn(vocab, hidden)
    ids = torch.tensor([[0, 4, 9], [3, 5, 1]])
    ref = F.embedding(ids, table)
    ctx = _pg_ctx(rank, tp)
    emb = VocabParallelEmbedding(vocab, hidden, ctx)
    s, e = shard_range(vocab, rank, tp)
    with torch.no_grad():
        emb.weight.copy_(table[s:e])
    return (emb(ids) - ref).abs().max().item()


def _tp2_embedding_uneven(rank, tp):
    torch.manual_seed(6)
    vocab, hidden = 9, 8  # 9 % 2 != 0 -> rank0 has 5 rows, rank1 has 4
    table = torch.randn(vocab, hidden)
    ids = torch.tensor([[0, 8, 4], [7, 2, 6]])
    ref = F.embedding(ids, table)
    ctx = _pg_ctx(rank, tp)
    emb = VocabParallelEmbedding(vocab, hidden, ctx)
    s, e = shard_range(vocab, rank, tp)
    assert (e - s) == (5 if rank == 0 else 4)
    with torch.no_grad():
        emb.weight.copy_(table[s:e])
    return (emb(ids) - ref).abs().max().item()


def _tp2_lm_head(rank, tp):
    torch.manual_seed(8)
    hidden, vocab = 8, 10
    W = torch.randn(vocab, hidden)
    x = torch.randn(2, 4, hidden)
    ref = F.linear(x, W)
    ctx = _pg_ctx(rank, tp)
    head = VocabParallelLMHead(hidden, vocab, ctx, gather_logits=True)
    s, e = shard_range(vocab, rank, tp)
    with torch.no_grad():
        head.weight.copy_(W[s:e])
    gathered = head(x)
    err = (gathered - ref).abs().max().item()
    head.gather_logits = False
    err_local = (head(x) - ref[..., s:e]).abs().max().item()
    return max(err, err_local)


def _tp2_argmax(rank, tp):
    """Distributed greedy argmax vs torch.argmax: global max on each rank,
    negatives, exact ties across shards, and vocab-padding-free uneven split."""
    ctx = _pg_ctx(rank, tp)
    hidden, vocab = 8, 9
    head = VocabParallelLMHead(hidden, vocab, ctx)
    s, e = shard_range(vocab, rank, tp)
    errs = []
    cases = []
    torch.manual_seed(11)
    W = torch.randn(vocab, hidden)
    x = torch.randn(3, hidden)
    cases.append(F.linear(x, W))
    # global max on rank 1 (token 8 lives in rank1's shard)
    logits = F.linear(x, W)
    logits[:, 8] += 100.0
    cases.append(logits)
    # global max on rank 0
    logits2 = F.linear(x, W)
    logits2[:, 2] += 100.0
    cases.append(logits2)
    # exact tie across shards: tokens 3 (rank0) and 6 (rank1) equal max
    logits3 = F.linear(x, W)
    lo = logits3.max(dim=-1).values.max()
    logits3[:, 3] = lo + 50
    logits3[:, 6] = lo + 50
    cases.append(logits3)
    # all negative logits
    cases.append(-F.linear(x, W).abs() - 1.0)

    for case in cases:
        got = head.distributed_argmax(case[..., s:e])
        want = case.argmax(dim=-1, keepdim=True)
        errs.append(int((got != want).sum().item()))
    return max(errs)


@pytest.mark.distributed
def test_tp2_vocab_layers_gloo():
    if os.environ.get("SKIP_DISTRIBUTED_TESTS"):
        pytest.skip("distributed tests disabled")
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_embedding)
    assert max(r.values()) < 1e-5, r
    r = run_tp2(_tp2_embedding_uneven)
    assert max(r.values()) < 1e-5, r
    r = run_tp2(_tp2_lm_head)
    assert max(r.values()) < 1e-5, r
    r = run_tp2(_tp2_argmax)
    assert max(r.values()) == 0, r


def test_embedding_tp1():
    torch.manual_seed(1)
    emb = VocabParallelEmbedding(10, 8, _make_ctx())
    with torch.no_grad():
        emb.weight.copy_(torch.randn(10, 8))
    ids = torch.randint(0, 10, (2, 3))
    torch.testing.assert_close(emb(ids), F.embedding(ids, emb.weight))


def test_argmax_tie_smaller_id_tp1():
    head = VocabParallelLMHead(4, 8, _make_ctx())
    logits = torch.zeros(1, 8)
    logits[0, 5] = 2.0
    logits[0, 2] = 2.0
    assert head.distributed_argmax(logits).item() == 2


def test_lm_head_tp1():
    head = VocabParallelLMHead(8, 10, _make_ctx(), gather_logits=True)
    W = torch.randn(10, 8)
    x = torch.randn(2, 8)
    with torch.no_grad():
        head.weight.copy_(W)
    torch.testing.assert_close(head(x), F.linear(x, W), atol=1e-5, rtol=1e-5)

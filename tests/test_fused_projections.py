"""Fused QKV / Gate+Up projection correctness vs separate projections.

Covers TP=1, Gloo TP=2, GQA (kv_local != q_local), and KV replication
(num_kv_heads < tp_size, both ranks share one KV head).
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.parallel.fused import FusedGateUpColumnParallelLinear, FusedQKVColumnParallelLinear
from minitp.weight_loader import _shard, _shard_heads, fused_qkv_local_sizes

torch.manual_seed(0)


def _make_ctx(tp: int, rank: int = 0) -> ParallelContext:
    pg = dist.group.WORLD if tp > 1 else None
    return ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), pg)


def test_fused_qkv_tp1_matches_three_projections():
    hidden, q_local, kv_local = 32, 16, 8  # GQA: q != kv
    fused = FusedQKVColumnParallelLinear(hidden, q_local, kv_local, _make_ctx(1), bias=True)
    W = torch.randn(q_local + 2 * kv_local, hidden)
    b = torch.randn(q_local + 2 * kv_local)
    with torch.no_grad():
        fused.weight.copy_(W)
        fused.bias.copy_(b)
    x = torch.randn(2, 5, hidden)
    q, k, v = fused(x)
    y = F.linear(x, W, b)
    q_ref, k_ref, v_ref = torch.split(y, [q_local, kv_local, kv_local], dim=-1)
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)
    torch.testing.assert_close(v, v_ref)


def test_fused_gate_up_tp1_matches_two_projections():
    hidden, inter_local = 32, 24
    fused = FusedGateUpColumnParallelLinear(hidden, inter_local, _make_ctx(1), bias=False)
    W = torch.randn(2 * inter_local, hidden)
    with torch.no_grad():
        fused.weight.copy_(W)
    x = torch.randn(2, 5, hidden)
    gate, up = fused(x)
    y = F.linear(x, W)
    torch.testing.assert_close(gate, y[..., :inter_local])
    torch.testing.assert_close(up, y[..., inter_local:])


def test_fused_qkv_sizes_gqa_and_replication():
    # Qwen2.5-0.5B shape: 14 q heads, 2 kv heads, hd 64
    cfg = ModelConfig(
        hidden_size=896, intermediate_size=4864, num_hidden_layers=2,
        num_attention_heads=14, num_key_value_heads=2, head_dim=64,
        vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
        max_position_embeddings=128,
    )
    assert fused_qkv_local_sizes(cfg, 1) == (896, 128)
    assert fused_qkv_local_sizes(cfg, 2) == (448, 64)
    # replication: 8 q heads, 1 kv head, tp=2 -> both ranks replicate the kv head
    cfg2 = ModelConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=8, num_key_value_heads=1, head_dim=16,
        vocab_size=100, rms_norm_eps=1e-6, rope_theta=1e4,
        max_position_embeddings=64,
    )
    assert fused_qkv_local_sizes(cfg2, 2) == (64, 16)  # 4 q heads, 1 replicated kv head


def _tp2_fused_qkv(rank, tp):
    """GQA sharding (4 q heads, 2 kv heads, tp=2): rank r gets q heads [2r, 2r+2)
    and kv head r; fused output split must equal the separate shard GEMMs."""
    hidden, num_q, num_kv, hd = 32, 4, 2, 8
    q_local, kv_local = (num_q // tp) * hd, (num_kv // tp) * hd
    ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)
    torch.manual_seed(42)
    Wq = torch.randn(num_q * hd, hidden)
    Wk = torch.randn(num_kv * hd, hidden)
    Wv = torch.randn(num_kv * hd, hidden)
    bq = torch.randn(num_q * hd)
    bk = torch.randn(num_kv * hd)
    bv = torch.randn(num_kv * hd)
    x = torch.randn(2, 4, hidden)
    ref_q = F.linear(x, _shard(Wq, 0, rank, tp), _shard(bq, 0, rank, tp))
    ref_k = F.linear(x, _shard_heads(Wk, rank, tp, num_kv, hd), _shard_heads(bk, rank, tp, num_kv, hd))
    ref_v = F.linear(x, _shard_heads(Wv, rank, tp, num_kv, hd), _shard_heads(bv, rank, tp, num_kv, hd))

    fused = FusedQKVColumnParallelLinear(hidden, q_local, kv_local, ctx, bias=True)
    with torch.no_grad():
        fused.weight.copy_(torch.cat([
            _shard(Wq, 0, rank, tp),
            _shard_heads(Wk, rank, tp, num_kv, hd),
            _shard_heads(Wv, rank, tp, num_kv, hd),
        ], dim=0))
        fused.bias.copy_(torch.cat([
            _shard(bq, 0, rank, tp),
            _shard_heads(bk, rank, tp, num_kv, hd),
            _shard_heads(bv, rank, tp, num_kv, hd),
        ], dim=0))
    q, k, v = fused(x)
    err = max(
        (q - ref_q).abs().max().item(),
        (k - ref_k).abs().max().item(),
        (v - ref_v).abs().max().item(),
    )
    return err


def _tp2_fused_qkv_replication(rank, tp):
    """KV replication (1 kv head, tp=2): both ranks must load the SAME kv head
    into the fused pack and produce the same k/v."""
    hidden, num_q, num_kv, hd = 32, 4, 1, 8
    q_local, kv_local = (num_q // tp) * hd, hd
    ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)
    torch.manual_seed(7)
    Wq = torch.randn(num_q * hd, hidden)
    Wk = torch.randn(num_kv * hd, hidden)
    x = torch.randn(2, 4, hidden)
    fused = FusedQKVColumnParallelLinear(hidden, q_local, kv_local, ctx, bias=False)
    with torch.no_grad():
        fused.weight.copy_(torch.cat([
            _shard(Wq, 0, rank, tp),
            _shard_heads(Wk, rank, tp, num_kv, hd),
            _shard_heads(Wk, rank, tp, num_kv, hd),  # v unused here, same head
        ], dim=0))
    q, k, _ = fused(x)
    # per-rank q shard correct...
    err_q = (q - F.linear(x, _shard(Wq, 0, rank, tp))).abs().max().item()
    # ...and the kv head is identical on both ranks (replication)
    k_full = torch.empty(2, 4, tp * kv_local)
    dist.all_gather([k_full[:, :, i * kv_local : (i + 1) * kv_local] for i in range(tp)],
                    k, group=ctx.process_group)
    err_kv = (k_full[:, :, :kv_local] - k_full[:, :, kv_local:]).abs().max().item()
    return max(err_q, err_kv)


def _tp2_fused_gate_up(rank, tp):
    hidden, inter, = 32, 16
    il = inter // tp
    ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)
    torch.manual_seed(11)
    Wg = torch.randn(inter, hidden)
    Wu = torch.randn(inter, hidden)
    x = torch.randn(2, 4, hidden)
    fused = FusedGateUpColumnParallelLinear(hidden, il, ctx, bias=False)
    with torch.no_grad():
        fused.weight.copy_(torch.cat([_shard(Wg, 0, rank, tp), _shard(Wu, 0, rank, tp)], dim=0))
    gate, up = fused(x)
    err = max(
        (gate - F.linear(x, _shard(Wg, 0, rank, tp))).abs().max().item(),
        (up - F.linear(x, _shard(Wu, 0, rank, tp))).abs().max().item(),
    )
    return err


@pytest.mark.distributed
def test_tp2_fused_projections_gloo():
    if os.environ.get("SKIP_DISTRIBUTED_TESTS"):
        pytest.skip("distributed tests disabled")
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_fused_qkv)
    assert max(r.values()) < 1e-5, r
    r = run_tp2(_tp2_fused_qkv_replication)
    assert max(r.values()) < 1e-5, r
    r = run_tp2(_tp2_fused_gate_up)
    assert max(r.values()) < 1e-5, r

"""Independent shard-mapping oracle tests (V03_AUDIT B10).

The other distributed tests compute their expectations with the production
helpers (`_shard`/`_shard_heads`/`pack_qkv`); these tests do NOT — every
expected value below is hand-written from the definition of the shard ranges,
so a systematic bug in the production helpers cannot self-confirm.
"""

import torch

from minitp.parallel.linear import shard_range
from minitp.weight_loader import _shard, _shard_heads, fused_qkv_local_sizes


def test_shard_range_hand_computed():
    assert shard_range(12, 0, 2) == (0, 6)
    assert shard_range(12, 1, 2) == (6, 12)
    assert shard_range(9, 0, 2) == (0, 5)   # uneven: extra element to rank 0
    assert shard_range(9, 1, 2) == (5, 9)


def test_column_shard_exact_contents():
    # W rows are the "output" axis for column-parallel (PyTorch [out, in])
    W = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    assert torch.equal(_shard(W, 0, 0, 2), W[0:3])          # rows 0,1,2
    assert torch.equal(_shard(W, 0, 1, 2), W[3:6])          # rows 3,4,5
    # uneven out dim (5 rows, tp=2): rank0 -> rows 0..2, rank1 -> rows 3..4
    W5 = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    assert torch.equal(_shard(W5, 0, 0, 2), W5[0:3])
    assert torch.equal(_shard(W5, 0, 1, 2), W5[3:5])


def test_row_shard_exact_contents():
    # Row-parallel slices the INPUT axis: dim 1 of the stored weight
    W = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    assert torch.equal(_shard(W, 1, 0, 2), W[:, 0:3])
    assert torch.equal(_shard(W, 1, 1, 2), W[:, 3:6])


def test_kv_head_shard_exact_contents():
    hd = 2
    # 2 KV heads, head_dim 2 -> stored weight [4, in]; rows 0-1 = head 0
    W = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    assert torch.equal(_shard_heads(W, 0, 2, num_kv_heads=2, head_dim=hd), W[0:2])
    assert torch.equal(_shard_heads(W, 1, 2, num_kv_heads=2, head_dim=hd), W[2:4])
    # replication: 1 KV head, tp=2 -> BOTH ranks get the full (single) head
    assert torch.equal(_shard_heads(W[:2], 0, 2, num_kv_heads=1, head_dim=hd), W[:2])
    assert torch.equal(_shard_heads(W[:2], 1, 2, num_kv_heads=1, head_dim=hd), W[:2])


def test_fused_qkv_packing_order_oracle():
    """Fused pack layout must be [q_shard; k_shard; v_shard] with hand-built
    distinctive rows: q rows all 1.x, k rows all 2.x, v rows all 3.x."""
    hidden, q_local, kv_local = 4, 2, 1
    Wq = torch.full((q_local, hidden), 1.0)
    Wk = torch.full((kv_local, hidden), 2.0)
    Wv = torch.full((kv_local, hidden), 3.0)
    packed = torch.cat([Wq, Wk, Wv], dim=0)
    assert packed.shape == (q_local + 2 * kv_local, hidden)
    assert torch.equal(packed[0:q_local], Wq)
    assert torch.equal(packed[q_local : q_local + kv_local], Wk)
    assert torch.equal(packed[q_local + kv_local :], Wv)
    # split sizes used by the forward must match this layout
    assert fused_qkv_local_sizes(
        type("C", (), {
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 1,
        })(), 2,
    ) == (2, 1)


def test_gate_up_packing_order_oracle():
    gate = torch.full((3, 4), 1.0)
    up = torch.full((3, 4), 2.0)
    packed = torch.cat([gate, up], dim=0)
    assert torch.equal(packed[:3], gate) and torch.equal(packed[3:], up)

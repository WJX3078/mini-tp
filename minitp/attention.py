"""Qwen2 attention with GQA tensor parallelism (per-rank local heads).

v0.2: q/k/v shards are packed into one fused parameter at load time — a
single GEMM per forward instead of three (FusedQKVColumnParallelLinear).
The split afterwards is a zero-copy view.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.kv_cache import KVCache
from minitp.parallel.fused import FusedQKVColumnParallelLinear
from minitp.parallel.linear import RowParallelLinear
from minitp.rope import RotaryEmbedding, apply_rope


class TPQwen2Attention(nn.Module):
    """QKV fused ColumnParallel (head sharded, no comm); attention runs on
    local heads only; o_proj is RowParallel followed by a single AllReduce."""

    def __init__(self, cfg: ModelConfig, ctx: ParallelContext) -> None:
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        tp = ctx.tp_size
        self.q_heads_local = cfg.num_attention_heads // tp
        self.kv_replication = cfg.kv_replication(tp)
        self.kv_heads_local = max(1, cfg.num_key_value_heads // tp)
        self.head_dim = cfg.head_dim
        self.q_local = self.q_heads_local * self.head_dim
        self.kv_local = self.kv_heads_local * self.head_dim

        q_rows, kv_rows = self.q_local, self.kv_local
        self.qkv_proj = FusedQKVColumnParallelLinear(
            cfg.hidden_size, q_rows, kv_rows, ctx, bias=True
        )
        self.o_proj = RowParallelLinear(cfg.hidden_size, cfg.hidden_size, ctx, bias=False)
        # ablation toggle: False reproduces the v0.1 three-GEMM pattern on the
        # SAME packed storage (weight row views), so only the GEMM count differs
        self.use_fused_qkv = True

    def forward(
        self,
        x: torch.Tensor,  # [B, T, hidden] replicated
        positions: torch.Tensor,  # [T]
        kv_cache: KVCache | None,
        layer_idx: int,
        rotary: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv_proj(x) if self.use_fused_qkv else self.qkv_proj.forward_unfused(x)
        q = qkv[0].view(b, t, self.q_heads_local, self.head_dim).transpose(1, 2)
        k = qkv[1].view(b, t, self.kv_heads_local, self.head_dim).transpose(1, 2)
        v = qkv[2].view(b, t, self.kv_heads_local, self.head_dim).transpose(1, 2)

        if rotary is not None:
            q, k = rotary.apply(q, k, positions)
        else:
            q, k = apply_rope(q, k, positions, self.cfg.rope_theta)

        if kv_cache is not None:
            cache_len = kv_cache.seq_len
            # NOTE: does not advance the cursor here — the model advances it
            # once per forward pass, otherwise layer 1 would write at the
            # position layer 0 just advanced past.
            k, v = kv_cache.update(layer_idx, k, v)
        else:
            cache_len = 0

        if self.q_heads_local != self.kv_heads_local:
            group = self.q_heads_local // self.kv_heads_local
            if self.q_heads_local % self.kv_heads_local != 0:
                raise ValueError(
                    f"local q heads {self.q_heads_local} not divisible by local kv heads "
                    f"{self.kv_heads_local}"
                )
            if self.kv_heads_local == 1:
                # single local KV head broadcast to all local Q heads: zero-copy
                k = k.expand(b, self.q_heads_local, k.shape[2], self.head_dim)
                v = v.expand(b, self.q_heads_local, v.shape[2], self.head_dim)
            else:
                k = k.repeat_interleave(group, dim=1)
                v = v.repeat_interleave(group, dim=1)

        if t == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        elif cache_len == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # q position i may attend to cache positions [0, cache_len) and new
            # positions up to cache_len + i (top-left causal would misalign).
            total = k.shape[2]
            mask = torch.ones(t, total, dtype=torch.bool, device=q.device).tril(
                diagonal=cache_len
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(b, t, self.q_local)
        return self.o_proj(out)

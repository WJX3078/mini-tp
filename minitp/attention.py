"""Qwen2 attention with GQA tensor parallelism (per-rank local heads)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.kv_cache import KVCache
from minitp.parallel.linear import ColumnParallelLinear, RowParallelLinear
from minitp.rope import apply_rope


class TPQwen2Attention(nn.Module):
    """QKV are ColumnParallel (output/head sharded, no comm); attention runs on
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

        self.q_proj = ColumnParallelLinear(cfg.hidden_size, cfg.hidden_size, ctx, bias=True)
        self.k_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, ctx, bias=True
        )
        self.v_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, ctx, bias=True
        )
        self.o_proj = RowParallelLinear(cfg.hidden_size, cfg.hidden_size, ctx, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, hidden] replicated
        positions: torch.Tensor,  # [T]
        kv_cache: KVCache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.q_heads_local, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads_local, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads_local, self.head_dim).transpose(1, 2)

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

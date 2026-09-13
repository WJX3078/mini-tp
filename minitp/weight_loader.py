"""TP weight loader: slice HF Qwen2 checkpoints onto per-rank shards.

PyTorch Linear weight layout is [out_features, in_features]:
- ColumnParallel shards along dim 0 (out).
- RowParallel shards along dim 1 (in).
- VocabParallelEmbedding/LMHead shard along dim 0 (vocab).
KV heads are sliced as contiguous head blocks so the contiguous Q-head block
of each rank shares its KV head(s) (see DESIGN §6).

v0.2: the per-rank q/k/v (and gate/up) shards are *packed* into single fused
parameters at load time (FusedQKV / FusedGateUp, docs/PERFORMANCE_AUDIT.md §7).
The pack functions below are the single source of truth for shard mapping —
tests reuse them, so loader and tests can never drift apart.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.layer import TPQwen2ForCausalLM


def _shard(t: torch.Tensor, dim: int, rank: int, tp: int) -> torch.Tensor:
    size = t.shape[dim]
    base, rem = divmod(size, tp)
    start = rank * base + min(rank, rem)
    n = base + (1 if rank < rem else 0)
    return t.narrow(dim, start, n).clone()


def _shard_heads(t: torch.Tensor, rank: int, tp: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Slice [num_kv_heads*head_dim, ...] into contiguous KV-head blocks per rank."""
    if num_kv_heads % tp == 0:
        heads_per_rank = num_kv_heads // tp
        start = rank * heads_per_rank
        return t.narrow(0, start * head_dim, heads_per_rank * head_dim).clone()
    # replication: rank r owns KV head r // (tp // num_kv_heads)
    group = tp // num_kv_heads
    return t.narrow(0, (rank // group) * head_dim, head_dim).clone()


def fused_qkv_local_sizes(cfg: ModelConfig, tp: int) -> tuple[int, int]:
    """(q_rows, kv_rows) per rank for the fused QKV parameter."""
    q_heads = cfg.num_attention_heads // tp
    kv_heads = max(1, cfg.num_key_value_heads // tp)
    return q_heads * cfg.head_dim, kv_heads * cfg.head_dim


def pack_qkv(
    state: dict[str, torch.Tensor],
    prefix: str,  # e.g. "model.layers.0.self_attn."
    rank: int,
    tp: int,
    cfg: ModelConfig,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build this rank's fused QKV [weight, bias] from HF q/k/v tensors."""
    q_w = _shard(state[f"{prefix}q_proj.weight"], 0, rank, tp)
    k_w = _shard_heads(state[f"{prefix}k_proj.weight"], rank, tp, cfg.num_key_value_heads, cfg.head_dim)
    v_w = _shard_heads(state[f"{prefix}v_proj.weight"], rank, tp, cfg.num_key_value_heads, cfg.head_dim)
    weight = torch.cat([q_w, k_w, v_w], dim=0)
    bias = None
    if f"{prefix}q_proj.bias" in state:
        q_b = _shard(state[f"{prefix}q_proj.bias"], 0, rank, tp)
        k_b = _shard_heads(state[f"{prefix}k_proj.bias"], rank, tp, cfg.num_key_value_heads, cfg.head_dim)
        v_b = _shard_heads(state[f"{prefix}v_proj.bias"], rank, tp, cfg.num_key_value_heads, cfg.head_dim)
        bias = torch.cat([q_b, k_b, v_b], dim=0)
    if dtype is not None:
        weight, bias = weight.to(dtype), bias.to(dtype) if bias is not None else None
    return weight, bias


def pack_gate_up(
    state: dict[str, torch.Tensor],
    prefix: str,  # e.g. "model.layers.0.mlp."
    rank: int,
    tp: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build this rank's fused gate|up weight [2*inter_local, hidden]."""
    gate = _shard(state[f"{prefix}gate_proj.weight"], 0, rank, tp)
    up = _shard(state[f"{prefix}up_proj.weight"], 0, rank, tp)
    weight = torch.cat([gate, up], dim=0)
    return weight.to(dtype) if dtype is not None else weight


def load_checkpoint_state(model_dir: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        state: dict[str, torch.Tensor] = {}
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        for shard_file in sorted(set(index.values())):
            state.update(load_file(os.path.join(model_dir, shard_file)))
        return state
    return load_file(os.path.join(model_dir, "model.safetensors"))


def load_qwen2_tp(
    model_dir: str,
    cfg: ModelConfig,
    ctx: ParallelContext,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> TPQwen2ForCausalLM:
    """Build a TP model and load only this rank's shards. NOTE (P0): this
    materializes the full CPU state_dict per rank; GPU only ever receives the
    local shard. Selective safetensors loading is deferred (audit §10)."""
    state = load_checkpoint_state(model_dir)
    tp = ctx.tp_size
    rank = ctx.tp_rank
    # construct directly in target dtype: avoids a full fp32 intermediate
    model = TPQwen2ForCausalLM(cfg, ctx).to(dtype=dtype)
    sd = model.state_dict()

    def put(name: str, tensor: torch.Tensor) -> None:
        sd[name].copy_(tensor.to(dtype=dtype))

    put("model.embed_tokens.weight", _shard(state["model.embed_tokens.weight"], 0, rank, tp))
    for i in range(cfg.num_hidden_layers):
        attn = f"model.layers.{i}.self_attn."
        mlp = f"model.layers.{i}.mlp."
        put(f"model.layers.{i}.input_layernorm.weight", state[f"model.layers.{i}.input_layernorm.weight"])
        put(
            f"model.layers.{i}.post_attention_layernorm.weight",
            state[f"model.layers.{i}.post_attention_layernorm.weight"],
        )
        qkv_w, qkv_b = pack_qkv(state, attn, rank, tp, cfg)
        put(f"{attn}qkv_proj.weight", qkv_w)
        if qkv_b is not None:
            put(f"{attn}qkv_proj.bias", qkv_b)
        put(f"{attn}o_proj.weight", _shard(state[f"{attn}o_proj.weight"], 1, rank, tp))
        put(f"{mlp}gate_up_proj.weight", pack_gate_up(state, mlp, rank, tp))
        put(f"{mlp}down_proj.weight", _shard(state[f"{mlp}down_proj.weight"], 1, rank, tp))
    put("model.norm.weight", state["model.norm.weight"])
    if not cfg.tie_word_embeddings:
        put("lm_head.weight", _shard(state["lm_head.weight"], 0, rank, tp))
    # tied: lm_head shares embed_tokens.weight, already loaded above

    model.load_state_dict(sd)
    del state
    return model.to(device if device is not None else ctx.device)

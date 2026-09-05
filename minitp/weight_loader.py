"""TP weight loader: slice HF Qwen2 checkpoints onto per-rank shards.

PyTorch Linear weight layout is [out_features, in_features]:
- ColumnParallel shards along dim 0 (out).
- RowParallel shards along dim 1 (in).
- VocabParallelEmbedding/LMHead shard along dim 0 (vocab).
KV heads are sliced as contiguous head blocks so the contiguous Q-head block
of each rank shares its KV head(s) (see DESIGN §6).
"""

from __future__ import annotations

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


def load_qwen2_tp(
    model_dir: str,
    cfg: ModelConfig,
    ctx: ParallelContext,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> TPQwen2ForCausalLM:
    """Build a TP model and load only this rank's shards. NOTE (P0): this
    materializes the full CPU state_dict per rank; GPU only ever receives the
    local shard. Selective safetensors loading is P1."""
    import json
    import os

    device = device if device is not None else ctx.device
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    state: dict[str, torch.Tensor] = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        for shard_file in sorted(set(index.values())):
            state.update(load_file(os.path.join(model_dir, shard_file)))
    else:
        state = load_file(os.path.join(model_dir, "model.safetensors"))

    tp = ctx.tp_size
    rank = ctx.tp_rank
    model = TPQwen2ForCausalLM(cfg, ctx).to(dtype=dtype)
    sd = model.state_dict()
    head_dim = cfg.head_dim
    kv_heads = cfg.num_key_value_heads

    def put(name: str, tensor: torch.Tensor) -> None:
        sd[name].copy_(tensor.to(dtype=dtype))

    put("model.embed_tokens.weight", _shard(state["model.embed_tokens.weight"], 0, rank, tp))
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        put(f"{p}input_layernorm.weight", state[f"{p}input_layernorm.weight"])
        put(f"{p}post_attention_layernorm.weight", state[f"{p}post_attention_layernorm.weight"])
        put(f"{p}self_attn.q_proj.weight", _shard(state[f"{p}self_attn.q_proj.weight"], 0, rank, tp))
        put(f"{p}self_attn.q_proj.bias", _shard(state[f"{p}self_attn.q_proj.bias"], 0, rank, tp))
        put(
            f"{p}self_attn.k_proj.weight",
            _shard_heads(state[f"{p}self_attn.k_proj.weight"], rank, tp, kv_heads, head_dim),
        )
        put(
            f"{p}self_attn.k_proj.bias",
            _shard_heads(state[f"{p}self_attn.k_proj.bias"], rank, tp, kv_heads, head_dim),
        )
        put(
            f"{p}self_attn.v_proj.weight",
            _shard_heads(state[f"{p}self_attn.v_proj.weight"], rank, tp, kv_heads, head_dim),
        )
        put(
            f"{p}self_attn.v_proj.bias",
            _shard_heads(state[f"{p}self_attn.v_proj.bias"], rank, tp, kv_heads, head_dim),
        )
        put(f"{p}self_attn.o_proj.weight", _shard(state[f"{p}self_attn.o_proj.weight"], 1, rank, tp))
        put(f"{p}mlp.gate_proj.weight", _shard(state[f"{p}mlp.gate_proj.weight"], 0, rank, tp))
        put(f"{p}mlp.up_proj.weight", _shard(state[f"{p}mlp.up_proj.weight"], 0, rank, tp))
        put(f"{p}mlp.down_proj.weight", _shard(state[f"{p}mlp.down_proj.weight"], 1, rank, tp))
    put("model.norm.weight", state["model.norm.weight"])
    if not cfg.tie_word_embeddings:
        put("lm_head.weight", _shard(state["lm_head.weight"], 0, rank, tp))
    # tied: lm_head shares embed_tokens.weight, already loaded above

    model.load_state_dict(sd)
    return model.to(device)

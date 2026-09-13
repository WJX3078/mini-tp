"""TP weight loader: slice HF Qwen2 checkpoints onto per-rank shards.

PyTorch Linear weight layout is [out_features, in_features]:
- ColumnParallel shards along dim 0 (out).
- RowParallel shards along dim 1 (in).
- VocabParallelEmbedding/LMHead shard along dim 0 (vocab).
KV heads are sliced as contiguous head blocks so the contiguous Q-head block
of each rank shares its KV head(s) (see DESIGN §6).

v0.2: per-rank q/k/v (and gate/up) shards are *packed* into single fused
parameters at load time (FusedQKV / FusedGateUp).

v0.3: two loaders with byte-identical results (regression-tested):
- ``loader="selective"`` (default): ``SelectiveTensorReader`` over
  ``safetensors.safe_open`` — each rank reads ONLY its own slices, never
  materializing the full state dict (host memory ~ local shards).
- ``loader="legacy"``: full CPU state dict, then slice (kept for
  correctness comparison; ~full-checkpoint host memory).

The shard *ranges* come from one set of helpers shared with the independent
oracle tests; mapping bugs cannot hide behind two divergent implementations.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file, safe_open

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.layer import TPQwen2ForCausalLM

# --------------------------------------------------------------------------
# shard range helpers (single source of truth)
# --------------------------------------------------------------------------

def shard_len(size: int, rank: int, tp: int) -> tuple[int, int]:
    """(start, length) of rank ``rank``'s contiguous chunk of ``size``."""
    base, rem = divmod(size, tp)
    start = rank * base + min(rank, rem)
    return start, base + (1 if rank < rem else 0)


def _shard(t: torch.Tensor, dim: int, rank: int, tp: int) -> torch.Tensor:
    start, n = shard_len(t.shape[dim], rank, tp)
    return t.narrow(dim, start, n).clone()


def kv_head_rows(num_kv_heads: int, head_dim: int, rank: int, tp: int) -> tuple[int, int]:
    """(row_start, row_count) of this rank's KV-head block in [kv*hd, ...]."""
    if num_kv_heads % tp == 0:
        heads = num_kv_heads // tp
        return rank * heads * head_dim, heads * head_dim
    group = tp // num_kv_heads  # replication: rank r owns head r // group
    return (rank // group) * head_dim, head_dim


def _shard_heads(t: torch.Tensor, rank: int, tp: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    start, n = kv_head_rows(num_kv_heads, head_dim, rank, tp)
    return t.narrow(0, start, n).clone()


def fused_qkv_local_sizes(cfg: ModelConfig, tp: int) -> tuple[int, int]:
    """(q_rows, kv_rows) per rank for the fused QKV parameter."""
    q_heads = cfg.num_attention_heads // tp
    kv_heads = max(1, cfg.num_key_value_heads // tp)
    return q_heads * cfg.head_dim, kv_heads * cfg.head_dim


# --------------------------------------------------------------------------
# selective safetensors reader
# --------------------------------------------------------------------------

class SelectiveTensorReader:
    """Reads individual (slices of) tensors from safetensors checkpoint files
    without materializing the whole state dict. Handles both single-file
    checkpoints and ``model.safetensors.index.json`` sharded layouts."""

    def __init__(self, model_dir: str) -> None:
        self._handles: dict[str, safe_open] = {}
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map: dict[str, str] = json.load(f)["weight_map"]
            self._weight_map: dict[str, str] = {
                name: os.path.join(model_dir, fname) for name, fname in weight_map.items()
            }
            self._single_file: str | None = None
        else:
            self._weight_map = {}
            self._single_file = os.path.join(model_dir, "model.safetensors")

    def _file_for(self, name: str) -> str:
        if name in self._weight_map:
            return self._weight_map[name]
        if self._single_file is not None:
            return self._single_file
        raise KeyError(f"tensor {name!r} not in checkpoint index")

    def _handle(self, path: str) -> safe_open:
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt", device="cpu")
        return self._handles[path]

    def names(self) -> list[str]:
        return list(self._weight_map)

    def has(self, name: str) -> bool:
        if self._single_file is not None:
            return name in self._handle(self._single_file).keys()
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self._file_for(name)).get_tensor(name)

    def get_slice(self, name: str, dim: int, start: int, length: int) -> torch.Tensor:
        """Read only ``tensor[dim][start:start+length]`` (no full-tensor copy).

        1-D tensors (biases, norms — negligible bytes) are read whole and
        narrowed in torch: PySafeSlice 1-D indexing is unreliable in
        safetensors 0.8 (returns garbage memory). All 2-D reads use the
        zero-copy slice API.
        """
        sl = self._handle(self._file_for(name)).get_slice(name)
        if len(sl.get_shape()) == 1:
            assert dim == 0
            return self.get(name).narrow(0, start, length)
        index = [slice(None)] * len(sl.get_shape())
        index[dim] = slice(start, start + length)
        return sl[tuple(index)]

    def close(self) -> None:
        self._handles.clear()


# --------------------------------------------------------------------------
# destination mapping
# --------------------------------------------------------------------------

def _qkv_slices(cfg: ModelConfig, rank: int, tp: int, prefix: str):
    """[(hf_name, dim, start, length, dest_row)] for this rank's fused QKV."""
    q_rows, kv_rows = fused_qkv_local_sizes(cfg, tp)
    plan = []
    q_start, q_len = shard_len(cfg.num_attention_heads * cfg.head_dim, rank, tp)
    plan.append((f"{prefix}q_proj.weight", 0, q_start, q_len, 0))
    k_start, k_len = kv_head_rows(cfg.num_key_value_heads, cfg.head_dim, rank, tp)
    plan.append((f"{prefix}k_proj.weight", 0, k_start, k_len, q_rows))
    v_start, v_len = kv_head_rows(cfg.num_key_value_heads, cfg.head_dim, rank, tp)
    plan.append((f"{prefix}v_proj.weight", 0, v_start, v_len, q_rows + kv_rows))
    return plan, q_rows, kv_rows


def _fill_dest(dest: torch.Tensor, reader: SelectiveTensorReader, plan, dtype) -> None:
    for name, dim, start, length, dest_row in plan:
        dest[dest_row : dest_row + length].copy_(reader.get_slice(name, dim, start, length).to(dtype))


def load_qwen2_tp(
    model_dir: str,
    cfg: ModelConfig,
    ctx: ParallelContext,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    loader: str = "selective",
) -> TPQwen2ForCausalLM:
    """Build a TP model and load only this rank's shards.

    loader="selective" (default, v0.3): per-slice reads via safe_open; the
    full state dict is never materialized. loader="legacy": full CPU state
    dict per rank (v0.1/v0.2 behavior, kept for comparison). Both produce
    bit-identical local parameters (tests/test_weight_loader.py).
    """
    if loader not in ("selective", "legacy"):
        raise ValueError(f"unknown loader {loader!r}")
    tp = ctx.tp_size
    rank = ctx.tp_rank
    model = TPQwen2ForCausalLM(cfg, ctx).to(dtype=dtype)
    sd = model.state_dict()

    def put(name: str, tensor: torch.Tensor) -> None:
        sd[name].copy_(tensor.to(dtype=dtype))

    if loader == "legacy":
        state = _load_full_state(model_dir)
        put("model.embed_tokens.weight", _shard(state["model.embed_tokens.weight"], 0, rank, tp))
        for i in range(cfg.num_hidden_layers):
            attn = f"model.layers.{i}.self_attn."
            mlp = f"model.layers.{i}.mlp."
            put(f"model.layers.{i}.input_layernorm.weight", state[f"model.layers.{i}.input_layernorm.weight"])
            put(f"model.layers.{i}.post_attention_layernorm.weight",
                state[f"model.layers.{i}.post_attention_layernorm.weight"])
            qkv_w, qkv_b = _pack_qkv_legacy(state, attn, rank, tp, cfg)
            put(f"{attn}qkv_proj.weight", qkv_w)
            if qkv_b is not None:
                put(f"{attn}qkv_proj.bias", qkv_b)
            put(f"{attn}o_proj.weight", _shard(state[f"{attn}o_proj.weight"], 1, rank, tp))
            gate = _shard(state[f"{mlp}gate_proj.weight"], 0, rank, tp)
            up = _shard(state[f"{mlp}up_proj.weight"], 0, rank, tp)
            put(f"{mlp}gate_up_proj.weight", torch.cat([gate, up], dim=0))
            put(f"{mlp}down_proj.weight", _shard(state[f"{mlp}down_proj.weight"], 1, rank, tp))
        put("model.norm.weight", state["model.norm.weight"])
        if not cfg.tie_word_embeddings:
            put("lm_head.weight", _shard(state["lm_head.weight"], 0, rank, tp))
        del state
    else:
        reader = SelectiveTensorReader(model_dir)
        try:
            ev_start, ev_len = shard_len(cfg.vocab_size, rank, tp)
            sd["model.embed_tokens.weight"].copy_(
                reader.get_slice("model.embed_tokens.weight", 0, ev_start, ev_len).to(dtype)
            )
            for i in range(cfg.num_hidden_layers):
                base = f"model.layers.{i}."
                attn, mlp = f"{base}self_attn.", f"{base}mlp."
                put(f"{base}input_layernorm.weight", reader.get(f"{base}input_layernorm.weight"))
                put(f"{base}post_attention_layernorm.weight",
                    reader.get(f"{base}post_attention_layernorm.weight"))
                plan, q_rows, kv_rows = _qkv_slices(cfg, rank, tp, attn)
                _fill_dest(sd[f"{attn}qkv_proj.weight"], reader, plan, dtype)
                if reader.has(f"{attn}q_proj.bias"):
                    bias_plan = [(n.replace(".weight", ".bias"), d, s, l, r) for n, d, s, l, r in plan]
                    _fill_dest(sd[f"{attn}qkv_proj.bias"], reader, bias_plan, dtype)
                o_start, o_len = shard_len(cfg.hidden_size, rank, tp)
                sd[f"{attn}o_proj.weight"].copy_(
                    reader.get_slice(f"{attn}o_proj.weight", 1, o_start, o_len).to(dtype)
                )
                g_rows = cfg.intermediate_size // tp
                g_start, g_len = shard_len(cfg.intermediate_size, rank, tp)
                gu = sd[f"{mlp}gate_up_proj.weight"]
                gu[:g_rows].copy_(reader.get_slice(f"{mlp}gate_proj.weight", 0, g_start, g_len).to(dtype))
                gu[g_rows:].copy_(reader.get_slice(f"{mlp}up_proj.weight", 0, g_start, g_len).to(dtype))
                d_start, d_len = shard_len(cfg.intermediate_size, rank, tp)
                sd[f"{mlp}down_proj.weight"].copy_(
                    reader.get_slice(f"{mlp}down_proj.weight", 1, d_start, d_len).to(dtype)
                )
            put("model.norm.weight", reader.get("model.norm.weight"))
            if not cfg.tie_word_embeddings:
                h_start, h_len = shard_len(cfg.vocab_size, rank, tp)
                sd["lm_head.weight"].copy_(
                    reader.get_slice("lm_head.weight", 0, h_start, h_len).to(dtype)
                )
        finally:
            reader.close()

    model.load_state_dict(sd)
    return model.to(device if device is not None else ctx.device)


def _load_full_state(model_dir: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        state: dict[str, torch.Tensor] = {}
        with open(index_path) as f:
            index = json.load(f)["weight_map"]
        for shard_file in sorted(set(index.values())):
            state.update(load_file(os.path.join(model_dir, shard_file)))
        return state
    return load_file(os.path.join(model_dir, "model.safetensors"))


def _pack_qkv_legacy(state, prefix: str, rank: int, tp: int, cfg: ModelConfig):
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
    return weight, bias

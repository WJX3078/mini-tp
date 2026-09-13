"""Selective safetensors loader: synthetic checkpoint, bit-identical to legacy.

Builds a tiny safetensors checkpoint (single-file AND index-sharded variants)
with untied LM head + biases, then asserts legacy and selective loaders
produce identical local parameters for TP=1, TP=2 GQA sharding, and TP=2 KV
replication. Loading is rank-local and collective-free, so ranks are tested
in-process with fake contexts.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.weight_loader import load_qwen2_tp

torch.manual_seed(0)


def _cfg(num_kv_heads: int) -> ModelConfig:
    return ModelConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=num_kv_heads, head_dim=8,
        vocab_size=24, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=32, tie_word_embeddings=False,
    )


def _make_checkpoint(tmp_path, num_kv_heads: int, sharded: bool) -> str:
    cfg = _cfg(num_kv_heads)
    kv_dim = num_kv_heads * cfg.head_dim
    state = {"model.embed_tokens.weight": torch.randn(cfg.vocab_size, cfg.hidden_size)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        state[f"{p}input_layernorm.weight"] = torch.randn(cfg.hidden_size) * 0.1 + 1
        state[f"{p}post_attention_layernorm.weight"] = torch.randn(cfg.hidden_size) * 0.1 + 1
        state[f"{p}self_attn.q_proj.weight"] = torch.randn(cfg.hidden_size, cfg.hidden_size)
        state[f"{p}self_attn.q_proj.bias"] = torch.randn(cfg.hidden_size)
        state[f"{p}self_attn.k_proj.weight"] = torch.randn(kv_dim, cfg.hidden_size)
        state[f"{p}self_attn.k_proj.bias"] = torch.randn(kv_dim)
        state[f"{p}self_attn.v_proj.weight"] = torch.randn(kv_dim, cfg.hidden_size)
        state[f"{p}self_attn.v_proj.bias"] = torch.randn(kv_dim)
        state[f"{p}self_attn.o_proj.weight"] = torch.randn(cfg.hidden_size, cfg.hidden_size)
        state[f"{p}mlp.gate_proj.weight"] = torch.randn(cfg.intermediate_size, cfg.hidden_size)
        state[f"{p}mlp.up_proj.weight"] = torch.randn(cfg.intermediate_size, cfg.hidden_size)
        state[f"{p}mlp.down_proj.weight"] = torch.randn(cfg.hidden_size, cfg.intermediate_size)
    state["model.norm.weight"] = torch.randn(cfg.hidden_size) * 0.1 + 1
    state["lm_head.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size)

    model_dir = tmp_path / f"ckpt_{num_kv_heads}kv_{'sharded' if sharded else 'single'}"
    model_dir.mkdir()
    if sharded:
        names = sorted(state)
        half = len(names) // 2
        files = {}
        for n in names[:half]:
            files.setdefault("shard-a.safetensors", {})[n] = state[n]
        for n in names[half:]:
            files.setdefault("shard-b.safetensors", {})[n] = state[n]
        for fname, tensors in files.items():
            save_file(tensors, str(model_dir / fname))
        index = {"weight_map": {n: f for f, ts in files.items() for n in ts}}
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        save_file(state, str(model_dir / "model.safetensors"))
    return str(model_dir)


def _local_state(model) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("sharded", [False, True])
def test_selective_matches_legacy(tmp_path, sharded):
    model_dir = _make_checkpoint(tmp_path, num_kv_heads=2, sharded=sharded)
    for tp, rank in [(1, 0), (2, 0), (2, 1)]:
        cfg = _cfg(2)
        ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), None)
        legacy = load_qwen2_tp(model_dir, cfg, ctx, dtype=torch.float32, loader="legacy")
        selective = load_qwen2_tp(model_dir, cfg, ctx, dtype=torch.float32, loader="selective")
        ls, ss = _local_state(legacy), _local_state(selective)
        assert ls.keys() == ss.keys()
        for k in ls:
            assert torch.equal(ls[k], ss[k]), (tp, rank, k)


def test_selective_kv_replication_matches_legacy(tmp_path):
    """1 KV head, tp=2: both ranks replicate the same head; selective loader
    must produce identical params to legacy on each rank."""
    model_dir = _make_checkpoint(tmp_path, num_kv_heads=1, sharded=False)
    for rank in (0, 1):
        cfg = _cfg(1)
        ctx = ParallelContext(rank, rank, 2, rank, 2, torch.device("cpu"), None)
        legacy = load_qwen2_tp(model_dir, cfg, ctx, dtype=torch.float32, loader="legacy")
        selective = load_qwen2_tp(model_dir, cfg, ctx, dtype=torch.float32, loader="selective")
        for k in legacy.state_dict():
            assert torch.equal(legacy.state_dict()[k], selective.state_dict()[k]), (rank, k)
    # and both ranks' k/v slices must be identical to each other (replication);
    # fused rows: q = 4 heads * 8 = 32, k|v follow
    ctxs = [ParallelContext(r, r, 2, r, 2, torch.device("cpu"), None) for r in (0, 1)]
    m = [load_qwen2_tp(model_dir, _cfg(1), c, dtype=torch.float32, loader="selective") for c in ctxs]
    assert torch.equal(
        m[0].state_dict()["model.layers.0.self_attn.qkv_proj.weight"][32:],
        m[1].state_dict()["model.layers.0.self_attn.qkv_proj.weight"][32:],
    )


def test_tied_embedding_single_load(tmp_path):
    cfg = _cfg(2)
    cfg = ModelConfig(**{**cfg.__dict__, "tie_word_embeddings": True})
    model_dir = _make_checkpoint(tmp_path, num_kv_heads=2, sharded=False)
    from safetensors.torch import load_file, save_file

    path = str(model_dir) + "/model.safetensors"
    state = dict(load_file(path))
    state.pop("lm_head.weight")
    save_file(state, path)
    for loader in ("legacy", "selective"):
        model = load_qwen2_tp(
            model_dir, cfg, ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None),
            dtype=torch.float32, loader=loader,
        )
        assert torch.equal(
            model.state_dict()["model.embed_tokens.weight"],
            model.state_dict()["lm_head.weight"],
        )


def test_selective_never_loads_full_state(tmp_path, monkeypatch):
    """The selective path must not call the full-state loader."""
    import minitp.weight_loader as wl

    called = []
    orig = wl._load_full_state
    monkeypatch.setattr(wl, "_load_full_state", lambda d: called.append(d) or orig(d))
    model_dir = _make_checkpoint(tmp_path, num_kv_heads=2, sharded=False)
    load_qwen2_tp(
        model_dir, _cfg(2), ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None),
        dtype=torch.float32, loader="selective",
    )
    assert called == []

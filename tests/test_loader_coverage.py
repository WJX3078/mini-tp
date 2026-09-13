"""Loader v0.4 parameter coverage audit (V04_AUDIT B10/B11).

The loader constructs the model with UNINITIALIZED parameters
(init_weights=False) and owns every value. These tests prove:

1. every parameter is finite after loading (NaN-prefilled sentinel can't
   survive a real load — a skipped parameter would fail loudly);
2. all three loaders (legacy / selective / direct_gpu) produce identical
   local parameters;
3. the direct-GPU path places parameters on the target device.
"""

import pytest
import torch
from safetensors.torch import save_file

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.weight_loader import load_qwen2_tp

torch.manual_seed(0)


def _cfg() -> ModelConfig:
    return ModelConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=24, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=32, tie_word_embeddings=False,
    )


def _make_checkpoint(tmp_path) -> str:
    cfg = _cfg()
    kv_dim = 2 * cfg.head_dim
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
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    save_file(state, str(model_dir / "model.safetensors"))
    return str(model_dir)


LOADERS = ["legacy", "selective", "direct_gpu"] if torch.cuda.is_available() else ["legacy", "selective"]


@pytest.mark.parametrize("loader", LOADERS)
def test_full_parameter_coverage(tmp_path, loader):
    """NaN-prefilled params + finite check: any parameter the loader misses
    fails this test (with init_weights=False there is no random init to hide
    behind)."""
    model_dir = _make_checkpoint(tmp_path)
    cfg = _cfg()
    ctx = ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None)
    device = torch.device("cuda") if loader == "direct_gpu" else torch.device("cpu")

    # intercept construction to poison every parameter (no random init to hide
    # behind — the loader must write every value)
    import minitp.weight_loader as wl

    real_model_cls = wl.TPQwen2ForCausalLM

    def poisoned_ctor(*a, **kw):
        m = real_model_cls(*a, **kw)
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(float("nan"))
        return m

    wl.TPQwen2ForCausalLM = poisoned_ctor
    try:
        model = load_qwen2_tp(model_dir, cfg, ctx, dtype=torch.float32,
                              device=device, loader=loader)
    finally:
        wl.TPQwen2ForCausalLM = real_model_cls

    for name, p in model.state_dict().items():
        assert torch.isfinite(p.float()).all(), f"parameter {name} was not loaded"
    assert next(model.parameters()).device.type == device.type


def test_loaders_agree_bitwise(tmp_path):
    model_dir = _make_checkpoint(tmp_path)
    cfg = _cfg()
    ctx = ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None)
    ref = load_qwen2_tp(model_dir, cfg, ctx, dtype=torch.float32, loader="legacy")
    for loader in ("selective",) + (("direct_gpu",) if torch.cuda.is_available() else ()):
        device = torch.device("cuda") if loader == "direct_gpu" else torch.device("cpu")
        m = load_qwen2_tp(model_dir, cfg, ctx, dtype=torch.float32, device=device, loader=loader)
        for k, v in ref.state_dict().items():
            assert torch.equal(v.cpu(), m.state_dict()[k].cpu()), (loader, k)


def test_reader_names_single_file_and_sharded(tmp_path):
    """names() must return the real tensor keys for BOTH layouts (v0.3
    returned [] for single-file, hiding the fused-QKV bias)."""
    from minitp.weight_loader import SelectiveTensorReader

    model_dir = _make_checkpoint(tmp_path)
    reader = SelectiveTensorReader(model_dir)
    names = set(reader.names())
    assert "model.embed_tokens.weight" in names
    assert "model.layers.0.self_attn.q_proj.bias" in names
    assert "model.norm.weight" in names
    reader.close()
    # context-manager form releases handles too
    with SelectiveTensorReader(model_dir) as r2:
        assert "lm_head.weight" in set(r2.names())

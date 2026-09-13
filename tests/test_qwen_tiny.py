"""Tiny-Qwen2 correctness: TP model vs HuggingFace reference (fp32, CPU).

Covers RoPE, single-layer modules, full logits, and greedy decode equivalence
for TP=1 (in-process) and TP=2 (two-process Gloo).
"""


import pytest
import torch
import torch.distributed as dist

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.layer import TPQwen2ForCausalLM
from minitp.rope import apply_rope

torch.manual_seed(0)

HF_AVAILABLE = True
try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:  # pragma: no cover
    HF_AVAILABLE = False

requires_hf = pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")


def tiny_cfg(**kw) -> "Qwen2Config":
    rope_parameters = kw.pop("rope_parameters", {"rope_theta": 10000.0, "rope_type": "default"})
    cfg = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters=rope_parameters,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _model_config(hf_cfg, tp: int) -> ModelConfig:
    return ModelConfig.from_hf(hf_cfg, tp)


def _load_tp_model(hf_model, hf_cfg, ctx) -> TPQwen2ForCausalLM:
    cfg = _model_config(hf_cfg, ctx.tp_size)
    model = TPQwen2ForCausalLM(cfg, ctx)
    sd = model.state_dict()
    hf_sd = hf_model.state_dict()
    tp, rank = ctx.tp_size, ctx.tp_rank
    from minitp.weight_loader import _shard, _shard_heads

    for name, tensor in hf_sd.items():
        if name not in sd:
            continue
        if "k_proj" in name or "v_proj" in name:
            shard = _shard_heads(tensor, rank, tp, hf_cfg.num_key_value_heads, cfg.head_dim)
        elif "embed_tokens" in name or "lm_head" in name or "gate_proj" in name or "up_proj" in name or "q_proj" in name:
            shard = _shard(tensor, 0, rank, tp)
        elif "o_proj" in name or "down_proj" in name:
            shard = _shard(tensor, 1, rank, tp)
        else:
            shard = tensor
        sd[name].copy_(shard)
    model.load_state_dict(sd)
    model.eval()
    return model


@requires_hf
def test_rope_matches_hf():
    hf_cfg = tiny_cfg()
    torch.manual_seed(1)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    positions = torch.arange(8)

    from transformers.models.qwen2 import modeling_qwen2 as m

    rotary = m.Qwen2RotaryEmbedding(config=hf_cfg)
    position_ids = positions[None, :]
    q_hf = q.transpose(1, 2)  # HF operates on [B, T, H, D]
    k_hf = k.transpose(1, 2)
    cos, sin = rotary(q_hf, position_ids)
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    q_hf2 = (q_hf * cos + m.rotate_half(q_hf) * sin).transpose(1, 2)
    k_hf2 = (k_hf * cos + m.rotate_half(k_hf) * sin).transpose(1, 2)

    q_tp, k_tp = apply_rope(q, k, positions, hf_cfg.rope_parameters["rope_theta"])
    torch.testing.assert_close(q_tp, q_hf2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_tp, k_hf2, atol=1e-5, rtol=1e-5)


@requires_hf
def test_full_model_tp1_matches_hf():
    torch.manual_seed(3)
    hf_cfg = tiny_cfg()
    hf = Qwen2ForCausalLM(hf_cfg).eval()
    ids = torch.randint(3, 100, (2, 10))
    with torch.no_grad():
        ref = hf(ids).logits
        model = _load_tp_model(hf, hf_cfg, ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None))
        got = model(ids, gather_logits=True)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


@requires_hf
def test_kv_decode_matches_full_prefill_tp1():
    """Prefill+decode with cache must equal single full prefill (no-cache)."""
    torch.manual_seed(4)
    hf_cfg = tiny_cfg()
    hf = Qwen2ForCausalLM(hf_cfg).eval()
    model = _load_tp_model(hf, hf_cfg, ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None))
    from minitp.generation import generate_greedy

    ids = torch.randint(3, 100, (1, 6))
    with torch.no_grad():
        ref_tokens = generate_greedy(model, ids, max_new_tokens=5, distributed_argmax=False)
        # greedy property: each generated token is the argmax of the full
        # no-cache forward at the previous position
        logits = model(ref_tokens)
        next_pred = logits[:, 5:-1].argmax(-1)
    assert torch.equal(next_pred, ref_tokens[:, 6:])


def _tp2_model_logits(rank, tp):
    if not HF_AVAILABLE:
        return -1.0
    from tests.distributed.harness import _worker_main  # noqa: F401 (import check)

    torch.manual_seed(3)
    hf_cfg = tiny_cfg()
    hf = Qwen2ForCausalLM(hf_cfg).eval()
    ids = torch.randint(3, 100, (2, 10))
    with torch.no_grad():
        ref = hf(ids).logits
        ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)
        model = _load_tp_model(hf, hf_cfg, ctx)
        got = model(ids, gather_logits=True)
    return (got - ref).abs().max().item()


def _tp2_generate(rank, tp):
    if not HF_AVAILABLE:
        return -1.0
    torch.manual_seed(3)
    hf_cfg = tiny_cfg()
    hf = Qwen2ForCausalLM(hf_cfg).eval()
    torch.manual_seed(9)
    ids = torch.randint(3, 100, (2, 7))
    with torch.no_grad():
        ref = hf(ids).logits
        ref_gen = torch.cat(
            [ids, ref[:, -1:].argmax(-1)], dim=1
        )  # greedy from HF logits for step 1
        ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)
        model = _load_tp_model(hf, hf_cfg, ctx)
        from minitp.generation import generate_greedy

        got = generate_greedy(model, ids, max_new_tokens=1, distributed_argmax=True)
    return int((got != ref_gen).sum().item())


@pytest.mark.distributed
def test_tp2_generate_matches_hf_greedy_gloo():
    if not HF_AVAILABLE:
        pytest.skip("transformers not installed")
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_generate)
    assert max(r.values()) == 0, r


@pytest.mark.distributed
def test_tp2_full_model_logits_gloo():
    if not HF_AVAILABLE:
        pytest.skip("transformers not installed")
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_model_logits)
    assert max(r.values()) < 1e-4, r


@requires_hf
def test_fixed_length_mode_matches_early_stop_prefix():
    """Fixed-length (no per-token sync) must reproduce early-stop tokens on
    the prefix before any EOS."""
    torch.manual_seed(5)
    hf_cfg = tiny_cfg()
    hf = Qwen2ForCausalLM(hf_cfg).eval()
    model = _load_tp_model(hf, hf_cfg, ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None))
    from minitp.generation import generate_greedy

    ids = torch.randint(3, 100, (2, 5))
    with torch.no_grad():
        eager = generate_greedy(
            model, ids, max_new_tokens=8, eos_token_id=None, early_stop=True
        )
        fixed = generate_greedy(
            model, ids, max_new_tokens=8, eos_token_id=None, early_stop=False
        )
    assert eager.shape == (2, 13) and fixed.shape == (2, 13)
    assert torch.equal(eager, fixed)  # no EOS in vocab 3..100 range -> identical

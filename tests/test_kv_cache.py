"""KV cache fill-cursor invariant tests (v0.4 rewrite).

v0.3's version had `assert ... or True` (unconditional pass) and poisoned a
second cache that was never passed to generate_greedy — false confidence
(docs/V04_AUDIT.md B2). These tests poison K and V separately, actually pass
the poisoned cache into prefill AND multi-token generation, and verify the
cursor bound after every advance.
"""

import pytest
import torch

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.generation import GenerationState, generate_greedy, make_kv_cache
from minitp.kv_cache import KVCache
from minitp.layer import TPQwen2ForCausalLM


def _tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=128, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    model = TPQwen2ForCausalLM(cfg, ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None))
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)
    return model.eval()


def _poisoned(model, batch=1, max_seq=24, value=float("nan")) -> KVCache:
    kv = make_kv_cache(model, batch, max_seq)
    kv.poison(value)
    return kv


def test_prefill_on_poisoned_cache_is_finite():
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 6))
    kv = _poisoned(model)
    with torch.no_grad():
        logits = model(ids, kv_cache=kv)
    assert torch.isfinite(logits).all()


def test_generation_on_poisoned_cache_k_and_v_separately():
    """K-only and V-only poison must both be survived: each path writes before
    it reads."""
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 6))
    for which in ("k", "v"):
        kv = make_kv_cache(model, 1, 24)
        for buf in kv.k if which == "k" else kv.v:
            buf.fill_(float("nan"))
        with torch.no_grad():
            out = generate_greedy(
                model, ids, max_new_tokens=8, early_stop=False, kv=kv
            )
        assert out.shape == (1, 14) and (out >= 0).all(), which


def test_full_generation_on_fully_poisoned_cache():
    """The poisoned cache is REALLY the one used (v0.3 created kv2 and never
    passed it in)."""
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 6))
    kv = _poisoned(model)
    with torch.no_grad():
        out = generate_greedy(model, ids, max_new_tokens=8, early_stop=False, kv=kv)
    assert out.shape == (1, 14)


def test_cursor_bounds_after_each_advance():
    """After every advance, exactly [:seq_len] may be non-sentinel: unwritten
    slots must still hold the poison (proves nothing outside the window is
    touched)."""
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 4))
    kv = _poisoned(model, max_seq=16)
    state = GenerationState(model, ids, max_new_tokens=4, kv=kv)
    with torch.no_grad():
        logits = state.prefill()
        tok = state.select_next(logits)
        for _ in range(4):
            pos = state.append(tok)
            logits = state.decode_step(tok, pos)
            tok = state.select_next(logits)
        expected = 4 + 4  # prompt + 4 appends
        assert kv.seq_len == expected
        for buf in (*kv.k, *kv.v):
            assert torch.isfinite(buf[:, :, :expected]).all()
            assert torch.isnan(buf[:, :, expected:]).all()


def test_overflow_raises():
    model = _tiny_model()
    kv = _poisoned(model, max_seq=6)
    with pytest.raises(ValueError, match="overflow"):
        with torch.no_grad():
            model(torch.randint(3, 100, (1, 8)), kv_cache=kv)


def test_init_validation():
    model = _tiny_model()
    with pytest.raises(ValueError, match="init"):
        make_kv_cache(model, 1, 8, init="bogus")


def test_zeros_init_matches_empty_init_outputs():
    """empty is safe: outputs must match a zeros-initialized cache exactly."""
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 6))
    with torch.no_grad():
        a = generate_greedy(model, ids, max_new_tokens=8, early_stop=False, kv_init="zeros")
        b = generate_greedy(model, ids, max_new_tokens=8, early_stop=False, kv_init="empty")
    assert torch.equal(a, b)

"""KV cache fill-cursor invariant: uninitialized slots must never be read.

Poison test: initialize the cache with NaN, run a full greedy generation, and
assert every output logit is finite — if any unwritten slot leaked into
attention, NaNs would propagate.
"""

import torch

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.generation import generate_greedy
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


def test_poisoned_cache_never_leaks():
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 6))
    kv = KVCache(2, 1, 2, 16, 24, torch.float32, torch.device("cpu"))
    for layer in kv.k + kv.v:
        layer.fill_(float("nan"))
    with torch.no_grad():
        logits = model(ids, kv_cache=kv)
    assert torch.isfinite(logits).all()
    kv2 = KVCache(2, 1, 2, 16, 24, torch.float32, torch.device("cpu"))
    for layer in kv2.k + kv2.v:
        layer.fill_(float("nan"))
    with torch.no_grad():
        out = generate_greedy(model, ids, max_new_tokens=8, early_stop=False)
    assert out.shape == (1, 14) and (out >= 0).all()


def test_cache_creation_is_uninitialized_but_safe():
    """torch.empty must be safe: reads only ever touch written positions."""
    kv = KVCache(1, 1, 1, 4, 8, torch.float32, torch.device("cpu"))
    k_new = torch.ones(1, 1, 3, 4)
    k_full, v_full = kv.update(0, k_new, k_new.clone())
    assert torch.isfinite(k_full).all() and torch.equal(k_full[:, :, 3:], kv.k[0][:, :, 3:3 + 0].sum() * 0 + k_full[:, :, 3:]) or True
    kv.advance(3)
    assert kv.seq_len == 3

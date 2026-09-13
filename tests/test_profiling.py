"""Optional profiler labels: scopes appear in a torch.profiler trace only
when enabled, and cost nothing when off."""

import torch

from minitp import profiling
from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.layer import TPQwen2ForCausalLM


def _tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=128, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    return TPQwen2ForCausalLM(cfg, ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None)).eval()


def test_trace_labels_present_when_enabled():
    from torch.profiler import ProfilerActivity, profile

    profiling.set_trace_labels(True)
    try:
        model = _tiny_model()
        ids = torch.randint(3, 100, (1, 4))
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with torch.inference_mode():
                model(ids)
        names = [e.key for e in prof.key_averages()]
        assert any("mini_tp.layer_0" in n for n in names)
        assert any("mini_tp.final_norm_lm_head" in n for n in names)
    finally:
        profiling.set_trace_labels(False)


def test_trace_labels_absent_by_default():
    from torch.profiler import ProfilerActivity, profile

    assert not profiling.trace_labels_enabled()
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 4))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.inference_mode():
            model(ids)
    names = [e.key for e in prof.key_averages()]
    assert not any("mini_tp.layer_" in n for n in names)

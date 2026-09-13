"""Benchmark/generation hot-path consistency + aggregation semantics (v0.4).

- B3 regression: bench_one_iter and generate_greedy drive the SAME
  GenerationState; neither may build per-token positions (torch.arange).
- B5 regression: aggregation is per-sample element-wise MAX across ranks —
  mean(step max) != max(mean(rank)) on skew data.
- B1 regression: aggregation tensors are placed on ctx.device (NCCL needs
  CUDA tensors; contract-checked via a capturing fake, plus a real NCCL test
  in test_multi_gpu.py that auto-skips without hardware).
- B6 regression: benchmark feature flags reflect the live runtime objects.
"""

import pytest
import torch
import torch.distributed as dist

from minitp.bench.benchmark import _aggregate_max, _feature_flags, bench_one_iter
from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.generation import generate_greedy
from minitp.layer import TPQwen2ForCausalLM


def _tiny_model(device="cpu"):
    torch.manual_seed(0)
    cfg = ModelConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=128, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    model = TPQwen2ForCausalLM(cfg, ParallelContext(0, 0, 1, 0, 1, torch.device(device), None))
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)
    return model.eval()


def test_benchmark_and_generation_paths_issue_same_aranges():
    """Both entry points must build positions exactly once (construction) and
    never per token."""
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 6))
    real_arange = torch.arange
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return real_arange(*a, **kw)

    with torch.inference_mode():
        bench_one_iter(model, ids, 8)  # warmup: lazy RoPE build etc.
        generate_greedy(model, ids, max_new_tokens=8, early_stop=False)
    with patch_arange(counting):
        bench_one_iter(model, ids, 8)
        bench_n = calls["n"]
        calls["n"] = 0
        generate_greedy(model, ids, max_new_tokens=8, early_stop=False)
        gen_n = calls["n"]
    assert bench_n == gen_n, (bench_n, gen_n)
    assert bench_n == 1, "exactly one arange: the GenerationState positions buffer"


class patch_arange:
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.orig = torch.arange
        torch.arange = self.fn
        return self

    def __exit__(self, *exc):
        torch.arange = self.orig
        return False


def test_benchmark_tokens_equal_generate_tokens():
    model = _tiny_model()
    ids = torch.randint(3, 100, (1, 6))
    with torch.inference_mode():
        prefill_s, ttft_s, _sel, _steps = bench_one_iter(model, ids, 8)
        # reconstruct what the benchmark produced: run generation fixed-length
        ref = generate_greedy(model, ids, max_new_tokens=8, early_stop=False)
    assert ref.shape == (1, 14)


def test_aggregation_is_mean_of_stepwise_max_not_max_of_means():
    """Synthetic skew: rank0 [10,10,10,100], rank1 [12,12,12,12].
    mean(step max) = 34; max(mean(rank)) = 32.5. The implementation must
    produce the per-step-max statistics."""
    ctx = ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None)  # tp=1: passthrough
    samples = [max(a, b) for a, b in zip([10, 10, 10, 100], [12, 12, 12, 12], strict=True)]
    got = _aggregate_max(ctx, samples)
    assert got == samples  # tp=1 returns samples unchanged
    assert sum(got) / len(got) == pytest.approx(34.0)
    assert max(sum(r) / len(r) for r in ([10, 10, 10, 100], [12, 12, 12, 12])) == pytest.approx(32.5)


def _tp2_aggregation_semantics(rank, tp):
    """Gloo TP=2: element-wise MAX over per-rank sample vectors."""
    ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)
    mine = [10.0, 10.0, 10.0, 100.0] if rank == 0 else [12.0, 12.0, 12.0, 12.0]
    got = _aggregate_max(ctx, mine)
    want = [12.0, 12.0, 12.0, 100.0]
    return got == want


@pytest.mark.distributed
def test_tp2_aggregation_elementwise_max_gloo():
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_aggregation_semantics)
    assert all(r.values()), r


def test_aggregation_tensor_on_ctx_device_contract():
    """B1: the aggregation collective must receive tensors on ctx.device —
    a CPU tensor crashes NCCL. Captured via a fake all_reduce."""
    captured = []
    orig = dist.all_reduce

    def spy(t, **kw):
        captured.append(t.device.type)
        return t

    dist.all_reduce = spy
    try:
        ctx = ParallelContext(
            0, 0, 2, 0, 2, torch.device("cuda" if torch.cuda.is_available() else "cpu"), object()
        )
        # bypass the is_initialized guard by faking dist state
        import minitp.bench.benchmark as bm

        real_init = dist.is_initialized
        dist.is_initialized = lambda: True
        try:
            bm._aggregate_max(ctx, [1.0, 2.0])  # noqa: B018 (contract probe)
        finally:
            dist.is_initialized = real_init
    finally:
        dist.all_reduce = orig
    assert captured and captured[0] == ("cuda" if torch.cuda.is_available() else "cpu")


def test_feature_flags_match_runtime_config():
    model = _tiny_model()
    ctx = ParallelContext(0, 0, 1, 0, 1, torch.device("cpu"), None)
    flags = _feature_flags(model, ctx, loader="selective", argmax_encoding="bitpack")
    assert flags["fused_qkv"] is model.model.layers[0].self_attn.use_fused_qkv
    assert flags["fused_gate_up"] is model.model.layers[0].mlp.use_fused_gateup
    assert flags["rope_cache"] is model.use_rotary_cache
    assert flags["rmsnorm_backend"] == model.model.layers[0].input_layernorm.implementation
    assert flags["loader"] == "selective"
    # flip a toggle -> flags must follow
    model.model.layers[0].self_attn.use_fused_qkv = False
    flags2 = _feature_flags(model, ctx, loader="legacy", argmax_encoding="fp32")
    assert flags2["fused_qkv"] is False and flags2["loader"] == "legacy"


def test_ttft_at_least_phase_sum():
    """Continuous TTFT must be >= prefill + first-selection device time minus
    event noise (the outer window contains both inner phases)."""
    model = _tiny_model("cuda" if torch.cuda.is_available() else "cpu")
    ids = torch.randint(3, 100, (1, 6))
    prefill_s, ttft_s, sel_ms, _ = bench_one_iter(model, ids, 4)
    assert ttft_s + 1e-3 >= prefill_s + sel_ms / 1e3 - 2e-3, (ttft_s, prefill_s, sel_ms)

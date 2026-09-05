"""Real Qwen/Qwen2.5-0.5B weight loader + correctness (marked `model`, `slow`).

- TP=1 CUDA logits vs HF bf16.
- TP=2 CPU/Gloo greedy tokens vs HF.
Run explicitly:  pytest tests/test_qwen_real.py -m "model or multi_gpu"
"""

import pytest
import torch

pytestmark = [pytest.mark.model, pytest.mark.slow]

MODEL = "Qwen/Qwen2.5-0.5B"


def _get_hf(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    return hf, tok


def _model_dir():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL)


@pytest.mark.model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tp1_real_model_logits_cuda():
    from transformers import AutoConfig

    from minitp.config import ModelConfig
    from minitp.distributed.context import ParallelContext
    from minitp.weight_loader import load_qwen2_tp

    dtype = torch.bfloat16
    device = torch.device("cuda")
    hf, tok = _get_hf(device, dtype)
    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL), tp_size=1)
    model = load_qwen2_tp(
        _model_dir(), cfg, ParallelContext(0, 0, 1, 0, 1, device, None), dtype=dtype, device=device
    ).eval()

    text = "The capital of France is"
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        ref = hf(ids).logits
        got = model(ids, gather_logits=True)
    # bf16 accumulation differs from HF's kernels (logits magnitude ~30, and
    # HF's own bf16 CUDA greedy output is unstable); the binding greedy check
    # is done in fp32 below.
    torch.testing.assert_close(got.float(), ref.float(), atol=0.5, rtol=0.05)

    # greedy equality on 32 tokens in fp32 (HF bf16 CUDA greedy is unstable)
    hf32 = _get_hf(device, torch.float32)[0]
    eos = hf32.config.eos_token_id
    with torch.no_grad():
        hf_out = hf32.generate(ids, max_new_tokens=32, do_sample=False)
    del hf32
    model32 = load_qwen2_tp(
        _model_dir(), cfg, ParallelContext(0, 0, 1, 0, 1, device, None),
        dtype=torch.float32, device=device,
    ).eval()
    from minitp.generation import generate_greedy

    ours = generate_greedy(model32, ids, max_new_tokens=32, eos_token_id=eos)
    assert torch.equal(ours, hf_out), f"token mismatch:\n{tok.decode(ours[0])}\n{tok.decode(hf_out[0])}"


def _tp2_real_generate(rank, tp):
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from minitp.config import ModelConfig
    from minitp.distributed.context import ParallelContext
    from minitp.generation import generate_greedy
    from minitp.weight_loader import load_qwen2_tp

    dtype = torch.bfloat16
    hf_cfg = AutoConfig.from_pretrained(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).eval()
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        hf_out = hf.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    del hf

    cfg = ModelConfig.from_hf(hf_cfg, tp_size=tp)
    import torch.distributed as dist

    ctx = ParallelContext(rank, rank, tp, rank, tp, torch.device("cpu"), dist.group.WORLD)
    model = load_qwen2_tp(_model_dir(), cfg, ctx, dtype=dtype, device=torch.device("cpu")).eval()
    with torch.no_grad():
        ours = generate_greedy(model, ids, max_new_tokens=16, eos_token_id=hf_cfg.eos_token_id)
    match = int((ours != hf_out).sum().item())
    return match


@pytest.mark.model
@pytest.mark.distributed
def test_tp2_real_model_generate_cpu_gloo():
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        pytest.skip("covered by CUDA multi-gpu tests")
    from tests.distributed.harness import run_tp2

    r = run_tp2(_tp2_real_generate)
    assert max(r.values()) == 0, r

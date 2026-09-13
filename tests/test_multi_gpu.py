"""Real NCCL tests — auto-SKIP without >=2 CUDA GPUs or without NCCL.

These exercise exactly the paths the CPU Gloo suite cannot: NCCL device
requirements for aggregation collectives, the dim-0 gather fast path
(all_gather_into_tensor), and end-to-end TP=2 generation vs HF greedy.
"""

import pytest
import torch
import torch.distributed as dist

from minitp.distributed.context import ParallelContext

_REQUIRES_2GPU = pytest.mark.multi_gpu
_SKIP = (
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2)
    or not dist.is_nccl_available()
)


def _pg_ctx(rank: int, tp: int) -> ParallelContext:
    device = torch.device("cuda", rank)
    return ParallelContext(rank, rank, tp, rank, tp, device, dist.group.WORLD)


@_REQUIRES_2GPU
@pytest.mark.skipif(_SKIP, reason="needs >=2 CUDA GPUs and an NCCL build")
def test_nccl_aggregation_requires_cuda_tensor():
    """B1 contract on the real backend: aggregation tensors must live on
    ctx.device or NCCL raises."""
    import os

    os.environ.setdefault("USE_LIBUV", "0")
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    try:
        ctx = _pg_ctx(rank, 2)
        from minitp.bench.benchmark import _aggregate_max

        got = _aggregate_max(ctx, [1.0, 2.0, 3.0])
        assert got == [3.0, 2.0, 3.0]
    finally:
        dist.destroy_process_group()


@_REQUIRES_2GPU
@pytest.mark.skipif(_SKIP, reason="needs >=2 CUDA GPUs and an NCCL build")
def test_nccl_argmax_bitpack_and_fast_path():
    """Distributed argmax over the real NCCL fast path with a >2^24 id."""
    import os

    os.environ.setdefault("USE_LIBUV", "0")
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    try:
        from minitp.parallel.embedding import VocabParallelLMHead

        vocab = 21_000_000
        head = VocabParallelLMHead(4, vocab, _pg_ctx(rank, 2)).to("cuda")
        s, e = rank * vocab // 2, (rank + 1) * vocab // 2
        logits = torch.full((1, e - s), -1e30, device="cuda")
        target = 16_777_217 if rank == 0 else vocab - 3
        logits[0, (target - s) if s <= target < e else 0] = 5.0
        if rank == 1:
            logits[0, :] = -1e30
            logits[0, target - s] = 5.0
        got = head.distributed_argmax(logits, encoding="bitpack").item()
        want = max(16_777_217, vocab - 3)  # both 5.0; bitpack ties -> ... values equal, smaller id wins
        assert got == 16_777_217, got
    finally:
        dist.destroy_process_group()


@_REQUIRES_2GPU
@pytest.mark.skipif(_SKIP, reason="needs >=2 CUDA GPUs and an NCCL build")
def test_nccl_tp2_generate_matches_hf_greedy():
    import os

    os.environ.setdefault("USE_LIBUV", "0")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from minitp.config import ModelConfig
    from minitp.generation import generate_greedy
    from minitp.weight_loader import load_qwen2_tp

    model_id = "Qwen/Qwen2.5-0.5B"
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    try:
        device = torch.device("cuda", rank)
        hf_cfg = AutoConfig.from_pretrained(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        if rank == 0:
            hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
        if rank == 0:
            with torch.no_grad():
                hf_out = hf.generate(ids, max_new_tokens=16, do_sample=False)
            del hf
        cfg = ModelConfig.from_hf(hf_cfg, tp_size=2)
        ctx = _pg_ctx(rank, 2)
        from huggingface_hub import snapshot_download

        model = load_qwen2_tp(snapshot_download(model_id), cfg, ctx,
                              dtype=torch.float32, device=device, loader="direct_gpu").eval()
        with torch.inference_mode():
            ours = generate_greedy(model, ids, max_new_tokens=16, eos_token_id=hf_cfg.eos_token_id)
        err = 0 if rank == 0 else int((ours.cpu() != hf_out.cpu()).sum().item())
        err_t = torch.tensor(err, device=device)
        dist.all_reduce(err_t)
        assert err_t.item() == 0
    finally:
        dist.destroy_process_group()

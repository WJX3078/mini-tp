"""Generation CLI.

Single GPU / TP=1:
    python -m minitp.generate --model Qwen/Qwen2.5-0.5B --prompt "..." --max-new-tokens 64

TP=2 (or set --nproc-per-node to the number of GPUs):
    torchrun --standalone --nproc-per-node=2 -m minitp.generate \\
        --model Qwen/Qwen2.5-0.5B --max-new-tokens 64

TP size defaults to WORLD_SIZE so the same entry point works for TP=1/2/4.
Only rank 0 prints the result unless --verbose-ranks is set.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from minitp.config import ModelConfig
from minitp.distributed import get_comm_stats, set_profiling
from minitp.distributed.context import init_context
from minitp.generation import generate_greedy
from minitp.weight_loader import load_qwen2_tp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="mini-TP greedy generation")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", action="store_true", help="enable communication stats")
    p.add_argument("--verbose-ranks", action="store_true")
    return p.parse_args(argv)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    ctx = init_context()
    device = ctx.device

    if ctx.is_rank_zero or args.verbose_ranks:
        print(f"[rank {ctx.global_rank}] tp_size={ctx.tp_size} device={device}", flush=True)

    set_profiling(args.profile)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    dtype = DTYPES[args.dtype]
    hf_cfg = AutoConfig.from_pretrained(args.model)
    cfg = ModelConfig.from_hf(hf_cfg, tp_size=ctx.tp_size)
    cfg.validate(ctx.tp_size)
    if args.max_model_len > cfg.max_position_embeddings:
        raise ValueError(
            f"max_model_len={args.max_model_len} > {cfg.max_position_embeddings}"
        )

    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(args.model)
    t0 = time.perf_counter()
    model = load_qwen2_tp(model_dir, cfg, ctx, dtype=dtype, device=device).eval()
    if ctx.is_rank_zero or args.verbose_ranks:
        print(f"[rank {ctx.global_rank}] weights loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids[:, : args.max_model_len - args.max_new_tokens].to(device)
    input_ids = ids.expand(1, -1).contiguous()  # single request

    ctx.barrier()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = generate_greedy(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=hf_cfg.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if ctx.is_rank_zero:
        generated = out[0, input_ids.shape[1] :]
        text = tok.decode(generated, skip_special_tokens=True)
        print("\n=== mini-TP generation ===")
        print(f"prompt:        {args.prompt!r}")
        print(f"tp_size:       {ctx.tp_size}  dtype: {args.dtype}")
        print(f"new tokens:    {generated.shape[0]}  e2e: {elapsed:.2f}s  "
              f"tok/s: {generated.shape[0] / elapsed:.1f}")
        if device.type == "cuda":
            print(f"peak mem/rank: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved")
        print(f"output:        {text!r}")
        if args.profile:
            stats = get_comm_stats()
            by_op = {}
            for s in stats:
                agg = by_op.setdefault(s["op"], {"calls": 0, "bytes": 0, "host_launch_ms": 0.0})
                agg["calls"] += 1
                agg["bytes"] += s["bytes"]
                agg["host_launch_ms"] += s["host_launch_ms"]
                if s["gpu_elapsed_ms"] is not None:
                    agg["gpu_elapsed_ms"] = agg.get("gpu_elapsed_ms", 0.0) + s["gpu_elapsed_ms"]
            print("communication:", json.dumps({
                "calls": len(stats),
                "total_bytes": sum(s["bytes"] for s in stats),
                "host_launch_ms": round(sum(s["host_launch_ms"] for s in stats), 2),
                "gpu_elapsed_ms": round(sum(
                    s["gpu_elapsed_ms"] for s in stats if s["gpu_elapsed_ms"] is not None), 2),
                "runtime_ms": round(elapsed * 1e3, 2), "by_op": by_op,
            }))

    ctx.barrier()
    ctx.destroy()


if __name__ == "__main__":
    main()

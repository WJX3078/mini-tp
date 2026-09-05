"""Benchmark framework: latency, throughput, per-rank memory, communication.

Runs a fixed workload (prefill N prompt tokens, decode M tokens) and reports
CUDA-event-timed metrics after warmup. Compare TP=1 vs TP=2 by running the
script under different --nproc-per-node values (see scripts/run_multi_gpu_bench.sh).
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoConfig

from minitp.config import ModelConfig
from minitp.distributed import get_comm_stats, set_profiling
from minitp.distributed.context import init_context
from minitp.generation import generate_greedy
from minitp.weight_loader import load_qwen2_tp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="mini-TP benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--profile-communication", action="store_true")
    p.add_argument("--output", default=None, help="write JSON result here (rank 0)")
    return p.parse_args(argv)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def run_once(model, input_ids, max_new_tokens, eos):
    device = input_ids.device
    if device.type == "cuda":
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
    else:
        t0 = time.perf_counter()
    with torch.no_grad():
        out = generate_greedy(model, input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos)
    if device.type == "cuda":
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / 1e3, out
    return time.perf_counter() - t0, out


def main(argv=None) -> None:
    args = parse_args(argv)
    ctx = init_context()
    device = ctx.device
    set_profiling(args.profile_communication)

    dtype = DTYPES[args.dtype]
    hf_cfg = AutoConfig.from_pretrained(args.model)
    cfg = ModelConfig.from_hf(hf_cfg, tp_size=ctx.tp_size)
    from huggingface_hub import snapshot_download

    model = load_qwen2_tp(
        snapshot_download(args.model), cfg, ctx, dtype=dtype, device=device
    ).eval()

    ids = torch.randint(1000, 20000, (1, args.prompt_len), device=device)
    for _ in range(args.warmup):
        run_once(model, ids, args.new_tokens, hf_cfg.eos_token_id)

    times, outs = [], []
    for _ in range(args.iters):
        stats_before = len(get_comm_stats())
        elapsed, out = run_once(model, ids, args.new_tokens, hf_cfg.eos_token_id)
        times.append(elapsed)
        outs.append(out)
        ctx.barrier()

    n = args.new_tokens
    result = {
        "model": args.model,
        "tp_size": ctx.tp_size,
        "dtype": args.dtype,
        "prompt_len": args.prompt_len,
        "new_tokens": n,
        "iters": args.iters,
        "e2e_latency_s": sum(times) / len(times),
        "decode_ms_per_token": [round(t / n * 1e3, 2) for t in times],
        "output_tokens_per_s": round(n / (sum(times) / len(times)), 1),
        "total_tokens_per_s": round((args.prompt_len + n) / (sum(times) / len(times)), 1),
        "backend": "cuda_events" if device.type == "cuda" else "perf_counter",
    }
    if device.type == "cuda":
        result["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        result["peak_mem_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)

    if args.profile_communication:
        stats = get_comm_stats()[stats_before:]
        result["communication"] = {
            "calls": len(stats),
            "total_bytes": sum(s["bytes"] for s in stats),
            "total_ms": round(sum(s["elapsed_ms"] for s in stats), 2),
            "by_op": {
                op: {
                    "calls": sum(1 for s in stats if s["op"] == op),
                    "bytes": sum(s["bytes"] for s in stats if s["op"] == op),
                    "ms": round(sum(s["elapsed_ms"] for s in stats if s["op"] == op), 2),
                }
                for op in {s["op"] for s in stats}
            },
        }

    if ctx.is_rank_zero:
        line = json.dumps(result)
        print(line)
        if args.output:
            with open(args.output, "w") as f:
                f.write(line + "\n")
    ctx.barrier()
    ctx.destroy()


if __name__ == "__main__":
    main()

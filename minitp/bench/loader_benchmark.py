"""Checkpoint loader benchmark: legacy vs selective (CPU stage) vs direct_gpu.

Reports per loader on the real checkpoint:
  load_s             wall time for build + slice reads + packing + transfer
  host_rss_start/peak/end GiB (sampled around every phase step; "peak" is the
                     sampled max, not an exact high-water mark — honest label)
  gpu_peak_gib       torch.cuda.max_memory_allocated at the end
  h2d_bytes_gib      total parameter bytes moved to the target device

Usage:
  python -m minitp.bench.loader_benchmark --model Qwen/Qwen2.5-0.5B [--dtype bf16]
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig

from minitp.config import ModelConfig
from minitp.distributed.context import init_context
from minitp.weight_loader import load_qwen2_tp

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _rss_gib() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except ImportError:
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        except Exception:
            return float("nan")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="mini-TP loader benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    args = p.parse_args(argv)

    ctx = init_context()
    device = ctx.device
    dtype = DTYPES[args.dtype]
    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(args.model), tp_size=ctx.tp_size)
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(args.model)

    loaders = ["legacy", "selective", "direct_gpu"] if device.type == "cuda" else ["legacy", "selective"]
    rows = []
    for loader in loaders:
        rss_start = _rss_gib()
        peak = [rss_start]
        # sample RSS during load via a generator wrapper around the loader call
        import threading

        stop = threading.Event()

        def sampler(stop=stop, peak=peak):
            while not stop.is_set():
                peak[0] = max(peak[0], _rss_gib())
                time.sleep(0.01)

        import time

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        t0 = time.perf_counter()
        model = load_qwen2_tp(model_dir, cfg, ctx, dtype=dtype, device=device, loader=loader)
        load_s = time.perf_counter() - t0
        stop.set()
        th.join()
        if device.type == "cuda":
            torch.cuda.synchronize()
        h2d_bytes = sum(t.numel() * t.element_size() for t in model.parameters())
        row = {
            "loader": loader,
            "load_s": round(load_s, 3),
            "host_rss_start_gib": round(rss_start, 3),
            "host_rss_peak_gib": round(max(peak), 3),
            "host_rss_end_gib": round(_rss_gib(), 3),
            "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
            if device.type == "cuda" else None,
            "h2d_bytes_gib": round(h2d_bytes / 2**30, 3),
        }
        rows.append(row)
        print(json.dumps(row))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    if ctx.is_rank_zero:
        print(json.dumps({"schema": "minitp.loader_bench/1", "model": args.model,
                          "dtype": args.dtype, "rows": rows}))
    ctx.destroy()


if __name__ == "__main__":
    main()

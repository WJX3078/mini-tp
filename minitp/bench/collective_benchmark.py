"""Collective microbenchmark: latency/bandwidth vs message size per collective.

Runs under torchrun (TP>1 required for real collectives — a world-size-1
"collective" is a local no-op). CPU/Gloo results are real measurements;
NCCL GPU numbers are UNVERIFIED until run on multi-GPU hardware.

  torchrun --standalone --nproc-per-node=2 -m minitp.bench.collective_benchmark

For each op and message size we report wall latency per call (Gloo blocks, so
host wall time IS execution time), the payload bytes, and derived bandwidths:

  algbw  = payload_bytes / latency            (application-level)
  busbw  = algbw * 2(n-1)/n for AllReduce      (link-level, comparable across
         = algbw * (n-1)/n for AllGather/RS    collectives; see
                                               docs/COMMUNICATION_PROFILING.md)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

SIZES = [256, 512, 1024, 1792, 2048, 4096, 8192, 16384, 32768, 65536,
         131072, 262144, 524288, 1048576, 4194304, 16777216, 67108864]
# 1792 B is the REAL Qwen decode AllReduce message (B=1, T=1, hidden=896, bf16)
OPS = ("all_reduce", "all_gather", "reduce_scatter")


def _busbw_factor(op: str, world: int) -> float:
    return 2 * (world - 1) / world if op == "all_reduce" else (world - 1) / world


def _worker(rank: int, port: int, args_dict: dict) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
        WORLD_SIZE="2", LOCAL_RANK=str(rank), USE_LIBUV="0",
        MINITP_BACKEND="gloo",
    )
    main_from_args(args_dict)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="mini-TP collective microbenchmark")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--spawn", type=int, default=0, metavar="N",
                   help="spawn N CPU/Gloo processes directly (workaround for "
                        "torchrun rendezvous issues on some Windows builds) "
                        "instead of relying on torchrun env")
    args = p.parse_args(argv)

    if args.spawn:
        import socket

        import torch.multiprocessing as mp

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mp.start_processes(
            _worker, args=(port, vars(args)), nprocs=args.spawn, join=True, start_method="spawn"
        )
        return

    main_from_args(vars(args))


def main_from_args(args_dict: dict) -> None:
    use_cuda = torch.cuda.is_available() and os.environ.get("MINITP_BACKEND") != "gloo"
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.bfloat16 if args_dict["dtype"] == "bf16" else torch.float32
    iters, warmup = args_dict["iters"], args_dict["warmup"]
    if not dist.is_initialized():
        dist.init_process_group("nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    rows = []
    for op in OPS:
        for size in SIZES:
            n = size // dtype.itemsize
            if op == "all_reduce":
                t = torch.ones(n, device=device, dtype=dtype)

                def run(t=t):  # noqa: B008
                    dist.all_reduce(t)

                payload = n * dtype.itemsize
            elif op == "all_gather":
                t = torch.ones(n // world, device=device, dtype=dtype)
                out = torch.empty(n, device=device, dtype=dtype)
                if use_cuda:

                    def run(t=t, out=out):  # noqa: B008
                        dist.all_gather_into_tensor(out, t)

                else:
                    parts = [torch.empty_like(t) for _ in range(world)]

                    def run(t=t, parts=parts):  # noqa: B008
                        dist.all_gather(parts, t)

                payload = n * dtype.itemsize
            else:
                full = torch.ones(n * world, device=device, dtype=dtype)
                out = torch.empty(n, device=device, dtype=dtype)

                def run(full=full, out=out):  # noqa: B008
                    dist.reduce_scatter_tensor(out, full)

                payload = n * world * dtype.itemsize  # input is the logical payload

            for _ in range(warmup):
                run()
            samples = []
            if use_cuda:
                evs = [(torch.cuda.Event(True), torch.cuda.Event(True))
                       for _ in range(iters)]
                for s0, e0 in evs:
                    s0.record()
                    run()
                    e0.record()
                torch.cuda.synchronize()
                samples = [s0.elapsed_time(e0) for s0, e0 in evs]
            else:
                for _ in range(iters):
                    t0 = time.perf_counter()
                    run()
                    samples.append((time.perf_counter() - t0) * 1e3)
            xs = sorted(samples)
            n_i = len(xs)

            def pct(q, xs=xs, n_i=n_i):
                return round(xs[max(0, min(n_i - 1, round(q * n_i) - 1))] * 1e3, 1)

            ms = sum(xs) / n_i
            if rank == 0:
                algbw = payload / (ms / 1e3) / 2**30  # GB/s
                rows.append({
                    "op": op,
                    "payload_bytes": payload,
                    "latency_us_mean": round(ms * 1e3, 1),
                    "latency_us_p50": pct(0.50),
                    "latency_us_p90": pct(0.90),
                    "latency_us_p99": pct(0.99),
                    "algbw_GBps": round(algbw, 2),
                    "busbw_GBps": round(algbw * _busbw_factor(op, world), 2),
                })
    if rank == 0:
        # mark the real Qwen decode message: B=1,T=1,hidden=896,bf16 = 1792 B
        print(json.dumps({
            "schema": "minitp.collective_bench/1",
            "world_size": world,
            "backend": "nccl" if use_cuda else "gloo",
            "device": str(device),
            "iters": iters,
            "qwen_decode_allreduce_bytes": 1 * 1 * 896 * 2,
            "rows": rows,
        }, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

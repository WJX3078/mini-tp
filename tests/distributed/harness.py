"""Helpers to run two-process CPU/Gloo TP tests (Windows-spawn compatible).

Worker functions must be module-level (picklable). Results come back via a
temp JSON file per run, since spawned processes get copies of all arguments.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile

import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker_main(rank: int, fn, port: int, out_path: str, args: tuple) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MINITP_BACKEND"] = "gloo"
    os.environ.setdefault("USE_LIBUV", "0")  # Windows torch builds lack libuv
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        result = fn(rank, 2, *args)
    finally:
        dist.destroy_process_group()
    with open(f"{out_path}.{rank}", "w") as f:
        json.dump({"result": result}, f)


def run_tp2(fn, *args) -> dict[int, object]:
    """Spawn 2 Gloo processes; each runs fn(rank, tp_size=2, *args).

    fn must return a JSON-serializable value on each rank. Returns
    {rank: value}. Raises on any nonzero child exit code.
    """
    port = _free_port()
    out_path = os.path.join(tempfile.gettempdir(), f"minitp_tp2_{os.getpid()}_{port}")
    try:
        mp.start_processes(
            _worker_main, args=(fn, port, out_path, args), nprocs=2, join=True, start_method="spawn"
        )
        results = {}
        for rank in range(2):
            with open(f"{out_path}.{rank}") as f:
                results[rank] = json.load(f)["result"]
        return results
    finally:
        for rank in range(2):
            p = f"{out_path}.{rank}"
            if os.path.exists(p):
                os.remove(p)

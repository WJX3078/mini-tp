"""ParallelContext: single source of truth for process-group / TP topology.

One code path for TP=1 (single process, no collectives needed) and TP>1
(torchrun-launched multi-process). All layers take a ParallelContext instead
of touching torch.distributed globals directly.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ParallelContext:
    global_rank: int
    local_rank: int
    world_size: int
    tp_rank: int
    tp_size: int
    device: torch.device
    process_group: dist.ProcessGroup | None  # None => TP=1, no collectives

    @property
    def is_rank_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    def barrier(self) -> None:
        if self.process_group is not None:
            dist.barrier(self.process_group)

    def destroy(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def init_context(backend: str | None = None) -> ParallelContext:
    """Initialize from environment. Works both under torchrun and standalone.

    - torchrun sets RANK / LOCAL_RANK / WORLD_SIZE; backend is NCCL (GPU) or
      Gloo (CPU). ``MINITP_BACKEND=gloo`` forces CPU gloo (used by tests).
    - Standalone (no env): TP=1 on cuda if available else cpu, no process group.
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    force_cpu = os.environ.get("MINITP_BACKEND", "").lower() == "gloo"
    if world == 1:
        device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
        return ParallelContext(rank, local_rank, 1, 0, 1, device, None)

    if backend is None:
        backend = "gloo" if force_cpu or not torch.cuda.is_available() else "nccl"
    if backend == "gloo":
        # Windows torch builds lack libuv; classic TCPStore is fine for gloo.
        os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world,
        timeout=dt.timedelta(minutes=10),
    )
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    # This project is single-node, single TP group: world == tp group.
    return ParallelContext(rank, local_rank, world, rank, world, device, dist.group.WORLD)

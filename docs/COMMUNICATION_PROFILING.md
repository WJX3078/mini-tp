# Communication Profiling in mini-TP

How `minitp/distributed/collectives.py` measures collectives, and how to read
the numbers. Enable with `--profile-communication` (or `set_profiling(True)`);
when disabled the wrappers are a single boolean check around the raw
`torch.distributed` call.

## CUDA/NCCL execution is asynchronous

```python
t0 = time.perf_counter()
dist.all_reduce(t)          # enqueues an NCCL kernel, returns immediately
t1 = time.perf_counter()    # t1 - t0 is HOST LAUNCH time, not GPU time
```

`dist.all_reduce` (NCCL) enqueues work and returns before the GPU has executed
it. The v0.1 profiler reported exactly this wall-clock delta as "communication
time" — that is a measurement bug: it measures enqueue latency (µs scale) and
only accidentally includes GPU time when the launch queue happens to be full.

## What v0.2 measures

Per collective, exactly one enqueue, bracketed by:

| metric | how | meaning |
|---|---|---|
| `host_launch_ms` | `perf_counter` around the enqueue | CPU cost of issuing the op (Python + c10d + NCCL enqueue) |
| `gpu_elapsed_ms` | paired `torch.cuda.Event(enable_timing=True)` recorded on the current stream immediately before/after the enqueue, resolved later | device-side span of the collective on that stream |

Events are resolved in `drain_comm_stats()` with **one**
`torch.cuda.synchronize()` at readout — never inside the generation hot path,
so profiling does not fragment kernel pipelining. The record also carries
`op`, `bytes`, `dtype`, `shape`, `world_size`.

On CPU/Gloo the collective blocks until done, so `host_launch_ms` *is*
execution time and `gpu_elapsed_ms` is `null`.

Caveats: the event pair measures the span on the current stream. c10d/NCCL
synchronizes its internal stream with the current stream around the kernel,
so the span captures the collective's device work, but if other work is
queued on the stream the *gap* before it starts is included. In mini-TP's
decode loop the stream is otherwise idle-ish at collectives, which is the
case that matters (latency-bound AllReduce).

## Collective data volume (theory)

For a ring AllReduce of a tensor with `s` bytes across `n` ranks, each rank
sends and receives `2·(n-1)/n · s` bytes; the algorithmic time floor at bus
bandwidth `B` is `t ≈ 2(n-1)/n · s / B`. AllGather and ReduceScatter each move
`(n-1)/n · s` per rank.

Derived metrics (reported by the benchmark):

- **algbw** = `s / gpu_elapsed` — application-level bandwidth of the op.
- **busbw** = `algbw × 2(n-1)/n` for AllReduce (`× (n-1)/n` for AG/RS) —
  the number comparable across collectives and against link bandwidth
  (PCIe ~16–32 GB/direction, NVLink much higher).

## Reading mini-TP's pattern

Replicated-residual TP does per layer per forward: 1 AllReduce after
`o_proj`, 1 after `down_proj` (each `B·T·896·dtype_bytes`), plus one
embedding AllReduce per forward and one O(tp) AllGather per greedy decode
step. So decode issues `2 × layers + 2` collectives per token — all small and
latency-bound, which is why `host_launch_ms` matters as much as `gpu_elapsed_ms`
on PCIe-attached consumer GPUs.

TP=2/TP=4 GPU numbers: **UNVERIFIED — requires real multi-GPU hardware.**
Run `scripts/run_multi_gpu_bench.sh` to produce them; the profiler emits the
same records on NCCL.

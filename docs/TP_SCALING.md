# TP Scaling Framework & Measurement Plan

How mini-TP measures tensor-parallel scaling, what is measured so far, and
what remains UNVERIFIED. The matrix runner is `scripts/run_multi_gpu_bench.sh`;
the collective floor it builds on is `minitp/bench/collective_benchmark.py`.

## Metrics per (TP, workload) cell

| metric | definition |
|---|---|
| TTFT | prefill + first token selection (distributed argmax included) |
| TPOT | mean latency per output token after the first (forward + selection) |
| decode tok/s | 1 / TPOT per rank batch |
| prefill tok/s | prompt tokens / prefill time |
| peak memory/rank | `max_memory_allocated` after load reset (compute phase) |
| max-rank latency | MAX across ranks via tensor-collective aggregation — the user-visible step latency is gated by the slowest rank |
| rank skew | max − min across ranks |
| collective time/token | sum of `collective_gpu_ms` per decode step (CUDA events) |
| communication % | collective_gpu_ms / wall step time |
| scaling efficiency | TPOT(TP=1) / (TP × TPOT(TP)) — how much of the ideal per-rank speedup survives communication |

## Matrix

Models: `Qwen/Qwen2.5-0.5B` (and a larger model when hardware allows).
Workloads: `B1 P128 O128`, `B1 P512 O128`, `B1 P2048 O128`, `B4 P512 O128`.
TP sizes: 1, 2, 4.

## Measured so far (single-GPU host)

- **TP=1**: full TTFT/TPOT/memory results in README (RTX 4060 Laptop GPU).
- **Collective floor, CPU/Gloo TP=2 over loopback TCP** (real measurements,
  `results/collective_gloo_tp2.json`): an AllReduce of 4 KB costs **~563 µs**
  and 64 MB sustains only **~0.11 GB/s busbw**. TCP loopback is
  latency-bound and bandwidth-starved — this is the quantitative argument for
  why (a) mini-TP's decode does only 2 AllReduces/layer and (b) real TP
  scaling needs NCCL over NVLink/PCIe, not sockets.
- The Qwen decode AllReduce payload is tiny — `B·T·hidden·dtype = 1×1×896×2 =
  1792 B` — squarely latency-bound; per-message latency, not bandwidth, is
  what TP=2 decode pays 48× per token.

## UNVERIFIED — requires >=2 CUDA GPUs

- TP=2 / TP=4 TTFT, TPOT, memory/rank, rank skew, communication %, scaling
  efficiency on NCCL. No TP>1 GPU number may be claimed from this host; the
  framework and scripts produce them as-is on multi-GPU hardware.
- The zero-copy GQA expand path (kv_heads_local == 1) and the
  `all_gather_into_tensor` dim-0 fast path are correctness-tested on Gloo and
  feature-gated to NCCL, but their **performance** on real GPUs is unmeasured.

# mini-TP — From-Scratch Tensor Parallel LLM Inference Runtime

An **educational, from-scratch tensor-parallel inference runtime**: the Transformer
sharding math, collective communication, weight loading, and KV-cache decode are all
implemented directly on `torch.distributed` (NCCL/Gloo). HuggingFace `transformers` is
used only for config/tokenizer download and as a correctness reference — no Megatron,
DeepSpeed, vLLM, accelerate, or `device_map`.

```bash
torchrun --standalone --nproc-per-node=2 examples/generate.py \
    --model Qwen/Qwen2.5-0.5B --tp-size 2
```

Target model: **Qwen2 / Qwen2.5** (validated on `Qwen/Qwen2.5-0.5B`: 24 layers,
hidden 896, 14 Q heads + 2 KV heads GQA, SwiGLU, RoPE, tied embeddings, vocab 151936).

## Architecture (30-second version)

Replicated-residual TP: after each RowParallel AllReduce every rank holds the identical
full hidden state; only weights and intermediate activations are sharded.

```text
Attention (per layer):
  x ─ RMSNorm ─ QKV ColumnParallel (no comm, heads sharded)
    ─ local GQA attention on local heads (no comm)
    ─ O RowParallel ─ AllReduce ─ (+ residual)

MLP (per layer):
  x ─ RMSNorm ─ gate/up ColumnParallel (intermediate stays sharded, no comm)
    ─ local SwiGLU ─ down RowParallel ─ AllReduce ─ (+ residual)

Embedding: vocab-row sharded, masked local lookup + AllReduce
LM Head:   vocab-column sharded; greedy sampling via distributed argmax
           (AllGather O(tp) per step, never O(vocab))
→ 2 AllReduces per layer per forward (48/token for 24 layers)
```

## Key features

- From-scratch **ColumnParallelLinear / RowParallelLinear** (correct PyTorch `[out, in]`
  weight-axis slicing; bias added exactly once after reduction)
- **Qwen2 GQA tensor parallelism** with per-rank local Q/KV heads and KV-head
  replication when `num_kv_heads < tp_size`
- **Sharded KV cache** — each rank stores only its own KV heads
- **VocabParallelEmbedding / VocabParallelLMHead** with uneven-vocab support
- **Distributed greedy argmax** (O(tp) AllGather per decode step, tie-broken to match
  `torch.argmax`)
- **TP weight loader** with per-tensor shard mapping (see
  [docs/QWEN_WEIGHT_MAPPING.md](docs/QWEN_WEIGHT_MAPPING.md))
- Instrumented collectives: calls / bytes / milliseconds per AllReduce and AllGather
- CPU **Gloo TP=2** test suite so correctness is verifiable without multiple GPUs

## Correctness (measured, not aspirational)

Verified against HuggingFace `Qwen2ForCausalLM` on the real `Qwen2.5-0.5B` checkpoint:

| Check | TP=1 | TP=2 |
|---|---|---|
| Column/Row parallel primitives (fp32, vs `F.linear`) | PASS | PASS (CPU Gloo) |
| Vocab embedding / LM head / distributed argmax | PASS | PASS (CPU Gloo) |
| Tiny-model logits vs HF (fp32) | PASS | PASS (CPU Gloo) |
| Real-model bf16 logits vs HF | PASS (≤0.5 abs on ~30-magnitude logits) | — |
| Real-model greedy tokens vs HF (fp32) | PASS (32/32 tokens, CUDA) | PASS (16/16 tokens, CPU Gloo) |

The bf16 greedy comparison against HF CUDA is intentionally not asserted: HF's own
bf16 CUDA greedy output is unstable (repetition loops); in fp32 both models agree
token-for-token, and the TP=1 CPU bf16 run also agrees with HF CPU.

## Performance engineering (v0.2)

v0.2 is a measurement-driven optimization pass over the v0.1 runtime
([docs/PERFORMANCE_AUDIT.md](docs/PERFORMANCE_AUDIT.md)): profiling showed decode
was **kernel-launch-bound** (~6,565 CUDA kernels per token, GEMMs only ~15 % of
GPU time), so the optimizations target launch count and host overhead — with
correctness gates on every change (logits + fp32 greedy token equality vs HF):

- **Fused QKV / fused gate+up** — per-rank shards packed into one GEMM at load
  time (7 → 4 GEMMs per layer per token; split is a zero-copy view)
- **RoPE cache** — cos/sin tables precomputed once (fp32) instead of rebuilt in
  all 24 layers on every forward
- **Allocation-free decode loop** — preallocated output buffer + positions,
  and a fixed-length benchmark mode with **zero GPU→CPU syncs per token**
  (early-stop EOS checking remains opt-in for interactive use)
- **Correct async communication profiler** — paired CUDA events split
  `host_launch_ms` vs `gpu_elapsed_ms` ([docs/COMMUNICATION_PROFILING.md](docs/COMMUNICATION_PROFILING.md))
- **Phase-separated benchmark** ([docs/BENCHMARK.md](docs/BENCHMARK.md)) —
  prefill / per-token decode p50-p99 / e2e / load measured independently

Microbenchmarks (this GPU, bf16, old vs new per hot-path op):

| op | old | new | speedup |
|---|---|---|---|
| RoPE apply (per layer) | 0.581 ms | 0.393 ms | 1.5× |
| QKV projection (per layer) | 0.116 ms | 0.051 ms | 2.3× |
| gate+up projection (per layer) | 0.062 ms | 0.052 ms | 1.2× |
| decode bookkeeping (per 256 tok, incl. v0.1's per-token sync) | 21.5 ms | 11.9 ms | 1.8× |

## Benchmark

**Measured on 1× consumer GPU (laptop, Windows, PyTorch 2.6, bf16, batch 1),
schema `minitp.bench/2` — decode timed per token, fixed-length, sync-free.**
TP=2/TP=4 numbers: `UNVERIFIED — requires >=2 CUDA GPUs`; run
`scripts/run_multi_gpu_bench.sh` on a multi-GPU host to produce them.

| Workload (prompt+gen) | TP | prefill tok/s | decode ms/token (mean) | decode tok/s | Peak mem/GPU (GiB) |
|---|---|---|---|---|---|
| 128 + 32 | 1 | 2,196 | 55.7 | 17.9 | 0.99 |
| 512 + 128 | 1 | 7,830 | 55.7 | 17.9 | 1.10 |
| 2048 + 32 | 1 | 18,147 | 55.1 | 18.2 | 1.55 |

Decode improved from **~76.8 ms/token (v0.1, e2e-derived) to ~55.7 ms/token
(v0.2, phase-measured)** — ~28 % faster; microbenchmarks attribute ≈6–7 ms to
the three fused/cached ops and removed per-token sync, the remainder to reduced
allocator churn and host dispatch. Memory model: weights ≈ 0.5B × 2 bytes / TP
= **0.93 GiB at TP=1, 0.47 GiB at TP=2** per rank; measured peaks include
activations, KV cache, and allocator overhead. At TP=2 the runtime does
2 AllReduces × 24 layers × `B·T·896·2` bytes per forward, which on PCIe
typically makes TP=2 slower than TP=1 for a 0.5B model — the profiler and
theoretical model exist to quantify exactly this trade-off.

## Usage

```bash
pip install -e ".[dev]"

# TP=1
python -m minitp.generate --model Qwen/Qwen2.5-0.5B --max-new-tokens 64

# TP=2 (NCCL)
torchrun --standalone --nproc-per-node=2 -m minitp.generate \
    --model Qwen/Qwen2.5-0.5B --max-new-tokens 64

# benchmark + communication profile
torchrun --standalone --nproc-per-node=2 -m minitp.bench.benchmark \
    --prompt-len 512 --new-tokens 128 --profile-communication

# tests (CPU Gloo TP=2 included; multi-GPU tests are marked and excluded by default)
pytest tests/ -m "not multi_gpu and not model"
```

## Project layout

```text
minitp/
  distributed/   context (process group/TP topology), instrumented collectives
  parallel/      Column/Row parallel linear, vocab-parallel embedding + LM head
  config.py      lightweight Qwen2 config + TP divisibility validation
  rope.py, attention.py, mlp.py, layer.py   TP Qwen2 modules
  kv_cache.py    per-rank contiguous KV cache
  weight_loader.py  HF checkpoint → per-rank shards
  generation.py  prefill + KV-cache decode + distributed greedy sampling
  generate.py    CLI          bench/  benchmark + communication/memory tooling
docs/            DESIGN, QWEN_WEIGHT_MAPPING, REVIEW
```

## Scope & known limitations

This is a **from-scratch educational distributed inference runtime**, not
production-ready. Single-node TP only; no pipeline/MoE/ZeRO/FSDP, no training, no
paged KV / continuous batching / speculative decoding, no quantization, no HTTP
serving (see the companion mini-vLLM project for serving/scheduling), no CUDA
graphs. P0 checkpoint loading materializes the full CPU state_dict per rank
(selective safetensors loading is a P1). `kv_heads < tp_size` replication is
implemented but untested against real checkpoints. All multi-GPU benchmark numbers
are marked `UNVERIFIED` until executed on real hardware.

See [docs/DESIGN.md](docs/DESIGN.md) for the full design, including the sharding
math, GQA head mapping, communication model, and correctness strategy.

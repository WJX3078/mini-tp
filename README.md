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

## Benchmark

**Environment for the numbers below: 1× consumer GPU (single-GPU laptop), PyTorch 2.6,
bf16, batch 1.** TP=2 numbers: `UNVERIFIED — requires >=2 CUDA GPUs`; run
`scripts/run_multi_gpu_bench.sh` on a multi-GPU host to fill them in.

| Workload (prompt+gen) | TP | E2E (s) | ms/token | tok/s (output) | Peak mem/GPU (GiB) |
|---|---|---|---|---|---|
| 128 + 32 | 1 | 2.46 | 76.8 | 13.0 | 0.98 |
| 512 + 128 | 1 | 7.88 | 61.6 | 16.2 | 1.09 |
| 2048 + 32 | 1 | 2.19 | 68.6 | 14.6 | 1.55 |

Memory model: weights ≈ 0.5B × 2 bytes / TP = **0.93 GiB at TP=1, 0.47 GiB at TP=2**
(per rank). Measured 0.98 GiB at TP=1 — the ~0.05 GiB delta is activations, KV cache,
norm weights, and allocator overhead. Communication profile: TP=1 does zero
collectives; at TP=2 the runtime does 2 AllReduces × 24 layers × `B·T·896·2` bytes
per forward, which on PCIe-attached GPUs typically makes TP=2 **slower** than TP=1
for a 0.5B model — the theoretical communication model and profiler exist to
quantify exactly this trade-off.

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
docs/            DESIGN, QWEN_WEIGHT_MAPPING, REVIEW, INTERVIEW_GUIDE, RESUME_BULLETS
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

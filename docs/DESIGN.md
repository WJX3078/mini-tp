# mini-TP Design: Tensor Parallel LLM Inference from Scratch

> Status: v0.1 design for Qwen2/Qwen2.5 inference. All tensor names/shapes below were read
> from the real `Qwen/Qwen2.5-0.5B` safetensors header, not from memory.

## 1. Motivation

Tensor Parallelism (TP) splits a single model's weights and computation across GPUs in the
same process group, so that (a) models too large for one GPU fit, (b) per-GPU memory drops
~1/TP for the sharded weights, and (c) latency-bound inference can use more silicon. The
goal of this project is to implement the sharding math, the collective communication, and
the weight loading **ourselves** on top of raw `torch.distributed`, and to prove
mathematical equivalence against the HuggingFace reference.

This is deliberately *not* a wrapper around Megatron/DeepSpeed/vLLM/accelerate. Transformers
is used only for config/tokenizer download and as a correctness reference.

## 2. Tensor Parallel Theory

For a linear layer `Y = XW` (`W ∈ R^{in×out}` mathematically; PyTorch stores `weight` as
`[out_features, in_features]` and computes `X @ weight.T + bias`). There are two ways to
shard:

- **Column Parallel** (Megatron terminology): shard the *output* dimension. Each rank holds
  `W_r = W[:, r*slice:(r+1)*slice]` (PyTorch: rows of `weight`), computes
  `Y_r = X W_r`, producing shard `r` of `Y`. No communication on the forward path; the
  output stays sharded unless gathered.
- **Row Parallel**: shard the *input* dimension. The input must already be sharded as
  `X = [X_0 | X_1]`; each rank computes `Y_r = X_r W_r` (a partial sum of the full output)
  and an **AllReduce** (or ReduceScatter) produces `Y = Σ_r Y_r`.

### Why Column→Row composition minimizes communication

A Transformer MLP is `down(gelu-ish(gate(X) * up(X)))`. Sharding `gate/up` by column keeps
the intermediate activation *sharded* — each rank does local elementwise work on its own
1/TP slice with zero communication. Only `down` (Row Parallel) needs one AllReduce at the
end. The alternative — ColumnParallel → AllGather → re-shard — costs an extra collective
per layer for no mathematical benefit. The rule: **communication happens once per
sublayer, where partial sums must become full activations.**

## 3. Qwen2.5-0.5B: real config and tensors

From `Qwen/Qwen2.5-0.5B/config.json`:

| field | value |
|---|---|
| hidden_size | 896 |
| num_hidden_layers | 24 |
| num_attention_heads | 14 (head_dim = 64) |
| num_key_value_heads | 2 → **GQA** (group size 7) |
| intermediate_size | 4864 |
| vocab_size | 151936 |
| rms_norm_eps | 1e-6 |
| rope_theta | 1e6 |
| tie_word_embeddings | **true** (no separate `lm_head.weight`) |

From the safetensors header (dtype BF16, one file):

```text
model.embed_tokens.weight                     [151936, 896]
model.layers.{i}.input_layernorm.weight       [896]
model.layers.{i}.post_attention_layernorm.weight [896]
model.layers.{i}.self_attn.q_proj.weight      [896, 896]     (+ bias [896])
model.layers.{i}.self_attn.k_proj.weight      [128, 896]     (+ bias [128])
model.layers.{i}.self_attn.v_proj.weight      [128, 896]     (+ bias [128])
model.layers.{i}.self_attn.o_proj.weight      [896, 896]     (no bias)
model.layers.{i}.mlp.gate_proj.weight         [4864, 896]    (no bias)
model.layers.{i}.mlp.up_proj.weight           [4864, 896]    (no bias)
model.layers.{i}.mlp.down_proj.weight         [896, 4864]    (no bias)
model.norm.weight                             [896]
# lm_head is absent: tied to embed_tokens
```

Qwen2/2.5 use fused but *separate* q/k/v projections (not one fused QKV tensor), each with
bias. RoPE is applied to Q and K after projection; attention is full (no sliding window for
this config); MLP activation is SiLU-gated (SwiGLU): `down(silu(gate(x)) * up(x))`.

### TP=2 divisibility check (all must pass)

| tensor | dim | TP=2 shard |
|---|---|---|
| hidden 896 | ÷2 | 448 (Q/o_proj columns/rows) |
| heads 14 | ÷2 | 7 Q heads/rank |
| kv heads 2 | ÷2 | 1 KV head/rank |
| intermediate 4864 | ÷2 | 2432 (gate/up out, down in) |
| vocab 151936 | ÷2 | 75968 (embedding rows, LM head cols) |

All divisible — no padding needed for Qwen2.5-0.5B. The code still handles
`vocab % tp != 0` with **uneven partitions** (contiguous chunks via `r*size <= i < (r+1)*size`
arithmetic, see §7), and hard-fails with an explanatory `ValueError` for undivisible
hidden/heads/intermediate (§10) rather than producing silently wrong results.

## 4. Column Parallel Linear

```python
class ColumnParallelLinear(nn.Module):
    # weight: [out_local, in]  (rows of the PyTorch weight = columns of math W)
    def __init__(self, in_size, out_size, ctx, bias=True, gather_output=False)
```

Forward: `Y_r = X W_r^T + b_r`. With `gather_output=True`, AllGather along the last dim
(debug/correctness mode; the default inference path keeps the output sharded). QKV and
gate/up projections use `gather_output=False` because their consumers (attention heads,
SwiGLU, RowParallel down/o_proj) operate naturally on shards.

**Bias**: also column-sharded (`[out_local]`), added locally per rank — each rank owns the
bias entries for its output slice, so the sum over ranks is exact. No double-counting.

## 5. Row Parallel Linear

```python
class RowParallelLinear(nn.Module):
    # weight: [out, in_local]
    def __init__(self, in_size, out_size, ctx, bias=True, input_is_parallel=True)
```

Forward: partial sum `Y_r = X_r W_r^T` → `all_reduce` → add **bias once, after the
reduction** (the classic bug: adding full bias per rank before AllReduce multiplies bias
by TP). Covered by a dedicated unit test. `reduce_output=False` skips the AllReduce (used
when composing two RowParallel ops back-to-back; not used in Qwen, kept for generality).
`input_is_parallel=False` first splits the (replicated) input along the last dim — used in
layer-level tests and for the "TP=1-equivalent full input" case.

## 6. Attention Sharding (incl. GQA)

Attention is inherently head-parallel: heads are independent until the output projection.
Qwen2.5-0.5B: 14 Q heads, 2 KV heads (GQA, each KV head serves 7 Q heads).

- `q_proj`: ColumnParallel over 14 heads → 7 local Q heads (448 dims).
- `k_proj`/`v_proj`: ColumnParallel over 2 KV heads → 1 local KV head (64 dims each).
  If `num_kv_heads < tp_size`, we **replicate** each KV head across
  `tp_size / num_kv_heads` ranks (standard Megatron behavior) rather than fail — the
  mapping is exact and stated in the error-free path; if `tp_size % num_kv_heads != 0`
  and neither divisibility holds, raise `ValueError`.
- **Head→rank mapping**: rank `r` owns Q heads `[r*7, (r+1)*7)` and KV head `r`. Because HF
  lays out k_proj rows as `[kv_head0_dims, kv_head1_dims]`, slicing rows contiguously by
  rank gives each rank exactly the KV head matching its contiguous Q-head block. This is
  the key GQA correctness property: **contiguous blocks of Q heads must map to the same
  contiguous block of KV heads** — true here because group size (7) divides the per-rank Q
  head count.
- Local attention via `F.scaled_dot_product_attention`, no communication.
- `o_proj`: RowParallel over the 448-dim local attention output → **AllReduce #1**.
- Q/K/V biases: column-sharded, local. RoPE applied to local Q/K (position-wise, no comm).

## 7. Embedding / LM Head (Vocab Parallel)

- **VocabParallelEmbedding**: embedding table `[vocab, hidden]` row-sharded by vocab.
  Each rank looks up only tokens in its `[vocab_start, vocab_end)` range; out-of-shard
  positions contribute zeros; **AllReduce** sums across ranks into the full embedding.
  Uneven vocab handled by contiguous arithmetic ranges (no padding).
- **VocabParallelLMHead**: hidden → vocab, column-sharded over vocab. Each rank produces
  `[B, T, vocab_local]` logits for its slice. Two modes:
  - `gather_logits=True` (debug/correctness): AllGather full vocab logits.
  - inference: **distributed greedy argmax** — each rank computes
    `(local_max, local_argmax_token_id)` and AllGather`s just those `2` values per row
    (O(tp) bytes instead of O(vocab)); global winner = max value, ties broken by smaller
    token id to match `torch.argmax` semantics. With tied embeddings, the LM head weight
    is the embedding shard (transposed use), loaded once.

## 8. Communication Pattern per Decoder Layer

Replicated-residual architecture: after each AllReduce, **every rank holds the identical
full hidden state**; only the weights/intermediates are sharded. Hence:

- RMSNorm: elementwise on full hidden — no communication.
- Residual add: full hidden — no communication.
- Attention block: QKV (col, no comm) → local heads → o_proj (row) → **AllReduce**.
- MLP block: gate/up (col) → SwiGLU local → down (row) → **AllReduce**.

→ **2 AllReduces per layer per forward pass** (24 layers → 48 AllReduces per token
step), each moving `B × T × 896 × dtype_bytes` per rank. Greedy sampling adds one
O(tp) AllGather per step if using distributed argmax. Prefill of `T` tokens amortizes the
per-layer AllReduce over T tokens; decode does one layer-AllReduce per token — which is
why TP helps decode latency only when interconnect bandwidth is good (§14).

## 9. KV Cache

Per-rank contiguous cache, one buffer per layer:

```text
k_cache: [batch, kv_heads_local, max_seq_len, head_dim]   (fp same dtype as model)
v_cache: [batch, kv_heads_local, max_seq_len, head_dim]
```

Only local KV heads are stored — with GQA/TP=2 that is 1 of 2 KV heads, i.e. the TP
memory benefit extends to the cache. Prefill writes positions `[0, T)`, decode writes
position `T + step`. A `seq_len` cursor tracks fill; attention uses an index slice, so
SDPA never attends to unfilled slots. No paging/block tables (out of scope).

## 10. Shape Validation

`ModelConfig.from_hf(config, tp_size)` asserts, with values in the message:

- `hidden_size % tp_size == 0`
- `num_attention_heads % tp_size == 0`
- `intermediate_size % tp_size == 0`
- `tp_size % num_kv_heads == 0` (KV replication) **or** `num_kv_heads % tp_size == 0`
  (KV sharding); otherwise `ValueError` — never silent wrong results.

## 11. Weight Loader

`minitp/weight_loader.py` maps HF names → local shards, **slicing on the correct
PyTorch axis** (`weight` is `[out, in]`):

| HF tensor | destination | partition dim | TP=2 local shape |
|---|---|---|---|
| `model.embed_tokens.weight` | embedding.weight | 0 (vocab) | [75968, 896] |
| `q_proj.weight` | q_proj.weight | 0 (out) | [448, 896] |
| `q_proj.bias` | q_proj.bias | 0 | [448] |
| `k_proj.weight` | k_proj.weight | 0 (out, kv-head blocks) | [64, 896] |
| `k_proj.bias` | k_proj.bias | 0 | [64] |
| `v_proj.*` | v_proj.* | 0 | same as k |
| `o_proj.weight` | o_proj.weight | 1 (in) | [896, 448] |
| `gate_proj.weight` | gate_proj.weight | 0 | [2432, 896] |
| `up_proj.weight` | up_proj.weight | 0 | [2432, 896] |
| `down_proj.weight` | down_proj.weight | 1 (in) | [896, 2432] |
| `input_layernorm/post_attention_layernorm/norm.weight` | replicated | — | [896] |
| (tied) `model.embed_tokens.weight` | lm_head.weight | 0 (vocab) | [75968, 896] |

Loader contract: rank `r` slices `[r*n_local, (r+1)*n_local)`, copies into a
pre-allocated local parameter; GPU only ever receives the local shard. Since
v0.3 the default is the *selective* loader (`safe_open` rank-local slices, no
full state dict); `direct_gpu` (v0.4) constructs on the target device;
`legacy` full-state loading is kept for comparison. See docs/LOADER_PIPELINE.md.

## 12. Correctness Strategy

Three tiers, all with `torch.manual_seed`, `model.eval()`, no dropout:

1. **Primitive**: random `X/W/b`, reference `F.linear`; TP gathered output ≡ reference
   (`atol=rtol=1e-5` fp32). Runs on CPU with **Gloo TP=2 two-process tests** so a
   single-GPU dev machine can develop.
2. **Layer/Model**: tiny random Qwen config (2 layers, hidden 64, 4 Q heads, 2 KV heads)
   with identical weights replicated vs sharded; compare against a non-TP reference
   implementation and against HF `Qwen2ForCausalLM` on the tiny config (fp32 logits).
3. **End-to-end**: real `Qwen2.5-0.5B`: TP=1 GPU and TP=2 (CPU-Gloo simulation or multi-GPU
   when available) vs HF logits (`atol` scaled per dtype: fp32 1e-4, bf16 ~1e-2 abs on
   logits) and **greedy token-ID equality** over ≥64 tokens on multiple prompts. Token
   ties from float nondeterminism must be investigated, not ignored.

## 13. Benchmark Strategy

`minitp/bench/`: batch=1, prompt lengths 128/512/2048, outputs 32/128. Metrics: prefill
latency, decode ms/token, E2E, tok/s, peak `max_memory_allocated/reserved` per rank,
collective calls/bytes/seconds via the collectives instrumentation (off by default,
`--profile-communication`). CUDA-event timing after warmup, loader excluded. All numbers
in README come from real runs; anything not executed on ≥2 GPUs is marked
`UNVERIFIED — requires >=2 CUDA GPUs`.

Theoretical comm model: 48 AllReduces/token of `B·896·dtype_bytes` (ring AllReduce moves
`2·(world-1)/world · size` bytes per rank) — compared against measured comm % to explain
why TP=2 on a 0.5B model may be *slower* than TP=1 (compute per rank too small to hide
latency; PCIe lacks NVLink bandwidth).

## 14. Limitations (explicit)

Single-node TP only; no PP/EP/MoE/ZeRO/FSDP; training not supported; contiguous (non-paged)
KV cache; no continuous batching, quantization, CUDA graphs, or serving; checkpoint loading
materializes the full CPU state_dict only in the legacy loader; the default
selective loader reads rank-local slices (docs/LOADER_PIPELINE.md).
KV-replication mode (`kv_heads < tp`)
untested against real checkpoints (Qwen2.5-0.5B has 2 KV heads, so TP=2/4 shard evenly).

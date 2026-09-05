# mini-TP Review Notes (Adversarial Pass)

Reviewer stance: *assume a hidden TP correctness bug exists; find it.* This documents
the issues actually found during development and the checks performed on the final
code. All items are fixed and covered by tests unless marked otherwise.

## Bugs found and fixed during development

1. **bf16 token-id corruption in distributed argmax** — `torch.stack([bf16_values,
   int64_ids])` promotes to bf16, which cannot represent ids > 256 exactly; token
   12095 silently became 12096 and TP=2 generation produced garbage. Fix: cast both
   to fp32 before stacking. Found via real-model TP=2 generate test; impossible to
   hit at TP=1. (`minitp/parallel/embedding.py`)
2. **KV cache cursor advanced per layer** — each layer's attention advanced the shared
   cursor, so layer 1 wrote where layer 0 had just advanced past (with 24 layers,
   layer 23 wrote at position `prompt + 23`). Fix: layers write at the cursor; the
   model advances once per forward. (`minitp/attention.py`, `minitp/layer.py`)
3. **RowParallel bias semantics** — bias is added once *after* AllReduce. Regression
   test feeds zero input with all-ones bias and asserts output is exactly 1.0, not
   `tp_size`. (`tests/test_column_row_parallel.py`)
4. **GQA local expansion missing** — local Q heads (7) > local KV heads (1) requires
   `repeat_interleave` before SDPA; without it SDPA errors or (if shapes happened to
   match) would silently attend across wrong groups. (`minitp/attention.py`)
5. **Local Q/KV head contiguity** — KV-head slices must be contiguous blocks aligned
   with the rank's contiguous Q-head block; verified for shard and replication modes
   (`tests/test_weight_loader.py`).

## Checklist pass over final code

- **Shard dimensions**: Column → dim 0 of `[out, in]`; Row → dim 1; embedding/LM head →
  dim 0 (vocab). Unit tests build `arange` weights and compare exact contents per rank.
- **Collective ordering / deadlock**: every collective (AllReduce in RowParallel,
  AllGather in gather_output/distributed argmax, embedding AllReduce) is executed
  unconditionally on all ranks — no rank-dependent control flow in the model path.
  Decode selection (`distributed_argmax`) is itself a collective, keeping ranks in
  lockstep. Single-process TP=1 path calls no collectives at all.
- **Double/missing AllReduce**: AllReduce appears exactly twice per decoder layer
  (after o_proj, after down_proj) plus the embedding AllReduce once per model. The
  communication profiler cross-checks counts (`--profile-communication`).
- **Bias duplication**: ColumnParallel bias is sharded (sums exactly); RowParallel bias
  added post-reduce; o_proj/down_proj have no bias in Qwen2.
- **LM head vocab offsets**: `vocab_start` from the same `shard_range` arithmetic as
  the loader; tie-breaking matches `torch.argmax` (smallest id wins).
- **Embedding mask**: out-of-shard tokens clamp to index 0 and are zero-masked before
  the AllReduce, so no cross-shard contamination; uneven vocab (9 % 2) covered.
- **KV cache shapes**: `[B, kv_heads_local, S, head_dim]` per rank; TP=2 stores half
  the KV heads. Prefill-with-cache uses `tril(diagonal=cache_len)` masks (top-left
  causal misaligns with cache offsets — caught in review of the first draft).
- **Device/group consistency**: single process group (`WORLD`) used everywhere; the
  `ParallelContext` carries the group and device; tensors created on the context device.
- **Rank divergence**: only rank 0 prints; token selection results are identical on
  all ranks by construction (collective).
- **Weight loading**: loader writes into pre-allocated local parameters; GPU receives
  only local shards (CPU-side full state_dict is a documented P0 limitation).

## Known remaining gaps (documented, not bugs)

- `num_kv_heads < tp_size` replication path is unit-tested for shard contents but has
  no real-checkpoint end-to-end test (Qwen2.5-0.5B has 2 KV heads).
- P1 items not yet implemented: selective safetensors loading, async AllReduce /
  reduce-scatter experiments, sequence parallelism.
- bf16-vs-HF greedy equality is not asserted on CUDA (HF's own bf16 greedy is
  unstable); fp32 is the asserted path.

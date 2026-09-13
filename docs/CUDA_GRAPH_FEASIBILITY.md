# CUDA Graph Feasibility Study (v0.3)

Question: should mini-TP capture its decode step in a CUDA graph? Short
answer: **not in the current design** — the KV-cache cursor changes the
execution structure every token, and the honest fixes all trade away the
project's educational clarity for gains this runtime cannot realize on its
target hardware. Analysis below; no implementation was attempted beyond this
study.

## Why the decode step is not graph-safe today

CUDA graph replay requires a static execution structure: same kernels, same
shapes, same memory addresses. The v0.3 decode step violates this in one
central place:

- `KVCache.seq_len` grows every token, so `kv_cache.update` writes at
  `[:, :, seq_len : seq_len+t]` — a *different slice address* every step.
- The attention source sequence is `k[:, :, :seq_len+t]` — a *different
  length* every step, so SDPA sees a new shape every token and the prefill
  mask branch (`tril(diagonal=cache_len)`) would need a new mask tensor
  every step.
- The distributed argmax AllGather buffer is shape-stable, but the model
  forward around it is not.

## Options considered

| option | idea | memory | wasted FLOPs | complexity | verdict |
|---|---|---|---|---|---|
| A: graph per length | capture one graph per `seq_len` (up to max_seq_len) | max_seq_len graphs × full activation pool each — prohibitive | none | high | rejected |
| B: length buckets | bucket seq_len (e.g. ×32), pad K/V reads to bucket end | tens of graphs | attention over padded slots (masked) | high | rejected for v0.3 |
| C: fixed-capacity KV + mask | always attend over `[0, max_len)` with a mask derived from a *device-side* cursor; static shapes | one graph | attention cost ∝ max_seq_len for every token (e.g. 8192 instead of 128) | medium | rejected: wasted FLOPs scale with context, exactly wrong for a latency-focused runtime |
| D: capture shape-stable islands only | graph the RMSNorm/QKV/SwiGLU subgraphs, keep SDPA + collectives eager | small | none | medium | **feasible future work**, but on this runtime the islands already run as fused single kernels (fused QKV/gate-up GEMMs; RMSNorm opt-in `F.rms_norm`), so the capture win is mostly launch overhead of ~4 kernels per layer — expected single-digit % on datacenter GPUs, unmeasurable behind this laptop's dispatch noise |

## Additional blockers specific to mini-TP

1. **Collectives inside graphs**: NCCL supports graph capture of collectives
   only under strict conditions (same collective sequence, dedicated
   allocator behavior). The O(tp) argmax AllGather would have to be captured
   with static buffers — doable — but is untestable on this single-GPU host,
   so any implementation would ship UNVERIFIED.
2. **Windows/WDDM**: graph capture requires `cudaStreamCaptureModeGlobal`
   care and behaves differently under WDDM; the dev host cannot validate.
3. **Educational scope**: a graph-captured decode obfuscates exactly the
   sharding/communication mechanics the project exists to teach.

## Conclusion

Full decode graph (A–C) is wrong for this runtime; island capture (D) is the
only defensible path and is deferred until (a) multi-GPU hardware is
available to measure the collective-in-graph interaction and (b) the kernel
count per token drops further, making launch overhead the dominant remaining
term. Until then the measured evidence says the biggest wins already shipped:
fused GEMMs, RoPE cache, and sync-free decode.

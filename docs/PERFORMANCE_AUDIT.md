# mini-TP Performance Audit (v0.1 → v0.2)

> Audit performed before any optimization. Environment: 1× consumer laptop GPU
> (single-GPU), PyTorch 2.6.0+cu124, Python 3.10, Qwen2.5-0.5B bf16, TP=1.
> Multi-GPU TP=2/4 items are marked UNVERIFIED where hardware was unavailable.

## Measured baseline (torch.profiler, 10 decode steps, batch=1, T=1, bf16)

- **≈ 6,565 CUDA kernel launches per decode token** (65,650 over 10 steps).
- Wall-clock decode: ~60–80 ms/token (v0.1 README). With ~6.5k launches that is
  **~10 µs per kernel — the decode path is launch/overhead-bound, not FLOP-bound**.
- GEMM work (`linear`+`mm`+`addmm`) is only **~15 % of GPU time**. The rest:
  dtype casts (`to`/`_to_copy` ≈ 11 %), elementwise (`mul`/`pow`/`add`), view churn
  (`slice` 433 calls/token, `as_strided` 1,037/token, `transpose` 361/token), and
  `repeat_interleave` copies (48 calls/token).
- `aten::linear` = 169 calls/token — matches 7 GEMMs × 24 layers + embedding/lm_head.
- RMSNorm trace shows `pow` 97/token (2 per layer): each RMSNorm is ~7 kernels
  (float cast, pow, mean, rsqrt, mul, mul, cast back) — all on B×1×896 tensors.

**Conclusion: the biggest lever is reducing per-token kernel count and host
dispatch, not making GEMMs faster.** Compute floor for 0.5B bf16 is
~1 GB weights / ~150 GB/s ≈ 7 ms/token; v0.1 is ~10× above it.

## 1. Decode hot path (`minitp/layer.py`, `minitp/generation.py`)

Per token, per layer: 2 RMSNorm (~14 kernels), QKV 3×GEMM (+3 bias), RoPE
(~15 kernels: arange, outer, cat, cos, sin, chunk/cat, 4 mul, 2 add, casts),
2× `repeat_interleave` (GQA expand), SDPA, o_proj GEMM, AllReduce, 2 GEMMs +
silu + mul for MLP, down GEMM, AllReduce, 2 residual adds. Model level:
embedding mask path (~6 kernels + AllReduce), final norm, lm_head.
**Bottleneck: kernel count, not math.**
Risk of change: numerics drift — gate every change on logits/greedy-token tests.
Verification: profiler kernel count before/after; greedy-token equality vs HF (fp32).

## 2. Prefill hot path

Same ops but amortized over T tokens; GEMMs are efficient at T≥128. Wasted work:
RoPE cos/sin are **rebuilt for every one of the 24 layers** over the full T range;
positions tensor rebuilt per forward. Communication: 48 AllReduces of
B×T×896×dtype_bytes — expected to dominate prefill at TP≥2 on PCIe (UNVERIFIED).
Verification: prefill tokens/s before/after RoPE caching (modest, ~5–8 % expected).

## 3. Temporary tensor allocation

- `generation.py`: `out = torch.cat([out, tok], 1)` **allocates and copies the full
  sequence every token** (O(T²) bytes over a generation).
- `layer.py` forward: `torch.arange(...)` per forward when `positions=None`.
- RoPE: 6+ temporaries per layer per forward (fp32 cos/sin are the largest).
- RMSNorm: 2 full-hidden fp32 casts per layer per forward.
- GQA: `repeat_interleave` materializes k/v copies per layer (12.6 ms/token
  self-CUDA in profile — measurable).
Verification: allocator stats (`torch.cuda.memory_stats()["allocation"]` count) and
kernel count; microbench old vs new loop.

## 4. Python overhead

~150 `nn.Module.__call__` dispatches per token + `KVCache.update` slicing +
loop-level `torch.cat`/`finished` bookkeeping. CPU time per token is in the same
order as GPU time on this machine (profiler-inflated, but indicative). Fix is
fewer ops (fusions below), not C++/graphs (CUDA graphs out of scope).

## 5. GPU→CPU synchronization

- `generation.py`: `bool(finished.all())` **syncs the GPU every token** and
  prevents kernel-queue pipelining. Fix: `early_stop=False` (fixed-length) mode
  for benchmarking with zero syncs; keep eager early-stop as opt-in for
  interactive use. EOS handling in fixed-length mode: tokens continue to be
  generated after EOS and are trimmed/ignored by the caller.
- `benchmark.py` v0.1: `torch.cuda.synchronize()` only at window end — OK.
Verification: profile shows zero `cudaStreamSynchronize` per decode token in
fixed-length mode.

## 6. NCCL communication pattern

2 AllReduces/layer/forward + 1 embedding AllReduce/forward + O(tp) argmax
AllGather per decode step — matches DESIGN §8. All are small (decode: B×896×2 B)
and latency-bound. **Profiler defect (P0):** v0.1 measures collectives with
`perf_counter` around the enqueue — that is *host launch time*, not GPU
execution time (CUDA/NCCL are async). Fix: paired CUDA events + deferred drain,
host/device split; see docs/COMMUNICATION_PROFILING.md. UNVERIFIED on multi-GPU.

## 7. GEMM count

Decode per layer: 7 (q,k,v,o,gate,up,down) → **5 after fused QKV** → **4 after
fused gate+up**. Per token (24 layers): 170 → 122 → 98 (+2 model-level).
Prefill: same count, amortized. Expected: fewer launches + better GEMM
efficiency (qkv fused [896→1088] GEMM instead of 3 thin ones).

## 8. Kernel launch count (decode, per token, measured)

| source | calls/token | after v0.2 |
|---|---|---|
| total | ≈ 6,565 | target ≈ 4,000 |
| GEMM (`linear`) | 169 | 121 |
| `repeat_interleave` | 48 | 0 (expand views) |
| RoPE construction+apply | ≈ 360 | ≈ 144 (cached cos/sin) |
| RMSNorm | ≈ 336 | same unless `F.rms_norm` adopted (P2) |

## 9. KV cache memory behavior

Contiguous per-layer `[B, kv_local, S, head_dim]` preallocated at
`prompt+max_new` — allocation-free at steady state, writes are 2 slice-copies per
layer per token. No resize/copy-on-grow. Memory: 2 layers… 24×2×B×1×S×64×2 B ≈
6 KB/token/layer at TP=2 — negligible vs weights. No change needed (P2: could
skip zero-init via `torch.empty` since fill-cursor guarantees reads only after
writes; small win, kept for safety in v0.2).

## 10. Checkpoint load memory behavior (CPU)

`load_qwen2_tp` materializes: full bf16 safetensors (1 GB) + fp32-constructed
model (2 GB) + bf16 cast (1 GB) ≈ **4 GB CPU peak per rank** for a 1 GB
checkpoint. Improvements kept minimal in v0.2: construct model directly in
target dtype (halves peak) and free the state dict before H2D copy; full
selective safetensors loading stays P1-deferred. GPU only ever receives local
shards (unchanged).

## 11. Benchmark methodology (v0.1 defects → v0.2 fixes)

Defects: (a) `E2E/new_tokens` was reported as “decode ms/token” — pollutes
decode with prefill; (b) no prefill/decode separation; (c) no percentiles;
(d) load time inside the timed window for the first (warmup) iteration only —
but not reported at all; (e) memory reported once including load peaks.
Fixes (P0): separate phases with CUDA events, per-token decode distribution
(p50/p90/p99), `--batch-size`, structured JSON, load time reported separately.
See docs/BENCHMARK.md.

## 12. Communication measurement methodology

v0.1: perf_counter around enqueue (wrong semantics — see §6). v0.2: per-op
paired CUDA events, `host_launch_ms` vs `gpu_elapsed_ms` split, one deferred
synchronize at drain time; Gloo/CPU reports wall time with `gpu_elapsed=null`.
Ring model: AllReduce moves `2·(n-1)/n·s` bytes per rank; effective bus
bandwidth = `s·2(n-1)/n / gpu_time`. UNVERIFIED beyond TP=1 (no multi-GPU).

## Optimization plan (impact × risk)

| # | change | expected (TP=1 decode) | risk | gate |
|---|---|---|---|---|
| 1 | Fused QKV GEMM | −48 GEMM launches/token, better GEMM shape | shard packing bugs | fused-vs-3-proj tests, TP=1 + Gloo TP=2 + replication |
| 2 | Fused gate+up | −24 launches/token | packing order bugs | fused-vs-2-proj tests |
| 3 | RoPE cache | −200+ launches/token | cos/sin convention drift | cached-vs-reference RoPE (T=1/T>1/offset/fp32/bf16) + tiny-model logits |
| 4 | Prealloc decode buffers + no-sync fixed-length mode | removes O(T²) copies + 1 sync/token | EOS semantics change (fixed-length) | generation tests; greedy-token equality vs HF |
| 5 | GQA expand views (no copy when kv_local==1) | −48 copies/token | stride handling in SDPA | tiny-model logits TP=1/TP=2 |
| 6 | Comm profiler rework | correctness of measurement | none (optional path) | profiler unit tests |
| 7 | Benchmark rewrite | trustworthy numbers | none | JSON schema test |

Non-goals (unchanged): CUDA graphs, paged KV, continuous batching, speculative
decoding, torch.compile (fragile across versions; may revisit), quantization.

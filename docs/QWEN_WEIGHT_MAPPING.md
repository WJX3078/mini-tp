# Qwen2.5-0.5B TP Weight Mapping

Tensor names and shapes below were read from the actual safetensors header of
`Qwen/Qwen2.5-0.5B` (BF16). PyTorch stores Linear weights as `[out_features,
in_features]`, so "Column" means slicing dim 0 and "Row" means dim 1 of the
stored tensor. Local shapes shown for TP=2.

| HF tensor name | shape | TP destination | partition dim | local shape (TP=2) |
|---|---|---|---|---|
| `model.embed_tokens.weight` | [151936, 896] | `model.embed_tokens.weight` | 0 (vocab) | [75968, 896] |
| `model.layers.{i}.input_layernorm.weight` | [896] | same | replicated | [896] |
| `model.layers.{i}.post_attention_layernorm.weight` | [896] | same | replicated | [896] |
| `model.layers.{i}.self_attn.q_proj.weight` | [896, 896] | `self_attn.qkv_proj.weight` rows [0:448) | 0 (out) | packed |
| `model.layers.{i}.self_attn.q_proj.bias` | [896] | `self_attn.qkv_proj.bias` rows [0:448) | 0 | packed |
| `model.layers.{i}.self_attn.k_proj.weight` | [128, 896] | `self_attn.qkv_proj.weight` rows [448:512) | 0 (kv-head blocks) | packed |
| `model.layers.{i}.self_attn.k_proj.bias` | [128] | `self_attn.qkv_proj.bias` rows [448:512) | 0 | packed |
| `model.layers.{i}.self_attn.v_proj.weight` | [128, 896] | `self_attn.qkv_proj.weight` rows [512:576) | 0 (kv-head blocks) | packed |
| `model.layers.{i}.self_attn.v_proj.bias` | [128] | `self_attn.qkv_proj.bias` rows [512:576) | 0 | packed |
| `model.layers.{i}.self_attn.o_proj.weight` | [896, 896] | `self_attn.o_proj.weight` | 1 (in) | [896, 448] |
| `model.layers.{i}.mlp.gate_proj.weight` | [4864, 896] | `mlp.gate_up_proj.weight` rows [0:2432) | 0 (out) | packed |
| `model.layers.{i}.mlp.up_proj.weight` | [4864, 896] | `mlp.gate_up_proj.weight` rows [2432:4864) | 0 (out) | packed |
| `model.layers.{i}.mlp.down_proj.weight` | [896, 4864] | `mlp.down_proj.weight` | 1 (in) | [896, 2432] |
| `model.norm.weight` | [896] | `model.norm.weight` | replicated | [896] |
| (`lm_head` absent — tied to `embed_tokens`; the LM head reuses the embedding shard) | | `lm_head.weight` (shared param) | 0 (vocab) | shared |

Notes:
- v0.2 packs the per-rank q/k/v shards into ONE fused parameter at load time
  (`FusedQKVColumnParallelLinear`, 576 rows at TP=2 = 448 q + 64 k + 64 v), and
  gate/up into one fused parameter (4864 rows = 2432 gate + 2432 up). One GEMM
  per projection group per forward instead of three/two; the split afterwards
  is a zero-copy view. Packing lives in `weight_loader.pack_qkv/pack_gate_up`,
  shared with the tests.
- k/v are sliced as contiguous KV-head blocks: with 2 KV heads and TP=2, rank r
  gets exactly head r (rows [r*64, (r+1)*64)), matching its contiguous block of
  7 Q heads (GQA group size 7).
- q/k/v carry bias in Qwen2; o_proj/gate/up/down do not.
- Verified by tests: shard-content unit tests (`tests/test_weight_loader.py`)
  and end-to-end token equality against HF greedy generation.

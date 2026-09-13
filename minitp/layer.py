"""RMSNorm, TP decoder layer, and the full TP Qwen2 model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minitp.attention import TPQwen2Attention
from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.kv_cache import KVCache
from minitp.mlp import TPQwen2MLP
from minitp.parallel.embedding import VocabParallelEmbedding, VocabParallelLMHead
from minitp.rope import RotaryEmbedding


class RMSNorm(nn.Module):
    """Applied to the full (replicated) hidden state — after each RowParallel
    AllReduce every rank holds identical hidden, so no communication here."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        # "reference" (7-kernel hand-written) vs "functional" (F.rms_norm,
        # single fused kernel). Default stays reference until the v0.3
        # experiment proves functional is faster AND numerically equivalent.
        self.implementation = "reference"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.implementation == "functional" and hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class TPQwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, ctx: ParallelContext) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = TPQwen2Attention(cfg, ctx)
        self.mlp = TPQwen2MLP(cfg, ctx)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None,
        layer_idx: int,
        rotary: RotaryEmbedding | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions, kv_cache, layer_idx, rotary)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TPQwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, ctx: ParallelContext) -> None:
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.rotary: RotaryEmbedding | None = None  # lazily built on first forward
        self.use_rotary_cache = True  # ablation toggle
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size, ctx)
        self.model.layers = nn.ModuleList(
            TPQwen2DecoderLayer(cfg, ctx) for _ in range(cfg.num_hidden_layers)
        )
        self.model.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(cfg.hidden_size, cfg.vocab_size, ctx)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        positions: torch.Tensor | None = None,  # [T]
        kv_cache: KVCache | None = None,
        gather_logits: bool = False,
    ) -> torch.Tensor:
        b, t = input_ids.shape
        if positions is None:
            offset = kv_cache.seq_len if kv_cache is not None and t == 1 else 0
            positions = torch.arange(offset, offset + t, device=input_ids.device)
        x = self.model.embed_tokens(input_ids)
        rotary = None
        if self.use_rotary_cache:
            if self.rotary is None or self.rotary.cos_cache.device != x.device:
                # shared across layers; built once on the model's device
                self.rotary = RotaryEmbedding(
                    self.cfg.head_dim, self.cfg.rope_theta,
                    self.cfg.max_position_embeddings, x.device,
                )
            rotary = self.rotary
        for i, layer in enumerate(self.model.layers):
            x = layer(x, positions, kv_cache, i, rotary)
        if kv_cache is not None:
            kv_cache.advance(t)
        x = self.model.norm(x)
        if gather_logits:
            self.lm_head.gather_logits = True
            logits = self.lm_head(x)
            self.lm_head.gather_logits = False
            return logits
        return self.lm_head(x)

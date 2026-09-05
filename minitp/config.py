"""Lightweight Qwen2 model config with TP divisibility validation."""

from __future__ import annotations

from dataclasses import dataclass

from minitp.distributed.context import ParallelContext


def _rope_theta(hf_config) -> float:
    theta = getattr(hf_config, "rope_theta", None)
    if theta is None:  # transformers >= 5.0 moved it into rope_parameters
        theta = hf_config.rope_parameters["rope_theta"]
    return float(theta)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool = True

    @classmethod
    def from_hf(cls, hf_config, tp_size: int = 1) -> ModelConfig:
        cfg = cls(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=getattr(hf_config, "head_dim", None)
            or hf_config.hidden_size // hf_config.num_attention_heads,
            vocab_size=hf_config.vocab_size,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=_rope_theta(hf_config),
            max_position_embeddings=hf_config.max_position_embeddings,
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", True),
        )
        cfg.validate(tp_size)
        return cfg

    def validate(self, tp_size: int) -> None:
        checks = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
        }
        for name, value in checks.items():
            if value % tp_size != 0:
                raise ValueError(
                    f"{name}={value} not divisible by tp_size={tp_size}; "
                    "tensor-parallel sharding would be incorrect"
                )
        # KV heads must either shard evenly across ranks (kv % tp == 0) or be
        # replicated evenly (tp % kv == 0). Anything else => wrong results.
        if self.num_key_value_heads % tp_size != 0 and tp_size % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} incompatible with "
                f"tp_size={tp_size}: need kv%tp==0 (shard) or tp%kv==0 (replicate)"
            )

    def local_heads(self, tp_size: int) -> tuple[int, int]:
        """(q_heads_per_rank, kv_heads_per_rank) after TP."""
        return self.num_attention_heads // tp_size, max(1, self.num_key_value_heads // tp_size)

    def kv_replication(self, tp_size: int) -> int:
        """How many ranks share each KV head (>1 means replication)."""
        if self.num_key_value_heads % tp_size == 0:
            return 1
        return tp_size // self.num_key_value_heads


def tiny_config(tp_size: int = 1) -> ModelConfig:
    """Tiny random Qwen2-ish config for fast layer-level tests."""
    cfg = ModelConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    cfg.validate(tp_size)
    return cfg


def context_for(tp_size: int, device: str = "cpu") -> ParallelContext:
    """Fake single-process context (tp_rank 0 only) for TP=1 tests."""
    import torch

    return ParallelContext(
        global_rank=0,
        local_rank=0,
        world_size=1,
        tp_rank=0,
        tp_size=tp_size if tp_size == 1 else 1,
        device=torch.device(device),
        process_group=None,
    )

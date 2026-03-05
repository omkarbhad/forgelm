"""
NanoLLM — Model Configuration
Matches LLaMA-3 architecture with configurable sizes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NanoLLMConfig:
    """Configuration for NanoLLM transformer model."""

    # Model architecture
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4          # GQA: fewer KV heads than query heads
    ffn_dim: Optional[int] = None  # Auto-computed if None: 8/3 * d_model
    max_seq_length: int = 2048
    rope_theta: float = 500000.0  # RoPE base frequency (LLaMA-3 uses 500K)

    # Training
    dropout: float = 0.0          # LLaMA uses 0 dropout
    bias: bool = False            # No bias in linear layers (LLaMA style)
    norm_eps: float = 1e-5        # RMSNorm epsilon

    # Precision
    dtype: str = "bfloat16"       # bf16 for Ampere+, fp16 for older

    def __post_init__(self):
        if self.ffn_dim is None:
            # LLaMA-3 formula: 8/3 * d_model, rounded to nearest multiple of 256
            raw = int(8 / 3 * self.d_model)
            self.ffn_dim = ((raw + 255) // 256) * 256

        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"


# Preset configurations
CONFIGS = {
    "tiny": NanoLLMConfig(
        d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
        max_seq_length=512, vocab_size=32000,
    ),
    "small": NanoLLMConfig(
        d_model=768, n_layers=12, n_heads=12, n_kv_heads=4,
        max_seq_length=2048, vocab_size=32000,
    ),
    "base": NanoLLMConfig(
        d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4,
        max_seq_length=2048, vocab_size=32000,
    ),
}

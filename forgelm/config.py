"""Model configuration and size presets."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ForgeConfig:
    vocab_size: int = 1024
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2          # grouped-query attention: fewer KV heads than query heads
    ffn_dim: Optional[int] = None  # default: 8/3 * d_model, rounded up to a multiple of 64
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = (int(8 / 3 * self.d_model) + 63) // 64 * 64
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


# vocab_size is overwritten with the real tokenizer size at training time.
PRESETS = {
    "tiny": dict(d_model=192, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=256),
    "small": dict(d_model=384, n_layers=8, n_heads=6, n_kv_heads=2, max_seq_len=512),
    "base": dict(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),
}

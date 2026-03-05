"""
NanoLLM — Grouped Query Attention with RoPE and KV Cache

Implements:
1. Grouped Query Attention (GQA) — fewer KV heads than query heads
2. Rotary Position Embeddings (RoPE) — relative position encoding
3. KV Cache — efficient autoregressive generation
4. FlashAttention-2 compatible — via scaled_dot_product_attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import NanoLLMConfig


def precompute_rope_frequencies(dim: int, max_seq_length: int, theta: float = 500000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin for Rotary Position Embeddings.

    RoPE applies a rotation matrix to query and key vectors based on their position.
    This gives the model relative position awareness without learned embeddings.

    Args:
        dim: Head dimension (must be even)
        max_seq_length: Maximum sequence length
        theta: Base frequency (LLaMA-3 uses 500000)

    Returns:
        cos, sin tensors of shape (max_seq_length, dim)
    """
    assert dim % 2 == 0, "Head dimension must be even for RoPE"

    # Compute frequencies: theta_i = 1 / (theta^(2i/dim)) for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Compute position * frequency for all positions
    positions = torch.arange(max_seq_length).float()
    angles = torch.outer(positions, freqs)  # (seq_len, dim/2)

    # Duplicate for full dimension: each pair shares the same angle
    cos = torch.cos(angles).repeat(1, 2)  # (seq_len, dim)
    sin = torch.sin(angles).repeat(1, 2)  # (seq_len, dim)

    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings to input tensor.

    For each pair of dimensions (x_2i, x_{2i+1}), applies:
        x_2i'    = x_2i * cos - x_{2i+1} * sin
        x_{2i+1}' = x_2i * sin + x_{2i+1} * cos

    Args:
        x: Input tensor of shape (batch, n_heads, seq_len, head_dim)
        cos: Cosine values (seq_len, head_dim)
        sin: Sine values (seq_len, head_dim)
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Rotate pairs: [-x1, x0, -x3, x2, ...] (interleaved rotation)
    x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)

    return x * cos + x_rotated * sin


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    During generation, we don't want to recompute K and V for all previous tokens.
    Instead, we cache them and only compute K, V for the new token.
    """

    def __init__(self):
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None

    def update(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value to cache and return full sequence."""
        if self.key_cache is None:
            self.key_cache = key
            self.value_cache = value
        else:
            self.key_cache = torch.cat([self.key_cache, key], dim=2)
            self.value_cache = torch.cat([self.value_cache, value], dim=2)
        return self.key_cache, self.value_cache

    def reset(self):
        """Clear the cache."""
        self.key_cache = None
        self.value_cache = None


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) — as used in LLaMA-3.

    Standard MHA has equal number of Q, K, V heads.
    GQA uses fewer K, V heads — each KV head is shared across multiple Q heads.
    This reduces KV cache memory by n_heads/n_kv_heads without quality loss.

    LLaMA-3 8B: 32 query heads, 8 KV heads → 4x less KV cache memory.

    Args:
        config: NanoLLMConfig with model dimensions
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # How many Q heads per KV head

        # Projections
        self.wq = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=config.bias)

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_cos",
            precompute_rope_frequencies(self.head_dim, config.max_seq_length, config.rope_theta)[0]
        )
        self.register_buffer(
            "rope_sin",
            precompute_rope_frequencies(self.head_dim, config.max_seq_length, config.rope_theta)[1]
        )

        # KV Cache (for inference)
        self.kv_cache = KVCache()

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match number of query heads.

        If n_kv_heads=8 and n_heads=32, each KV head is repeated 4 times.
        Shape: (batch, n_kv_heads, seq, head_dim) → (batch, n_heads, seq, head_dim)
        """
        if self.n_rep == 1:
            return x
        batch, n_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, self.n_rep, seq_len, head_dim)
        return x.reshape(batch, self.n_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with GQA + RoPE.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Causal attention mask
            use_cache: Whether to use KV cache (for generation)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V — position is encoded in attention, not values)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # KV Cache for generation
        if use_cache:
            k, v = self.kv_cache.update(k, v)

        # Repeat KV heads to match Q heads
        k = self._repeat_kv(k)  # (batch, n_heads, seq, head_dim)
        v = self._repeat_kv(v)

        # Scaled dot-product attention (uses FlashAttention-2 when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and not use_cache),  # Auto causal mask
            dropout_p=0.0,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(attn_output)

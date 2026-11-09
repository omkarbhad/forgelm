"""Decoder-only transformer, LLaMA style, written from scratch.

RMSNorm (pre-norm) -> grouped-query attention with rotary embeddings -> SwiGLU MLP,
tied input/output embeddings, KV cache for generation.

The model is stateless: a KV cache is passed in explicitly, never stored on the module.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ForgeConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope(head_dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Rotate pairs (x[i], x[i + d/2]) by position-dependent angles.

    x: (batch, heads, seq, head_dim). Positions are start .. start + seq - 1, so tokens
    decoded one at a time with a KV cache get their true absolute position.
    """
    t = x.shape[2]
    c = cos[start : start + t].to(x.dtype)
    s = sin[start : start + t].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class KVCache:
    """Keys/values of one attention layer for everything decoded so far."""

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v


def new_cache(config: ForgeConfig) -> List[KVCache]:
    return [KVCache() for _ in range(config.n_layers)]


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ForgeConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: Optional[KVCache] = None):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        start = cache.length if cache is not None else 0
        q, k = apply_rope(q, cos, sin, start), apply_rope(k, cos, sin, start)
        if cache is not None:
            k, v = cache.update(k, v)

        # each KV head serves n_rep consecutive query heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        s = k.shape[2]  # total keys = start + t
        if start == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif t == 1:
            out = F.scaled_dot_product_attention(q, k, v)  # a single new token sees all of the past
        else:
            # chunk of new tokens after a cached prefix: query i may see keys j <= start + i
            allowed = torch.arange(s, device=x.device)[None, :] <= (start + torch.arange(t, device=x.device))[:, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ForgeConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ForgeConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class ForgeLM(nn.Module):
    def __init__(self, cfg: ForgeConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: shrink residual-branch outputs so the residual stream does not grow with depth
        for name, p in self.named_parameters():
            if name.endswith(("attn.wo.weight", "mlp.w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights are counted once

    def forward(self, idx, targets=None, cache: Optional[List[KVCache]] = None):
        """idx: (batch, seq) token ids. Returns (logits, loss); loss is None without targets.

        targets are the *next* tokens (same shape as idx), so no shifting happens here.
        """
        start = cache[0].length if cache is not None else 0
        if start + idx.shape[1] > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {start + idx.shape[1]} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, self.rope_cos, self.rope_sin, cache[i] if cache is not None else None)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, top_p=None, eos_id=None):
        """Autoregressive sampling with a KV cache. temperature=0 means greedy."""
        was_training = self.training
        self.eval()
        max_new_tokens = min(max_new_tokens, self.cfg.max_seq_len - idx.shape[1])
        cache = new_cache(self.cfg)
        logits, _ = self(idx, cache=cache)  # prefill the prompt in one pass
        for _ in range(max_new_tokens):
            nxt = _sample(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            logits, _ = self(nxt, cache=cache)  # one token at a time, history comes from the cache
        self.train(was_training)
        return idx


def _sample(logits, temperature, top_k, top_p):
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        drop = probs.cumsum(-1) - probs > top_p  # drop tokens once the mass before them already exceeds top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

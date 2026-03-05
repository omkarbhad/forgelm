"""
NanoLLM — Full Transformer Model

Decoder-only transformer matching the LLaMA-3 architecture:
- Pre-normalization (RMSNorm before attention and FFN)
- Grouped Query Attention (GQA) with RoPE
- SwiGLU Feed-Forward Network
- No bias in any linear layers
- KV Cache for efficient generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import NanoLLMConfig
from .attention import GroupedQueryAttention
from .feedforward import SwiGLU
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-normalization.

    Architecture (LLaMA-3 style):
        x → RMSNorm → GQA → + residual → RMSNorm → SwiGLU → + residual
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        h = x + self.attention(self.attention_norm(x), mask=mask, use_cache=use_cache)
        # Pre-norm FFN with residual
        out = h + self.ffn(self.ffn_norm(h))
        return out


class NanoLLM(nn.Module):
    """
    Complete decoder-only transformer language model.

    Components:
    - Token embeddings (no learned positional embeddings — RoPE handles position)
    - N transformer blocks
    - Final RMSNorm
    - Language model head (weight-tied with token embeddings)
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings (position is handled by RoPE in attention)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Output projection (weight-tied with embeddings)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            targets: Target token IDs for loss computation (batch, seq_len)
            use_cache: Whether to use KV cache (for generation)

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Token embeddings
        h = self.tok_embeddings(input_ids)

        # Pass through transformer blocks
        for layer in self.layers:
            h = layer(h, use_cache=use_cache)

        # Final norm + output projection
        h = self.norm(h)
        logits = self.output(h)

        result = {"logits": logits}

        # Compute loss if targets provided
        if targets is not None:
            # Shift logits and targets for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-1,
            )
            result["loss"] = loss

        return result

    def reset_cache(self):
        """Reset KV cache for all layers."""
        for layer in self.layers:
            layer.attention.kv_cache.reset()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache.

        Uses top-k + top-p (nucleus) sampling with temperature.
        KV cache avoids recomputing attention for previous tokens.
        """
        self.reset_cache()

        # Process the full prompt first (prefill)
        output = self.forward(input_ids, use_cache=True)

        for _ in range(max_new_tokens):
            logits = output["logits"][:, -1, :]  # Last token logits

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Only pass the new token through (KV cache handles history)
            output = self.forward(next_token, use_cache=True)

        self.reset_cache()
        return input_ids

    def count_parameters(self) -> int:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

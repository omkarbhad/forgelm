"""
NanoLLM — SwiGLU Feed-Forward Network

LLaMA-3 uses SwiGLU instead of the standard ReLU/GELU FFN.

Standard FFN:   FFN(x) = GELU(xW1) · W2
SwiGLU FFN:     FFN(x) = (SiLU(xW_gate) ⊙ xW_up) · W_down

SwiGLU has 3 weight matrices instead of 2, but empirically produces
better results. The FFN hidden dimension is adjusted to 8/3 * d_model
(rounded to nearest 256) to keep parameter count comparable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NanoLLMConfig


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in LLaMA-3.

    Architecture:
        gate = SiLU(x @ W_gate)
        up   = x @ W_up
        out  = (gate ⊙ up) @ W_down

    Where ⊙ is element-wise multiplication and SiLU(x) = x * sigmoid(x).
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__()

        self.w_gate = nn.Linear(config.d_model, config.ffn_dim, bias=config.bias)
        self.w_up = nn.Linear(config.d_model, config.ffn_dim, bias=config.bias)
        self.w_down = nn.Linear(config.ffn_dim, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward pass.

        The gating mechanism (SiLU on one path, linear on another)
        allows the network to learn which features to pass through.
        """
        gate = F.silu(self.w_gate(x))  # SiLU = x * sigmoid(x)
        up = self.w_up(x)
        return self.w_down(gate * up)   # Element-wise gate then project down

"""
NanoLLM — RMSNorm (Root Mean Square Layer Normalization)

LLaMA uses RMSNorm instead of LayerNorm.
RMSNorm is simpler and faster — it skips the mean-centering step.

LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
RMSNorm:   y = x / sqrt(mean(x^2) + eps) * gamma

RMSNorm has no bias (beta) and no mean subtraction.
Empirically performs as well as LayerNorm but ~10-15% faster.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes by the RMS of the input (no mean subtraction).
    Used in LLaMA, Mistral, Gemma, and most modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale (gamma)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        # x^2 → mean over last dim → add eps → rsqrt → multiply
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm.

        Cast to float32 for numerical stability during norm computation,
        then cast back to original dtype.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

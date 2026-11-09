import pytest
import torch

from forgelm import ForgeConfig, ForgeLM


@pytest.fixture
def cfg():
    return ForgeConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return ForgeLM(cfg).eval()

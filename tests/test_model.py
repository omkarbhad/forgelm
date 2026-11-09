import pytest
import torch

from forgelm import ForgeConfig, ForgeLM
from forgelm.model import apply_rope, new_cache, precompute_rope


def test_config_validation():
    with pytest.raises(ValueError):
        ForgeConfig(d_model=30, n_heads=4)  # not divisible
    with pytest.raises(ValueError):
        ForgeConfig(n_heads=6, n_kv_heads=4)  # kv heads must divide heads


def test_shapes_and_loss(model, cfg):
    x = torch.randint(0, cfg.vocab_size, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(float(cfg.vocab_size))).item(), rel=0.2)  # ~ln(V) at init


def test_gqa_has_fewer_kv_params(cfg):
    m = ForgeLM(cfg)
    attn = m.layers[0].attn
    assert attn.wk.weight.shape[0] == cfg.n_kv_heads * cfg.head_dim < attn.wq.weight.shape[0]


def test_tied_embeddings(model):
    assert model.lm_head.weight is model.tok_emb.weight


def test_no_future_leakage(model, cfg):
    """Changing token t must not change the logits at any position before t (causal masking)."""
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    y = x.clone()
    y[0, 7] = (y[0, 7] + 1) % cfg.vocab_size
    lx, _ = model(x)
    ly, _ = model(y)
    assert torch.allclose(lx[:, :7], ly[:, :7], atol=1e-6)
    assert not torch.allclose(lx[:, 7:], ly[:, 7:], atol=1e-6)


def test_rope_is_relative():
    """The attention score between positions m and n depends only on m - n."""
    cos, sin = precompute_rope(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        return (apply_rope(q, cos, sin, start=m) * apply_rope(k, cos, sin, start=n)).sum()

    assert torch.allclose(score(5, 2), score(20, 17), atol=1e-5)
    assert not torch.allclose(score(5, 2), score(5, 3), atol=1e-5)


def test_kv_cache_matches_full_forward(model, cfg):
    """Prefill + one-token steps must reproduce a single full forward pass."""
    x = torch.randint(0, cfg.vocab_size, (2, 14))
    full, _ = model(x)
    cache = new_cache(cfg)
    parts = [model(x[:, :6], cache=cache)[0]]
    for t in range(6, 14):
        parts.append(model(x[:, t : t + 1], cache=cache)[0])
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4)


def test_kv_cache_chunk_after_prefix(model, cfg):
    """A multi-token chunk after a cached prefix (the masked path) must also match."""
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    full, _ = model(x)
    cache = new_cache(cfg)
    a, _ = model(x[:, :5], cache=cache)
    b, _ = model(x[:, 5:9], cache=cache)
    c, _ = model(x[:, 9:], cache=cache)
    assert torch.allclose(full, torch.cat([a, b, c], dim=1), atol=1e-4)


def test_greedy_generate_matches_naive_loop(model, cfg):
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    fast = model.generate(prompt, 10, temperature=0)
    ids = prompt
    for _ in range(10):
        logits, _ = model(ids)
        ids = torch.cat([ids, logits[:, -1:].argmax(-1)], dim=1)
    assert torch.equal(fast, ids)


def test_generate_stops_at_max_seq_len(model, cfg):
    out = model.generate(torch.zeros(1, 30, dtype=torch.long), 100, temperature=1.0, top_k=5, top_p=0.9)
    assert out.shape[1] == cfg.max_seq_len


def test_can_overfit_one_sequence(cfg):
    """If the whole stack is wired correctly, it can memorize a short sequence."""
    torch.manual_seed(0)
    m = ForgeLM(cfg)
    seq = torch.randint(0, cfg.vocab_size, (1, 17))
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1

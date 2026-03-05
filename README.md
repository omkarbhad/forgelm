# NanoLLM 🔬

Decoder-only transformer built from scratch — matching the LLaMA-3 architecture with modern components.

Not a wrapper. Not a tutorial. A complete, trainable implementation demonstrating deep understanding of LLM internals.

## Architecture

```
Input Tokens
    ↓
BPE Tokenizer (custom-trained)
    ↓
Token Embeddings (NO positional embeddings — RoPE is applied in attention)
    ↓
┌────────────────────────────────┐
│  Transformer Block (×N)        │
│  ├── RMSNorm (pre-norm)       │
│  ├── Grouped Query Attention   │  ← GQA + RoPE + KV Cache
│  │   └── FlashAttention-2     │
│  ├── RMSNorm (pre-norm)       │
│  └── SwiGLU FFN               │  ← SwiGLU replaces ReLU/GELU
└────────────────────────────────┘
    ↓
RMSNorm
    ↓
Output Logits → Vocabulary
```

## What Makes This Different from nanoGPT

| Component | nanoGPT (2023) | NanoLLM (2025) |
|-----------|---------------|----------------|
| Attention | Standard MHA | **Grouped Query Attention (GQA)** |
| Position | Learned embeddings | **Rotary Position Embeddings (RoPE)** |
| Normalization | LayerNorm | **RMSNorm** (faster, no mean computation) |
| Activation | GELU | **SwiGLU** (gated linear unit) |
| Memory | Standard attention | **FlashAttention-2** (IO-aware) |
| Inference | Naive autoregressive | **KV Cache** (no recomputation) |
| Training | Single GPU | **DDP** (distributed data parallel) |
| Precision | fp32 | **bf16 mixed precision** (AMP) |

## Project Structure

```
NanoLLM/
├── README.md
├── model/
│   ├── __init__.py
│   ├── config.py           # Model configuration (NanoLLMConfig)
│   ├── attention.py         # GQA + RoPE + FlashAttention-2 + KV Cache
│   ├── feedforward.py       # SwiGLU FFN
│   ├── normalization.py     # RMSNorm
│   ├── transformer.py       # Full transformer block + model
│   └── tokenizer.py         # BPE tokenizer (train + encode + decode)
├── data/
│   ├── prepare.py           # Download + preprocess corpus
│   └── dataloader.py        # Streaming dataloader with packing
├── train/
│   ├── trainer.py           # Training loop (DDP + AMP + gradient accumulation)
│   ├── train.py             # Main training script
│   └── generate.py          # Text generation with KV cache
├── notebooks/
│   ├── 01_attention_deep_dive.ipynb
│   ├── 02_training_curves.ipynb
│   └── 03_generation_demo.ipynb
└── requirements.txt
```

## Key Implementations

### Grouped Query Attention (GQA)
```python
# Standard MHA: n_heads query, n_heads key, n_heads value
# GQA: n_heads query, n_kv_heads key, n_kv_heads value (n_kv_heads < n_heads)
# LLaMA-3 8B: 32 query heads, 8 KV heads → 4x less KV memory
```

### Rotary Position Embeddings (RoPE)
```python
# Apply rotation to query and key vectors based on position
# Allows relative position awareness without learned embeddings
# Extrapolates to longer sequences than training length
```

### SwiGLU Activation
```python
# FFN(x) = SiLU(xW_gate) ⊙ (xW_up)  then  → W_down
# 3 weight matrices instead of 2 — but empirically better
# Hidden dim adjusted: 8/3 * d_model (rounded to multiple of 256)
```

## Quick Start

```bash
# Train BPE tokenizer
python model/tokenizer.py --corpus data/corpus.txt --vocab_size 32000

# Train model (single GPU)
python train/train.py --config small  # 125M params

# Train model (multi-GPU DDP)
torchrun --nproc_per_node=4 train/train.py --config base  # 350M params

# Generate text
python train/generate.py --checkpoint outputs/model.pt --prompt "The transformer architecture"
```

## Model Configs

| Config | Params | Layers | d_model | Heads | KV Heads | FFN dim |
|--------|--------|--------|---------|-------|----------|---------|
| tiny   | 15M    | 6      | 384     | 6     | 2        | 1024    |
| small  | 125M   | 12     | 768     | 12    | 4        | 2048    |
| base   | 350M   | 24     | 1024    | 16    | 4        | 2730    |

## License

MIT

# ForgeLM

A decoder-only language model **built from scratch in PyTorch**: tokenizer, transformer, training loop and generation. No `transformers`, no `tokenizers`, no wrappers. Only `torch`.

It is also my long-term base for machine-learning and LLM experiments, so it is kept small, tested and readable.

## What is in it

| Piece | Where | Notes |
|---|---|---|
| Byte-level **BPE tokenizer**, trained from scratch | `forgelm/tokenizer.py` | lossless round-trip for any text, special tokens, save/load |
| **Transformer** (LLaMA-style) | `forgelm/model.py` | RMSNorm pre-norm, grouped-query attention, rotary embeddings (RoPE), SwiGLU MLP, tied embeddings |
| **KV cache** generation | `forgelm/model.py` | stateless model, cache passed in; temperature, top-k, top-p |
| **Training loop** | `forgelm/train.py` | AdamW with decay groups, warmup + cosine LR, grad clipping, best-checkpoint saving, bf16 on CUDA |
| Data pipeline | `forgelm/data.py` | download, tokenize, random-window batches |
| **Tests** | `tests/` | causality, cache equivalence, RoPE, tokenizer, overfit check |

## Quickstart

```bash
git clone https://github.com/omkarbhad/forgelm && cd forgelm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                            # 23 tests
python -m forgelm.data                            # download tiny Shakespeare, train the tokenizer
python -m forgelm.train --preset tiny --max-steps 2000 --lr 2e-3
python -m forgelm.generate --run runs/tiny --prompt "ROMEO:"
```

Device is picked automatically (CUDA, then Apple MPS, then CPU). Bring your own corpus with `python -m forgelm.data --text my.txt`.

## Result

Training results are not published yet. A tiny 3M-parameter run on tiny Shakespeare takes about 13 minutes on an Apple-silicon Mac (see Quickstart); numbers and samples will be added here once a full run is done.

## Architecture

```
tokens -> embedding
       -> N x [ RMSNorm -> grouped-query attention (+RoPE, KV cache) -> residual
                RMSNorm -> SwiGLU                                      -> residual ]
       -> RMSNorm -> linear head (weights tied to the embedding) -> logits
```

Presets live in `forgelm/config.py` (`tiny`, `small`, `base`). Grouped-query attention shares each key/value head across several query heads, which shrinks the KV cache by `n_heads / n_kv_heads`.

## Testing what matters

Language-model bugs rarely crash; they quietly make the model cheat or generate garbage. The tests target exactly those:

- **No future leakage**: changing token `t` must not change any logit before `t` (a missing causal mask makes a model look great in training and useless at inference).
- **KV cache equals full forward**: prefill plus one-token steps, and chunks after a cached prefix, reproduce a single full pass. Catches wrong RoPE positions and wrong masks during cached decoding.
- **RoPE is relative**: the score between positions `m` and `n` depends only on `m - n`.
- **Greedy cached generation equals a naive loop.**
- **The model can overfit one sequence**, which proves the whole stack is wired correctly.

I checked that these tests fail when the bugs are reintroduced.

## Roadmap

- [ ] Larger corpus and a proper data loader (streaming, document packing)
- [ ] Multi-GPU training (DDP), gradient accumulation
- [ ] Evaluation harness (perplexity on held-out sets, small benchmarks)
- [ ] Instruction fine-tuning and sampling improvements (repetition penalty, beam search)
- [ ] Faster inference (preallocated KV cache, quantization)
- [ ] Ablations: GQA vs MHA, RoPE vs learned positions, SwiGLU vs GELU

## History

This started as a notebook-based encoder-decoder transformer and an earlier `nanollm` prototype. Both had the same silent flaw: the decoder was not causally masked, so training leaked the answer and inference produced empty or wrong output. ForgeLM is the clean rewrite with tests that prevent that class of bug.

## License

MIT

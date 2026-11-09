"""Dataset preparation and batching. Default corpus: tiny Shakespeare (~1 MB)."""

import os
import urllib.request

import torch

from .tokenizer import BPETokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(data_dir: str = "data", vocab_size: int = 1024, text_path: str = None) -> None:
    """Train the tokenizer on the corpus and save train/val token tensors + tokenizer.json."""
    os.makedirs(data_dir, exist_ok=True)
    if text_path is None:
        text_path = os.path.join(data_dir, "input.txt")
        if not os.path.exists(text_path):
            print(f"downloading {SHAKESPEARE_URL}")
            urllib.request.urlretrieve(SHAKESPEARE_URL, text_path)
    text = open(text_path, encoding="utf-8").read()

    tok = BPETokenizer.train(text, vocab_size)
    tok.save(os.path.join(data_dir, "tokenizer.json"))
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    cut = int(0.9 * len(ids))
    torch.save({"train": ids[:cut], "val": ids[cut:]}, os.path.join(data_dir, "tokens.pt"))
    print(f"{len(text):,} chars -> {len(ids):,} tokens ({len(text) / len(ids):.2f} chars/token), "
          f"vocab {tok.vocab_size}, train {cut:,} / val {len(ids) - cut:,}")


def load(data_dir: str = "data"):
    tok = BPETokenizer.load(os.path.join(data_dir, "tokenizer.json"))
    tokens = torch.load(os.path.join(data_dir, "tokens.pt"))
    return tok, tokens["train"], tokens["val"]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device):
    """Random windows. y is x shifted left by one: y[t] is the token that follows x[t]."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="download corpus, train tokenizer, tokenize")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--text", default=None, help="path to your own .txt corpus")
    a = p.parse_args()
    prepare(a.data_dir, a.vocab_size, a.text)

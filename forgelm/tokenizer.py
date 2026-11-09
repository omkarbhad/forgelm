"""Byte-level BPE tokenizer, trained from scratch.

Text is split into words with a regex, each word becomes raw UTF-8 bytes, and the most
frequent adjacent pair is merged repeatedly. Ids 0-255 are the bytes, then one id per merge,
then special tokens. Any string round-trips losslessly because every byte has a token.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

# words, numbers, punctuation runs and whitespace, each keeping a leading space
SPLIT = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")
EOT = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, merges: Optional[Sequence[Tuple[int, int]]] = None, special_tokens: Sequence[str] = (EOT,)):
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self.special_tokens = list(special_tokens)
        self._build()

    def _build(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.special_ids = {s: 256 + len(self.merges) + i for i, s in enumerate(self.special_tokens)}
        self._special_by_id = {i: s for s, i in self.special_ids.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in self.special_tokens) + ")") if self.special_tokens else None
        )
        self._cache: Dict[str, List[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_id(self) -> Optional[int]:
        return self.special_ids.get(EOT)

    # ---- training -------------------------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int, special_tokens: Sequence[str] = (EOT,)) -> "BPETokenizer":
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size is too small for 256 bytes plus the special tokens")
        # count unique words once; merging then works on the (small) word table, not the full text
        words = Counter(SPLIT.findall(text))
        table = {tuple(w.encode("utf-8")): c for w, c in words.items()}
        merges: List[Tuple[int, int]] = []
        for step in range(n_merges):
            pairs: Counter = Counter()
            for word, count in table.items():
                for pair in zip(word, word[1:]):
                    pairs[pair] += count
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))  # ties: smallest ids, deterministic
            new_id = 256 + step
            merges.append(best)
            table = {_merge(word, best, new_id): count for word, count in table.items()}
        return cls(merges, special_tokens)

    # ---- encode / decode ------------------------------------------------------------------
    def _encode_word(self, word: str) -> List[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, 256 + self.ranks[pair])
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        chunks = self._special_re.split(text) if self._special_re else [text]
        for chunk in chunks:
            if chunk in self.special_ids:
                out.append(self.special_ids[chunk])
            else:
                for word in SPLIT.findall(chunk):
                    out.extend(self._encode_word(word))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        parts = []
        for i in ids:
            i = int(i)
            parts.append(self._special_by_id[i].encode("utf-8") if i in self._special_by_id else self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---- persistence ----------------------------------------------------------------------
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges, "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["merges"], d["special_tokens"])


def _merge(ids: Sequence[int], pair: Tuple[int, int], new_id: int):
    """Replace every non-overlapping occurrence of `pair` in ids with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out) if isinstance(ids, tuple) else out

"""Dependency-free reference models and a synthetic corpus.

These exist so the suite can be exercised end to end without PyTorch and so
that each metric has a known-answer baseline:

* :class:`BigramModel` uses exactly one token of history, so its context gain
  is zero by construction and its name binding is at chance.
* :class:`InductionModel` mixes the bigram with an in-context copying
  mechanism (a one-layer induction head): it does use long context.

:func:`synthetic_stories` produces short templated stories with recurring
named characters, enough for the probes to find positions to test.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence

import numpy as np

from .model import LogitsModel


class WordTokenizer:
    """Word-level tokenizer with ``<pad>``, ``<bos>``, ``<eos>``, ``<unk>`` specials.

    ``decode`` joins words with single spaces and attaches closing punctuation to
    the previous token, so incremental decoding yields consistent character spans.
    """

    TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+|[^\w\s]")
    SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")
    _NO_SPACE_BEFORE = set(".,!?;:')")

    def __init__(self, vocab: Sequence[str]) -> None:
        self.itos = list(self.SPECIALS) + [w for w in vocab if w not in self.SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)

    @classmethod
    def fit(cls, texts: Iterable[str], max_vocab: int | None = None) -> WordTokenizer:
        counts: dict[str, int] = {}
        for text in texts:
            for token in cls.TOKEN_RE.findall(text):
                counts[token] = counts.get(token, 0) + 1
        vocab = sorted(counts, key=lambda w: (-counts[w], w))
        return cls(vocab[:max_vocab] if max_vocab else vocab)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in self.TOKEN_RE.findall(text)]

    def decode(self, ids: Sequence[int]) -> str:
        out: list[str] = []
        quote_open = False
        glue_next = False
        for i in ids:
            if i < len(self.SPECIALS):
                continue
            token = self.itos[i]
            if token == '"':
                if not quote_open and out:
                    out.append(" ")
                out.append(token)
                glue_next = not quote_open
                quote_open = not quote_open
                continue
            if out and not glue_next and token[0] not in self._NO_SPACE_BEFORE:
                out.append(" ")
            out.append(token)
            glue_next = False
        return "".join(out)


class UniformModel(LogitsModel):
    """Assigns equal probability to every token: perplexity equals the vocabulary size."""

    def __init__(self, tokenizer: WordTokenizer, **kwargs: object) -> None:
        kwargs.setdefault("name", "uniform")
        super().__init__(tokenizer, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, **kwargs)  # type: ignore[arg-type]
        self.vocab_size = len(tokenizer)

    def logits(self, ids: np.ndarray) -> np.ndarray:
        return np.zeros((*ids.shape, self.vocab_size), dtype=np.float32)


class BigramModel(LogitsModel):
    """Add-``alpha`` smoothed bigram model: the "local statistics only" baseline."""

    def __init__(self, tokenizer: WordTokenizer, table: np.ndarray, **kwargs: object) -> None:
        kwargs.setdefault("name", "bigram")
        super().__init__(tokenizer, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, **kwargs)  # type: ignore[arg-type]
        self.table = table  # [V, V] log P(next | current)

    @classmethod
    def fit(
        cls,
        tokenizer: WordTokenizer,
        texts: Iterable[str],
        *,
        alpha: float = 0.05,
        **kwargs: object,
    ) -> BigramModel:
        v = len(tokenizer)
        counts = np.full((v, v), alpha, dtype=np.float64)
        for text in texts:
            ids = [tokenizer.bos_id, *tokenizer.encode(text), tokenizer.eos_id]
            np.add.at(counts, (ids[:-1], ids[1:]), 1.0)
        table = np.log(counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)
        return cls(tokenizer, table, **kwargs)

    def logits(self, ids: np.ndarray) -> np.ndarray:
        return self.table[ids]


class InductionModel(BigramModel):
    """Bigram mixed with in-context copying.

    ``P(next | ctx) = (1 - lam) * bigram(next | cur) + lam * copy(next | ctx)`` where
    ``copy`` is the empirical distribution of the tokens that followed earlier
    occurrences of the current token in the context (falls back to the bigram
    when the current token has not been seen before).  This is the mechanism of
    an induction head and is the simplest model that benefits from long context.
    """

    def __init__(
        self, tokenizer: WordTokenizer, table: np.ndarray, *, lam: float = 0.5, **kwargs: object
    ) -> None:
        kwargs.setdefault("name", "induction")
        super().__init__(tokenizer, table, **kwargs)
        self.lam = lam

    def logits(self, ids: np.ndarray) -> np.ndarray:
        probs = np.exp(self.table[ids]).astype(np.float64)  # [B, T, V]
        for b in range(ids.shape[0]):
            followers: dict[int, dict[int, int]] = {}
            row = ids[b]
            for t in range(row.shape[0]):
                current = int(row[t])
                seen = followers.get(current)
                if seen:
                    copy = np.zeros(probs.shape[-1], dtype=np.float64)
                    for token, count in seen.items():
                        copy[token] = count
                    copy /= copy.sum()
                    probs[b, t] = (1.0 - self.lam) * probs[b, t] + self.lam * copy
                if t + 1 < row.shape[0]:
                    nxt = int(row[t + 1])
                    followers.setdefault(current, {})
                    followers[current][nxt] = followers[current].get(nxt, 0) + 1
        return np.log(probs).astype(np.float32)


# --------------------------------------------------------------------------- synthetic corpus

_NAMES = ["Lily", "Tom", "Mia", "Ben", "Sam", "Anna", "Max", "Zoe", "Leo", "Ella"]
_PLACES = ["park", "garden", "beach", "forest", "kitchen", "farm", "library"]
_OBJECTS = ["ball", "kite", "cake", "boat", "hat", "book", "drum", "shell"]
_ANIMALS = ["dog", "cat", "bird", "frog", "duck", "rabbit"]
_OPENINGS = [
    "Once upon a time, {A} went to the {place} with {B}.",
    "One sunny day, {A} and {B} walked to the {place}.",
    "{A} lived near a big {place}. One morning {B} came to visit.",
]
_SENTENCES = [
    "{A} found a {obj} under a tree.",
    "{B} wanted to play with the {obj} too.",
    '"Can I try the {obj}?" asked {B}.',
    '"Yes, we can share," said {A} with a smile.',
    "They played with the {obj} all afternoon.",
    "Then a little {animal} ran past them.",
    "{B} laughed and chased the {animal} around the {place}.",
    "{A} sat down to rest and looked at the sky.",
    "The {animal} hid behind a rock, and {A} found it.",
    "After a while, {A} felt hungry.",
    '"Let us go home," said {B}.',
    "{A} picked up the {obj} and they walked back together.",
    "On the way, {B} told {A} a funny story about a {animal}.",
    "At home, {A} gave the {obj} to {B} as a gift.",
    "{B} said thank you and hugged {A}.",
    "That night, {A} dreamed about the {place} and the little {animal}.",
    "The next day, {B} came back and they played again.",
]


def synthetic_stories(n: int, *, seed: int = 0, length: tuple[int, int] = (12, 20)) -> list[str]:
    """Templated children's stories with two recurring named characters."""
    rng = random.Random(seed)
    stories: list[str] = []
    for _ in range(n):
        a, b = rng.sample(_NAMES, 2)
        slots = {
            "A": a,
            "B": b,
            "place": rng.choice(_PLACES),
            "obj": rng.choice(_OBJECTS),
            "animal": rng.choice(_ANIMALS),
        }
        parts = [rng.choice(_OPENINGS).format(**slots)]
        for _ in range(rng.randint(*length)):
            parts.append(rng.choice(_SENTENCES).format(**slots))
        stories.append(" ".join(parts))
    return stories

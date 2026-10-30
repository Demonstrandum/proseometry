"""The model abstraction every metric in proseometry is written against.

A :class:`Model` bundles a language model with its tokenizer and its scoring
conventions (BOS/EOS ids, context limit, batch size).  Subclasses implement two
primitives:

* :meth:`Model._token_logprobs` -- teacher-forced log-probabilities of a batch, and
* :meth:`Model._next_token_logits` -- the next-token distribution after a batch of
  prefixes (used for sampling when the model has no faster ``generate``).

Everything else (BOS handling, padding, batching, continuation scoring, nucleus
sampling) is implemented once, here.  :class:`LogitsModel` reduces the two
primitives to a single ``logits`` method for models that are easy to express in
numpy; the adapters in :mod:`proseometry.adapters` do the same for PyTorch and
Hugging Face models.

Conventions
-----------
* Every context fed to the model starts with ``bos_id`` when one is configured.
  Scores returned by :meth:`Model.score` are aligned to the *original* tokens:
  entry ``t`` is ``log P(x_t | x_<t)``.  Without a BOS token the first token has
  no prediction and its entry is ``NaN``.
* Character offsets used elsewhere in the package are relative to
  ``model.decode(ids)`` -- the text as the model sees it -- which for a
  round-tripping tokenizer is the original text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface: text to ids and back."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax along ``axis``."""
    m = np.max(logits, axis=axis, keepdims=True)
    z = logits - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def sample_token(
    logits: np.ndarray, *, temperature: float, top_p: float, rng: np.random.Generator
) -> int:
    """Sample one token id from ``logits`` with temperature and nucleus (top-p) truncation.

    ``temperature <= 0`` is greedy decoding.  ``top_p`` keeps the smallest set of
    tokens whose cumulative probability reaches ``top_p`` (Holtzman et al. 2020).
    """
    if temperature <= 0:
        return int(np.argmax(logits))
    probs = np.exp(log_softmax(logits.astype(np.float64) / temperature))
    if top_p < 1.0:
        order = np.argsort(-probs, kind="stable")
        cumulative = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cumulative, top_p)) + 1
        keep = order[:cutoff]
        truncated = np.zeros_like(probs)
        truncated[keep] = probs[keep]
        probs = truncated / truncated.sum()
    return int(rng.choice(len(probs), p=probs))


@dataclass
class TokenizedDoc:
    """A document as a model sees it: ids, the decoded text, and each token's character span."""

    ids: list[int]
    text: str
    spans: list[tuple[int, int]]

    def __len__(self) -> int:
        return len(self.ids)


class Model(ABC):
    """A language model under test.

    Parameters
    ----------
    tokenizer:
        Anything with ``encode(str) -> list[int]`` and ``decode(ids) -> str``.
    name:
        Label used in reports.
    bos_id:
        If set, prepended to every context fed to the model (and never scored).
    eos_id:
        If set, sampling stops when it is produced.
    max_context:
        Longest sequence (including BOS) the model accepts.  Scoring truncates
        documents to fit; sampling keeps the most recent tokens.
    batch_size:
        Sequences per forward pass.  Memory scales with ``batch_size * T * V``
        for models that materialise full logits, so keep it small for big vocabularies.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        name: str = "model",
        bos_id: int | None = None,
        eos_id: int | None = None,
        max_context: int | None = None,
        batch_size: int = 4,
    ) -> None:
        self.tokenizer = tokenizer
        self.name = name
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_context = max_context
        self.batch_size = max(1, batch_size)

    # ------------------------------------------------------------------ primitives
    @abstractmethod
    def _token_logprobs(self, ids: np.ndarray) -> np.ndarray:
        """Teacher-forced log-probabilities for a right-padded batch.

        ``ids`` is an int64 array of shape ``[B, T]``.  Return a float array of shape
        ``[B, T - 1]`` whose entry ``[b, t]`` is ``log P(ids[b, t + 1] | ids[b, :t + 1])``.
        Padding is on the right, so for a causal model it never affects real positions.
        """

    @abstractmethod
    def _next_token_logits(self, ids: np.ndarray) -> np.ndarray:
        """Next-token logits after each prefix in an unpadded ``[B, T]`` batch: shape ``[B, V]``."""

    # ------------------------------------------------------------------ tokenization
    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    @property
    def max_doc_tokens(self) -> int | None:
        """Longest document (excluding BOS) that fits in one forward pass, if limited."""
        if self.max_context is None:
            return None
        return self.max_context - (1 if self.bos_id is not None else 0)

    def tokenize_doc(self, text: str, max_tokens: int | None = None) -> TokenizedDoc:
        """Encode ``text`` (truncated to ``max_tokens`` and the context limit) with char spans."""
        from .text import token_char_spans

        ids = self.encode(text)
        limit = self.max_doc_tokens
        for cap in (max_tokens, limit):
            if cap is not None:
                ids = ids[:cap]
        decoded = self.decode(ids)
        return TokenizedDoc(ids=ids, text=decoded, spans=token_char_spans(self.tokenizer, ids))

    # ------------------------------------------------------------------ scoring
    def _with_bos(self, seq: Sequence[int]) -> list[int]:
        return [self.bos_id, *seq] if self.bos_id is not None else list(seq)

    @property
    def _pad_id(self) -> int:
        for candidate in (self.eos_id, self.bos_id):
            if candidate is not None:
                return candidate
        return 0

    def score(self, seqs: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Teacher-forced ``log P(x_t | x_<t)`` for every token of every sequence.

        Returns one float32 array per input sequence, aligned to its tokens.  Entry
        0 is ``NaN`` when the model has no BOS token (nothing predicts the first token).
        Sequences are sorted by length and batched to limit padding.
        """
        seqs = [list(s) for s in seqs]
        out: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * len(seqs)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for start in range(0, len(order), self.batch_size):
            index = order[start : start + self.batch_size]
            batch = [self._with_bos(seqs[i]) for i in index]
            width = max(len(b) for b in batch)
            if width < 2:
                logprobs = np.zeros((len(batch), 0), dtype=np.float32)
            else:
                array = np.full((len(batch), width), self._pad_id, dtype=np.int64)
                for row, b in enumerate(batch):
                    array[row, : len(b)] = b
                logprobs = np.asarray(self._token_logprobs(array), dtype=np.float32)
            for row, i in enumerate(index):
                n = len(seqs[i])
                result = np.full(n, np.nan, dtype=np.float32)
                if self.bos_id is None:
                    if n > 1:
                        result[1:] = logprobs[row, : n - 1]
                else:
                    result[:] = logprobs[row, :n]
                out[i] = result
        return out

    def score_continuations(
        self, prefix: Sequence[int], continuations: Sequence[Sequence[int]]
    ) -> np.ndarray:
        """Total log-probability of each continuation given ``prefix``: shape ``[C]``.

        Without a BOS token the prefix must be non-empty, otherwise the first
        continuation token has no prediction and the total is ``NaN``.
        """
        prefix = list(prefix)
        scored = self.score([prefix + list(c) for c in continuations])
        return np.array([float(np.sum(lp[len(prefix) :])) for lp in scored], dtype=np.float64)

    # ------------------------------------------------------------------ sampling
    def generate(
        self,
        prompt: Sequence[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[int]:
        """Sample a continuation of ``prompt`` (returned without the prompt).

        This default re-runs the model on the whole context at every step (no KV
        cache); adapters override it with a native generation routine when one exists.
        """
        rng = np.random.default_rng(seed)
        context = list(prompt)
        generated: list[int] = []
        window = self.max_doc_tokens
        for _ in range(max_new_tokens):
            visible = context[-window:] if window is not None else context
            array = np.asarray([self._with_bos(visible)], dtype=np.int64)
            logits = np.asarray(self._next_token_logits(array))[0]
            token = sample_token(logits, temperature=temperature, top_p=top_p, rng=rng)
            generated.append(token)
            context.append(token)
            if self.eos_id is not None and token == self.eos_id:
                break
        return generated

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class LogitsModel(Model):
    """A :class:`Model` defined by a single ``logits`` method.

    Implement ``logits(ids) -> [B, T, V]`` in numpy and both scoring primitives
    follow.  Convenient for n-gram baselines, toy models, and tests; for neural
    models prefer the PyTorch adapters, which keep the vocabulary axis on device.
    """

    @abstractmethod
    def logits(self, ids: np.ndarray) -> np.ndarray:
        """Next-token logits at every position: ``[B, T]`` ids to ``[B, T, V]`` floats."""

    def _token_logprobs(self, ids: np.ndarray) -> np.ndarray:
        logprobs = log_softmax(np.asarray(self.logits(ids[:, :-1]), dtype=np.float64))
        targets = ids[:, 1:]
        return np.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]

    def _next_token_logits(self, ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.logits(ids))[:, -1]

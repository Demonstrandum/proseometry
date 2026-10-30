"""Teacher-forced metrics: bits per byte, context-ablation gain, key tokens and LongPPL.

All of these start from :func:`context_scores`, which scores a document once
with its full context and once per short-context window ``k``.  The short
score of token ``x_t`` conditions only on ``x_{t-k..t-1}`` (plus BOS), so
``full - short`` is the long-short difference (LSD) of Fang et al. (2025): the
individual treatment effect of giving the model more than ``k`` tokens of
history.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .model import Model, TokenizedDoc
from .stats import Interval, bootstrap_mean
from .text import merge_spans


@dataclass
class ContextScores:
    """Per-token log-probabilities of one document under full and truncated contexts."""

    doc: TokenizedDoc
    full: np.ndarray
    short: dict[int, np.ndarray] = field(default_factory=dict)

    def lsd(self, k: int) -> np.ndarray:
        """Long-short difference: full minus ``k``-token-context log-prob (0 for ``t < k``)."""
        return self.full - self.short[k]

    @property
    def scored(self) -> np.ndarray:
        return ~np.isnan(self.full)


def short_context_logprobs(
    model: Model, ids: Sequence[int], full: np.ndarray, k: int, *, stride: int = 1
) -> np.ndarray:
    """``log P(x_t | x_{t-k..t-1})`` for every token, reusing ``full`` where ``t < k``.

    With ``stride == 1`` every token sees exactly ``k`` predecessors (one window per
    token).  A larger stride scores ``stride`` consecutive tokens per window, so
    they see between ``k`` and ``k + stride - 1`` predecessors: ``stride`` times
    cheaper, slightly optimistic about the short context.
    """
    ids = list(ids)
    n = len(ids)
    out = np.array(full, dtype=np.float32, copy=True)
    if n <= k:
        return out
    stride = max(1, stride)
    windows: list[list[int]] = []
    targets: list[range] = []
    for block_start in range(k, n, stride):
        block_end = min(block_start + stride, n)
        windows.append(ids[block_start - k : block_end])
        targets.append(range(block_start, block_end))
    for window_scores, target in zip(model.score(windows), targets, strict=True):
        out[target.start : target.stop] = window_scores[k : k + len(target)]
    return out


def context_scores(
    model: Model, doc: TokenizedDoc, ks: Iterable[int] = (), *, stride: int = 1
) -> ContextScores:
    """Score ``doc`` under its full context and under each short window in ``ks``."""
    (full,) = model.score([doc.ids])
    scores = ContextScores(doc=doc, full=full)
    for k in sorted(set(int(k) for k in ks)):
        scores.short[k] = short_context_logprobs(model, doc.ids, full, k, stride=stride)
    return scores


# --------------------------------------------------------------------------- bits per byte


@dataclass
class LikelihoodSummary:
    """Corpus-level likelihood in tokenizer-independent units."""

    nll_sum: float
    n_tokens: int
    n_bytes: int
    n_docs: int
    per_doc_bits_per_byte: list[float]

    @property
    def bits_per_byte(self) -> float:
        return self.nll_sum / (self.n_bytes * math.log(2)) if self.n_bytes else math.nan

    @property
    def nll_per_token(self) -> float:
        return self.nll_sum / self.n_tokens if self.n_tokens else math.nan

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll_per_token) if self.n_tokens else math.nan

    def bits_per_byte_interval(self, seed: int = 0) -> Interval:
        return bootstrap_mean(self.per_doc_bits_per_byte, seed=seed)


def likelihood_summary(scores: Sequence[ContextScores]) -> LikelihoodSummary:
    """Aggregate full-context scores into bits per byte and per-token perplexity.

    Bytes are counted over the *scored* text only: with no BOS token the first
    token is unpredictable and both its loss and its bytes are excluded.
    """
    nll_sum = 0.0
    n_tokens = 0
    n_bytes = 0
    per_doc: list[float] = []
    for sc in scores:
        mask = sc.scored
        if not mask.any():
            continue
        first = int(np.argmax(mask))
        text = sc.doc.text[sc.doc.spans[first][0] :]
        doc_bytes = len(text.encode("utf-8"))
        doc_nll = float(-np.sum(sc.full[mask]))
        nll_sum += doc_nll
        n_tokens += int(mask.sum())
        n_bytes += doc_bytes
        per_doc.append(doc_nll / (doc_bytes * math.log(2)) if doc_bytes else math.nan)
    return LikelihoodSummary(nll_sum, n_tokens, n_bytes, len(per_doc), per_doc)


# --------------------------------------------------------------------------- context gain


@dataclass
class ContextGain:
    """How much history beyond the last ``k`` tokens helps, averaged over tokens with ``t >= k``."""

    k: int
    gain_nats: float
    n_tokens: int
    per_doc_gain: list[float]

    @property
    def ppl_reduction(self) -> float:
        """Relative perplexity reduction from long context: ``1 - PPL_long / PPL_short``."""
        return 1.0 - math.exp(-self.gain_nats) if not math.isnan(self.gain_nats) else math.nan

    def interval(self, seed: int = 0) -> Interval:
        return bootstrap_mean(self.per_doc_gain, seed=seed)


def context_gain(scores: Sequence[ContextScores], k: int) -> ContextGain:
    """Mean LSD over tokens that have more than ``k`` tokens of history."""
    total = 0.0
    count = 0
    per_doc: list[float] = []
    for sc in scores:
        if k not in sc.short:
            raise KeyError(f"document was not scored with short window k={k}")
        lsd = sc.lsd(k)[k:]
        lsd = lsd[~np.isnan(lsd)]
        if lsd.size == 0:
            continue
        total += float(lsd.sum())
        count += int(lsd.size)
        per_doc.append(float(lsd.mean()))
    return ContextGain(k, total / count if count else math.nan, count, per_doc)


# --------------------------------------------------------------------------- key tokens / LongPPL


def key_token_mask(
    scores: ContextScores, k: int, *, alpha: float = 2.0, beta: float = -2.0
) -> np.ndarray:
    """Key tokens in the sense of LongPPL (Fang et al. 2025, eq. 4).

    A token is *key* when its long-short difference exceeds ``alpha`` (long context
    materially helps) and its full-context log-probability exceeds ``beta`` (it is
    actually predictable given that context, so it is not just noise).  Tokens
    with fewer than ``k`` predecessors are never key.
    """
    lsd = scores.lsd(k)
    with np.errstate(invalid="ignore"):  # NaN (unscored first token) compares False
        mask = (lsd > alpha) & (scores.full > beta)
    mask[:k] = False
    return mask


def key_spans(doc: TokenizedDoc, mask: np.ndarray) -> list[tuple[int, int]]:
    """Character spans of the key tokens, merged, for transfer across tokenizers."""
    return merge_spans([doc.spans[i] for i in np.flatnonzero(mask)])


def mask_from_spans(doc: TokenizedDoc, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    """Tokens of ``doc`` that overlap any character span in ``spans``."""
    mask = np.zeros(len(doc.ids), dtype=bool)
    if not spans:
        return mask
    j = 0
    ordered = sorted(spans)
    for i, (start, end) in enumerate(doc.spans):
        while j < len(ordered) and ordered[j][1] <= start:
            j += 1
        if j < len(ordered) and ordered[j][0] < end:
            mask[i] = True
    return mask


@dataclass
class LongPPL:
    """Perplexity restricted to key tokens, plus how many tokens qualified."""

    nll_sum: float
    n_key_tokens: int
    n_tokens: int

    @property
    def long_ppl(self) -> float:
        return math.exp(self.nll_sum / self.n_key_tokens) if self.n_key_tokens else math.nan

    @property
    def key_fraction(self) -> float:
        return self.n_key_tokens / self.n_tokens if self.n_tokens else math.nan


def long_ppl(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> LongPPL:
    """LongPPL from ``(full_logprobs, key_mask)`` pairs, one per document."""
    nll_sum = 0.0
    n_key = 0
    n_total = 0
    for full, mask in pairs:
        valid = mask & ~np.isnan(full)
        nll_sum += float(-np.sum(full[valid]))
        n_key += int(valid.sum())
        n_total += int((~np.isnan(full)).sum())
    return LongPPL(nll_sum, n_key, n_total)

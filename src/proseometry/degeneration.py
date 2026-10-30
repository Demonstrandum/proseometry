"""Surface statistics of free-running text and their distance from human text.

Repetition is the degenerate attractor of likelihood training and is invisible
under teacher forcing, so these statistics are computed on *generated*
continuations.  Each is a per-sample number; a model is compared with people by
the overlap (``1 - KS``) between its per-sample distribution and the human one,
which is scale-free and penalises both looping and word salad.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .stats import ks_statistic, nanmean
from .text import words


def seq_rep_n(text: str, n: int = 4) -> float:
    """Fraction of repeated word n-grams in one text: ``1 - unique / total`` (Welleck et al. 2020).

    0 means every n-gram is distinct; a looping model approaches 1.  Texts with
    fewer than ``n`` words return 0.
    """
    tokens = words(text)
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    unique = len({tuple(tokens[i : i + n]) for i in range(total)})
    return 1.0 - unique / total


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Corpus-level diversity: distinct word n-grams over total n-grams across all ``texts``."""
    seen: set[tuple[str, ...]] = set()
    total = 0
    for text in texts:
        tokens = words(text)
        for i in range(len(tokens) - n + 1):
            seen.add(tuple(tokens[i : i + n]))
            total += 1
    return len(seen) / total if total else math.nan


def compression_ratio(text: str) -> float:
    """``len(zlib(text)) / len(text)`` in bytes: low for repetitive text, high for noise."""
    raw = text.encode("utf-8")
    if not raw:
        return math.nan
    return len(zlib.compress(raw, 9)) / len(raw)


def type_token_ratio(text: str) -> float:
    tokens = words(text)
    return len(set(tokens)) / len(tokens) if tokens else math.nan


#: Per-sample statistics compared against human text by default.
SURFACE_STATS: Mapping[str, Callable[[str], float]] = {
    "seq_rep_4": lambda t: seq_rep_n(t, 4),
    "compression_ratio": compression_ratio,
}


@dataclass
class StatComparison:
    """A per-sample statistic on model text versus human text."""

    name: str
    model_values: np.ndarray
    human_values: np.ndarray

    @property
    def model_mean(self) -> float:
        return nanmean(self.model_values)

    @property
    def human_mean(self) -> float:
        return nanmean(self.human_values)

    @property
    def ks(self) -> float:
        return ks_statistic(self.model_values, self.human_values)

    @property
    def overlap(self) -> float:
        """``1 - KS``: 1 when the model's distribution of this statistic matches people's."""
        ks = self.ks
        return math.nan if math.isnan(ks) else 1.0 - ks


def compare_statistic(
    name: str,
    statistic: Callable[[str], float],
    model_texts: Sequence[str],
    human_texts: Sequence[str],
) -> StatComparison:
    return StatComparison(
        name=name,
        model_values=np.array([statistic(t) for t in model_texts], dtype=np.float64),
        human_values=np.array([statistic(t) for t in human_texts], dtype=np.float64),
    )


def compare_surface(
    model_texts: Sequence[str],
    human_texts: Sequence[str],
    stats: Mapping[str, Callable[[str], float]] = SURFACE_STATS,
) -> dict[str, StatComparison]:
    """Compare every statistic in ``stats`` between model and human continuations."""
    return {
        name: compare_statistic(name, fn, model_texts, human_texts) for name, fn in stats.items()
    }

"""Small statistical helpers: two-sample overlap, bootstrap intervals, geometric means."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _clean(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return array[~np.isnan(array)]


def ks_statistic(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic ``sup_x |F_a(x) - F_b(x)|``.

    Returns ``NaN`` if either sample is empty after dropping ``NaN`` values.
    """
    a, b = _clean(a), _clean(b)
    if a.size == 0 or b.size == 0:
        return math.nan
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def overlap(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """``1 - KS``: 1 when two samples are drawn from the same distribution, 0 when disjoint.

    This is the package's scale-free notion of "indistinguishable from human" for a
    per-sample statistic.  It penalises deviations in either direction, so a
    model can lose points for being more repetitive *or* less repetitive than people.
    """
    ks = ks_statistic(a, b)
    return math.nan if math.isnan(ks) else 1.0 - ks


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:  # pragma: no cover - formatting
        if math.isnan(self.mean):
            return "n/a"
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for the mean of ``values`` (``NaN`` entries dropped)."""
    array = _clean(values)
    if array.size == 0:
        return Interval(math.nan, math.nan, math.nan, 0)
    if array.size == 1:
        v = float(array[0])
        return Interval(v, v, v, 1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return Interval(float(array.mean()), float(low), float(high), int(array.size))


def geometric_mean(
    values: Sequence[float], weights: Sequence[float] | None = None, *, floor: float = 0.0
) -> float:
    """Weighted geometric mean of non-negative ``values``; ``NaN`` entries are skipped.

    Each value is clipped below at ``floor`` first, so that a single zero does not
    erase the information carried by the other values.
    """
    pairs = [
        (v, 1.0 if weights is None else weights[i])
        for i, v in enumerate(values)
        if not math.isnan(v)
    ]
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return math.nan
    clipped = [(max(v, floor), w) for v, w in pairs]
    if any(v <= 0 and w > 0 for v, w in clipped):
        return 0.0
    return math.exp(sum(w * math.log(v) for v, w in clipped if w > 0) / total)


def nanmean(values: Sequence[float] | np.ndarray) -> float:
    array = _clean(values)
    return float(array.mean()) if array.size else math.nan

r"""``goodness(text)``: a reference-free quality score for one generated text.

The PQI needs a model and a held-out corpus.  Sometimes you only have the text a
model produced and want a decent number for it.  With no model and no paired
human continuation, the only principled question left is *how typical is this
text of human prose?* -- and that can be asked of every text-only statistic
that a degenerate model gets wrong:

===================== ========================= ==================================
statistic             failure direction         catches
===================== ========================= ==================================
``seq_rep_4``         too high                  looping, copied phrases
``compression_ratio`` too high                  word salad: text zlib cannot compress
``sentence_length``   too low *or* too high     fragments; run-ons / no punctuation
``new_name_rate``     too high                  a new character every sentence
``name_reuse``        too low                   never returning to a character
===================== ========================= ==================================

Each statistic is scored by its **typicality**: take the empirical p-value of
the text's value under a reference distribution of the same statistic over
human windows of the same length (one-sided in the declared failure direction,
or two-sided), and divide by the conventional significance level
``tolerance = 0.05``, capping at 1.  A value anywhere in the typical 95% of human
text scores 1; a value in the extreme tail decays linearly to 0 at the edge of
the human range.  The **goodness** of a text is its *weakest* typicality: every
statistic is a separate disqualifier, so a text is as good as its worst check,
and adding a check can only make the test stricter.  Human text scores 1 on most
passages and below 1 on roughly ``1 - 0.95 ** k`` of them for ``k`` checks.

The default :class:`Reference` ships with the package: quantiles computed by
``scripts/build_reference.py`` over windows of English narrative prose -- 14
public-domain novels (Project Gutenberg) balanced against simple children's
stories (TinyStories validation text) -- at several window lengths.  It is a
*broad* reference.  For any domain you care about -- your own training
distribution, news, dialogue -- calibrate with
``Reference.from_texts(my_human_texts)`` (a few hundred texts, seconds) and pass
it in; typicality is then relative to *your* prose, and the checks get sharper.

Text-only statistics cannot see meaning: "Tom thought about it for a moment
with his paw" is typical on every check above.  The optional **fluency check**
adds one more statistic, ``bits_per_byte`` of the text under a small reference
language model (GPT-2 by default), scored the same way against human windows
under the same model.  Nonsense is far less predictable than prose: in our
TinyStories check the 1M-parameter model's continuations exceed the human 95th
percentile 86% of the time while the coherent 33M model's do so 9% of the time.
``goodness(text, fluency=True)`` loads GPT-2 (``proseometry[torch]``);
``Reference.from_texts(texts, scorer=my_lm)`` calibrates for any other scorer.
Treat ``goodness`` as a fast filter for degeneration, incoherence and -- with
the fluency check -- gibberish; for the full picture of a *model* use the PQI.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .degeneration import compression_ratio, seq_rep_n
from .distribution import MissingDependency
from .likelihood import bits_per_byte
from .model import Model
from .stats import nanmean
from .text import WORD_RE, NameExtractor, find_names, word_count

# --------------------------------------------------------------------------- statistics

_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)|\n+")


def sentence_length(text: str) -> float:
    """Mean words per sentence; sentences end at ``.``, ``!``, ``?`` or a line break."""
    words = word_count(text)
    if words == 0:
        return math.nan
    sentences = [s for s in _SENTENCE_END_RE.split(text) if WORD_RE.search(s)]
    return words / max(1, len(sentences))


def new_name_rate(text: str, *, extractor: NameExtractor = find_names) -> float:
    """Distinct names per 100 words."""
    words = word_count(text)
    if words == 0:
        return math.nan
    return 100.0 * len({m.name for m in extractor(text)}) / words


def name_reuse(text: str, *, extractor: NameExtractor = find_names) -> float:
    """``1 - distinct names / name mentions``; ``NaN`` with fewer than two mentions."""
    mentions = extractor(text)
    if len(mentions) < 2:
        return math.nan
    return 1.0 - len({m.name for m in mentions}) / len(mentions)


@dataclass(frozen=True)
class TextStatistic:
    name: str
    fn: Callable[[str], float]
    direction: str
    """``"high"`` (large values are the failure), ``"low"``, or ``"both"``."""


TEXT_STATISTICS: tuple[TextStatistic, ...] = (
    TextStatistic("seq_rep_4", lambda t: seq_rep_n(t, 4), "high"),
    TextStatistic("compression_ratio", compression_ratio, "high"),
    TextStatistic("sentence_length", sentence_length, "both"),
    TextStatistic("new_name_rate", new_name_rate, "high"),
    TextStatistic("name_reuse", name_reuse, "low"),
)
"""The fixed text-only statistics behind :func:`goodness`, with their failure directions."""

FLUENCY = TextStatistic("bits_per_byte", lambda _text: math.nan, "high")
"""The optional fluency statistic: bits per byte under a reference language model."""

DIRECTIONS: dict[str, str] = {s.name: s.direction for s in (*TEXT_STATISTICS, FLUENCY)}


def text_statistics(text: str) -> dict[str, float]:
    """All :data:`TEXT_STATISTICS` of one text."""
    return {s.name: float(s.fn(text)) for s in TEXT_STATISTICS}


# --------------------------------------------------------------------------- reference

DEFAULT_LENGTHS: tuple[int, ...] = (50, 100, 200, 400, 800)
N_QUANTILES = 201
DEFAULT_TOLERANCE = 0.05
"""Significance level below which a statistic's p-value starts to count against the text."""


def p_value(value: float, reference_values: np.ndarray, direction: str) -> float:
    """Empirical p-value of ``value`` under sorted ``reference_values`` in the failure direction.

    ``"high"`` is the upper tail ``P(X >= value)``, ``"low"`` the lower tail
    ``P(X <= value)``, ``"both"`` twice the smaller of the two (capped at 1).  Tails
    are inclusive, so a value on a mass point shared with many human texts is not
    penalised.  ``NaN`` in, ``NaN`` out.
    """
    if math.isnan(value) or reference_values.size == 0:
        return math.nan
    n = reference_values.size
    lower = np.searchsorted(reference_values, value, side="right") / n  # P(X <= value)
    upper = 1.0 - np.searchsorted(reference_values, value, side="left") / n  # P(X >= value)
    if direction == "high":
        return float(upper)
    if direction == "low":
        return float(lower)
    if direction == "both":
        return float(min(1.0, 2.0 * min(lower, upper)))
    raise ValueError(f"unknown direction {direction!r}")


def typicality(
    value: float, reference_values: np.ndarray, direction: str, tolerance: float = DEFAULT_TOLERANCE
) -> float:
    """``min(1, p_value / tolerance)``: 1 in the typical range, linear decay to 0 in the tail."""
    p = p_value(value, reference_values, direction)
    return p if math.isnan(p) else float(min(1.0, p / tolerance))


@dataclass
class Reference:
    """Quantiles of each text statistic over human windows, by window length in words.

    ``quantiles[length][stat]`` is a sorted array of :data:`N_QUANTILES` values.
    Build one for your own domain with :meth:`from_texts`; the package default is
    :meth:`default`.
    """

    quantiles: dict[int, dict[str, np.ndarray]]
    name: str = "custom"
    info: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        lengths: Sequence[int] = DEFAULT_LENGTHS,
        windows_per_length: int = 2000,
        groups: Sequence[str] | None = None,
        min_group_texts: int = 20,
        scorer: Model | None = None,
        lm_windows_per_length: int = 500,
        seed: int = 0,
        name: str = "custom",
        extractor: NameExtractor = find_names,
    ) -> Reference:
        """Calibrate on human ``texts``: sample windows of each length and record quantiles.

        With a ``scorer`` the reference also records bits per byte under that model
        for the first ``lm_windows_per_length`` windows of each length, enabling the
        fluency check of :func:`goodness` for texts scored by the same model.

        Windows start at sentence boundaries, like the passages a model produces.
        Within a group, texts are drawn in
        proportion to the number of windows they can supply; when ``groups`` labels
        each text with a source, the sources are sampled equally often, so a corpus
        of short texts is not drowned out by a few long ones.  A source takes part
        at a window length only if at least ``min_group_texts`` of its texts are
        long enough (a handful of long texts must not stand in for a whole source);
        if no source qualifies, all eligible texts are pooled.
        """
        texts = list(texts)
        labels = list(groups) if groups is not None else ["all"] * len(texts)
        if len(labels) != len(texts):
            raise ValueError("groups must label every text")
        rng = np.random.default_rng(seed)
        spans_by_text = [list(WORD_RE.finditer(t)) for t in texts]
        starts_by_text = [
            _sentence_starts(t, spans) for t, spans in zip(texts, spans_by_text, strict=True)
        ]
        stats = _statistics_with_extractor(extractor)
        quantiles: dict[int, dict[str, np.ndarray]] = {}
        counts: dict[int, int] = {}
        for length in lengths:
            pools: dict[str, tuple[list[int], np.ndarray]] = {}
            candidates: dict[str, list[int]] = {label: [] for label in dict.fromkeys(labels)}
            usable_starts: dict[int, np.ndarray] = {}
            for i, spans in enumerate(spans_by_text):
                starts = starts_by_text[i][starts_by_text[i] <= len(spans) - length]
                if starts.size:
                    candidates[labels[i]].append(i)
                    usable_starts[i] = starts
            qualified = {k: v for k, v in candidates.items() if len(v) >= min_group_texts}
            if not qualified:
                pooled = [i for members in candidates.values() for i in members]
                qualified = {"all": pooled} if pooled else {}
            for label, eligible in qualified.items():
                weights = np.array([usable_starts[i].size for i in eligible], float)
                pools[label] = (eligible, weights / weights.sum())
            if not pools:
                continue
            pool_labels = list(pools)
            windows: list[str] = []
            for _ in range(windows_per_length):
                eligible, weights = pools[pool_labels[int(rng.integers(0, len(pool_labels)))]]
                i = int(rng.choice(eligible, p=weights))
                spans = spans_by_text[i]
                start = int(rng.choice(usable_starts[i]))
                windows.append(texts[i][spans[start].start() : spans[start + length - 1].end()])
            values = {s.name: np.array([s.fn(w) for w in windows], dtype=np.float64) for s in stats}
            if scorer is not None:
                values[FLUENCY.name] = bits_per_byte(scorer, windows[:lm_windows_per_length])
            quantiles[length] = {k: _quantiles(v) for k, v in values.items()}
            counts[length] = windows_per_length
        if not quantiles:
            raise ValueError("no text is long enough for any requested window length")
        info: dict[str, Any] = {
            "windows_per_length": counts,
            "n_texts": len(texts),
            "groups": sorted(set(labels)),
        }
        if scorer is not None:
            info["scorer"] = scorer.name
            info["lm_windows_per_length"] = lm_windows_per_length
        return cls(quantiles, name=name, info=info)

    @classmethod
    def default(cls) -> Reference:
        """The reference shipped with the package (public-domain English narrative prose)."""
        global _DEFAULT
        if _DEFAULT is None:
            path = resources.files("proseometry.data") / "reference_english_prose.json"
            _DEFAULT = cls.load(path)
        return _DEFAULT

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "info": self.info,
            "statistics": [s.name for s in TEXT_STATISTICS],
            "quantiles": {
                str(length): {
                    k: [None if math.isnan(x) else float(x) for x in v.tolist()]
                    for k, v in stats.items()
                }
                for length, stats in self.quantiles.items()
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Any) -> Reference:
        data = json.loads(
            Path(path).read_text(encoding="utf-8")
            if isinstance(path, (str, Path))
            else path.read_text(encoding="utf-8")
        )
        quantiles = {
            int(length): {
                k: np.array([math.nan if x is None else x for x in v], dtype=np.float64)
                for k, v in stats.items()
            }
            for length, stats in data["quantiles"].items()
        }
        return cls(quantiles, name=data.get("name", "custom"), info=data.get("info", {}))

    # ------------------------------------------------------------------ lookup
    @property
    def lengths(self) -> list[int]:
        return sorted(self.quantiles)

    @property
    def scorer(self) -> str | None:
        """Name of the language model whose bits per byte this reference holds, if any."""
        return self.info.get("scorer")

    def has(self, stat: str) -> bool:
        return all(stat in self.quantiles[length] for length in self.quantiles)

    def quantiles_for(self, stat: str, n_words: int) -> np.ndarray:
        """Reference quantiles of ``stat`` at ``n_words``, interpolated in log-length."""
        lengths = self.lengths
        n = min(max(n_words, lengths[0]), lengths[-1])
        if n in self.quantiles:
            return self.quantiles[n][stat]
        hi_index = int(np.searchsorted(lengths, n, side="left"))
        lo, hi = lengths[hi_index - 1], lengths[hi_index]
        q_lo, q_hi = self.quantiles[lo][stat], self.quantiles[hi][stat]
        weight = (math.log(n) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return _nan_sorted((1.0 - weight) * q_lo + weight * q_hi)

    def p_value(self, stat: str, value: float, n_words: int) -> float:
        """Empirical p-value of ``value`` for ``stat`` against human windows of ``n_words``."""
        return p_value(value, _drop_nan(self.quantiles_for(stat, n_words)), DIRECTIONS[stat])

    def typicality(
        self, stat: str, value: float, n_words: int, tolerance: float = DEFAULT_TOLERANCE
    ) -> float:
        p = self.p_value(stat, value, n_words)
        return p if math.isnan(p) else float(min(1.0, p / tolerance))


_DEFAULT: Reference | None = None


def _sentence_starts(text: str, spans: Sequence[re.Match[str]]) -> np.ndarray:
    """Indices of words that begin a sentence (text start, or after ``.!?`` or a line break)."""
    starts = [0] if spans else []
    for i in range(1, len(spans)):
        gap = text[spans[i - 1].end() : spans[i].start()]
        if "\n" in gap or any(c in gap for c in ".!?"):
            starts.append(i)
    return np.array(starts, dtype=np.int64)


def _quantiles(values: np.ndarray) -> np.ndarray:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.full(N_QUANTILES, np.nan)
    return np.quantile(values, np.linspace(0.0, 1.0, N_QUANTILES))


def _drop_nan(values: np.ndarray) -> np.ndarray:
    return values[~np.isnan(values)]


def _nan_sorted(values: np.ndarray) -> np.ndarray:
    return np.sort(values[~np.isnan(values)])


def _statistics_with_extractor(extractor: NameExtractor) -> tuple[TextStatistic, ...]:
    if extractor is find_names:
        return TEXT_STATISTICS
    swapped = {
        "new_name_rate": lambda t: new_name_rate(t, extractor=extractor),
        "name_reuse": lambda t: name_reuse(t, extractor=extractor),
    }
    return tuple(
        TextStatistic(s.name, swapped.get(s.name, s.fn), s.direction) for s in TEXT_STATISTICS
    )


# --------------------------------------------------------------------------- goodness

MIN_WORDS = 25


@dataclass
class Goodness:
    """The goodness of one text: its statistics, their p-values and typicality, and the score."""

    values: dict[str, float]
    p_values: dict[str, float]
    typicality: dict[str, float]
    n_words: int
    reference: str
    tolerance: float
    notes: list[str] = field(default_factory=list)

    @property
    def available(self) -> dict[str, float]:
        return {k: v for k, v in self.typicality.items() if not math.isnan(v)}

    @property
    def score(self) -> float:
        """The weakest typicality across the checks; ``NaN`` if none could be computed."""
        return min(self.available.values()) if self.available else math.nan

    @property
    def weakest(self) -> str | None:
        """Name of the statistic that set the score."""
        available = self.available
        return min(available, key=available.__getitem__) if available else None

    def summary(self) -> str:
        if math.isnan(self.score):
            return "goodness n/a (" + "; ".join(self.notes) + ")"
        checks = ", ".join(f"{k} {v:.2f}" for k, v in self.typicality.items() if not math.isnan(v))
        return f"goodness {self.score:.3f} (weakest: {self.weakest}; {checks})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "weakest": self.weakest,
            "values": dict(self.values),
            "p_values": dict(self.p_values),
            "typicality": dict(self.typicality),
            "n_words": self.n_words,
            "reference": self.reference,
            "tolerance": self.tolerance,
            "notes": list(self.notes),
        }


DEFAULT_SCORER = "gpt2"
_DEFAULT_SCORER: Model | None = None


def default_scorer() -> Model:
    """GPT-2 (124M) as a :class:`Model`, loaded once; needs ``proseometry[torch]``."""
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        try:
            from .adapters import HFModel
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise MissingDependency(
                "the fluency check needs torch and transformers: install `proseometry[torch]`"
            ) from exc
        _DEFAULT_SCORER = HFModel.from_pretrained(DEFAULT_SCORER, name=DEFAULT_SCORER)
    return _DEFAULT_SCORER


def goodness_many(
    texts: Iterable[str],
    reference: Reference | None = None,
    *,
    scorer: Model | None = None,
    fluency: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    extractor: NameExtractor = find_names,
) -> list[Goodness]:
    """Score texts for degeneration and incoherence against human prose.

    ``reference`` defaults to the shipped English-prose reference; build one with
    :meth:`Reference.from_texts` to calibrate to your own domain.  ``tolerance`` is
    the p-value below which a check starts to count against the text.

    The fluency check runs when ``scorer`` is given or ``fluency=True`` (which uses
    GPT-2); the reference must have been calibrated with the same scorer.  Texts
    shorter than :data:`MIN_WORDS` words get a ``NaN`` score and a note.
    """
    texts = list(texts)
    ref = reference or Reference.default()
    if fluency and scorer is None:
        scorer = default_scorer()
    if scorer is not None:
        if not ref.has(FLUENCY.name):
            raise ValueError(
                f"reference {ref.name!r} has no fluency quantiles; build it with "
                "Reference.from_texts(..., scorer=...)"
            )
        if ref.scorer != scorer.name:
            raise ValueError(
                f"reference {ref.name!r} was calibrated with scorer {ref.scorer!r}, "
                f"not {scorer.name!r}; bits per byte are only comparable under one model"
            )
    stats = _statistics_with_extractor(extractor)
    lm_values = bits_per_byte(scorer, texts) if scorer is not None else None

    results: list[Goodness] = []
    for index, text in enumerate(texts):
        n_words = word_count(text)
        values = {s.name: float(s.fn(text)) for s in stats}
        if lm_values is not None:
            values[FLUENCY.name] = float(lm_values[index])
        notes: list[str] = []
        if n_words < MIN_WORDS:
            notes.append(f"only {n_words} words; at least {MIN_WORDS} are needed for a score")
            p_values = {k: math.nan for k in values}
        else:
            p_values = {k: ref.p_value(k, v, n_words) for k, v in values.items()}
            if n_words > ref.lengths[-1] * 2:
                notes.append(
                    f"{n_words} words is far beyond the reference's longest window "
                    f"({ref.lengths[-1]}); score passages of a few hundred words instead"
                )
        typ = {k: (p if math.isnan(p) else min(1.0, p / tolerance)) for k, p in p_values.items()}
        results.append(
            Goodness(
                values=values,
                p_values=p_values,
                typicality=typ,
                n_words=n_words,
                reference=ref.name,
                tolerance=tolerance,
                notes=notes,
            )
        )
    return results


def goodness(
    text: str,
    reference: Reference | None = None,
    *,
    scorer: Model | None = None,
    fluency: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    extractor: NameExtractor = find_names,
) -> Goodness:
    """Score one text; see :func:`goodness_many` for the parameters."""
    return goodness_many(
        [text], reference, scorer=scorer, fluency=fluency, tolerance=tolerance, extractor=extractor
    )[0]


def mean_goodness(results: Sequence[Goodness]) -> float:
    """Mean score over results with a defined score."""
    return nanmean([r.score for r in results])


__all__ = [
    "DEFAULT_LENGTHS",
    "DEFAULT_SCORER",
    "DEFAULT_TOLERANCE",
    "DIRECTIONS",
    "FLUENCY",
    "Goodness",
    "MIN_WORDS",
    "Reference",
    "TEXT_STATISTICS",
    "TextStatistic",
    "default_scorer",
    "goodness",
    "goodness_many",
    "mean_goodness",
    "name_reuse",
    "new_name_rate",
    "p_value",
    "sentence_length",
    "text_statistics",
    "typicality",
]

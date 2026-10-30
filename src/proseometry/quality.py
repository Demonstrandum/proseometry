r"""The Prose Quality Index (PQI): one number for "sensible text that uses its context".

Perplexity fails in two independent ways: it averages over all tokens (so the
few tokens that need long context vanish) and it conditions on human text (so
free-running degeneration is invisible).  PQI is built from measurements that
do not share those blind spots, each expressed on a common scale:

* every sub-measure lies in ``[0, 1]`` and **1 means human level** -- either
  "indistinguishable from human text on this view" (MAUVE, ``1 - KS`` overlaps)
  or "perfect chance-corrected skill" (name binding), so no population of
  models is needed to normalise anything;
* sub-measures are grouped into three complementary components, averaged
  arithmetically within a component (they are alternative views of one
  property) and **geometrically across components** (they are different
  properties, and failing any one of them is fatal to quality, so a strength in
  one cannot buy forgiveness for a collapse in another).

Components
----------
``realism``
    Does free-running text read like human text?  MAUVE between model and human
    continuations; optionally the judge's coherence score.
``surface``
    Is the repetition / entropy profile human?  Overlap of per-sample ``seq-rep-4``
    and compression ratio with the human distribution.
``memory``
    Does the model use and retain what came before?  Chance-corrected name-binding
    skill (teacher-forced), overlap of callback and new-name rates in generated
    continuations with human continuations, and optionally the judge's consistency.

.. math::

    \mathrm{PQI} = \exp\Big(\sum_c w_c \ln \max(\bar m_c, \epsilon)\Big), \qquad \sum_c w_c = 1

where :math:`\bar m_c` is the mean of the available sub-measures of component
``c`` and :math:`\epsilon` (default 0.01) keeps a total failure on one axis from
erasing the ordering carried by the others.

PQI is designed for *ranking models under one fixed protocol*: same prompts,
human reference, decoding settings, sample counts and MAUVE featuriser.  Its
absolute value moves with those choices; the components are always reported so
that a score can be read, not just compared.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .stats import geometric_mean, nanmean

COMPONENTS: tuple[str, ...] = ("realism", "surface", "memory")
DEFAULT_WEIGHTS: Mapping[str, float] = {name: 1.0 for name in COMPONENTS}
DEFAULT_FLOOR = 0.01


@dataclass
class Component:
    """One axis of quality: a named set of sub-measures in ``[0, 1]``."""

    name: str
    measures: dict[str, float] = field(default_factory=dict)

    @property
    def available(self) -> dict[str, float]:
        return {k: v for k, v in self.measures.items() if not math.isnan(v)}

    @property
    def value(self) -> float:
        """Arithmetic mean of the available sub-measures (``NaN`` if none)."""
        return nanmean(list(self.measures.values()))


@dataclass
class QualityIndex:
    score: float
    components: dict[str, Component]
    weights: dict[str, float]
    floor: float

    @property
    def missing(self) -> list[str]:
        """Components that had no available sub-measure and were skipped."""
        return [name for name, c in self.components.items() if math.isnan(c.value)]

    @property
    def partial(self) -> bool:
        return bool(self.missing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "partial": self.partial,
            "missing": self.missing,
            "weights": dict(self.weights),
            "floor": self.floor,
            "components": {
                name: {"value": c.value, "measures": dict(c.measures)}
                for name, c in self.components.items()
            },
        }

    def summary(self) -> str:
        parts = ", ".join(
            f"{name} {c.value:.3f}" if not math.isnan(c.value) else f"{name} n/a"
            for name, c in self.components.items()
        )
        head = "n/a" if math.isnan(self.score) else f"{self.score:.3f}"
        return f"PQI {head} ({parts})" + (" [partial]" if self.partial else "")


def _clip_unit(value: float) -> float:
    if value is None or math.isnan(value):
        return math.nan
    return min(1.0, max(0.0, float(value)))


def prose_quality_index(
    *,
    realism: Mapping[str, float] | None = None,
    surface: Mapping[str, float] | None = None,
    memory: Mapping[str, float] | None = None,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    floor: float = DEFAULT_FLOOR,
) -> QualityIndex:
    """Combine sub-measures into a :class:`QualityIndex`.

    Each mapping holds sub-measure values in ``[0, 1]`` (values outside are
    clipped, ``NaN`` marks an unavailable measure).  Components with no available
    measure are left out of the geometric mean and listed in ``missing``.
    """
    groups = {"realism": realism or {}, "surface": surface or {}, "memory": memory or {}}
    components = {
        name: Component(name, {k: _clip_unit(v) for k, v in values.items()})
        for name, values in groups.items()
    }
    names = list(components)
    score = geometric_mean(
        [components[n].value for n in names],
        [float(weights.get(n, 0.0)) for n in names],
        floor=floor,
    )
    return QualityIndex(
        score=score,
        components=components,
        weights={n: float(weights.get(n, 0.0)) for n in names},
        floor=floor,
    )

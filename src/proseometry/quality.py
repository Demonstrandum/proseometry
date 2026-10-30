r"""The Prose Quality Index (PQI): one fixed statistic for "sensible text that uses its context".

Perplexity fails in two independent ways: it averages over all tokens (so the
few tokens that need long context vanish) and it conditions on human text (so
free-running degeneration is invisible).  PQI is a *fixed* set of measurements
that do not share those blind spots, combined in a fixed way, computable from a
model and a held-out corpus alone: :func:`prose_quality_index`.

Definition
----------
Three components, each the arithmetic mean of its sub-measures; the index is
their geometric mean.  Every sub-measure lies in ``[0, 1]`` and **1 means human
level** -- either "indistinguishable from human text on this view" (MAUVE,
``1 - KS`` overlaps of per-sample statistics) or "perfect chance-corrected
skill" (name binding).  Nothing is normalised against a population of models.

``realism``
    *MAUVE* between ``prompt + model continuation`` and ``prompt + human
    continuation`` (Pillutla et al. 2021).  Does free-running text read like
    human text?  This is the axis that catches fluent nonsense.
``surface``
    Overlap with the human distribution of per-sample *seq-rep-4* (Welleck et
    al. 2020) and *zlib compression ratio*.  Is the repetition / entropy profile
    human?  Two-sided: loops and word salad both lose.
``memory``
    *Name-binding skill* -- at long-range recurrences of a character's name, the
    chance-corrected accuracy of preferring the right name among those
    introduced earlier -- plus overlap of the *callback rate* and *new-name rate*
    of generated continuations with human continuations.  Does the model use
    and retain what came before?

.. math::

    \mathrm{PQI} = \Big(\prod_{c} \max(\bar m_c, \epsilon)\Big)^{1/3}

with :math:`\bar m_c` the mean of component ``c``'s sub-measures and a floor
:math:`\epsilon = 0.01` so that a total failure on one axis still leaves the
ordering carried by the others.  Arithmetic within a component because its
sub-measures are alternative views of one property; geometric across components
because they are different properties and failing any one is fatal to quality --
a strength in one cannot buy forgiveness for a collapse in another.

What is deliberately left out
-----------------------------
*Context gain* and *LongPPL* are reported by the suite as diagnostics but are not
in the index: the first has no human-level anchor in ``[0, 1]``, the second needs
an external evaluator model and so is not a function of the model and corpus
alone.  *LLM-judge* scores are excluded because they are expensive, depend on a
third-party model and prompt, and are not a formal statistic.  Bits per byte is
the thing PQI exists to complement.

A PQI is either complete or it is not: :attr:`PQI.score` is ``NaN`` when any
component could not be measured (MAUVE not installed, no eligible prompts, no
recurring names), and :attr:`PQI.partial_score` gives the geometric mean of what
is available for diagnostics.  Two partial indices with different missing sets
are not comparable.

PQI ranks models *under one fixed protocol* (:class:`PQIConfig`): same held-out
texts, prompt and continuation lengths, decoding settings, sample counts and
MAUVE featuriser.  Its absolute value moves with those choices, which is why the
components are always reported next to it.  :class:`CompositeIndex` is the
generic machinery underneath, available for indices of your own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .consistency import compare_name_usage
from .degeneration import SURFACE_STATS, StatComparison, compare_surface
from .distribution import MauveResult, MissingDependency, mauve_available, mauve_score
from .generation import GenerationSet, sample_continuations
from .model import Model
from .recall import BindingResult, name_binding
from .stats import geometric_mean, nanmean
from .text import NameExtractor, find_names

DEFAULT_FLOOR = 0.01


# --------------------------------------------------------------------------- generic index


@dataclass
class Component:
    """One axis of an index: a named set of sub-measures in ``[0, 1]`` (``NaN`` = unavailable)."""

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
class CompositeIndex:
    """Weighted geometric mean of components that are arithmetic means of ``[0, 1]`` measures.

    This is the generic combiner; :class:`PQI` is the fixed instance that
    proseometry stands behind.  Build your own with :func:`composite_index`.
    """

    components: dict[str, Component]
    weights: dict[str, float]
    floor: float = DEFAULT_FLOOR

    @property
    def missing(self) -> list[str]:
        """Weighted components that have no available sub-measure."""
        return [
            name
            for name, c in self.components.items()
            if self.weights.get(name, 0.0) > 0 and math.isnan(c.value)
        ]

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def partial_score(self) -> float:
        """Geometric mean over the components that *are* available (diagnostic only)."""
        names = list(self.components)
        return geometric_mean(
            [self.components[n].value for n in names],
            [self.weights.get(n, 0.0) for n in names],
            floor=self.floor,
        )

    @property
    def score(self) -> float:
        """The index; ``NaN`` unless every weighted component is available."""
        return self.partial_score if self.complete else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "partial_score": self.partial_score,
            "complete": self.complete,
            "missing": self.missing,
            "weights": dict(self.weights),
            "floor": self.floor,
            "components": {
                name: {"value": c.value, "measures": dict(c.measures)}
                for name, c in self.components.items()
            },
        }

    def summary(self, label: str = "index") -> str:
        parts = ", ".join(
            f"{name} {c.value:.3f}" if not math.isnan(c.value) else f"{name} n/a"
            for name, c in self.components.items()
        )
        if self.complete:
            return f"{label} {self.score:.3f} ({parts})"
        partial = "n/a" if math.isnan(self.partial_score) else f"{self.partial_score:.3f}"
        return f"{label} n/a -- missing {', '.join(self.missing)}; partial {partial} ({parts})"


def _clip_unit(value: float | None) -> float:
    if value is None or math.isnan(value):
        return math.nan
    return min(1.0, max(0.0, float(value)))


def composite_index(
    components: Mapping[str, Mapping[str, float]],
    *,
    weights: Mapping[str, float] | None = None,
    floor: float = DEFAULT_FLOOR,
) -> CompositeIndex:
    """Build a :class:`CompositeIndex` from ``{component: {measure: value}}``.

    Values outside ``[0, 1]`` are clipped; ``NaN`` marks an unavailable measure.
    Weights default to equal.
    """
    built = {
        name: Component(name, {k: _clip_unit(v) for k, v in measures.items()})
        for name, measures in components.items()
    }
    resolved = {name: float((weights or {}).get(name, 1.0)) for name in built}
    return CompositeIndex(built, resolved, floor)


# --------------------------------------------------------------------------- the PQI

PQI_MEASURES: Mapping[str, tuple[str, ...]] = {
    "realism": ("mauve",),
    "surface": ("seq_rep_4_overlap", "compression_ratio_overlap"),
    "memory": ("binding_skill", "callback_rate_overlap", "new_name_rate_overlap"),
}
"""The fixed sub-measures of the PQI, by component."""


@dataclass
class PQIConfig:
    """The measurement protocol behind the PQI.  Keep it fixed across the models you compare."""

    max_doc_tokens: int = 1024
    """Documents are truncated to this many tokens (and to the model's context)."""

    # Name binding (teacher-forced) ------------------------------------------
    binding_min_gap: int = 64
    """A name must have been absent for this many tokens to count as a long-range callback."""
    binding_max_positions: int | None = 32
    """Cap on probed positions per document (seeded random subset), to bound cost."""

    # Generation --------------------------------------------------------------
    prompt_tokens: int = 128
    continuation_tokens: int = 256
    samples_per_prompt: int = 1
    max_prompts: int | None = None
    """Use at most this many eligible documents as prompts (seeded random subset)."""
    temperature: float = 1.0
    top_p: float = 0.95

    # MAUVE -------------------------------------------------------------------
    mauve: bool = True
    """Compute MAUVE.  Without it the ``realism`` component is missing and the PQI is partial."""
    mauve_featurizer: str = "gpt2-large"
    mauve_device_id: int = -1
    mauve_max_text_length: int = 512
    mauve_batch_size: int = 8
    mauve_include_prompt: bool = True
    """Score ``prompt + continuation`` (as in the MAUVE paper), not the continuation alone."""

    seed: int = 0


@dataclass
class PQI(CompositeIndex):
    """The Prose Quality Index with the measurements it was computed from.

    Construct with :func:`prose_quality_index` (measures everything) or
    :meth:`PQI.from_measurements` (if you already have the pieces).
    """

    binding: BindingResult | None = None
    surface: dict[str, StatComparison] = field(default_factory=dict)
    name_usage: dict[str, StatComparison] = field(default_factory=dict)
    mauve: MauveResult | None = None
    generation: GenerationSet | None = None
    config: PQIConfig | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_measurements(
        cls,
        *,
        binding: BindingResult | None,
        surface: Mapping[str, StatComparison],
        name_usage: Mapping[str, StatComparison],
        mauve: MauveResult | None,
        generation: GenerationSet | None = None,
        config: PQIConfig | None = None,
        notes: Sequence[str] = (),
    ) -> PQI:
        """Assemble the PQI from its constituent measurements.

        Missing inputs become ``NaN`` sub-measures, so the result reports exactly
        which component could not be measured.
        """

        def overlap(group: Mapping[str, StatComparison], key: str) -> float:
            return group[key].overlap if key in group else math.nan

        measures = {
            "realism": {"mauve": mauve.mauve if mauve is not None else math.nan},
            "surface": {
                "seq_rep_4_overlap": overlap(surface, "seq_rep_4"),
                "compression_ratio_overlap": overlap(surface, "compression_ratio"),
            },
            "memory": {
                "binding_skill": binding.skill if binding is not None else math.nan,
                "callback_rate_overlap": overlap(name_usage, "callback_rate"),
                "new_name_rate_overlap": overlap(name_usage, "new_name_rate"),
            },
        }
        assert {k: tuple(v) for k, v in measures.items()} == {
            k: tuple(v) for k, v in PQI_MEASURES.items()
        }
        base = composite_index(measures, floor=DEFAULT_FLOOR)
        return cls(
            components=base.components,
            weights=base.weights,
            floor=base.floor,
            binding=binding,
            surface=dict(surface),
            name_usage=dict(name_usage),
            mauve=mauve,
            generation=generation,
            config=config,
            notes=list(notes),
        )

    def summary(self, label: str = "PQI") -> str:
        return super().summary(label)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["evidence"] = {
            "binding_positions": self.binding.n_positions if self.binding else 0,
            "prompts": len(self.generation.prompts) if self.generation else 0,
            "samples": len(self.generation) if self.generation else 0,
            "mauve_samples": self.mauve.n_model if self.mauve else 0,
        }
        data["notes"] = list(self.notes)
        return data


def prose_quality_index(
    model: Model,
    texts: Sequence[str],
    config: PQIConfig | None = None,
    *,
    extractor: NameExtractor = find_names,
) -> PQI:
    """Measure the Prose Quality Index of ``model`` on held-out ``texts``.

    Runs the name-binding probe, samples continuations, compares their surface
    and name-usage statistics with the human continuations, computes MAUVE, and
    combines the pieces as defined in this module.  Raises
    :class:`MissingDependency` if ``config.mauve`` is set but ``mauve-text`` is
    not installed; set ``mauve=False`` to accept a partial index instead.
    """
    cfg = config or PQIConfig()
    if cfg.mauve and not mauve_available():
        raise MissingDependency(
            "the PQI needs MAUVE: install `proseometry[mauve]`, "
            "or pass PQIConfig(mauve=False) for a partial index"
        )
    notes: list[str] = []
    docs = [model.tokenize_doc(t, cfg.max_doc_tokens) for t in texts]

    binding = name_binding(
        model,
        docs,
        min_gap=cfg.binding_min_gap,
        max_positions_per_doc=cfg.binding_max_positions,
        extractor=extractor,
        seed=cfg.seed,
    )
    if binding.n_positions == 0:
        notes.append("name binding: no recurring names with a long enough gap were found")

    generation = sample_continuations(
        model,
        docs,
        prompt_tokens=cfg.prompt_tokens,
        continuation_tokens=cfg.continuation_tokens,
        samples_per_prompt=cfg.samples_per_prompt,
        max_prompts=cfg.max_prompts,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=cfg.seed,
    )
    surface: dict[str, StatComparison] = {}
    usage: dict[str, StatComparison] = {}
    mauve: MauveResult | None = None
    if generation is None:
        notes.append(
            "generation skipped: no document has prompt_tokens + continuation_tokens tokens"
        )
    else:
        surface = compare_surface(generation.model, generation.human, SURFACE_STATS)
        usage = compare_name_usage(
            generation.prompts,
            generation.model,
            generation.human,
            model_prompt_index=generation.prompt_index,
            extractor=extractor,
        )
        if cfg.mauve:
            mauve = mauve_for(generation, cfg)
            if len(generation.prompts) < 1000:
                notes.append(
                    f"MAUVE on {len(generation.prompts)} prompts is optimistic and noisy; "
                    "a few thousand samples per side is the recommended regime"
                )
    return PQI.from_measurements(
        binding=binding,
        surface=surface,
        name_usage=usage,
        mauve=mauve,
        generation=generation,
        config=cfg,
        notes=notes,
    )


def mauve_for(generation: GenerationSet, cfg: PQIConfig) -> MauveResult:
    """MAUVE between the human and model sides of ``generation`` under ``cfg``."""
    if cfg.mauve_include_prompt:
        p_text, q_text = generation.human_full(), generation.model_full()
    else:
        p_text, q_text = generation.human, generation.model
    return mauve_score(
        p_text,
        q_text,
        featurize_model_name=cfg.mauve_featurizer,
        device_id=cfg.mauve_device_id,
        max_text_length=cfg.mauve_max_text_length,
        batch_size=cfg.mauve_batch_size,
        seed=cfg.seed + 25,
        verbose=False,
    )

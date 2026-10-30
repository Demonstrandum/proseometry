"""Run every metric under one fixed protocol and produce a comparable report.

:class:`Suite` holds the protocol: the PQI's :class:`~proseometry.quality.PQIConfig`
plus the diagnostics-only settings (context-gain windows, LongPPL evaluator
parameters, optional judge).  Calling :meth:`Suite.run` on several models with
the same suite and the same held-out texts gives reports whose numbers --
including the PQI -- are directly comparable.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .consistency import compare_name_usage
from .degeneration import SURFACE_STATS, StatComparison, compare_surface
from .distribution import MauveResult, MissingDependency, mauve_available
from .generation import GenerationSet, sample_continuations
from .judge import Judge, JudgeResult, judge_continuations
from .likelihood import (
    ContextGain,
    ContextScores,
    LikelihoodSummary,
    LongPPL,
    context_gain,
    context_scores,
    key_spans,
    key_token_mask,
    likelihood_summary,
    long_ppl,
    mask_from_spans,
)
from .model import Model, TokenizedDoc
from .quality import PQI, PQIConfig, mauve_for
from .recall import BindingResult, name_binding
from .text import NameExtractor, find_names


@dataclass
class SuiteConfig(PQIConfig):
    """The PQI protocol plus the suite's diagnostics.  Keep it fixed across compared models."""

    context_windows: tuple[int, ...] = (16, 64, 256)
    """Short-context sizes ``k`` for the context-gain curve (diagnostic, not in the PQI)."""
    short_stride: int = 1
    """1 = exact ``k``-token contexts; larger = cheaper, slightly longer contexts."""

    long_ppl_k: int = 64
    long_ppl_alpha: float = 2.0
    long_ppl_beta: float = -2.0
    """LongPPL key-token selection (diagnostic; needs an evaluator model)."""

    verbose: bool = True


@dataclass
class Report:
    model: str
    config: SuiteConfig
    n_docs: int
    likelihood: LikelihoodSummary
    context_gain: dict[int, ContextGain]
    long_ppl: LongPPL | None
    binding: BindingResult
    generation: GenerationSet | None
    surface: dict[str, StatComparison]
    name_usage: dict[str, StatComparison]
    mauve: MauveResult | None
    judge: JudgeResult | None
    pqi: PQI
    notes: list[str] = field(default_factory=list)
    scores: list[ContextScores] | None = None

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict[str, Any]:
        def comparisons(group: dict[str, StatComparison]) -> dict[str, dict[str, float]]:
            return {
                k: {
                    "model_mean": v.model_mean,
                    "human_mean": v.human_mean,
                    "ks": v.ks,
                    "overlap": v.overlap,
                }
                for k, v in group.items()
            }

        return {
            "model": self.model,
            "n_docs": self.n_docs,
            "pqi": self.pqi.to_dict(),
            "likelihood": {
                "bits_per_byte": self.likelihood.bits_per_byte,
                "perplexity": self.likelihood.perplexity,
                "nll_per_token": self.likelihood.nll_per_token,
                "n_tokens": self.likelihood.n_tokens,
                "n_bytes": self.likelihood.n_bytes,
            },
            "context_gain": {
                str(k): {
                    "gain_nats": g.gain_nats,
                    "ppl_reduction": g.ppl_reduction,
                    "n_tokens": g.n_tokens,
                }
                for k, g in self.context_gain.items()
            },
            "long_ppl": None
            if self.long_ppl is None
            else {
                "long_ppl": self.long_ppl.long_ppl,
                "n_key_tokens": self.long_ppl.n_key_tokens,
                "key_fraction": self.long_ppl.key_fraction,
            },
            "binding": {
                "accuracy": self.binding.accuracy,
                "chance": self.binding.chance,
                "recency_accuracy": self.binding.recency_accuracy,
                "skill": self.binding.skill,
                "recency_skill": self.binding.recency_skill,
                "true_name_nll": self.binding.true_name_nll,
                "n_positions": self.binding.n_positions,
            },
            "generation": None
            if self.generation is None
            else {
                "n_prompts": len(self.generation.prompts),
                "n_samples": len(self.generation.model),
                "prompt_tokens": self.config.prompt_tokens,
                "continuation_tokens": self.config.continuation_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
            "surface": comparisons(self.surface),
            "name_usage": comparisons(self.name_usage),
            "mauve": None if self.mauve is None else self.mauve.to_dict(),
            "judge": None if self.judge is None else self.judge.to_dict(),
            "notes": list(self.notes),
            "config": asdict(self.config),
        }

    def to_json(self, **kwargs: Any) -> str:
        """JSON with ``NaN`` rendered as ``null`` and numpy scalars converted."""
        kwargs.setdefault("indent", 2)
        return json.dumps(_jsonable(self.to_dict()), **kwargs)

    def to_markdown(self) -> str:
        rows: list[tuple[str, str]] = []
        lik = self.likelihood
        rows.append(("documents / scored tokens", f"{self.n_docs} / {lik.n_tokens}"))
        rows.append(("bits per byte", _fmt(lik.bits_per_byte)))
        rows.append(("perplexity (per token)", _fmt(lik.perplexity, 2)))
        for k, g in sorted(self.context_gain.items()):
            rows.append(
                (
                    f"context gain beyond k={k}",
                    f"{_fmt(g.gain_nats)} nats/token ({_pct(g.ppl_reduction)} PPL reduction)",
                )
            )
        if self.long_ppl is not None:
            rows.append(
                (
                    f"LongPPL (k={self.config.long_ppl_k})",
                    f"{_fmt(self.long_ppl.long_ppl, 2)} on {self.long_ppl.n_key_tokens} key tokens "
                    f"({_pct(self.long_ppl.key_fraction)} of tokens)",
                )
            )
        b = self.binding
        rows.append(
            (
                f"name binding (gap >= {self.config.binding_min_gap})",
                f"accuracy {_fmt(b.accuracy)} vs chance {_fmt(b.chance)} / recency "
                f"{_fmt(b.recency_accuracy)}; skill {_fmt(b.skill)}; n={b.n_positions}",
            )
        )
        for name, cmp in {**self.surface, **self.name_usage}.items():
            rows.append(
                (
                    name,
                    f"model {_fmt(cmp.model_mean)} vs human {_fmt(cmp.human_mean)}; "
                    f"overlap {_fmt(cmp.overlap)}",
                )
            )
        if self.mauve is not None:
            rows.append(
                (
                    "MAUVE",
                    f"{_fmt(self.mauve.mauve)} (n={self.mauve.n_model} vs {self.mauve.n_human})",
                )
            )
        if self.judge is not None:
            rows.append(
                (
                    "judge (1-5, not in PQI)",
                    ", ".join(f"{c} {_fmt(self.judge.mean(c), 2)}" for c in self.judge.criteria)
                    + f" (parsed {self.judge.n_parsed}/{len(self.judge.per_sample)})",
                )
            )
        lines = [f"## {self.model} -- {self.pqi.summary()}", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
        if self.notes:
            lines += ["", *[f"- {note}" for note in self.notes]]
        return "\n".join(lines)


class Suite:
    """Runs the whole metric suite on a :class:`Model` under a fixed :class:`SuiteConfig`.

    Parameters
    ----------
    config:
        The protocol.
    evaluator:
        A *stronger* reference model used only to select LongPPL key tokens.  Fang
        et al. (2025) show self-evaluation does not discriminate, so this should
        not be one of the models being compared.  Omit to skip LongPPL.
    judge:
        Optional ``Callable[[str], str]`` LLM judge (see :mod:`proseometry.judge`).
        Reported as a diagnostic; never part of the PQI.
    extractor:
        Name detector used by the binding probe and the callback statistics.
    """

    def __init__(
        self,
        config: SuiteConfig | None = None,
        *,
        evaluator: Model | None = None,
        judge: Judge | None = None,
        extractor: NameExtractor = find_names,
    ) -> None:
        self.config = config or SuiteConfig()
        self.evaluator = evaluator
        self.judge = judge
        self.extractor = extractor
        self._key_span_cache: dict[str, list[list[tuple[int, int]]]] = {}
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------ logging
    def _log(self, message: str) -> None:
        if self.config.verbose:
            elapsed = time.monotonic() - self._t0
            print(f"[proseometry +{elapsed:7.1f}s] {message}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ pieces
    def _evaluator_key_spans(self, texts: Sequence[str]) -> list[list[tuple[int, int]]]:
        assert self.evaluator is not None
        cfg = self.config
        digest = hashlib.sha1()
        for t in texts:
            digest.update(t.encode("utf-8"))
            digest.update(b"\x00")
        key = "|".join(
            [
                digest.hexdigest(),
                self.evaluator.name,
                str(cfg.max_doc_tokens),
                str(cfg.long_ppl_k),
                str(cfg.long_ppl_alpha),
                str(cfg.long_ppl_beta),
                str(cfg.short_stride),
            ]
        )
        if key not in self._key_span_cache:
            self._log(f"selecting LongPPL key tokens with evaluator {self.evaluator.name!r}")
            spans: list[list[tuple[int, int]]] = []
            for text in texts:
                doc = self.evaluator.tokenize_doc(text, cfg.max_doc_tokens)
                sc = context_scores(self.evaluator, doc, [cfg.long_ppl_k], stride=cfg.short_stride)
                mask = key_token_mask(
                    sc, cfg.long_ppl_k, alpha=cfg.long_ppl_alpha, beta=cfg.long_ppl_beta
                )
                spans.append(key_spans(doc, mask))
            self._key_span_cache[key] = spans
        return self._key_span_cache[key]

    def generate(self, model: Model, docs: Sequence[TokenizedDoc]) -> GenerationSet | None:
        """Sample continuations under the suite's protocol (see :func:`sample_continuations`)."""
        cfg = self.config
        return sample_continuations(
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

    # ------------------------------------------------------------------ main entry
    def run(self, model: Model, texts: Sequence[str], *, name: str | None = None) -> Report:
        """Evaluate ``model`` on held-out ``texts`` and return a :class:`Report`."""
        cfg = self.config
        label = name or model.name
        texts = list(texts)
        notes: list[str] = []
        if self.evaluator is model:
            notes.append(
                "evaluator is the evaluated model: LongPPL key tokens are self-selected, "
                "which Fang et al. (2025) found not to discriminate between models"
            )

        self._log(f"[{label}] tokenizing {len(texts)} documents")
        docs = [model.tokenize_doc(t, cfg.max_doc_tokens) for t in texts]

        windows = set(cfg.context_windows)
        if self.evaluator is not None:
            windows.add(cfg.long_ppl_k)
        self._log(f"[{label}] scoring full context and short windows {sorted(windows)}")
        scores = [context_scores(model, d, windows, stride=cfg.short_stride) for d in docs]
        likelihood = likelihood_summary(scores)
        gains = {k: context_gain(scores, k) for k in cfg.context_windows}

        long = None
        if self.evaluator is not None:
            spans = self._evaluator_key_spans(texts)
            long = long_ppl(
                (sc.full, mask_from_spans(sc.doc, sp)) for sc, sp in zip(scores, spans, strict=True)
            )
            if long.n_key_tokens == 0:
                notes.append("LongPPL: evaluator selected no key tokens (try lower alpha or k)")

        self._log(f"[{label}] probing name binding")
        binding = name_binding(
            model,
            docs,
            min_gap=cfg.binding_min_gap,
            max_positions_per_doc=cfg.binding_max_positions,
            extractor=self.extractor,
            seed=cfg.seed,
        )
        if binding.n_positions == 0:
            notes.append("name binding: no recurring names with a long enough gap were found")

        self._log(f"[{label}] sampling continuations")
        generation = self.generate(model, docs)
        surface: dict[str, StatComparison] = {}
        usage: dict[str, StatComparison] = {}
        mauve: MauveResult | None = None
        judge: JudgeResult | None = None
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
                extractor=self.extractor,
            )
            if cfg.mauve:
                if mauve_available():
                    self._log(f"[{label}] computing MAUVE with {cfg.mauve_featurizer}")
                    try:
                        mauve = mauve_for(generation, cfg)
                    except (RuntimeError, ValueError, MissingDependency) as exc:
                        notes.append(f"MAUVE failed; the rest of the report is unaffected: {exc}")
                    if mauve is not None and len(generation.prompts) < 1000:
                        notes.append(
                            f"MAUVE on {len(generation.prompts)} prompts is optimistic and noisy; "
                            "a few thousand samples per side is the recommended regime"
                        )
                else:
                    notes.append(
                        "MAUVE skipped: `mauve-text` is not installed (proseometry[mauve]); "
                        "the PQI is partial"
                    )
            if self.judge is not None:
                self._log(f"[{label}] judging {len(generation)} continuations")
                judge = judge_continuations(
                    self.judge, generation.model_prompts(), generation.model
                )

        pqi = PQI.from_measurements(
            binding=binding,
            surface=surface,
            name_usage=usage,
            mauve=mauve,
            generation=generation,
            config=cfg,
            notes=notes,
        )
        self._log(f"[{label}] done: {pqi.summary()}")
        return Report(
            model=label,
            config=cfg,
            n_docs=likelihood.n_docs,
            likelihood=likelihood,
            context_gain=gains,
            long_ppl=long,
            binding=binding,
            generation=generation,
            surface=surface,
            name_usage=usage,
            mauve=mauve,
            judge=judge,
            pqi=pqi,
            notes=notes,
            scores=scores,
        )


def compare(reports: Sequence[Report]) -> str:
    """One markdown table, one row per report, sorted by PQI (partial indices marked with *)."""

    def rank(r: Report) -> float:
        value = r.pqi.score if r.pqi.complete else r.pqi.partial_score
        return -value if not math.isnan(value) else math.inf

    ordered = sorted(reports, key=rank)
    has_long = any(r.long_ppl is not None for r in reports)
    has_mauve = any(r.mauve is not None for r in reports)
    any_partial = any(not r.pqi.complete for r in reports)
    header = ["model", "PQI", "realism", "surface", "memory", "bits/byte", "PPL"]
    ks = sorted({k for r in reports for k in r.context_gain})
    header += [f"PPL drop @k={k}" for k in ks]
    if has_long:
        header.append("LongPPL")
    header += ["binding skill", "seq-rep-4 (model/human)"]
    if has_mauve:
        header.append("MAUVE")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in ordered:
        c = r.pqi.components
        pqi_cell = _fmt(r.pqi.score) if r.pqi.complete else f"{_fmt(r.pqi.partial_score)}*"
        row = [
            r.model,
            pqi_cell,
            _fmt(c["realism"].value),
            _fmt(c["surface"].value),
            _fmt(c["memory"].value),
            _fmt(r.likelihood.bits_per_byte),
            _fmt(r.likelihood.perplexity, 2),
        ]
        row += [_pct(r.context_gain[k].ppl_reduction) if k in r.context_gain else "" for k in ks]
        if has_long:
            row.append(_fmt(r.long_ppl.long_ppl, 2) if r.long_ppl else "")
        rep = r.surface.get("seq_rep_4")
        row += [
            _fmt(r.binding.skill),
            f"{_fmt(rep.model_mean)} / {_fmt(rep.human_mean)}" if rep else "",
        ]
        if has_mauve:
            row.append(_fmt(r.mauve.mauve) if r.mauve else "")
        lines.append("| " + " | ".join(row) + " |")
    if any_partial:
        lines.append("")
        lines.append(
            "\\* partial PQI: a component is missing (see each report's notes); "
            "only comparable across rows missing the same component"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- helpers


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100 * value:.1f}%"


def _jsonable(value: Any) -> Any:
    """Recursively convert to plain JSON types; NaN/inf become ``None``."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

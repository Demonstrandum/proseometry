"""Name binding: can the model tell *which* character comes back after a long gap?

Perplexity averages over every token; this probe looks only at the positions
where a name recurs after being absent for at least ``min_gap`` tokens.  At each
such position the model must assign the highest probability, among all names
introduced earlier in the document, to the one that actually appears.  Scoring
the choice *among candidates* (rather than against the whole vocabulary) keeps
legitimate alternatives such as pronouns from counting as errors: the question
is not "would you mention a name here" but "if you do, do you remember who".

Two reference points are reported alongside accuracy: chance (``1 / |candidates|``)
and a recency heuristic that always picks the most recently mentioned name.
The chance-corrected *skill* ``(accuracy - chance) / (1 - chance)`` is what the
quality index uses, so a model with no long-range memory scores 0 and a model
that never confuses its characters scores 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import Model, TokenizedDoc
from .stats import Interval, bootstrap_mean, nanmean
from .text import NameExtractor, find_names, tokens_covering


@dataclass
class BindingRecord:
    """One probed position."""

    doc: int
    token_index: int
    true_name: str
    candidates: list[str]
    logprobs: list[float]
    recency_pick: str

    @property
    def predicted(self) -> str:
        return self.candidates[int(np.argmax(self.logprobs))]

    @property
    def correct(self) -> bool:
        return self.predicted == self.true_name

    @property
    def recency_correct(self) -> bool:
        return self.recency_pick == self.true_name

    @property
    def chance(self) -> float:
        return 1.0 / len(self.candidates)

    @property
    def true_nll(self) -> float:
        return -self.logprobs[self.candidates.index(self.true_name)]


@dataclass
class BindingResult:
    records: list[BindingRecord]
    n_docs: int

    @property
    def n_positions(self) -> int:
        return len(self.records)

    @property
    def accuracy(self) -> float:
        return nanmean([float(r.correct) for r in self.records])

    @property
    def chance(self) -> float:
        return nanmean([r.chance for r in self.records])

    @property
    def recency_accuracy(self) -> float:
        return nanmean([float(r.recency_correct) for r in self.records])

    @property
    def true_name_nll(self) -> float:
        return nanmean([r.true_nll for r in self.records])

    @property
    def skill(self) -> float:
        return skill_score(self.accuracy, self.chance)

    @property
    def recency_skill(self) -> float:
        return skill_score(self.recency_accuracy, self.chance)

    def accuracy_interval(self, seed: int = 0) -> Interval:
        return bootstrap_mean([float(r.correct) for r in self.records], seed=seed)


def skill_score(accuracy: float, chance: float) -> float:
    """Chance-corrected accuracy, clipped at 0: 0 = no better than guessing, 1 = perfect."""
    if math.isnan(accuracy) or math.isnan(chance) or chance >= 1.0:
        return math.nan
    return max(0.0, (accuracy - chance) / (1.0 - chance))


@dataclass
class _Position:
    mention_start: int
    token_start: int
    true_name: str
    candidates: list[str]
    recency_pick: str


def binding_positions(
    doc: TokenizedDoc, *, min_gap: int, extractor: NameExtractor = find_names
) -> list[_Position]:
    """Recurrences of a name after >= ``min_gap`` tokens of absence, with >= 2 candidates."""
    positions: list[_Position] = []
    last_end: dict[str, int] = {}  # name -> token index just after its latest mention
    for mention in extractor(doc.text):
        covering = tokens_covering(doc.spans, mention.start, mention.end)
        if covering is None:
            continue
        token_start, token_end = covering
        if token_start == 0:
            last_end[mention.name] = token_end
            continue
        is_recurrence = mention.name in last_end and len(last_end) >= 2
        if is_recurrence and last_end[mention.name] <= token_start - min_gap:
            candidates = sorted(last_end)
            recency = max(candidates, key=lambda name: last_end[name])
            positions.append(
                _Position(mention.start, token_start, mention.name, candidates, recency)
            )
        last_end[mention.name] = max(token_end, last_end.get(mention.name, 0))
    return positions


def name_binding(
    model: Model,
    docs: Sequence[TokenizedDoc],
    *,
    min_gap: int = 64,
    max_positions_per_doc: int | None = 32,
    extractor: NameExtractor = find_names,
    seed: int = 0,
) -> BindingResult:
    """Run the name-binding probe over tokenized documents.

    For each probed position every candidate name is scored as a continuation of
    the document prefix, tokenized in the same local context as the true mention
    (the characters of the mention's first token that precede the name, usually a
    space, are prepended before encoding).  Candidates are compared on total
    log-probability, which is the probability the model would *produce* that name.
    """
    rng = np.random.default_rng(seed)
    records: list[BindingRecord] = []
    for doc_index, doc in enumerate(docs):
        positions = binding_positions(doc, min_gap=min_gap, extractor=extractor)
        if max_positions_per_doc is not None and len(positions) > max_positions_per_doc:
            chosen = sorted(rng.choice(len(positions), size=max_positions_per_doc, replace=False))
            positions = [positions[i] for i in chosen]
        if not positions:
            continue
        sequences: list[list[int]] = []
        prefix_lengths: list[int] = []
        for pos in positions:
            lead = doc.text[doc.spans[pos.token_start][0] : pos.mention_start]
            prefix = doc.ids[: pos.token_start]
            for name in pos.candidates:
                sequences.append(prefix + model.encode(lead + name))
                prefix_lengths.append(len(prefix))
        scored = model.score(sequences)
        cursor = 0
        for pos in positions:
            logprobs = []
            for _ in pos.candidates:
                lp = scored[cursor][prefix_lengths[cursor] :]
                logprobs.append(float(np.sum(lp)))
                cursor += 1
            records.append(
                BindingRecord(
                    doc=doc_index,
                    token_index=pos.token_start,
                    true_name=pos.true_name,
                    candidates=list(pos.candidates),
                    logprobs=logprobs,
                    recency_pick=pos.recency_pick,
                )
            )
    return BindingResult(records, n_docs=len(docs))

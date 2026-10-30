from __future__ import annotations

import math

import numpy as np
import pytest

from proseometry.model import LogitsModel
from proseometry.recall import binding_positions, name_binding, skill_score
from proseometry.toy import BigramModel, WordTokenizer

FILLER = " ".join(["the dog ran and the cat sat"] * 4)  # 28 tokens of names-free filler
STORY = (
    f"Lily went to the park. Tom came too. {FILLER}. "
    f"Then Lily smiled at Tom. {FILLER}. Tom laughed."
)


class FavouriteModel(LogitsModel):
    """Puts almost all probability on one token, whatever the context."""

    def __init__(self, tokenizer: WordTokenizer, favourite: str) -> None:
        super().__init__(tokenizer, name="favourite", bos_id=tokenizer.bos_id)
        self.favourite = tokenizer.stoi[favourite]
        self.vocab = len(tokenizer)

    def logits(self, ids: np.ndarray) -> np.ndarray:
        out = np.zeros((*ids.shape, self.vocab), dtype=np.float32)
        out[..., self.favourite] = 10.0
        return out


@pytest.fixture
def story_tokenizer() -> WordTokenizer:
    return WordTokenizer.fit([STORY])


def test_binding_positions_require_gap_and_two_candidates(story_tokenizer: WordTokenizer) -> None:
    model = FavouriteModel(story_tokenizer, "Lily")
    doc = model.tokenize_doc(STORY)
    # Token positions: Lily@0, Tom@6, Lily@40, Tom@43, Tom@74.
    positions = binding_positions(doc, min_gap=20)
    names = [(p.true_name, p.candidates, p.recency_pick) for p in positions]
    assert names == [
        ("Lily", ["Lily", "Tom"], "Tom"),  # Lily absent for 39 tokens, Tom seen more recently
        ("Tom", ["Lily", "Tom"], "Lily"),  # Tom absent for 36 tokens, Lily seen more recently
        ("Tom", ["Lily", "Tom"], "Tom"),  # Tom absent for 30 tokens, still the most recent name
    ]
    assert [p.token_start for p in positions] == [40, 43, 74]
    # Raising the gap drops the recurrences whose previous mention is too close.
    assert [p.true_name for p in binding_positions(doc, min_gap=38)] == ["Lily"]
    assert binding_positions(doc, min_gap=1000) == []


def test_name_binding_with_favourite_model(story_tokenizer: WordTokenizer) -> None:
    model = FavouriteModel(story_tokenizer, "Lily")
    doc = model.tokenize_doc(STORY)
    result = name_binding(model, [doc], min_gap=20, max_positions_per_doc=None)
    assert result.n_positions == 3
    assert [r.predicted for r in result.records] == ["Lily", "Lily", "Lily"]
    assert result.accuracy == pytest.approx(1 / 3)
    assert result.chance == pytest.approx(0.5)
    assert result.recency_accuracy == pytest.approx(1 / 3)
    assert result.skill == 0.0  # below chance clips to zero
    assert result.true_name_nll > 0

    tom = FavouriteModel(story_tokenizer, "Tom")
    tom_result = name_binding(tom, [doc], min_gap=20)
    assert tom_result.accuracy == pytest.approx(2 / 3)
    assert tom_result.skill == pytest.approx(1 / 3)


def test_skill_score() -> None:
    assert skill_score(1.0, 0.5) == 1.0
    assert skill_score(0.5, 0.5) == 0.0
    assert skill_score(0.3, 0.5) == 0.0  # clipped
    assert skill_score(0.75, 0.5) == pytest.approx(0.5)
    assert math.isnan(skill_score(math.nan, 0.5))
    assert math.isnan(skill_score(1.0, 1.0))


def test_name_binding_on_corpus_is_near_chance_for_bigram(
    bigram: BigramModel, held_out: list[str]
) -> None:
    docs = [bigram.tokenize_doc(t) for t in held_out]
    result = name_binding(bigram, docs, min_gap=12, max_positions_per_doc=16, seed=0)
    assert result.n_positions > 20
    assert result.chance == pytest.approx(0.5)
    assert 0.3 <= result.accuracy <= 0.75
    assert result.recency_accuracy < result.chance  # the probe targets non-recent names
    assert all(len(r.candidates) == 2 for r in result.records)

from __future__ import annotations

import math

import numpy as np
import pytest

from proseometry import toy
from proseometry.goodness import (
    DIRECTIONS,
    FLUENCY,
    MIN_WORDS,
    TEXT_STATISTICS,
    Reference,
    goodness,
    goodness_many,
    mean_goodness,
    name_reuse,
    new_name_rate,
    p_value,
    sentence_length,
    text_statistics,
    typicality,
)

HUMAN = (
    "Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom "
    "liked to play near the pit. One day, Tom lost his red ball. He was very sad. Tom asked his "
    "friend, Sam, to help him search for the ball. They looked high and low, but they could not "
    'find the ball. Tom said, "I think my ball fell into the pit." Sam and Tom went close to the '
    "pit. They were scared, but they wanted to find the red ball. They looked into the pit, but "
    'it was too dark to see. Tom said, "We must go in and search for my ball." They went into '
    "the pit to search. It was dark and scary. They could not find the ball."
)
LOOP = "Tom and Sam went to the park. " * 25
RUN_ON = " ".join(["the children walked along the river and saw many things and went home"] * 8)


def test_text_statistics() -> None:
    assert sentence_length("One two three. Four five. Six!") == pytest.approx(2.0)
    assert sentence_length("no punctuation at all here") == 5.0
    assert math.isnan(sentence_length(""))
    # "Then" only ever starts a sentence, so it is not a name; Lily, Tom and Zoe are.
    assert new_name_rate("Then Lily met Tom and Tom met Zoe. " * 10) == pytest.approx(300 / 80)
    assert name_reuse("Lily met Tom. Then Lily left.") == pytest.approx(1 - 2 / 3)
    assert math.isnan(name_reuse("Lily left."))
    stats = text_statistics(HUMAN)
    assert set(stats) == {s.name for s in TEXT_STATISTICS}
    assert set(DIRECTIONS) == set(stats) | {FLUENCY.name}


def test_p_value_and_typicality_directions() -> None:
    ref = np.arange(1, 101, dtype=float)  # 1..100
    assert p_value(100.0, ref, "high") == pytest.approx(0.01)
    assert p_value(1.0, ref, "high") == 1.0
    assert p_value(1.0, ref, "low") == pytest.approx(0.01)
    assert p_value(50.0, ref, "both") == pytest.approx(1.0)
    assert p_value(3.0, ref, "both") == pytest.approx(0.06)
    assert p_value(1000.0, ref, "high") == 0.0
    assert math.isnan(p_value(math.nan, ref, "high"))
    with pytest.raises(ValueError):
        p_value(1.0, ref, "sideways")
    # Inclusive tails: a value on a mass point shared with half the reference is typical.
    massed = np.array([0.0] * 50 + list(range(1, 51)), dtype=float)
    assert p_value(0.0, massed, "low") == pytest.approx(0.5)
    assert typicality(97.0, ref, "high", tolerance=0.05) == pytest.approx(0.8)
    assert typicality(20.0, ref, "high", tolerance=0.05) == 1.0


def test_default_reference_scores_human_text_as_typical_and_degenerate_text_as_not() -> None:
    ref = Reference.default()
    assert ref.lengths == [50, 100, 200, 400, 800]
    assert ref.scorer == "gpt2" and ref.has(FLUENCY.name)
    human = goodness(HUMAN)
    assert human.score >= 0.8  # a simple children's story under a broad reference
    assert all(v == 1.0 for k, v in human.typicality.items() if k != "seq_rep_4")
    assert set(human.typicality) == {s.name for s in TEXT_STATISTICS}  # no fluency by default
    loop = goodness(LOOP)
    assert loop.score == 0.0 and loop.weakest == "seq_rep_4"
    run_on = goodness(RUN_ON)
    assert run_on.score == 0.0 and run_on.typicality["sentence_length"] == 0.0
    assert "seq_rep_4" in loop.summary() and loop.to_dict()["weakest"] == "seq_rep_4"


def test_short_text_gets_nan_and_a_note() -> None:
    result = goodness("Hello there, Tom.")
    assert math.isnan(result.score) and result.weakest is None
    assert any(str(MIN_WORDS) in note for note in result.notes)
    assert "n/a" in result.summary()
    assert math.isnan(mean_goodness([result]))
    assert mean_goodness([result, goodness(HUMAN)]) == goodness(HUMAN).score


def test_reference_from_texts_round_trips_and_calibrates(tmp_path) -> None:
    stories = toy.synthetic_stories(300, seed=3)
    ref = Reference.from_texts(stories, lengths=(50, 100, 200), windows_per_length=300, name="toy")
    assert ref.lengths == [50, 100, 200] and ref.scorer is None
    assert all(ref.quantiles[50][s.name].shape == (201,) for s in TEXT_STATISTICS)
    path = tmp_path / "ref.json"
    ref.save(path)
    loaded = Reference.load(path)
    assert loaded.name == "toy" and loaded.lengths == ref.lengths
    np.testing.assert_allclose(loaded.quantiles[100]["seq_rep_4"], ref.quantiles[100]["seq_rep_4"])
    # Texts from the calibration distribution are typical under it; a loop is not.
    scores = [r.score for r in goodness_many(toy.synthetic_stories(40, seed=9), loaded)]
    assert np.mean(scores) > 0.8
    assert goodness(LOOP, loaded).score == 0.0
    # Interpolation between buckets and clamping outside them both work.
    assert loaded.quantiles_for("sentence_length", 75).shape == (201,)
    assert loaded.quantiles_for("sentence_length", 5000) is loaded.quantiles[200]["sentence_length"]


def test_reference_groups_balance_sources_and_skip_thin_groups() -> None:
    short = toy.synthetic_stories(100, seed=1, length=(6, 8))
    long = [" ".join(toy.synthetic_stories(12, seed=i)) for i in range(25)]
    ref = Reference.from_texts(
        short + long,
        groups=["short"] * len(short) + ["long"] * len(long),
        lengths=(50, 400),
        windows_per_length=200,
        min_group_texts=20,
    )
    assert ref.lengths == [50, 400] and ref.info["groups"] == ["long", "short"]
    with pytest.raises(ValueError):
        Reference.from_texts(short, groups=["a"], lengths=(50,))
    with pytest.raises(ValueError):
        Reference.from_texts(["too short"], lengths=(50,))


def test_fluency_requires_matching_scorer() -> None:
    ref = Reference.from_texts(
        toy.synthetic_stories(60, seed=2), lengths=(50,), windows_per_length=50
    )
    scorer = toy.BigramModel.fit(toy.WordTokenizer.fit(["a b"]), ["a b"])
    with pytest.raises(ValueError, match="no fluency quantiles"):
        goodness(HUMAN, ref, scorer=scorer)
    other = toy.BigramModel.fit(toy.WordTokenizer.fit(["a b"]), ["a b"], name="other")
    calibrated = Reference.from_texts(
        toy.synthetic_stories(60, seed=2), lengths=(50,), windows_per_length=50, scorer=scorer
    )
    assert calibrated.scorer == "bigram" and calibrated.has(FLUENCY.name)
    with pytest.raises(ValueError, match="calibrated with scorer"):
        goodness(HUMAN, calibrated, scorer=other)


def test_fluency_check_with_a_toy_scorer_flags_unpredictable_text() -> None:
    stories = toy.synthetic_stories(400, seed=5)
    tokenizer = toy.WordTokenizer.fit(stories)
    scorer = toy.InductionModel.fit(tokenizer, stories[:300])
    ref = Reference.from_texts(
        stories[300:],
        lengths=(50, 100),
        windows_per_length=200,
        scorer=scorer,
        lm_windows_per_length=200,
    )
    typical = goodness(toy.synthetic_stories(1, seed=77)[0], ref, scorer=scorer)
    assert FLUENCY.name in typical.typicality and typical.typicality[FLUENCY.name] == 1.0
    gibberish = goodness(" ".join(reversed(stories[0].split())), ref, scorer=scorer)
    assert gibberish.typicality[FLUENCY.name] < 1.0
    assert gibberish.score < typical.score

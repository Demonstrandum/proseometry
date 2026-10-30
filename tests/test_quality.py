from __future__ import annotations

import math

import numpy as np
import pytest

from proseometry import PQI, PQIConfig, composite_index, prose_quality_index
from proseometry.degeneration import StatComparison
from proseometry.distribution import MauveResult, MissingDependency, mauve_available
from proseometry.judge import JudgeResult, judge_continuations, parse_scores
from proseometry.quality import PQI_MEASURES
from proseometry.recall import BindingRecord, BindingResult
from proseometry.toy import BigramModel, InductionModel

# --------------------------------------------------------------------------- generic combiner


def test_composite_index_is_geometric_mean_of_component_means() -> None:
    index = composite_index(
        {"a": {"x": 0.8}, "b": {"y": 0.6, "z": 0.4}, "c": {"w": 0.2}}  # b has mean 0.5
    )
    assert index.complete
    assert index.score == pytest.approx((0.8 * 0.5 * 0.2) ** (1 / 3))
    assert index.components["b"].value == pytest.approx(0.5)
    assert "index" in index.summary()


def test_composite_index_is_nan_when_incomplete_but_keeps_partial_score() -> None:
    index = composite_index({"a": {"x": math.nan}, "b": {"y": 0.9}, "c": {"z": 0.4}})
    assert not index.complete and index.missing == ["a"]
    assert math.isnan(index.score)
    assert index.partial_score == pytest.approx(math.sqrt(0.9 * 0.4))
    assert "missing a" in index.summary()
    assert index.to_dict()["score"] != index.to_dict()["score"]  # NaN


def test_composite_index_floor_weights_and_clipping() -> None:
    weak = composite_index({"a": {"x": 0.9}, "b": {"y": 0.9}, "c": {"z": 0.0}})
    weaker = composite_index({"a": {"x": 0.5}, "b": {"y": 0.5}, "c": {"z": 0.0}})
    assert weak.score == pytest.approx((0.9 * 0.9 * 0.01) ** (1 / 3))
    assert weak.score > weaker.score > 0.0
    weighted = composite_index(
        {"a": {"x": 1.7}, "b": {"y": -0.2}, "c": {"z": 1.0}}, weights={"a": 1.0, "b": 0.0, "c": 1.0}
    )
    assert weighted.components["a"].value == 1.0 and weighted.components["b"].value == 0.0
    assert weighted.score == pytest.approx(1.0)  # zero-weight component is ignored
    assert weighted.missing == []


# --------------------------------------------------------------------------- the fixed PQI


def _comparison(name: str, model: float, human: float) -> StatComparison:
    # 1 - KS of two point masses: 1.0 when equal, 0.0 when different.
    return StatComparison(name, np.array([model] * 5), np.array([human] * 5))


def _binding(correct: int, total: int) -> BindingResult:
    records = [
        BindingRecord(
            0, i, "Lily", ["Lily", "Tom"], [0.0, -1.0] if i < correct else [-1.0, 0.0], "Tom"
        )
        for i in range(total)
    ]
    return BindingResult(records, n_docs=1)


def test_pqi_from_measurements_uses_exactly_the_fixed_measure_set() -> None:
    pqi = PQI.from_measurements(
        binding=_binding(correct=4, total=4),
        surface={
            "seq_rep_4": _comparison("seq_rep_4", 0.01, 0.01),
            "compression_ratio": _comparison("compression_ratio", 0.5, 0.6),
        },
        name_usage={
            "callback_rate": _comparison("callback_rate", 3.0, 3.0),
            "new_name_rate": _comparison("new_name_rate", 1.0, 1.0),
        },
        mauve=MauveResult(0.5, 0.2, 0.5, 10, 10),
    )
    assert {k: tuple(c.measures) for k, c in pqi.components.items()} == dict(PQI_MEASURES)
    assert pqi.complete
    assert pqi.components["realism"].value == 0.5
    assert pqi.components["surface"].value == pytest.approx(0.5)  # overlaps 1.0 and 0.0
    assert pqi.components["memory"].value == pytest.approx(1.0)  # skill 1, overlaps 1, 1
    assert pqi.score == pytest.approx((0.5 * 0.5 * 1.0) ** (1 / 3))
    assert pqi.summary().startswith("PQI 0.630")
    assert pqi.to_dict()["evidence"]["binding_positions"] == 4


def test_pqi_without_mauve_is_partial_not_silently_redefined() -> None:
    pqi = PQI.from_measurements(
        binding=_binding(correct=2, total=4),  # accuracy 0.5 = chance -> skill 0
        surface={"seq_rep_4": _comparison("seq_rep_4", 0.0, 0.0)},  # compression missing
        name_usage={},
        mauve=None,
    )
    assert pqi.missing == ["realism"]
    assert math.isnan(pqi.score)
    assert math.isnan(pqi.components["surface"].measures["compression_ratio_overlap"])
    assert pqi.components["surface"].value == 1.0  # mean of the available overlap only
    assert pqi.components["memory"].value == 0.0
    assert pqi.partial_score == pytest.approx(math.sqrt(1.0 * 0.01))


@pytest.fixture(scope="module")
def pqi_config() -> PQIConfig:
    return PQIConfig(
        max_doc_tokens=512,
        binding_min_gap=12,
        binding_max_positions=8,
        prompt_tokens=40,
        continuation_tokens=60,
        max_prompts=6,
        mauve=False,
    )


def test_prose_quality_index_runs_from_model_and_texts(
    pqi_config: PQIConfig, bigram: BigramModel, induction: InductionModel, held_out: list[str]
) -> None:
    for model in (bigram, induction):
        pqi = prose_quality_index(model, held_out, pqi_config)
        assert isinstance(pqi, PQI)
        assert pqi.missing == ["realism"] and math.isnan(pqi.score)
        assert 0.0 <= pqi.partial_score <= 1.0
        assert pqi.binding is not None and pqi.binding.n_positions > 0
        assert pqi.generation is not None and len(pqi.generation) == 6
        assert set(pqi.surface) == {"seq_rep_4", "compression_ratio"}
        assert set(pqi.name_usage) == {"callback_rate", "new_name_rate"}
        assert pqi.config is pqi_config


def test_prose_quality_index_requires_mauve_unless_told_otherwise(
    bigram: BigramModel, held_out: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("proseometry.quality.mauve_available", lambda: False)
    with pytest.raises(MissingDependency):
        prose_quality_index(bigram, held_out[:2], PQIConfig())


@pytest.mark.network
def test_prose_quality_index_complete_with_tiny_featurizer(
    pqi_config: PQIConfig, induction: InductionModel, held_out: list[str]
) -> None:
    if not mauve_available():
        pytest.skip("mauve-text not installed")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import dataclasses

    config = dataclasses.replace(
        pqi_config, mauve=True, mauve_featurizer="hf-internal-testing/tiny-random-gpt2"
    )
    try:
        pqi = prose_quality_index(induction, held_out, config)
    except RuntimeError as exc:  # pragma: no cover - offline
        pytest.skip(f"featuriser unavailable: {exc}")
    assert pqi.complete
    assert 0.0 <= pqi.score <= 1.0
    assert pqi.mauve is not None and pqi.mauve.n_model == 6


# --------------------------------------------------------------------------- judge (not in PQI)


def test_parse_scores_accepts_json_anywhere_and_rejects_garbage() -> None:
    assert parse_scores('Sure! {"grammar": 5, "coherence": 3.5, "consistency": 9}') == {
        "grammar": 5.0,
        "coherence": 3.5,
        "consistency": 5.0,
    }
    assert parse_scores('{"grammar": "five"}') is None
    assert parse_scores("no json here") is None


def test_judge_continuations_aggregates_and_normalises() -> None:
    replies = iter(['{"grammar": 5, "coherence": 4, "consistency": 3}', "oops"])
    result = judge_continuations(lambda _prompt: next(replies), ["p1", "p2"], ["c1", "c2"])
    assert isinstance(result, JudgeResult)
    assert result.n_parsed == 1
    assert result.mean("coherence") == 4.0
    assert result.normalized("consistency") == pytest.approx(0.5)
    assert result.to_dict()["n_total"] == 2

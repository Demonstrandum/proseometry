from __future__ import annotations

import math

import pytest

from proseometry.judge import JudgeResult, judge_continuations, parse_scores
from proseometry.quality import prose_quality_index


def test_pqi_is_geometric_mean_of_component_means() -> None:
    index = prose_quality_index(
        realism={"mauve": 0.8},
        surface={"seq_rep_4_overlap": 0.6, "compression_ratio_overlap": 0.4},  # mean 0.5
        memory={"binding_skill": 0.2},
    )
    assert index.score == pytest.approx((0.8 * 0.5 * 0.2) ** (1 / 3))
    assert not index.partial
    assert index.components["surface"].value == pytest.approx(0.5)
    assert "PQI" in index.summary()


def test_pqi_skips_missing_components_and_flags_partial() -> None:
    index = prose_quality_index(realism={"mauve": math.nan}, surface={"x": 0.9}, memory={"y": 0.4})
    assert index.partial and index.missing == ["realism"]
    assert index.score == pytest.approx(math.sqrt(0.9 * 0.4))
    assert (
        index.to_dict()["components"]["realism"]["value"]
        != index.to_dict()["components"]["realism"]["value"]
    )  # NaN


def test_pqi_floor_keeps_ordering_when_one_axis_collapses() -> None:
    weak = prose_quality_index(realism={"m": 0.9}, surface={"s": 0.9}, memory={"b": 0.0})
    weaker = prose_quality_index(realism={"m": 0.5}, surface={"s": 0.5}, memory={"b": 0.0})
    assert weak.score == pytest.approx((0.9 * 0.9 * 0.01) ** (1 / 3))
    assert weak.score > weaker.score > 0.0


def test_pqi_clips_values_to_unit_interval_and_applies_weights() -> None:
    index = prose_quality_index(
        realism={"m": 1.7},
        surface={"s": -0.2},
        memory={"b": 1.0},
        weights={"realism": 1.0, "surface": 0.0, "memory": 1.0},
    )
    assert index.components["realism"].value == 1.0
    assert index.components["surface"].value == 0.0
    assert index.score == pytest.approx(1.0)


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

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from proseometry import Suite, SuiteConfig, compare
from proseometry.toy import BigramModel, InductionModel


@pytest.fixture(scope="module")
def config() -> SuiteConfig:
    return SuiteConfig(
        max_doc_tokens=512,
        context_windows=(2, 8),
        long_ppl_k=16,
        long_ppl_alpha=0.5,
        long_ppl_beta=-4.0,
        binding_min_gap=12,
        binding_max_positions=8,
        prompt_tokens=40,
        continuation_tokens=60,
        max_prompts=6,
        mauve=False,
        verbose=False,
    )


@pytest.fixture(scope="module")
def reports(config, bigram: BigramModel, induction: InductionModel, held_out: list[str]):
    suite = Suite(config, evaluator=induction)
    return suite.run(bigram, held_out), suite.run(induction, held_out)


def test_reports_have_expected_structure(reports, held_out: list[str]) -> None:
    for report in reports:
        assert report.n_docs == len(held_out)
        assert report.likelihood.n_tokens > 0
        assert set(report.context_gain) == {2, 8}
        assert report.long_ppl is not None and report.long_ppl.n_key_tokens > 0
        assert report.binding.n_positions > 0
        assert report.generation is not None and len(report.generation) == 6
        assert set(report.surface) == {"seq_rep_4", "compression_ratio"}
        assert set(report.name_usage) == {"callback_rate", "new_name_rate"}
        assert report.mauve is None
        assert report.pqi.missing == ["realism"] and math.isnan(report.pqi.score)
        assert 0.0 <= report.pqi.partial_score <= 1.0
        assert report.judge is None


def test_bigram_has_no_context_gain_and_induction_does(reports) -> None:
    bigram_report, induction_report = reports
    for k in (2, 8):
        assert abs(bigram_report.context_gain[k].gain_nats) < 1e-6
        assert induction_report.context_gain[k].gain_nats > 0.0
    assert induction_report.long_ppl.long_ppl < bigram_report.long_ppl.long_ppl


def test_self_evaluation_is_flagged(reports) -> None:
    bigram_report, induction_report = reports
    assert not any("evaluator is the evaluated model" in n for n in bigram_report.notes)
    assert any("evaluator is the evaluated model" in n for n in induction_report.notes)


def test_report_exports(reports) -> None:
    report = reports[1]
    data = json.loads(report.to_json())
    assert data["model"] == "induction"
    assert data["pqi"]["components"]["realism"]["value"] is None  # NaN -> null
    assert data["pqi"]["score"] is None and data["pqi"]["partial_score"] > 0
    assert data["context_gain"]["8"]["gain_nats"] > 0
    markdown = report.to_markdown()
    assert markdown.startswith("## induction -- PQI")
    assert "LongPPL" in markdown and "name binding" in markdown
    table = compare(list(reports))
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(rows) == 3  # header + two models
    assert "*" in rows[1] and "partial PQI" in table  # partial indices are marked
    by_partial = sorted(reports, key=lambda r: -r.pqi.partial_score)
    assert rows[1].startswith(f"| {by_partial[0].model} ")
    assert report.generation.examples(1).startswith("--- prompt")


def test_generation_rejects_protocol_that_exceeds_model_context(
    config, bigram: BigramModel, held_out
) -> None:
    bigram.max_context = 64
    try:
        with pytest.raises(ValueError, match="exceeds the context"):
            Suite(config).run(bigram, held_out[:2])
    finally:
        bigram.max_context = None


def test_generation_skipped_when_documents_are_too_short(
    config, bigram: BigramModel, held_out
) -> None:
    short = dataclasses.replace(config, prompt_tokens=10_000)
    report = Suite(short).run(bigram, held_out[:3])
    assert report.generation is None
    assert report.surface == {} and report.name_usage == {}
    assert any("generation skipped" in n for n in report.notes)
    assert report.pqi.missing == ["realism", "surface"]

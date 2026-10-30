from __future__ import annotations

import math

import numpy as np
import pytest

from proseometry.consistency import compare_name_usage, name_usage
from proseometry.degeneration import (
    compare_surface,
    compression_ratio,
    distinct_n,
    seq_rep_n,
    type_token_ratio,
)
from proseometry.stats import bootstrap_mean, geometric_mean, ks_statistic, overlap


def test_seq_rep_n_extremes() -> None:
    loop = " ".join(["the dog ran away"] * 10)
    assert seq_rep_n(loop, 4) == pytest.approx(1 - 4 / 37)
    assert seq_rep_n("one two three four five six seven", 4) == 0.0
    assert seq_rep_n("too short", 4) == 0.0


def test_distinct_n_and_ttr() -> None:
    assert distinct_n(["a b a b", "a b"], 2) == pytest.approx(2 / 4)
    assert type_token_ratio("a a b") == pytest.approx(2 / 3)
    assert math.isnan(distinct_n([""], 2))


def test_compression_ratio_orders_repetitive_below_varied() -> None:
    loop = "the same words again and again. " * 20
    varied = (
        "Yesterday the harbour filled with fog while gulls argued over a dropped sandwich, "
        "and a tram bell rang twice somewhere uphill, muffled, before the kettle finally boiled."
    )
    assert compression_ratio(loop) < compression_ratio(varied)
    assert math.isnan(compression_ratio(""))


def test_ks_and_overlap() -> None:
    a = np.arange(10, dtype=float)
    assert ks_statistic(a, a) == 0.0
    assert overlap(a, a) == 1.0
    assert ks_statistic(a, a + 100) == 1.0
    assert overlap(a, a + 100) == 0.0
    assert math.isnan(ks_statistic([], a))
    assert ks_statistic([math.nan, 1.0], [1.0]) == 0.0


def test_compare_surface_overlap_is_symmetric_and_two_sided() -> None:
    human = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 5
    loops = ["go go go go go go go go go go"] * 5
    result = compare_surface(loops, human)
    assert set(result) == {"seq_rep_4", "compression_ratio"}
    assert result["seq_rep_4"].overlap == 0.0
    assert result["seq_rep_4"].model_mean > result["seq_rep_4"].human_mean
    same = compare_surface(human, human)
    assert same["seq_rep_4"].overlap == 1.0


def test_name_usage_counts_callbacks_and_new_names() -> None:
    prompt = "Lily and Tom went to the park. "
    continuation = "Lily laughed. Then Tom and Lily met Zoe, and Zoe waved."
    usage = name_usage(prompt, continuation)
    assert usage.callback_mentions == 3
    assert usage.new_names == 1
    assert usage.n_words == 11
    assert usage.callback_rate == pytest.approx(300 / 11)
    # A sentence-initial mention in the continuation is recognised through the prompt.
    assert name_usage("Ben sat down near Zoe. ", "Zoe smiled.").callback_mentions == 1
    # ...but a word that is only ever sentence-initial is not treated as a name.
    assert name_usage("Zoe sat down. ", "Zoe smiled.").callback_mentions == 0


def test_compare_name_usage_aligns_samples_to_prompts() -> None:
    prompts = ["Lily met Tom. ", "Ben saw Mia. "]
    human = ["Lily waved at Tom.", "Ben waved at Mia."]
    model = ["Lily waved.", "Zoe waved.", "Ben waved at Mia and Zoe."]
    out = compare_name_usage(prompts, model, human, model_prompt_index=[0, 0, 1])
    assert out["callback_rate"].model_values.shape == (3,)
    assert out["new_name_rate"].human_mean == 0.0
    assert out["new_name_rate"].model_mean > 0.0


def test_bootstrap_and_geometric_mean() -> None:
    interval = bootstrap_mean([1.0, 2.0, 3.0, 4.0], n_boot=200, seed=1)
    assert interval.low <= interval.mean <= interval.high
    assert interval.n == 4
    assert geometric_mean([0.25, 1.0]) == pytest.approx(0.5)
    assert geometric_mean([0.25, math.nan, 1.0]) == pytest.approx(0.5)
    assert geometric_mean([0.0, 1.0]) == 0.0
    assert geometric_mean([0.0, 1.0], floor=0.01) == pytest.approx(0.1)
    assert geometric_mean([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.5)
    assert math.isnan(geometric_mean([math.nan]))

from __future__ import annotations

import math

import numpy as np
import pytest

from proseometry.likelihood import (
    context_gain,
    context_scores,
    key_spans,
    key_token_mask,
    likelihood_summary,
    long_ppl,
    mask_from_spans,
    short_context_logprobs,
)
from proseometry.toy import BigramModel, InductionModel, UniformModel, WordTokenizer


def test_uniform_model_bits_per_byte(tokenizer: WordTokenizer, held_out: list[str]) -> None:
    model = UniformModel(tokenizer)
    docs = [model.tokenize_doc(t) for t in held_out[:3]]
    summary = likelihood_summary([context_scores(model, d) for d in docs])
    n_tokens = sum(len(d) for d in docs)
    n_bytes = sum(len(d.text.encode("utf-8")) for d in docs)
    assert summary.n_tokens == n_tokens and summary.n_bytes == n_bytes
    assert summary.perplexity == pytest.approx(len(tokenizer), rel=1e-5)
    assert summary.bits_per_byte == pytest.approx(
        n_tokens * math.log2(len(tokenizer)) / n_bytes, rel=1e-5
    )


def test_short_context_equals_full_for_bigram(bigram: BigramModel, held_out: list[str]) -> None:
    doc = bigram.tokenize_doc(held_out[0])
    (full,) = bigram.score([doc.ids])
    for k, stride in ((1, 1), (4, 1), (4, 3), (16, 5)):
        short = short_context_logprobs(bigram, doc.ids, full, k, stride=stride)
        np.testing.assert_allclose(short, full, rtol=1e-5, atol=1e-6)


def test_short_context_window_longer_than_doc_returns_full(bigram: BigramModel) -> None:
    ids = bigram.encode("Lily went to the park.")
    (full,) = bigram.score([ids])
    short = short_context_logprobs(bigram, ids, full, k=len(ids) + 5)
    np.testing.assert_array_equal(short, full)


def test_context_gain_zero_for_bigram_positive_for_induction(
    bigram: BigramModel, induction: InductionModel, held_out: list[str]
) -> None:
    ks = (2, 8)
    bigram_scores = [context_scores(bigram, bigram.tokenize_doc(t), ks) for t in held_out[:4]]
    induction_scores = [
        context_scores(induction, induction.tokenize_doc(t), ks) for t in held_out[:4]
    ]
    for k in ks:
        assert abs(context_gain(bigram_scores, k).gain_nats) < 1e-6
        gain = context_gain(induction_scores, k)
        assert gain.gain_nats > 0.0
        assert 0.0 < gain.ppl_reduction < 1.0
        assert gain.n_tokens == sum(max(0, len(s.doc) - k) for s in induction_scores)


def test_short_context_exactness_vs_stride(induction: InductionModel, held_out: list[str]) -> None:
    doc = induction.tokenize_doc(held_out[2])
    (full,) = induction.score([doc.ids])
    exact = short_context_logprobs(induction, doc.ids, full, 8, stride=1)
    strided = short_context_logprobs(induction, doc.ids, full, 8, stride=4)
    # the first token of every stride block sees exactly k tokens in both cases
    np.testing.assert_allclose(exact[8::4], strided[8::4], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(exact[:8], full[:8])


def test_key_token_mask_thresholds() -> None:
    from proseometry.likelihood import ContextScores
    from proseometry.model import TokenizedDoc

    doc = TokenizedDoc(
        ids=[1, 2, 3, 4, 5], text="a b c d e", spans=[(0, 1), (1, 3), (3, 5), (5, 7), (7, 9)]
    )
    full = np.array([math.nan, -0.5, -3.0, -0.2, -1.0])
    short = np.array([math.nan, -0.5, -6.0, -3.5, -1.5])
    scores = ContextScores(doc, full, {2: short})
    mask = key_token_mask(scores, 2, alpha=2.0, beta=-2.0)
    # token 2: LSD 3 but LCL -3 < beta -> not key
    # token 3: LSD 3.3 and LCL -0.2 -> key
    # token 4: LSD 0.5 < alpha -> not key
    assert mask.tolist() == [False, False, False, True, False]
    assert key_spans(doc, mask) == [(5, 7)]
    assert mask_from_spans(doc, [(5, 7)]).tolist() == mask.tolist()
    assert mask_from_spans(doc, [(6, 8)]).tolist() == [False, False, False, True, True]


def test_long_ppl_restricts_to_key_tokens() -> None:
    full = np.array([math.nan, -1.0, -2.0, -3.0])
    mask = np.array([True, False, True, True])
    result = long_ppl([(full, mask)])
    assert result.n_key_tokens == 2 and result.n_tokens == 3
    assert result.long_ppl == pytest.approx(math.exp(2.5))
    assert result.key_fraction == pytest.approx(2 / 3)


def test_long_ppl_transfer_between_models(
    bigram: BigramModel, induction: InductionModel, held_out: list[str]
) -> None:
    """Key tokens picked by one model select the same positions in another (same tokenizer)."""
    text = held_out[3]
    ev_doc = induction.tokenize_doc(text)
    ev_scores = context_scores(induction, ev_doc, [4])
    mask = key_token_mask(ev_scores, 4, alpha=0.5, beta=-3.0)
    assert mask.any()
    spans = key_spans(ev_doc, mask)
    model_doc = bigram.tokenize_doc(text)
    transferred = mask_from_spans(model_doc, spans)
    assert transferred.tolist() == mask.tolist()

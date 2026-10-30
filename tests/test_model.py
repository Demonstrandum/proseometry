from __future__ import annotations

import math

import numpy as np
import pytest

from proseometry.model import LogitsModel, sample_token
from proseometry.toy import BigramModel, UniformModel, WordTokenizer


class NoBosBigram(BigramModel):
    """Same table as BigramModel but scored without a BOS token."""

    def __init__(self, base: BigramModel) -> None:
        LogitsModel.__init__(self, base.tokenizer, name="nobos", bos_id=None, eos_id=base.eos_id)
        self.table = base.table


def test_score_matches_direct_bigram_lookup(bigram: BigramModel, held_out: list[str]) -> None:
    ids = bigram.encode(held_out[0])
    (scores,) = bigram.score([ids])
    assert scores.shape == (len(ids),)
    assert not np.isnan(scores).any()  # BOS predicts the first token
    expected = [bigram.table[bigram.bos_id, ids[0]]] + [
        bigram.table[a, b] for a, b in zip(ids[:-1], ids[1:], strict=True)
    ]
    np.testing.assert_allclose(scores, expected, rtol=1e-5, atol=1e-6)


def test_score_without_bos_marks_first_token_nan(bigram: BigramModel, held_out: list[str]) -> None:
    model = NoBosBigram(bigram)
    ids = model.encode(held_out[0])
    (scores,) = model.score([ids])
    assert math.isnan(scores[0])
    (with_bos,) = bigram.score([ids])
    np.testing.assert_allclose(scores[1:], with_bos[1:], rtol=1e-5, atol=1e-6)


def test_batching_does_not_change_scores(bigram: BigramModel, held_out: list[str]) -> None:
    seqs = [bigram.encode(t)[: 10 + 7 * i] for i, t in enumerate(held_out[:5])]
    bigram.batch_size = 1
    one_by_one = bigram.score(seqs)
    bigram.batch_size = 3
    batched = bigram.score(seqs)
    bigram.batch_size = 4
    for a, b in zip(one_by_one, batched, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_score_handles_empty_and_single_token_sequences(bigram: BigramModel) -> None:
    empty, single = bigram.score([[], [7]])
    assert empty.shape == (0,)
    assert single.shape == (1,) and not math.isnan(single[0])


def test_score_continuations_sums_continuation_tokens(
    bigram: BigramModel, held_out: list[str]
) -> None:
    ids = bigram.encode(held_out[1])
    prefix, cont = ids[:20], ids[20:26]
    (total,) = bigram.score_continuations(prefix, [cont])
    (full,) = bigram.score([ids[:26]])
    assert total == pytest.approx(float(full[20:26].sum()), rel=1e-5)


def test_uniform_model_perplexity_is_vocab_size(tokenizer: WordTokenizer) -> None:
    model = UniformModel(tokenizer)
    (scores,) = model.score([[5, 6, 7, 8]])
    np.testing.assert_allclose(np.exp(-scores), len(tokenizer))


def test_generate_respects_length_eos_and_seed(bigram: BigramModel) -> None:
    prompt = bigram.encode("Once upon a time, Lily")
    a = bigram.generate(prompt, 30, temperature=1.0, top_p=0.95, seed=3)
    b = bigram.generate(prompt, 30, temperature=1.0, top_p=0.95, seed=3)
    assert a == b
    assert 1 <= len(a) <= 30
    assert bigram.eos_id not in a[:-1]
    greedy = bigram.generate(prompt, 5, temperature=0.0)
    assert len(greedy) <= 5


def test_sample_token_nucleus_and_greedy() -> None:
    rng = np.random.default_rng(0)
    logits = np.log(np.array([0.6, 0.3, 0.07, 0.03]))
    assert sample_token(logits, temperature=0.0, top_p=1.0, rng=rng) == 0
    draws = {sample_token(logits, temperature=1.0, top_p=0.5, rng=rng) for _ in range(50)}
    assert draws == {0}  # p=0.5 keeps only the top token
    draws = {sample_token(logits, temperature=1.0, top_p=0.8, rng=rng) for _ in range(200)}
    assert draws == {0, 1}  # 0.6 < 0.8 <= 0.6 + 0.3: the top two tokens

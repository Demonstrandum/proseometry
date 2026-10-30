from __future__ import annotations

import pytest

from proseometry import toy


@pytest.fixture(scope="session")
def stories() -> list[str]:
    return toy.synthetic_stories(60, seed=1)


@pytest.fixture(scope="session")
def tokenizer(stories: list[str]) -> toy.WordTokenizer:
    return toy.WordTokenizer.fit(stories)


@pytest.fixture(scope="session")
def bigram(tokenizer: toy.WordTokenizer, stories: list[str]) -> toy.BigramModel:
    return toy.BigramModel.fit(tokenizer, stories[:50])


@pytest.fixture(scope="session")
def induction(tokenizer: toy.WordTokenizer, stories: list[str]) -> toy.InductionModel:
    return toy.InductionModel.fit(tokenizer, stories[:50], lam=0.5)


@pytest.fixture(scope="session")
def held_out(stories: list[str]) -> list[str]:
    return stories[50:]

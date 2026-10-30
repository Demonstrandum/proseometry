"""Generation-side memory: does a continuation keep using the characters it was given?

For a prompt and its continuation we count, per 100 words of continuation,

* ``callback_rate`` -- mentions of names that were introduced in the prompt, and
* ``new_name_rate`` -- distinct names that never appeared in the prompt.

A model with no working memory invents new characters and rarely refers back; a
looping model refers back far too often.  Both are caught by comparing the
per-sample distributions with those of the real human continuations of the same
prompts (see :func:`compare_name_usage`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .degeneration import StatComparison
from .text import NameExtractor, find_names, word_count


@dataclass(frozen=True)
class NameUsage:
    n_words: int
    callback_mentions: int
    new_names: int

    @property
    def callback_rate(self) -> float:
        return 100.0 * self.callback_mentions / max(self.n_words, 1)

    @property
    def new_name_rate(self) -> float:
        return 100.0 * self.new_names / max(self.n_words, 1)


def name_usage(
    prompt: str, continuation: str, *, extractor: NameExtractor = find_names
) -> NameUsage:
    """Count prompt-name callbacks and newly introduced names in ``continuation``.

    Names are detected on ``prompt + continuation`` so that a sentence-initial
    mention in the continuation is still recognised as a known character.
    """
    boundary = len(prompt)
    mentions = extractor(prompt + continuation)
    prompt_names = {m.name for m in mentions if m.start < boundary}
    callbacks = 0
    new_names: set[str] = set()
    for m in mentions:
        if m.start < boundary:
            continue
        if m.name in prompt_names:
            callbacks += 1
        else:
            new_names.add(m.name)
    return NameUsage(word_count(continuation), callbacks, len(new_names))


def compare_name_usage(
    prompts: Sequence[str],
    model_continuations: Sequence[str],
    human_continuations: Sequence[str],
    *,
    model_prompt_index: Sequence[int] | None = None,
    extractor: NameExtractor = find_names,
) -> dict[str, StatComparison]:
    """Compare callback and new-name rates between model and human continuations.

    ``human_continuations[i]`` continues ``prompts[i]``.  When several model samples
    were drawn per prompt, ``model_prompt_index[j]`` gives the prompt of sample ``j``;
    by default the two lists are aligned one-to-one.
    """
    if model_prompt_index is None:
        model_prompt_index = list(range(len(model_continuations)))
    model_usage = [
        name_usage(prompts[p], c, extractor=extractor)
        for p, c in zip(model_prompt_index, model_continuations, strict=True)
    ]
    human_usage = [
        name_usage(p, c, extractor=extractor)
        for p, c in zip(prompts, human_continuations, strict=True)
    ]
    return {
        "callback_rate": StatComparison(
            "callback_rate",
            np.array([u.callback_rate for u in model_usage]),
            np.array([u.callback_rate for u in human_usage]),
        ),
        "new_name_rate": StatComparison(
            "new_name_rate",
            np.array([u.new_name_rate for u in model_usage]),
            np.array([u.new_name_rate for u in human_usage]),
        ),
    }

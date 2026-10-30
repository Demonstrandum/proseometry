"""Text utilities shared by the metrics: words, character spans, and a name detector.

The name detector is a deliberately simple heuristic that needs no model or
downloads: a capitalised word is a *name* if it occurs at least once somewhere
other than the start of a sentence.  That is enough for narrative prose, where
the long-range references that matter are characters and places.  Every
function that uses it accepts an ``extractor`` argument, so a NER model (spaCy,
Stanza, ...) can be substituted for other domains.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .model import Tokenizer

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CAPITALISED_RE = re.compile(r"[A-Z][a-z]+")

#: Characters skipped when looking back for the end of the previous sentence.
_QUOTES_AND_BRACKETS = set("\"'\u201c\u201d\u2018\u2019()[]*_")
#: Capitalised words that are never names even mid-sentence.
DEFAULT_STOPLIST = frozenset({"I"})


@dataclass(frozen=True)
class Mention:
    """One occurrence of a name: the surface form and its character span ``[start, end)``."""

    name: str
    start: int
    end: int


NameExtractor = Callable[[str], list[Mention]]


def words(text: str) -> list[str]:
    """Lower-cased word and punctuation tokens, independent of any model tokenizer."""
    return [w.lower() for w in WORD_RE.findall(text)]


def word_count(text: str) -> int:
    return sum(1 for w in WORD_RE.findall(text) if w[0].isalnum())


def is_sentence_initial(text: str, start: int) -> bool:
    """True if the word starting at ``start`` begins a sentence (or line, or the text)."""
    i = start - 1
    while i >= 0:
        c = text[i]
        if c == "\n":
            return True
        if c.isspace() or c in _QUOTES_AND_BRACKETS:
            i -= 1
            continue
        return c in ".!?"
    return True


def find_names(text: str, *, stoplist: frozenset[str] = DEFAULT_STOPLIST) -> list[Mention]:
    """Heuristic proper-noun detector.

    A capitalised word (``[A-Z][a-z]+`` forming a whole word) counts as a name if
    at least one of its occurrences is not sentence-initial.  All occurrences of
    such words, including sentence-initial ones, are returned in order.  Words
    that only ever start sentences are dropped because they cannot be told apart
    from ordinary capitalisation.
    """
    candidates: list[tuple[str, int, int, bool]] = []
    for m in CAPITALISED_RE.finditer(text):
        start, end = m.span()
        if start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            continue
        if end < len(text) and (text[end].isalpha() or text[end].isdigit()):
            continue
        word = m.group()
        if word in stoplist:
            continue
        candidates.append((word, start, end, is_sentence_initial(text, start)))
    names = {word for word, _, _, initial in candidates if not initial}
    return [Mention(word, s, e) for word, s, e, _ in candidates if word in names]


def token_char_spans(tokenizer: Tokenizer, ids: Sequence[int]) -> list[tuple[int, int]]:
    """Character span of each token within ``tokenizer.decode(ids)``.

    Computed by incremental decoding, which is tokenizer-agnostic and handles
    leading-space markers correctly (decoding a token in isolation would not).
    Spans are contiguous and non-decreasing; a token that completes a multi-byte
    character may get a zero-width span.
    """
    ids = list(ids)
    spans: list[tuple[int, int]] = []
    previous = 0
    for i in range(1, len(ids) + 1):
        current = max(previous, len(tokenizer.decode(ids[:i])))
        spans.append((previous, current))
        previous = current
    return spans


def tokens_covering(
    spans: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[int, int] | None:
    """Token index range ``[first, last)`` of tokens overlapping characters ``[start, end)``."""
    if not spans or end <= start:
        return None
    ends = [e for _, e in spans]
    starts = [s for s, _ in spans]
    first = bisect_right(ends, start)  # first token whose end is beyond `start`
    last = bisect_left(starts, end)  # first token whose start is at/after `end`
    if first >= last:
        return None
    return first, last


def merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of (possibly overlapping or adjacent) character spans, sorted."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

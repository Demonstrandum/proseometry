from __future__ import annotations

from proseometry.text import (
    find_names,
    is_sentence_initial,
    merge_spans,
    token_char_spans,
    tokens_covering,
    words,
)
from proseometry.toy import WordTokenizer

Q = chr(34)  # a double quote, kept out of the source for readability of the test strings


def test_find_names_ignores_sentence_initial_only_words():
    text = f"Once upon a time, Lily and Tom went to the park. The park was big. {Q}Hi,{Q} said Tom."
    mentions = find_names(text)
    assert {m.name for m in mentions} == {"Lily", "Tom"}
    assert [m.name for m in mentions] == ["Lily", "Tom", "Tom"]
    for m in mentions:
        assert text[m.start : m.end] == m.name


def test_find_names_includes_sentence_initial_occurrences_of_known_names():
    text = "Lily went home. Then Lily slept. Lily dreamed."
    mentions = find_names(text)
    assert [m.name for m in mentions] == ["Lily", "Lily", "Lily"]


def test_find_names_handles_quotes_newlines_and_pronoun_i():
    text = f"{Q}Hello,{Q} said Max.\nMax waved. I waved back at Max."
    names = [m.name for m in find_names(text)]
    assert names == ["Max", "Max", "Max"]
    assert is_sentence_initial(text, text.index("Max waved"))
    assert not is_sentence_initial(text, text.rindex("Max"))


def test_find_names_requires_whole_words():
    # 'McDonald' is not a [A-Z][a-z]+ word, so neither 'Mc' nor 'Donald' is matched.
    assert [m.name for m in find_names("The McDonald farm belonged to Tom.")] == ["Tom"]


def test_words_lowercases_and_splits_punctuation():
    assert words("Hi, Lily!") == ["hi", ",", "lily", "!"]


def test_token_char_spans_partition_decoded_text():
    sentence = f"Tom said, {Q}Hi Lily!{Q} Then Lily went to the park."
    tok = WordTokenizer.fit([sentence])
    ids = tok.encode(sentence)
    text = tok.decode(ids)
    assert text == sentence
    spans = token_char_spans(tok, ids)
    assert len(spans) == len(ids)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s, e), i in zip(spans, ids, strict=True):
        assert text[s:e].strip() == tok.itos[i]
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 == s2


def test_tokens_covering_and_merge_spans():
    spans = [(0, 4), (4, 9), (9, 10), (10, 15)]
    assert tokens_covering(spans, 5, 9) == (1, 2)
    assert tokens_covering(spans, 4, 10) == (1, 3)
    assert tokens_covering(spans, 0, 1) == (0, 1)
    assert tokens_covering(spans, 20, 25) is None
    assert merge_spans([(5, 9), (0, 4), (4, 6), (12, 14)]) == [(0, 9), (12, 14)]

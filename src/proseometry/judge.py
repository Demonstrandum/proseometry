"""Optional LLM-judge rubric for continuations (grammar, coherence, consistency).

Eldan & Li (2023, TinyStories) graded small-model stories with a fixed rubric
because the training loss did not track what they could see.  This module
provides the rubric, response parsing and aggregation; the judge itself is any
``Callable[[str], str]`` that maps a prompt to a completion, so it works with any
API or a local model.  :func:`openai_compatible_judge` builds one for any
OpenAI-style ``/chat/completions`` endpoint using only the standard library.

Scores are 1-5 and are rescaled to ``[0, 1]`` for the quality index.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .stats import nanmean

Judge = Callable[[str], str]

CRITERIA: tuple[str, ...] = ("grammar", "coherence", "consistency")

RUBRIC = """You are grading the continuation of a story written by a small language model.

OPENING (written by a person):
<<<
{prompt}
>>>

CONTINUATION (written by the model):
<<<
{continuation}
>>>

Score the CONTINUATION only, on a 1-5 scale for each criterion:
- grammar: well-formed sentences and words (1 = word salad, 5 = flawless).
- coherence: sentences follow from one another and make sense; no loops or
  contradictions within the continuation (1 = nonsense or repetition, 5 = clear).
- consistency: the continuation respects the opening -- the same characters,
  names, setting and established facts, with no unexplained new protagonists
  (1 = ignores the opening, 5 = fully consistent).

Answer with a single JSON object and nothing else, for example
{{"grammar": 4, "coherence": 3, "consistency": 2}}
"""

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_scores(response: str, criteria: Sequence[str] = CRITERIA) -> dict[str, float] | None:
    """Extract the rubric scores from a judge response; ``None`` if it cannot be parsed."""
    for match in _JSON_RE.finditer(response):
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        scores: dict[str, float] = {}
        for criterion in criteria:
            value = data.get(criterion)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                break
            scores[criterion] = float(min(5.0, max(1.0, value)))
        else:
            return scores
    return None


@dataclass
class JudgeResult:
    per_sample: list[dict[str, float] | None]
    criteria: tuple[str, ...] = CRITERIA
    raw_responses: list[str] = field(default_factory=list)

    @property
    def n_parsed(self) -> int:
        return sum(1 for s in self.per_sample if s is not None)

    def mean(self, criterion: str) -> float:
        """Mean 1-5 score for ``criterion`` over parsed responses."""
        return nanmean([s[criterion] for s in self.per_sample if s is not None])

    def normalized(self, criterion: str) -> float:
        """Mean score rescaled to ``[0, 1]`` (``(score - 1) / 4``)."""
        m = self.mean(criterion)
        return math.nan if math.isnan(m) else (m - 1.0) / 4.0

    def to_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {"n_parsed": self.n_parsed, "n_total": len(self.per_sample)}
        for c in self.criteria:
            out[c] = self.mean(c)
        return out


def judge_continuations(
    judge: Judge,
    prompts: Sequence[str],
    continuations: Sequence[str],
    *,
    rubric: str = RUBRIC,
    criteria: Sequence[str] = CRITERIA,
    keep_responses: bool = False,
) -> JudgeResult:
    """Grade each ``(prompt, continuation)`` pair with ``judge`` and aggregate."""
    per_sample: list[dict[str, float] | None] = []
    raw: list[str] = []
    for prompt, continuation in zip(prompts, continuations, strict=True):
        response = judge(rubric.format(prompt=prompt, continuation=continuation))
        per_sample.append(parse_scores(response, criteria))
        if keep_responses:
            raw.append(response)
    return JudgeResult(per_sample, tuple(criteria), raw)


def openai_compatible_judge(
    model: str,
    *,
    base_url: str = "https://api.openai.com/v1",
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> Judge:
    """A judge backed by an OpenAI-style chat-completions endpoint.

    The API key is read from the environment variable ``api_key_env`` at call time
    and never stored.  Works with any compatible server (OpenAI, vLLM, Ollama, ...).
    """

    def call(prompt: str) -> str:
        key = os.environ.get(api_key_env, "")
        body = json.dumps(
            {
                "model": model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {key}"} if key else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]

    return call

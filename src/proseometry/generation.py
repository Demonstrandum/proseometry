"""Free-running continuations with paired human references.

Every generation-side metric in proseometry uses the same protocol: prompt the
model with the first ``prompt_tokens`` of a held-out document, sample
``continuation_tokens`` more, and keep the document's *real* next
``continuation_tokens`` tokens as the human reference.  Because both
continuations have the same token length and share the prompt, every statistic
computed on them is directly comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import Model, TokenizedDoc


@dataclass
class GenerationSet:
    """Prompts with their human and model continuations under one decoding protocol."""

    prompts: list[str]
    human: list[str]
    model: list[str]
    prompt_index: list[int]
    """``prompt_index[j]`` is the prompt that model sample ``j`` continues."""
    doc_index: list[int]
    """Index of each prompt's document within the corpus it was drawn from."""

    def __len__(self) -> int:
        return len(self.model)

    def model_prompts(self) -> list[str]:
        """The prompt of each model sample, aligned with ``model``."""
        return [self.prompts[i] for i in self.prompt_index]

    def human_full(self) -> list[str]:
        """``prompt + human continuation`` for each prompt."""
        return [p + h for p, h in zip(self.prompts, self.human, strict=True)]

    def model_full(self) -> list[str]:
        """``prompt + model continuation`` for each sample."""
        return [p + m for p, m in zip(self.model_prompts(), self.model, strict=True)]

    def examples(self, n: int = 3) -> str:
        """A few prompt / human / model triples for eyeballing."""
        chunks = []
        for j in range(min(n, len(self.model))):
            p = self.prompt_index[j]
            chunks.append(
                f"--- prompt {p} ---\n{self.prompts[p]}\n"
                f"--- human ---\n{self.human[p]}\n"
                f"--- model ---\n{self.model[j]}\n"
            )
        return "\n".join(chunks)


def sample_continuations(
    model: Model,
    docs: Sequence[TokenizedDoc],
    *,
    prompt_tokens: int,
    continuation_tokens: int,
    samples_per_prompt: int = 1,
    max_prompts: int | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
) -> GenerationSet | None:
    """Sample continuations for every document long enough to supply prompt + reference.

    Documents shorter than ``prompt_tokens + continuation_tokens`` are skipped;
    ``max_prompts`` picks a seeded random subset of the eligible ones.  Returns
    ``None`` when no document is eligible.  Sampling stops at the model's EOS
    token, which is removed from the continuation.
    """
    need = prompt_tokens + continuation_tokens
    limit = model.max_doc_tokens
    if limit is not None and need > limit:
        raise ValueError(
            f"prompt_tokens + continuation_tokens = {need} exceeds the context of "
            f"{model.name!r} ({limit} tokens excluding BOS); lower them in the config"
        )
    eligible = [i for i, d in enumerate(docs) if len(d) >= need]
    if max_prompts is not None and len(eligible) > max_prompts:
        rng = np.random.default_rng(seed)
        eligible = sorted(rng.choice(eligible, size=max_prompts, replace=False).tolist())
    if not eligible:
        return None
    prompts: list[str] = []
    human: list[str] = []
    samples: list[str] = []
    prompt_index: list[int] = []
    for j, i in enumerate(eligible):
        doc = docs[i]
        prompt_ids = doc.ids[:prompt_tokens]
        prompt_text = model.decode(prompt_ids)
        prompts.append(prompt_text)
        human.append(model.decode(doc.ids[:need])[len(prompt_text) :])
        for s in range(samples_per_prompt):
            out = model.generate(
                prompt_ids,
                continuation_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed * 7919 + i * 131 + s,
            )
            if model.eos_id is not None and model.eos_id in out:
                out = out[: out.index(model.eos_id)]
            samples.append(model.decode(prompt_ids + list(out))[len(prompt_text) :])
            prompt_index.append(j)
    return GenerationSet(prompts, human, samples, prompt_index, eligible)

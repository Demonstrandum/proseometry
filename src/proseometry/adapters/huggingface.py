"""Adapter for Hugging Face ``transformers`` causal language models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .pytorch import TorchModel


class HFTokenizer:
    """Wrap a ``PreTrainedTokenizer``: ``encode`` adds no special tokens, ``decode`` is faithful."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )


class HFModel(TorchModel):
    """A ``transformers`` model plus tokenizer, with native cached generation.

    ``bos_id`` defaults to the tokenizer's BOS token, falling back to EOS (GPT-2
    style, where ``<|endoftext|>`` delimits documents); ``max_context`` defaults to
    the config's position limit.  Pass explicit values to override either.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        name: str | None = None,
        device: Any = None,
        batch_size: int = 4,
        bos_id: int | None | str = "auto",
        eos_id: int | None | str = "auto",
        max_context: int | None = None,
    ) -> None:
        wrapped = tokenizer if isinstance(tokenizer, HFTokenizer) else HFTokenizer(tokenizer)
        raw = wrapped.tokenizer
        if bos_id == "auto":
            bos_id = raw.bos_token_id if raw.bos_token_id is not None else raw.eos_token_id
        if eos_id == "auto":
            eos_id = raw.eos_token_id
        if max_context is None:
            config = getattr(model, "config", None)
            for attribute in ("max_position_embeddings", "n_positions", "max_sequence_length"):
                value = getattr(config, attribute, None)
                if isinstance(value, int) and value > 0:
                    max_context = value
                    break
        super().__init__(
            model,
            wrapped,
            name=name or str(getattr(model, "name_or_path", "hf-model")),
            device=device,
            bos_id=bos_id,  # type: ignore[arg-type]
            eos_id=eos_id,  # type: ignore[arg-type]
            max_context=max_context,
            batch_size=batch_size,
        )
        self.hf_tokenizer = raw

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        name: str | None = None,
        device: Any = None,
        dtype: Any = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> HFModel:
        """Load a checkpoint (local path or hub id) with ``AutoModelForCausalLM``."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        if dtype is not None:
            kwargs["dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        return cls(model, tokenizer, name=name or path, device=device, batch_size=batch_size)

    def generate(
        self,
        prompt: Sequence[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[int]:
        """Sample with ``model.generate`` (KV-cached).  ``seed`` sets torch's global RNG."""
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        ids = torch.as_tensor([self._with_bos(prompt)], dtype=torch.long, device=self.device)
        sampling: dict[str, Any] = (
            {"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)}
            if temperature > 0
            else {"do_sample": False}
        )
        with torch.inference_mode():
            out = self.module.generate(
                ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._pad_id,
                eos_token_id=self.eos_id,
                **sampling,
            )
        return out[0, ids.shape[1] :].tolist()

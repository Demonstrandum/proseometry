"""Adapter tests.  They need the `torch` extra; the Hugging Face test also needs network access."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from proseometry.adapters import HFModel, TorchModel  # noqa: E402
from proseometry.toy import BigramModel, WordTokenizer  # noqa: E402


class TableModule(torch.nn.Module):
    """A bigram table as an nn.Module, so TorchModel can be checked against BigramModel exactly."""

    def __init__(self, table: np.ndarray) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.as_tensor(table), requires_grad=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table[ids]


def test_torch_model_matches_numpy_reference(
    tokenizer: WordTokenizer, bigram: BigramModel, held_out: list[str]
) -> None:
    wrapped = TorchModel(
        TableModule(bigram.table),
        tokenizer,
        name="torch-bigram",
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        batch_size=3,
    )
    seqs = [bigram.encode(t)[: 15 + 4 * i] for i, t in enumerate(held_out[:4])]
    for a, b in zip(wrapped.score(seqs), bigram.score(seqs), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    prompt = seqs[0][:5]
    assert wrapped.generate(prompt, 4, temperature=0.0) == bigram.generate(
        prompt, 4, temperature=0.0
    )


@pytest.mark.network
def test_hf_model_scores_and_generates() -> None:
    pytest.importorskip("transformers")
    try:
        model = HFModel.from_pretrained("hf-internal-testing/tiny-random-gpt2", name="tiny-gpt2")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"could not download test checkpoint: {exc}")
    assert model.bos_id is not None and model.max_context is not None
    doc = model.tokenize_doc("Lily and Tom went to the park. Tom found a ball.", max_tokens=32)
    assert doc.text.startswith("Lily and Tom")
    (scores,) = model.score([doc.ids])
    assert scores.shape == (len(doc.ids),) and np.isfinite(scores).all()
    # Token-level scores agree with the model's own loss on the same input.
    import torch as _torch

    ids = _torch.tensor([[model.bos_id, *doc.ids]])
    with _torch.inference_mode():
        logits = model.module(ids).logits[0, :-1].float()
    reference = _torch.log_softmax(logits, -1).gather(-1, ids[0, 1:, None])[:, 0].numpy()
    np.testing.assert_allclose(scores, reference, rtol=1e-4, atol=1e-4)
    out = model.generate(doc.ids[:8], 6, temperature=1.0, top_p=0.9, seed=0)
    assert 1 <= len(out) <= 6
    assert model.generate(doc.ids[:8], 6, temperature=1.0, top_p=0.9, seed=0) == out

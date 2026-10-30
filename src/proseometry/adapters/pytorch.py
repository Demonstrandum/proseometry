"""Adapter for any PyTorch causal language model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..model import Model, Tokenizer


def _default_forward(module: Any, ids: Any) -> Any:
    """Call ``module(ids)`` and pull logits out of the common return shapes."""
    out = module(ids)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class TorchModel(Model):
    """Wrap a ``torch.nn.Module`` whose forward pass yields ``[B, T, V]`` logits.

    Parameters
    ----------
    module:
        The network.  It is switched to ``eval()`` mode.
    tokenizer:
        Anything with ``encode``/``decode``.
    forward:
        ``forward(module, ids) -> logits`` for modules with a non-standard call
        signature.  The default calls ``module(ids)`` and accepts a tensor, a
        tuple whose first element is the logits, or an object with ``.logits``.
    device:
        Where to run; defaults to the device of the module's parameters.

    Log-softmax and target gathering happen on device, so only ``[B, T]``
    numbers cross to numpy.
    """

    def __init__(
        self,
        module: Any,
        tokenizer: Tokenizer,
        *,
        name: str = "model",
        forward: Callable[[Any, Any], Any] | None = None,
        device: Any = None,
        bos_id: int | None = None,
        eos_id: int | None = None,
        max_context: int | None = None,
        batch_size: int = 4,
    ) -> None:
        import torch

        super().__init__(
            tokenizer,
            name=name,
            bos_id=bos_id,
            eos_id=eos_id,
            max_context=max_context,
            batch_size=batch_size,
        )
        self.module = module.eval()
        self._forward = forward or _default_forward
        if device is None:
            try:
                device = next(module.parameters()).device
            except StopIteration:
                device = "cpu"
        self.device = torch.device(device)
        self.module.to(self.device)

    def _logits(self, ids: np.ndarray) -> Any:
        import torch

        tensor = torch.as_tensor(np.ascontiguousarray(ids), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            return self._forward(self.module, tensor).float()

    def _token_logprobs(self, ids: np.ndarray) -> np.ndarray:
        import torch

        logprobs = torch.log_softmax(self._logits(ids[:, :-1]), dim=-1)
        targets = torch.as_tensor(
            np.ascontiguousarray(ids[:, 1:]), dtype=torch.long, device=self.device
        )
        gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return gathered.cpu().numpy()

    def _next_token_logits(self, ids: np.ndarray) -> np.ndarray:
        return self._logits(ids)[:, -1].cpu().numpy()

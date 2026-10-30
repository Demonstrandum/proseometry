"""Adapters turning PyTorch modules and Hugging Face checkpoints into :class:`proseometry.Model`.

Importing this package does not import ``torch``; the classes import it lazily
so the core suite stays dependency-light.  Install ``proseometry[torch]``.
"""

from .huggingface import HFModel, HFTokenizer
from .pytorch import TorchModel

__all__ = ["HFModel", "HFTokenizer", "TorchModel"]

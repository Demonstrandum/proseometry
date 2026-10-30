"""proseometry -- measure what perplexity misses.

A suite of language-model metrics for long-context use and free-running text
quality, plus the Prose Quality Index (PQI) that combines them.

Typical use::

    from proseometry import Suite, SuiteConfig, compare
    from proseometry.adapters import HFModel

    suite = Suite(SuiteConfig(prompt_tokens=128, continuation_tokens=256))
    reports = [suite.run(HFModel.from_pretrained(p, name=n), held_out_texts)
               for n, p in checkpoints.items()]
    print(compare(reports))
"""

from . import (
    consistency,
    degeneration,
    distribution,
    judge,
    likelihood,
    quality,
    recall,
    stats,
    text,
    toy,
)
from .model import LogitsModel, Model, TokenizedDoc, Tokenizer
from .quality import QualityIndex, prose_quality_index
from .suite import GenerationSet, Report, Suite, SuiteConfig, compare

__version__ = "0.1.0"

__all__ = [
    "GenerationSet",
    "LogitsModel",
    "Model",
    "QualityIndex",
    "Report",
    "Suite",
    "SuiteConfig",
    "TokenizedDoc",
    "Tokenizer",
    "__version__",
    "compare",
    "consistency",
    "degeneration",
    "distribution",
    "judge",
    "likelihood",
    "prose_quality_index",
    "quality",
    "recall",
    "stats",
    "text",
    "toy",
]

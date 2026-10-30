"""proseometry -- measure what perplexity misses.

A suite of language-model metrics for long-context use and free-running text
quality, plus the Prose Quality Index (PQI) that combines them.

Three entry points, from quick to thorough::

    from proseometry import goodness, prose_quality_index, Suite

    goodness(text).score                       # one generated text, no model needed
    prose_quality_index(model, held_out).score # the PQI of a model on a held-out corpus
    Suite(config).run(model, held_out)         # the PQI plus every diagnostic, as a report
"""

from . import (
    consistency,
    degeneration,
    distribution,
    generation,
    judge,
    likelihood,
    quality,
    recall,
    stats,
    text,
    toy,
)
from .generation import GenerationSet, sample_continuations
from .goodness import Goodness, Reference, default_scorer, goodness, goodness_many, mean_goodness
from .model import LogitsModel, Model, TokenizedDoc, Tokenizer
from .quality import PQI, CompositeIndex, PQIConfig, composite_index, prose_quality_index
from .suite import Report, Suite, SuiteConfig, compare

__version__ = "0.1.0"

__all__ = [
    "PQI",
    "CompositeIndex",
    "GenerationSet",
    "Goodness",
    "LogitsModel",
    "Model",
    "PQIConfig",
    "Reference",
    "Report",
    "Suite",
    "SuiteConfig",
    "TokenizedDoc",
    "Tokenizer",
    "__version__",
    "compare",
    "composite_index",
    "consistency",
    "default_scorer",
    "degeneration",
    "distribution",
    "generation",
    "goodness",
    "goodness_many",
    "judge",
    "likelihood",
    "mean_goodness",
    "prose_quality_index",
    "quality",
    "recall",
    "sample_continuations",
    "stats",
    "text",
    "toy",
]

"""Distributional realism: MAUVE between model and human text.

MAUVE (Pillutla et al. 2021) embeds both corpora, quantises the embeddings, and
summarises the divergence frontier between the two distributions.  It penalises
text people never write (loops, nonsense; type-I error) and modes of human text
the model cannot produce (blandness; type-II error), and it tracks human
judgements of "human-like" and "sensible" better than generation perplexity.

This module wraps the reference implementation (``pip install mauve-text``, or
``proseometry[mauve]``).  Absolute MAUVE values depend on the sample size,
featuriser and number of clusters, so compare models only under one fixed
protocol -- which is what :class:`proseometry.suite.Suite` enforces.

Process isolation
-----------------
``torch``, ``faiss`` and ``scikit-learn`` each bundle their own OpenMP runtime.
When all three live in one process, whichever runtime initialised *second*
crashes inside its parallel regions (observed as a segfault on macOS arm64) --
in faiss' k-means if torch was imported first, in the GPT-2 forward pass
otherwise.  A crash there would take a long evaluation down with it, so:

* the k-means quantisation runs in a subprocess that never imports torch
  (``isolate=True``, the default), and
* featurisation runs in-process only when torch is already loaded or no sibling
  runtime is; if ``faiss``/``sklearn`` were imported before ``torch`` it moves to
  a subprocess with the safe import order.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class MissingDependency(ImportError):
    """Raised when an optional dependency is required but not installed."""


_INSTALL_HINT = "MAUVE needs the `mauve-text` package: install `proseometry[mauve]`."


@dataclass
class MauveResult:
    mauve: float
    frontier_integral: float
    mauve_star: float
    n_human: int
    n_model: int
    divergence_curve: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mauve": self.mauve,
            "frontier_integral": self.frontier_integral,
            "mauve_star": self.mauve_star,
            "n_human": self.n_human,
            "n_model": self.n_model,
        }


def mauve_available() -> bool:
    """Whether ``mauve-text`` is installed (checked without importing it, see below)."""
    return importlib.util.find_spec("mauve") is not None


def _run_worker(command: str, inputs: Path, outputs: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "proseometry._mauve_worker", command, str(inputs), str(outputs)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not outputs.exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(
            f"MAUVE {command} subprocess failed with exit code {proc.returncode}:\n{tail}"
        )


def featurization_is_safe_in_process(loaded: Sequence[str] | None = None) -> bool:
    """True unless a sibling OpenMP runtime (faiss, scikit-learn) was imported before torch.

    ``loaded`` is the list of imported module names in import order; it defaults
    to ``sys.modules``, whose insertion order is import order.  Importing ``mauve``
    itself pulls in faiss, scikit-learn and then torch -- the unsafe order -- which
    is why this module never imports it except to featurise.
    """
    order = list(sys.modules if loaded is None else loaded)
    siblings = [m for m in ("faiss", "sklearn") if m in order]
    if not siblings:
        return True
    if "torch" not in order:
        return False
    torch_position = order.index("torch")
    return all(order.index(m) > torch_position for m in siblings)


def featurize(
    texts: Sequence[str],
    *,
    featurize_model_name: str = "gpt2-large",
    device_id: int = -1,
    max_text_length: int = 512,
    batch_size: int = 8,
    verbose: bool = False,
    isolate: bool | None = None,
) -> np.ndarray:
    """Embed ``texts`` with MAUVE's featuriser (terminal hidden state of a GPT-2 model).

    Needs ``torch`` and ``transformers``.  ``device_id=-1`` means CPU.  Returns an
    array of shape ``[n, d]`` for :func:`mauve_from_features`.  ``isolate=None``
    runs in a subprocess only when :func:`featurization_is_safe_in_process` is false.
    """
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError("no non-empty texts to featurize")
    if isolate or (isolate is None and not featurization_is_safe_in_process()):
        if not mauve_available():
            raise MissingDependency(_INSTALL_HINT)
        options = {
            "featurize_model_name": featurize_model_name,
            "device_id": device_id,
            "max_text_length": max_text_length,
            "batch_size": batch_size,
            "verbose": verbose,
            "isolate": False,
        }
        with tempfile.TemporaryDirectory(prefix="proseometry-mauve-") as tmp:
            inputs = Path(tmp) / "texts.json"
            outputs = Path(tmp) / "features.npy"
            inputs.write_text(json.dumps({"texts": texts, "options": options}), encoding="utf-8")
            _run_worker("featurize", inputs, outputs)
            return np.load(outputs)
    try:
        # Import torch before mauve: mauve pulls in faiss and scikit-learn, each bundling
        # its own OpenMP runtime, and torch's runtime must be the one initialised first
        # or its parallel regions crash (observed on macOS arm64).
        import torch  # noqa: F401
        from mauve.compute_mauve import get_features_from_input
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependency(
            "MAUVE featurisation needs `torch`, `transformers` and `mauve-text`: "
            "install `proseometry[all]`, or pass precomputed features."
        ) from exc
    return np.asarray(
        get_features_from_input(
            None,
            None,
            texts,
            featurize_model_name,
            max_text_length,
            device_id,
            "text",
            batch_size,
            verbose=verbose,
        )
    )


def _result(out: dict[str, Any], n_human: int, n_model: int) -> MauveResult:
    curve = out.get("divergence_curve")
    return MauveResult(
        mauve=float(out["mauve"]),
        frontier_integral=float(out["frontier_integral"]),
        mauve_star=float(out.get("mauve_star", math.nan)),
        n_human=n_human,
        n_model=n_model,
        divergence_curve=None if curve is None else np.asarray(curve, dtype=np.float64),
    )


def mauve_from_features(
    human_features: np.ndarray,
    model_features: np.ndarray,
    *,
    num_buckets: int | str = "auto",
    seed: int = 25,
    isolate: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> MauveResult:
    """Quantise two feature sets and compute MAUVE.

    With ``isolate=True`` (default) the clustering runs in a fresh interpreter that
    never imports ``torch``, so MAUVE cannot crash an evaluation that has large
    models loaded.  Extra keyword arguments go to ``mauve.compute_mauve``
    (``kmeans_num_redo``, ``mauve_scaling_factor``, ...).
    """
    if not mauve_available():
        raise MissingDependency(_INSTALL_HINT)
    p = np.ascontiguousarray(human_features, dtype=np.float32)
    q = np.ascontiguousarray(model_features, dtype=np.float32)
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1]:
        raise ValueError("features must be [n, d] arrays with a common d")
    options = {"num_buckets": num_buckets, "seed": seed, "verbose": verbose, **kwargs}

    if not isolate:
        import mauve

        out = mauve.compute_mauve(p_features=p, q_features=q, **options)
        return _result(vars(out), len(p), len(q))

    with tempfile.TemporaryDirectory(prefix="proseometry-mauve-") as tmp:
        inputs = Path(tmp) / "features.npz"
        outputs = Path(tmp) / "result.json"
        np.savez(inputs, p=p, q=q, options=json.dumps(options))
        _run_worker("cluster", inputs, outputs)
        return _result(json.loads(outputs.read_text(encoding="utf-8")), len(p), len(q))


def mauve_score(
    human_texts: Sequence[str] | None = None,
    model_texts: Sequence[str] | None = None,
    *,
    human_features: np.ndarray | None = None,
    model_features: np.ndarray | None = None,
    featurize_model_name: str = "gpt2-large",
    device_id: int = -1,
    max_text_length: int = 512,
    batch_size: int = 8,
    num_buckets: int | str = "auto",
    seed: int = 25,
    isolate: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> MauveResult:
    """Compute MAUVE between human and model samples.

    Pass raw texts (featurised with ``featurize_model_name``; ``device_id=-1`` means
    CPU) or precomputed ``*_features`` arrays of shape ``[n, d]`` from any encoder.
    A few thousand samples per side is the recommended regime; with fewer samples
    MAUVE is optimistic and noisy, so keep ``seed`` and all other arguments fixed
    across the models you compare.
    """
    if human_features is None:
        if human_texts is None:
            raise ValueError("provide human_texts or human_features")
        human_features = featurize(
            human_texts,
            featurize_model_name=featurize_model_name,
            device_id=device_id,
            max_text_length=max_text_length,
            batch_size=batch_size,
            verbose=verbose,
        )
    if model_features is None:
        if model_texts is None:
            raise ValueError("provide model_texts or model_features")
        model_features = featurize(
            model_texts,
            featurize_model_name=featurize_model_name,
            device_id=device_id,
            max_text_length=max_text_length,
            batch_size=batch_size,
            verbose=verbose,
        )
    return mauve_from_features(
        human_features,
        model_features,
        num_buckets=num_buckets,
        seed=seed,
        isolate=isolate,
        verbose=verbose,
        **kwargs,
    )

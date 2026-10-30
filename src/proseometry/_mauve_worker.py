"""Subprocess entry points for the two halves of MAUVE.

``python -m proseometry._mauve_worker cluster features.npz result.json`` quantises
precomputed features and computes MAUVE in a process that never imports torch.

``python -m proseometry._mauve_worker featurize texts.json features.npz`` embeds
texts in a process that imports torch *before* mauve (and therefore before faiss
and scikit-learn), which is the import order under which featurisation is safe.

See :mod:`proseometry.distribution` for why the halves are kept apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def cluster(inputs: Path, outputs: Path) -> None:
    data = np.load(inputs, allow_pickle=False)
    options = json.loads(str(data["options"]))

    import mauve

    out = mauve.compute_mauve(p_features=data["p"], q_features=data["q"], **options)
    curve = getattr(out, "divergence_curve", None)
    outputs.write_text(
        json.dumps(
            {
                "mauve": float(out.mauve),
                "frontier_integral": float(out.frontier_integral),
                "mauve_star": float(getattr(out, "mauve_star", float("nan"))),
                "divergence_curve": None if curve is None else np.asarray(curve).tolist(),
            }
        ),
        encoding="utf-8",
    )


def featurize(inputs: Path, outputs: Path) -> None:
    spec = json.loads(inputs.read_text(encoding="utf-8"))

    from .distribution import featurize as embed

    np.save(outputs, embed(spec["texts"], **spec["options"]))


def main(argv: list[str]) -> int:
    command, inputs, outputs = argv[0], Path(argv[1]), Path(argv[2])
    if command == "cluster":
        cluster(inputs, outputs)
    elif command == "featurize":
        featurize(inputs, outputs)
    else:
        raise SystemExit(f"unknown command {command!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

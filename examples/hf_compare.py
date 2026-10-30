"""Compare Hugging Face checkpoints with proseometry.

Usage::

    uv run --extra torch python examples/hf_compare.py \
        --texts held_out.txt --separator '<|endoftext|>' \
        --model small=path/or/hub-id --model large=path/or/hub-id \
        --evaluator path/or/hub-id-of-a-stronger-model --out reports/

Every model is run under the same :class:`SuiteConfig`, so the PQI values in the
final table are comparable.  Pass ``--mauve`` to add MAUVE (downloads the
featuriser, ``gpt2-large`` by default, and needs ``proseometry[mauve]``).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from proseometry import Suite, SuiteConfig, compare
from proseometry.adapters import HFModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--texts", required=True, type=pathlib.Path, help="held-out documents")
    parser.add_argument("--separator", default="\n\n", help="document separator in --texts")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--model", action="append", required=True, metavar="NAME=PATH", help="repeatable"
    )
    parser.add_argument("--evaluator", default=None, help="stronger model for LongPPL key tokens")
    parser.add_argument("--device", default=None, help="e.g. cuda, mps, cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-doc-tokens", type=int, default=1024)
    parser.add_argument(
        "--windows", default="16,64,256", help="short-context sizes for context gain"
    )
    parser.add_argument("--short-stride", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--continuation-tokens", type=int, default=256)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--binding-gap", type=int, default=64)
    parser.add_argument("--mauve", action="store_true", help="compute MAUVE (needs mauve-text)")
    parser.add_argument("--mauve-featurizer", default="gpt2-large")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="directory for JSON reports")
    parser.add_argument("--examples", type=int, default=2, help="sample continuations to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = [t.strip() for t in args.texts.read_text(encoding="utf-8").split(args.separator)]
    texts = [t for t in texts if t]
    if args.max_docs:
        texts = texts[: args.max_docs]
    print(f"{len(texts)} held-out documents", file=sys.stderr)

    config = SuiteConfig(
        max_doc_tokens=args.max_doc_tokens,
        context_windows=tuple(int(k) for k in args.windows.split(",")),
        short_stride=args.short_stride,
        binding_min_gap=args.binding_gap,
        prompt_tokens=args.prompt_tokens,
        continuation_tokens=args.continuation_tokens,
        samples_per_prompt=args.samples_per_prompt,
        max_prompts=args.max_prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        mauve=args.mauve,
        mauve_featurizer=args.mauve_featurizer,
    )
    evaluator = None
    if args.evaluator:
        evaluator = HFModel.from_pretrained(
            args.evaluator,
            name=f"evaluator:{args.evaluator}",
            device=args.device,
            batch_size=args.batch_size,
        )
    suite = Suite(config, evaluator=evaluator)

    reports = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            name, path = spec, spec
        if evaluator is not None and path == args.evaluator:
            model = evaluator  # same checkpoint: reuse it, and let the suite flag self-evaluation
        else:
            model = HFModel.from_pretrained(
                path, name=name, device=args.device, batch_size=args.batch_size
            )
        report = suite.run(model, texts, name=name)
        reports.append(report)
        print(report.to_markdown())
        if report.generation is not None and args.examples:
            print()
            print(report.generation.examples(args.examples))
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{name}.json").write_text(report.to_json(), encoding="utf-8")
        del model

    print()
    print(compare(reports))


if __name__ == "__main__":
    main()

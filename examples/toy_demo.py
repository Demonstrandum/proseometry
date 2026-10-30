"""Run the whole suite on two dependency-free toy models in a few seconds.

    uv run python examples/toy_demo.py

The bigram model uses one token of history; the induction model adds in-context
copying.  Expect zero context gain for the bigram, positive gain for the
induction model, and a higher PQI for the latter.  MAUVE is skipped (it needs
the ``mauve`` extra and a featuriser download); the PQI is therefore partial.
"""

from __future__ import annotations

from proseometry import Suite, SuiteConfig, compare, toy


def main() -> None:
    stories = toy.synthetic_stories(80, seed=1)
    train, held_out = stories[:60], stories[60:]
    tokenizer = toy.WordTokenizer.fit(stories)
    bigram = toy.BigramModel.fit(tokenizer, train)
    induction = toy.InductionModel.fit(tokenizer, train, lam=0.5)

    config = SuiteConfig(
        max_doc_tokens=512,
        context_windows=(2, 8, 32),
        binding_min_gap=12,
        binding_max_positions=16,
        prompt_tokens=40,
        continuation_tokens=80,
        mauve=False,
        verbose=False,
    )
    # The induction model doubles as the LongPPL evaluator here; in practice use a
    # stronger model that is not one of those being compared.
    suite = Suite(config, evaluator=induction)
    reports = [suite.run(model, held_out) for model in (bigram, induction)]

    for report in reports:
        print(report.to_markdown())
        print()
    print(compare(reports))
    print()
    print(reports[1].generation.examples(1))


if __name__ == "__main__":
    main()

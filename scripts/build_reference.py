"""Build the default goodness() reference from human narrative prose.

    uv run python scripts/build_reference.py [--out PATH]

The default output is ``src/proseometry/data/reference_english_prose.json``.

Two sources, sampled equally often per window length (``groups`` in
``Reference.from_texts``):

* 14 public-domain novels from Project Gutenberg, header and footer stripped;
* the first few megabytes of the TinyStories validation split (Eldan & Li 2023,
  CDLA-Sharing-1.0) -- simple children's stories, so that short sentences and
  the repetition natural to that register are part of "typical human prose".

With ``--scorer`` (default ``gpt2``; needs ``proseometry[torch]``) the reference
also records bits per byte of human windows under that model, which enables
``goodness(text, fluency=True)``.  Pass ``--scorer none`` to skip it.

Only summary quantiles are stored; no text is redistributed.  Downloads are
cached in ``--cache``.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.request

from proseometry.goodness import DEFAULT_LENGTHS, Reference

BOOKS: dict[int, str] = {
    11: "Alice's Adventures in Wonderland",
    16: "Peter Pan",
    45: "Anne of Green Gables",
    55: "The Wonderful Wizard of Oz",
    74: "The Adventures of Tom Sawyer",
    84: "Frankenstein",
    98: "A Tale of Two Cities",
    113: "The Secret Garden",
    120: "Treasure Island",
    236: "The Jungle Book",
    271: "Black Beauty",
    514: "Little Women",
    1342: "Pride and Prejudice",
    1661: "The Adventures of Sherlock Holmes",
}

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)
TINYSTORIES_BYTES = 3_000_000

_START = re.compile(r"\*\*\* START OF.*?\*\*\*", re.S)
_END = re.compile(r"\*\*\* END OF.*", re.S)


def fetch(book_id: int, cache: pathlib.Path) -> str:
    path = cache / f"pg{book_id}.txt"
    if not path.exists():
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8-sig", errors="replace")


def fetch_tinystories(cache: pathlib.Path) -> list[str]:
    """Complete stories from the first ``TINYSTORIES_BYTES`` of the validation file."""
    path = cache / "tinystories_valid_head.txt"
    if not path.exists():
        print(f"downloading {TINYSTORIES_URL} (first {TINYSTORIES_BYTES} bytes)", file=sys.stderr)
        request = urllib.request.Request(
            TINYSTORIES_URL, headers={"Range": f"bytes=0-{TINYSTORIES_BYTES - 1}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            path.write_bytes(response.read())
    raw = path.read_text(encoding="utf-8", errors="ignore")
    stories = [s.strip() for s in raw.split("<|endoftext|>")]
    return [" ".join(s.split()) for s in stories[1:-1] if s.strip()]  # drop partial ends


def body(raw: str) -> str:
    """The text between the Gutenberg start and end markers, with paragraphs re-flowed."""
    start = _START.search(raw)
    text = raw[start.end() :] if start else raw
    text = _END.sub("", text)
    paragraphs = re.split(r"\n\s*\n", text)
    flowed = [" ".join(p.split()) for p in paragraphs]
    return "\n".join(p for p in flowed if len(p.split()) >= 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("src/proseometry/data/reference_english_prose.json"),
    )
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path(".cache/gutenberg"))
    parser.add_argument("--windows-per-length", type=int, default=3000)
    parser.add_argument(
        "--scorer", default="gpt2", help="HF model id for the fluency check, or 'none'"
    )
    parser.add_argument("--lm-windows-per-length", type=int, default=800)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    novels = [body(fetch(book_id, args.cache)) for book_id in BOOKS]
    stories = fetch_tinystories(args.cache)
    texts = novels + stories
    groups = ["novels"] * len(novels) + ["stories"] * len(stories)
    scorer = None
    if args.scorer.lower() != "none":
        from proseometry.adapters import HFModel

        scorer = HFModel.from_pretrained(args.scorer, name=args.scorer, device=args.device)
    reference = Reference.from_texts(
        texts,
        lengths=DEFAULT_LENGTHS,
        windows_per_length=args.windows_per_length,
        groups=groups,
        scorer=scorer,
        lm_windows_per_length=args.lm_windows_per_length,
        seed=args.seed,
        name="english-prose",
    )
    reference.info.update(
        {
            "sources": [f"Project Gutenberg #{k}: {v}" for k, v in BOOKS.items()]
            + [
                f"TinyStoriesV2-GPT4-valid.txt "
                f"(first {TINYSTORIES_BYTES} bytes, {len(stories)} stories)"
            ],
            "total_words": {
                "novels": sum(len(t.split()) for t in novels),
                "stories": sum(len(t.split()) for t in stories),
            },
            "lengths": list(DEFAULT_LENGTHS),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    reference.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    for length in reference.lengths:
        row = {k: round(float(v[len(v) // 2]), 3) for k, v in reference.quantiles[length].items()}
        print(f"  medians @ {length:>4} words: {row}")


if __name__ == "__main__":
    main()

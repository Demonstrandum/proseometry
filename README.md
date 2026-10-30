# proseometry

Language-model evaluation for what perplexity misses: **long-context use**,
**degeneration**, and **distributional realism** -- plus one combined number,
the **Prose Quality Index (PQI)**.

Perplexity is a mean of teacher-forced next-token losses. That makes it blind in
two independent ways:

* **It averages.** Only a small fraction of tokens -- a character's name 500 tokens
  later, the object introduced on page one -- actually need distant context; the
  rest are predicted from the last few tokens. A model that is slightly better at
  local statistics beats one that is much better at long-range use on the mean
  (Fang et al. 2025 measure ~zero correlation between PPL and long-context benchmarks).
* **It never lets the model speak.** Conditioning on the human prefix hides what
  happens when a model feeds on its own output. Repetition is the degenerate
  attractor of likelihood training and is invisible under teacher forcing; worse,
  a repeating loop is the perplexity *minimiser*, so low generation perplexity is a
  warning sign, not a virtue.

So the same checkpoint can have low perplexity while writing repetitive nonsense,
and a rival can have higher perplexity while writing coherent stories with
long-range callbacks. proseometry measures those properties directly.

## Install

Not on PyPI yet; install from the repository (or a local checkout with `uv add --editable path/to/proseometry`):

```sh
uv add "proseometry @ git+https://github.com/Demonstrandum/proseometry"          # core: numpy only
uv add "proseometry[torch] @ git+https://github.com/Demonstrandum/proseometry"   # PyTorch / Hugging Face adapters
uv add "proseometry[mauve] @ git+https://github.com/Demonstrandum/proseometry"   # MAUVE via the reference `mauve-text`
uv add "proseometry[all] @ git+https://github.com/Demonstrandum/proseometry"
```

Inside the checkout: `uv sync --all-extras` and `uv run pytest`.

## Quick start

```python
from proseometry import Suite, SuiteConfig, compare
from proseometry.adapters import HFModel

held_out = open("valid.txt").read().split("<|endoftext|>")

suite = Suite(
    SuiteConfig(prompt_tokens=64, continuation_tokens=128, context_windows=(8, 32, 128)),
    evaluator=HFModel.from_pretrained("a/stronger-model"),   # optional: selects LongPPL key tokens
)
reports = [
    suite.run(HFModel.from_pretrained(path, name=name, device="cuda"), held_out)
    for name, path in {"run-17": "ckpt/17", "run-23": "ckpt/23"}.items()
]
for report in reports:
    print(report.to_markdown())          # every metric with the human reference beside it
    print(report.generation.examples(2)) # read a few continuations: always do this
print(compare(reports))                  # one table, sorted by PQI
```

`examples/hf_compare.py` wraps this in a CLI; `examples/toy_demo.py` runs the
suite on dependency-free toy models in a few seconds.

Your own PyTorch model needs one line:

```python
from proseometry.adapters import TorchModel
model = TorchModel(module, tokenizer, name="run-17", bos_id=1, eos_id=2, max_context=2048)
```

`module(ids)` must return `[B, T, V]` logits (or an object with `.logits`); pass
`forward=lambda m, ids: ...` otherwise. The tokenizer only needs `encode`/`decode`.
Anything else can subclass `proseometry.Model` and implement two array methods.

## The metrics

All teacher-forced metrics start from one pass over each held-out document with
its full context and one per short window `k` where every token sees only its
last `k` predecessors (plus BOS). Generation-side metrics prompt the model with
the first `prompt_tokens` of each document and compare its continuation with the
*real* continuation of the same length, so every statistic has a paired human
reference.

| Metric | What it measures | Blind spot it covers |
|---|---|---|
| **Bits per byte / perplexity** | Ordinary likelihood, in tokenizer-independent units. | Baseline; comparable across vocabularies. |
| **Context gain** $G_k$ | Mean of $\log p(x_t \mid x_{<t}) - \log p(x_t \mid x_{t-k..t-1})$ over tokens with more than $k$ of history; reported in nats and as relative perplexity reduction. A curve over `context_windows`. | How much a model *actually uses* history beyond $k$ tokens. A local model is flat at 0. |
| **LongPPL** (Fang et al. 2025) | Perplexity on *key tokens* only: those whose log-prob rises by more than $\alpha$ with long context and that are predictable ($> \beta$) given it, selected by a separate **evaluator** model. Key spans transfer across tokenizers by character offsets. | The averaging problem. In the example below it widens a 2.6x PPL gap to 25x. |
| **Name binding** | At positions where a character's name recurs after $\geq$ `binding_min_gap` tokens of absence, the model must give the highest probability -- among all names introduced earlier -- to the right one. Reports accuracy, chance, a recency baseline, and chance-corrected **skill**. | "Does it remember *who*" without penalising pronouns; the recency baseline shows the probe really targets non-recent names. |
| **seq-rep-4, compression ratio** | Per-sample repeated-4-gram fraction (Welleck et al. 2020) and zlib ratio of generated text, compared with the human continuations by **overlap = 1 - KS**. | Free-running degeneration -- in *both* directions: looping and word salad. |
| **Callback / new-name rate** | Per 100 words of continuation: mentions of prompt characters, and distinct names never seen in the prompt; overlap with human continuations. | Generation-side memory: inventing a new protagonist, or parroting one. |
| **MAUVE** (Pillutla et al. 2021) | Divergence-frontier gap between model and human text in a GPT-2 embedding space; penalises text people never write and modes the model cannot produce. | "Does it read like a person wrote it" -- the semantic nonsense that surface statistics miss. |
| **Judge** (optional) | Fixed 1-5 rubric for grammar, coherence and consistency with the opening (Eldan & Li 2023), via any `Callable[[str], str]`. | Same as MAUVE, with a human-readable rationale; useful to spot-check. |

## The Prose Quality Index

PQI answers "is this sensible text that uses its context?" with one number built
only from measurements that do not share perplexity's blind spots. Two rules make
the combination meaningful rather than a weighted soup:

1. **Every sub-measure lives in [0, 1] and 1 means human level.** Distributional
   measures are *indistinguishability from human text* on some view (MAUVE; the
   `1 - KS` overlaps). Name binding is a *skill score*, `(accuracy - chance) / (1 - chance)`.
   Nothing is z-scored against a population of models, so a single model gets a
   meaningful score and rankings do not shift when a new model is added.
2. **Arithmetic mean within a component, geometric mean across components.**
   Sub-measures inside a component are alternative views of one property;
   components are different properties, and failing any one of them is fatal to
   quality, so a strength in one cannot buy forgiveness for a collapse in another.

| Component | Sub-measures | Property |
|---|---|---|
| `realism` | MAUVE; judge coherence (if a judge is configured) | reads like human text |
| `surface` | seq-rep-4 overlap; compression-ratio overlap | human repetition / entropy profile |
| `memory` | name-binding skill; callback-rate overlap; new-name-rate overlap; judge consistency | uses and retains what came before |

$$
\mathrm{PQI} = \exp\Big(\sum_c w_c \ln \max(\bar m_c, \epsilon)\Big), \qquad \sum_c w_c = 1
$$

with $\bar m_c$ the mean of the available sub-measures of component $c$, equal
weights by default, and a floor $\epsilon = 0.01$ so that a total failure on one
axis still leaves the ordering carried by the others. A component with no
available sub-measure (e.g. MAUVE not installed) is skipped and the report is
marked **partial** -- a partial PQI is blind on that axis, exactly like
perplexity is, so treat it as such.

How to read it: a repetitive low-perplexity model scores well on `memory` and
badly on `surface`/`realism`; a fluent model that forgets its characters scores
well on `realism` and badly on `memory`; the geometric mean ranks both below a
model that is merely decent on everything. PQI is for **ranking models under one
fixed protocol** (same held-out texts, prompt/continuation lengths, decoding
settings, sample counts, MAUVE featuriser). Its absolute value moves with those
choices, which is why the components are always reported next to it.

## Example: TinyStories-1M vs TinyStories-33M

120 TinyStories validation stories, 40 prompts of 64 tokens with 128-token
continuations (temperature 1.0, nucleus 0.95), the 33M model as LongPPL evaluator
(so its own LongPPL is self-selected and flagged; use a third, stronger model in
practice). Produced by `examples/hf_compare.py` on a laptop CPU in ~19 minutes.

| model | PQI | surface | memory | bits/byte | PPL | PPL drop @k=8 | @k=32 | @k=128 | LongPPL | binding acc. (chance 0.35, recency 0.13) | skill | seq-rep-4 (model/human) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 33M | 0.816* | 0.787 | 0.846 | 0.435 | 3.38 | 69.0% | 32.7% | 6.5% | 1.91 | 0.910 | 0.862 | 0.021 / 0.016 |
| 1M | 0.697* | 0.662 | 0.734 | 0.775 | 8.76 | 62.1% | 22.6% | 3.1% | 48.07 | 0.791 | 0.678 | 0.014 / 0.016 |

\*partial: MAUVE was off in this run. Computed separately with the `gpt2`
featuriser on 150 prompts, MAUVE is **0.61 vs 0.26**, which completes the index
at roughly **0.74 vs 0.50**.

Things worth noticing: LongPPL turns a 2.6x perplexity gap into a 25x gap on the
1.3% of tokens that depend on long context; the recency heuristic scores far
*below* chance, so the binding probe is really asking about names that were not
the last ones mentioned; the 1M model's `memory` axis is respectable -- it keeps
using "Tom" and "Sam" -- even though its text is nonsense ("Tom thought about it
for a moment with his paw"), which is precisely what the `realism` axis exists to
catch. Read a few continuations with `report.generation.examples()` before
trusting any number.

## Protocol notes

* **Sample, don't argmax.** Greedy decoding makes nearly every LM loop, so a greedy
  comparison measures the decoder. Fix one decoding for all models (default
  temperature 1.0, top-p 0.95) or sweep temperature and compare frontiers.
* **Sample sizes.** MAUVE wants a few thousand samples per side and is optimistic
  and noisy below that; the overlaps and binding accuracy come with bootstrap
  intervals (`report.binding.accuracy_interval()`, `report.likelihood.bits_per_byte_interval()`).
* **Evaluator.** LongPPL needs a stronger reference model; Fang et al. show that
  letting a model select its own key tokens does not discriminate. The suite
  flags self-evaluation when it can detect it (same object).
* **Names.** The default name detector is a heuristic (a capitalised word that
  appears at least once mid-sentence); it suits narrative prose. For other
  domains pass `Suite(..., extractor=my_ner)` returning `text.Mention` spans.
* **Cost.** Exact short-context scoring costs about $k$ forward tokens per token
  for each window; `short_stride=s` divides that by $s$ at the price of contexts
  between $k$ and $k+s-1$ tokens long. The name probe scores every candidate as a
  continuation of the prefix; cap it with `binding_max_positions`.
* **Units.** Compare models with different tokenizers in bits per byte, never in
  per-token perplexity. LongPPL key tokens are transferred from the evaluator to
  each model by character offsets, so it works across vocabularies; the name
  probe and the surface statistics operate on each model's own decoded text.
* **MAUVE process isolation.** `torch`, `faiss` and `scikit-learn` each ship an
  OpenMP runtime; together in one process they can segfault (observed on macOS).
  proseometry runs MAUVE's clustering in a torch-free subprocess and moves
  featurisation to a subprocess when `faiss`/`sklearn` were imported before `torch`,
  so a long run cannot be lost at the last step. `isolate=False` turns this off.

## Extending

* **Any model:** subclass `Model` and implement `_token_logprobs(ids) -> [B, T-1]`
  and `_next_token_logits(ids) -> [B, V]` (or `LogitsModel.logits(ids) -> [B, T, V]`
  in numpy). Override `generate` if you have a cached sampler.
* **Any judge:** `Suite(..., judge=fn)` with `fn(prompt) -> str`;
  `judge.openai_compatible_judge(model=...)` builds one for OpenAI-style endpoints
  and reads the key from an environment variable at call time.
* **Any statistic:** `degeneration.compare_statistic(name, fn, model_texts, human_texts)`
  gives the overlap for any per-sample function, and `quality.prose_quality_index`
  accepts arbitrary sub-measure dictionaries.

## Development

```sh
uv sync --all-extras
uv run pytest           # 51 tests; the Hugging Face adapter test downloads a tiny checkpoint
uv run ruff check src tests examples && uv run ruff format --check src tests examples
```

## References

* Fang, Wang, Liu, Zhang, Jegelka, Gao, Ding, Wang. *What is Wrong with Perplexity for Long-context Language Modeling?* ICLR 2025. [arXiv:2410.23771](https://arxiv.org/abs/2410.23771) -- LongPPL, key tokens.
* Pillutla, Swayamdipta, Zellers, Thickstun, Welleck, Choi, Harchaoui. *MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers.* NeurIPS 2021. [arXiv:2102.01454](https://arxiv.org/abs/2102.01454)
* Holtzman, Buys, Du, Forbes, Choi. *The Curious Case of Neural Text Degeneration.* ICLR 2020 -- nucleus sampling, degeneration.
* Welleck, Kulikov, Roller, Dinan, Cho, Weston. *Neural Text Generation with Unlikelihood Training.* ICLR 2020 -- seq-rep-n.
* Sun, Krishna, Mattarella-Micke, Iyyer. *Do Long-Range Language Models Actually Use Long-Range Context?* EMNLP 2021 -- context ablation.
* Eldan & Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* 2023 -- rubric grading of small-model stories.

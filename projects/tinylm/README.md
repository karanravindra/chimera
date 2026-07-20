# tinylm

A research-scale language model trained end to end, from tokenizer and pretraining
through instruction tuning, preference tuning, and reinforcement learning. The target
is a small assistant that can handle arbitrary prompts, follow instructions, reason
with everyday commonsense, stay grounded in supplied context, and track multi-turn
corrections. It is a portfolio research project rather than a production knowledge
assistant: a few reliable, inspectable capabilities matter more than broad memorization.

## Capability target

- Follow both transformation instructions (summarize, rewrite, extract, format) and
  open-ended requests.
- Answer from supplied passages and say when the requested information is absent.
- Carry context across multiple turns, including corrections and revised constraints.
- Apply everyday causal, physical, and social commonsense.
- Scale from the initial 512-token experiments to an 8192-token context. Long-context
  stages will use batches of roughly 8–16, length-bucketed so compute is not dominated
  by padding.

## Training roadmap

| stage                 | purpose                                                 | data emphasis                                                                     |
| --------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [pretrain](pretrain/) | language, commonsense, representation learning          | FineWeb-Edu, Cosmopedia, TinyStories, Stack Exchange, Wikipedia/Wikibooks         |
| late pretrain         | introduce long documents and conversation structure     | long web/reference documents plus 1–3% filtered ChatML conversations              |
| context extension     | reach 8192 tokens without losing short-context ability  | length-bucketed long documents and complete grounded conversations                |
| [SFT](sft/)           | instruction following and grounded assistant behavior   | CoQA, QuAC, No Robots, filtered OASST1, selected Tulu tasks                       |
| preference tuning     | response quality, correction handling, calibrated style | OASST1 ratings and curated chosen/rejected responses                              |
| RLVR                  | objectively checkable behavior                          | formatting constraints, extraction, grounded QA, and instruction-following checks |

Pretraining may include a small amount of full-token chat data so role and turn
structure are familiar. Assistant-only loss masking remains an SFT responsibility;
otherwise causal pretraining teaches the model to generate user turns as readily as
assistant turns.

## Dataset priorities

1. **CoQA** — grounded, free-form conversational QA over passages; preserves follow-up
   questions and references to earlier turns.
1. **QuAC** — grounded information-seeking dialogues, including unanswerable questions;
   useful for learning not to guess beyond the context.
1. **OpenAssistant OASST1** — human-authored conversation trees with follow-ups,
   corrections, alternate replies, and quality ratings. Keep high-rated English
   root-to-leaf paths; reserve the ratings for preference tuning.
1. **No Robots** — human-authored summarization, rewriting, extraction,
   classification, brainstorming, and open-ended instructions. Its CC BY-NC 4.0
   license must be reviewed before any use beyond this noncommercial research demo.
1. **SODA** — social-commonsense dialogue; sample lightly because it is synthetic and
   can imprint a repetitive conversational style.
1. **Selected Tulu 3 or Tasksource Instruct subsets** — broad instruction coverage,
   filtered to grounded QA, transformation, formatting, and ordinary conversation.
   Do not ingest the full mixtures blindly: much of their math, code, multilingual,
   and benchmark-derived content is outside this model's target.
1. **Dolma Stack Exchange and Wikipedia/Wikibooks slices** — natural explanatory QA
   and clean long-form reference text for continued pretraining.

Mix by tokens rather than examples. Preserve complete conversations as documents, and
keep train/evaluation splits plus benchmark-derived sources auditable to avoid claiming
zero-shot performance on tasks seen during training.

## Tokenizer plan

Train one immutable tokenizer on a fixed, future-facing corpus, compare a small suite
of vocabulary sizes with model-aware pilots, and freeze the winner before rerunning
pretraining. Every later stage—SFT, preference tuning, and RLVR—must reuse the exact
same artifact.

### Contract

- UTF-8 byte-level BPE with no unknown token, normalization, or lowercasing.
- Lossless round trips for arbitrary Unicode, whitespace, Markdown, and structured
  text.
- Reserve the canonical chat, reasoning, and tool markers at stable low IDs from the
  first training run; never append special tokens after pretraining.
- Keep `split_digits=False` initially. Commonsense and grounded assistance matter more
  than arithmetic, and unsplit digits compress dates and numbers better. Revisit only
  if arithmetic becomes an RLVR target.
- Save and identify the tokenizer by a content hash, not a mutable path or the current
  data-mixture name.

### Fixed training corpus

Sample approximately 250M characters with a fixed seed and explicit per-source
character budgets. This corpus represents the model's lifetime inputs rather than any
single pretraining run:

| source family                             | character share |
| ----------------------------------------- | --------------: |
| FineWeb-Edu                               |             35% |
| Cosmopedia                                |             20% |
| TinyStories                               |             15% |
| Stack Exchange                            |             10% |
| Wikipedia/Wikibooks                       |              5% |
| CoQA, QuAC, and SQuAD                     |            7.5% |
| OASST1, No Robots, and filtered UltraChat |            7.5% |

Cache the sampled corpus once and train every candidate from those identical bytes.
Preserve complete documents and conversations, render conversations with the canonical
ChatML template, and hold out separate source-stratified documents for evaluation.
Record dataset revisions, requested and realized character shares, seed, and a corpus
hash. Do not let `ConcatTextDataModule` resample tokenizer data from each experiment's
mixture.

### Candidate suite

Train 8,192-, 12,288-, and 16,384-token vocabularies. Skip 32k at this model scale:
with tied embeddings at width 384, every additional 8k vocabulary entries cost about
3.1M parameters as well as extra output-loss compute. The existing 16k tokenizer's
roughly 4.0–4.7 characters/token is already a reasonable compression ceiling, so a
larger vocabulary must justify its parameter cost through downstream results.

Evaluate each candidate on the same held-out corpus:

- Characters and UTF-8 bytes per token, aggregate and per source.
- Mean and p95 tokens per document/conversation, plus the fraction fitting within
  512, 2,048, and 8,192 tokens.
- Vocabulary utilization and rare/dead-token rates.
- Atomic special-token encoding and stable IDs.
- Exact round trips and representative tokenizations for Unicode, Markdown, JSON,
  URLs, contractions, dates, and ChatML.
- Embedding parameters, their share of the full model, tokenizer throughput, and the
  effective amount of text visible in each context window.

Reject a tokenizer with a severe chat, grounded-QA, or structured-text regression even
when its aggregate compression is better.

### Model-aware selection

Compression alone does not choose the tokenizer. Run otherwise identical 1k–2k-step
pretraining pilots for every candidate using the same backbone, data mixture, seed,
schedule, global token batch, and tokenizer-specific caches. Compare held-out BPB—not
loss per token—alongside throughput, peak VRAM, total parameters, BLiMP, LAMBADA,
grounded prompt probes, and prompt-format robustness. Select for the complete system;
8k or 12k is the expected winner, while 16k must show a meaningful BPB or capability
gain to pay for its embedding budget.

### Frozen artifact

Store the winner as:

```text
/mnt/ai/data/tinylm/tokenizer/v1/
├── tokenizer.json
├── corpus.jsonl
├── meta.json
└── evaluation.json
```

`meta.json` records the tokenizer hash, vocabulary and special-token IDs, dataset
revisions, realized mixture, seed, training options, corpus hash, and `tokenizers`
version. Add a required `tokenizer_path` to `ConcatTextDataModule`; all tokenized-data
caches remain keyed by the tokenizer content fingerprint.

Implementation starts with `projects/tinylm/data/train_tokenizer.py`, adapted from the
archived suite trainer but using deterministic weighted sampling. It should build the
corpus once, train all candidates, validate round trips and special tokens, and write a
single comparison report. Only after the winner is frozen should the full sources be
retokenized and the key pretraining mixtures rerun.

## Evaluation direction

The public demo accepts arbitrary prompts, so curated examples cannot hide brittle
behavior. In addition to capability benchmarks, evaluate a small adversarial prompt
set covering equivalent prompt phrasings, corrections, missing-information questions,
conflicting instructions, exact output constraints, and short-versus-long context.
See the pretraining and SFT READMEs for current results and stage-specific TODOs.

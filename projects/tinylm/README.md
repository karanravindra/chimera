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

## Evaluation direction

The public demo accepts arbitrary prompts, so curated examples cannot hide brittle
behavior. In addition to capability benchmarks, evaluate a small adversarial prompt
set covering equivalent prompt phrasings, corrections, missing-information questions,
conflicting instructions, exact output constraints, and short-versus-long context.
See the pretraining and SFT READMEs for current results and stage-specific TODOs.

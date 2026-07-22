# tinylm / sft

SFT of the pretrained ~6M GPT (`../pretrain`) into a simple-QA chat assistant.
Same rails as pretrain — packed seq-512 streams, FlexAttention causal+document
masking, CCE (+ logit softcap 30), Muon+AdamW with warmup+cosine — but the
stream is ChatML conversations (`chimera.data.text.chat_template`) with loss ONLY on
assistant tokens (labels -100 elsewhere), and the model/tokenizer come from the
pretrain run (the chat special tokens were reserved in the vocab from day one).

## Layout

- `train.py` — raw PyTorch SFT loop; loads `../pretrain`'s `model.py` + base
  checkpoint, saves to `/mnt/ai/runs/tinylm/sft/chimera_gpt6m_sft.pt`
- Data layer: `chimera.data.text.chat_sft` (`ChatSFTDataModule` + per-dataset
  subclasses; packed loss-masked streams, cached keyed on tokenizer hash)

## Run

```sh
cd projects/tinylm/sft
uv run python train.py
```

## Datasets

Simple-QA brief: bulk single-turn QA + grounded QA + a little chat style.
Full smol-smoltalk deliberately skipped (code/math/long multi-turn — beyond
6M capacity).

| id  | dataset                         | module                            | role                           |
| --- | ------------------------------- | --------------------------------- | ------------------------------ |
| gqc | GooAQ (chat)                    | `GooAQChatDataModule`             | closed-book simple QA (bulk)   |
| sqc | SQuAD (chat)                    | `SQuADChatDataModule`             | grounded QA (model's strength) |
| evc | smoltalk/everyday-conversations | `EverydayConversationsDataModule` | multi-turn chat style          |

### Target mixture

The next SFT stage targets a grounded, multi-turn assistant for arbitrary prompts rather
than bulk closed-book QA. Add the following sources in priority order:

| dataset                       | target role                                                     | handling                                                                         |
| ----------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| CoQA                          | grounded conversational QA                                      | keep each passage and complete dialogue together                                 |
| QuAC                          | grounded follow-ups and unanswerable questions                  | preserve multi-turn state and explicit no-answer cases                           |
| OpenAssistant OASST1          | arbitrary prompts, corrections, conversation repair             | high-rated English root-to-leaf paths only; retain ratings for preference tuning |
| No Robots                     | summarization, rewriting, extraction, formatting, open requests | review CC BY-NC 4.0 restrictions before broader deployment                       |
| selected Tulu 3 tasks         | instruction breadth                                             | keep grounded QA, transformations, and constraint-following subsets              |
| SODA / everyday conversations | social commonsense and chat continuity                          | low weight; avoid synthetic-style dominance                                      |

Starting token-level mixture for the first ablation:

| source                        | share |
| ----------------------------- | ----: |
| CoQA                          |   25% |
| QuAC                          |   20% |
| No Robots                     |   15% |
| filtered OASST1               |   15% |
| selected Tulu tasks           |   10% |
| SQuAD                         |    8% |
| SODA / everyday conversations |    5% |
| GooAQ                         |    2% |

Mix by supervised assistant tokens, not row count. At 8192 context, preserve complete
CoQA/QuAC conversations instead of flattening every question into an independent row.
The tiny-model mix should prefer high-quality targeted subsets over ingesting full
general SFT collections. See the project-level [capability target](../README.md#capability-target).

### Later stages

- Preference tuning: build chosen/rejected pairs from OASST1 ratings and curated model
  failures, especially unsupported answers and failures to follow corrections.
- RLVR: reward exact formatting, extraction, answer support in the supplied context,
  explicit missing-information behavior, and verifiable instruction constraints.
- Keep evaluation prompts and benchmark-derived task instances out of every training
  mixture unless the resulting metric is explicitly labeled supervised rather than
  zero-shot.

## Results

One row per run — append-only, as in `../pretrain`. Metrics: masked val loss
(assistant tokens only) + qualitative generations; benchmark deltas vs the base
checkpoint noted per run (SFT shouldn't tank them).

| run      | steps | base ckpt | mix              | val_loss  | notes                                                                                                   |
| -------- | ----- | --------- | ---------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| gc-ctx4k | 700   | ctx4k     | grounded-core    | **2.49**  | **best assistant** — grounded ✓ ("red", "Mediterranean"); ctx keeps passage visible                     |
| gc-ctx2k | 700   | ctx2k     | grounded-core    | 2.54      | grounded ✓ ("red"); seq-2048                                                                             |
| gc-base  | 700   | tok16k_4b | grounded-core    | 2.93      | chat clean; grounded ✗ ("Tom's ball"→"white") — 512 truncates the passage                               |
| gc-ctx8k | 700   | ctx8k     | grounded-core    | 3.79      | worst — overshoots; diverged at LR 0.005 (→ln V), stable at 0.002 but 8k base too eroded + sparse signal |
| full-ft  | 700   | curric    | gqc44 sqc55 evc1 | 2.975     | chat format lands; grounded extraction regressed (Tom's ball: base "red" → "white")                     |
| lora-r16 | 700   | curric    | gqc44 sqc55 evc1 | 3.293     | 3.3% trainable, AdamW 1e-3 no-wd; identical greeting behavior, same extraction regression ("blue kite") |

grounded-core mix (`gc-*`, 2026-07-21): CoQA + QuAC (grounded multi-turn, lead) +
SQuAD (capped below them, per the pilot's terse-span lesson) + a little GooAQ +
everyday. All on the pinned 16k vocab + the 4BT `tok16k_4b` base or a context stage,
SFT'd at each base's NATIVE seq_len (512/2k/4k/8k). Realized mix was QuAC-heavy (CoQA
capped out at its full ~5M tokens — small dataset).

**Headline: context extension pays off downstream even though it was a wash in
pretraining.** Keeping the grounding passage inside a longer context window improves
grounded extraction — base-512 SFT truncates the passage and hallucinates ("What color
is Tom's ball?"→"white"✗, "Nile flows into?"→"Atlantic"✗), while ctx2k/ctx4k SFT read it
correctly ("red"✓, "Mediterranean"✓). Masked val loss: base 2.93 → ctx2k 2.54 → **ctx4k
2.49 (best)** → ctx8k 3.79. **There is a SWEET SPOT at ctx4k; 8k overshoots** — the 8k
base is the most short-context-eroded (4× positional jump at 6M) and seq-8192 SFT on
short grounded dialogs is mostly filler/sparse supervision. See
[pretrain context-extension results](../pretrain/README.md#results-2026-07-21).

Phase-2 breadth (`gc-ctx4k-p2`, 2026-07-21): added instruction breadth to the ctx4k
base — No Robots (summarize/rewrite/extract, CC BY-NC) + a light SODA sample (social
dialog). **Negative result at 6M: breadth diluted rather than added.** The grounded probe
regressed ("Tom's ball" went from "red" to ignoring the question), summarize/rewrite still
fail (model too small), so the focused grounded-only ctx4k-SFT stays the best assistant.
Masked val 3.66 is NOT comparable to the grounded runs' ~2.5 — No Robots/SODA are 40–66%
supervised (long generative targets) vs the grounded core's short extractive spans, so
absolute loss is higher by task, not by quality. Also: the high-supervision mix **diverged
at the default LR 0.005** (collapsed to ln(V) in <100 steps — the large gradients from long
supervised spans); needed `MUON_LR=0.002`. OASST1 (→ preference tuning) and Tulu (needs
source filtering) were deferred. Verdict: don't broaden SFT at 6M — it's a scale lever, not
a data one. Modules (`NoRobots/SODAChatDataModule`) are kept for a larger base.

Two fixes landed this round: (1) `GPT.sample` now stops at `<|im_end|>` (the per-turn
marker) not just EOS (end-of-conversation, which never appears in a single reply) — the
old behavior ran past the answer into a degenerate loop; residual is the 6M model not
always emitting `<|im_end|>` on harder answers (→ run-ons, model-side). (2) `sft/train.py`
gained `TINYLM_SEQ_LEN/BATCH_SIZE/BASE_CKPT/RUN_TAG/MUON_LR/ADAMW_LR` env knobs; the
ctx8k SFT needed `MUON_LR=0.002` to avoid divergence at seq-8192.

full-ft (2026-07-19): masked val 2.975. Generations: greeting canned-correct, closed-book
fluent-but-circular (capacity), grounded extraction WORSE than base — suspect sqc's terse
span answers at 55% of the mix; try capping SQuAD-chat below GooAQ next.

lora-r16 (2026-07-19): plateaus ~0.32 behind full-ft; chat format transfers fully at 30x
fewer trained params. GOTCHA (2 failed attempts): flex_attn breaks when base weights are
requires_grad\_(False)-frozen — zero grads + forward pinned at uniform (loss = ln V =
9.7041). Freeze by optimizer exclusion instead (see chimera.models.lora docstring).

# tinylm / sft

SFT of the pretrained ~6M GPT (`../pretrain`) into a simple-QA chat assistant.
Same rails as pretrain — packed seq-512 streams, FlexAttention causal+document
masking, CCE (+ logit softcap 30), Muon+AdamW with warmup+cosine — but the
stream is ChatML conversations (`chimera.data.chat_template`) with loss ONLY on
assistant tokens (labels -100 elsewhere), and the model/tokenizer come from the
pretrain run (the chat special tokens were reserved in the vocab from day one).

## Layout

- `train.py` — raw PyTorch SFT loop; loads `../pretrain`'s `model.py` + base
  checkpoint, saves to `/mnt/ai/runs/tinylm/sft/chimera_gpt6m_sft.pt`
- Data layer: `chimera.data.chat_sft` (`ChatSFTDataModule` + per-dataset
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
| full-ft  | 700   | curric    | gqc44 sqc55 evc1 | **2.975** | chat format lands; grounded extraction regressed (Tom's ball: base "red" → "white")                     |
| lora-r16 | 700   | curric    | gqc44 sqc55 evc1 | 3.293     | 3.3% trainable, AdamW 1e-3 no-wd; identical greeting behavior, same extraction regression ("blue kite") |

full-ft (2026-07-19): masked val 2.975. Generations: greeting canned-correct, closed-book
fluent-but-circular (capacity), grounded extraction WORSE than base — suspect sqc's terse
span answers at 55% of the mix; try capping SQuAD-chat below GooAQ next.

lora-r16 (2026-07-19): plateaus ~0.32 behind full-ft; chat format transfers fully at 30x
fewer trained params. GOTCHA (2 failed attempts): flex_attn breaks when base weights are
requires_grad\_(False)-frozen — zero grads + forward pinned at uniform (loss = ln V =
9.7041). Freeze by optimizer exclusion instead (see chimera.models.lora docstring).

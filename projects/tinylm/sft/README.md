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

| id  | dataset                        | module                            | role                          |
|-----|--------------------------------|-----------------------------------|-------------------------------|
| gqc | GooAQ (chat)                   | `GooAQChatDataModule`             | closed-book simple QA (bulk)  |
| sqc | SQuAD (chat)                   | `SQuADChatDataModule`             | grounded QA (model's strength)|
| evc | smoltalk/everyday-conversations| `EverydayConversationsDataModule` | multi-turn chat style         |

## Results

One row per run — append-only, as in `../pretrain`. Metrics: masked val loss
(assistant tokens only) + qualitative generations; benchmark deltas vs the base
checkpoint noted per run (SFT shouldn't tank them).

| run | steps | base ckpt | mix | val_loss | notes |
|-----|-------|-----------|-----|----------|-------|
| full-ft | 700 | curric | gqc44 sqc55 evc1 | **2.975** | chat format lands; grounded extraction regressed (Tom's ball: base "red" → "white") |
| lora-r16 | 700 | curric | gqc44 sqc55 evc1 | 3.293 | 3.3% trainable, AdamW 1e-3 no-wd; identical greeting behavior, same extraction regression ("blue kite") |

full-ft (2026-07-19): masked val 2.975. Generations: greeting canned-correct, closed-book
fluent-but-circular (capacity), grounded extraction WORSE than base — suspect sqc's terse
span answers at 55% of the mix; try capping SQuAD-chat below GooAQ next.

lora-r16 (2026-07-19): plateaus ~0.32 behind full-ft; chat format transfers fully at 30x
fewer trained params. GOTCHA (2 failed attempts): flex_attn breaks when base weights are
requires_grad_(False)-frozen — zero grads + forward pinned at uniform (loss = ln V =
9.7041). Freeze by optimizer exclusion instead (see chimera.models.lora docstring).

# tiny-llm / gpt — pretraining

Pretrain a tiny (5–20M param) decoder-only GPT on the tiny-llm mixture
(see [`../data`](../data)), logging to wandb. Reuses the shared chimera rails
(`GPT`, `MixtureDataModule`, `Muon`, loggers/callbacks); only the metric
conventions are project-specific (`TinyLMModule`).

## Files
```
train.py    entry point (argparse). Builds GPT + Muon + data + trainer, logs to wandb.
module.py   TinyLMModule — per-source val/<src>/bpb logging (see Evals).
bpb.py      per-source + aggregate bytes/token measurement (cached).
```

## Evals — `val/<src>/bpb`

The mix serves **one val loader per source**, so we log the headline
**`val/<src>/bpb`** (e.g. `val/tinystories/bpb`, `val/fineweb-edu/bpb`) — slash-
separated so wandb groups them under one **val** panel. Each source is
normalized by **its own bytes/token** (`bpb.measure`), because:

- **bpb is tokenizer-invariant** — the *only* metric comparable across the
  4k/8k/16k tokenizers (loss/token isn't; a bigger vocab deflates it).
- bytes/token differs per source, so a single global normalizer would make
  cross-source bpb wrong.

Also logged: aggregate **`val/loss`** (nats — the objective + `ModelCheckpoint`
monitor) and **`val/bpb`**; `train/loss` + `train/bpb` on-step; `train/lr`.
**`bpt` is dropped** (it's just `loss/ln2`), as is per-source loss (redundant
with per-source bpb). A final `trainer.test()` emits `test/<src>/bpb`.

Deferred (add later as extra test-phase evals): BLiMP, LAMBADA, TinyStories
generative judge, SciQ/PIQA. bpb is the headline for now.

## Model family (muP)

`--arch` presets (W-H-K-L, head_dim=32, GQA n_kv=1, depth-6; keeps head_dim
fixed + scales width so the swept LRs transfer). Param counts @ 8k vocab, tied:

| preset | dims | 4k | 8k | 16k |
|---|---|---:|---:|---:|
| tiny | 256-8-1-6 | 5.1M | 6.1M | 8.2M |
| **small** (default) | 320-10-1-6 | 7.6M | **8.9M** | 11.5M |
| base | 448-14-1-6 | 14.1M | 15.9M | 19.6M |

Embedding share rises sharply with vocab (tiny@16k = 51%!, base@4k = 13%) — the
reason vocab is really an embedding-budget decision at this scale. Default:
**`small` + 8k tokenizer** (~8.9M, 29.5% embeddings).

## Prerequisite: build the packed mix

`train.py` needs `/mnt/ai/data/tiny-llm/mix/<mix>/{train,val}.bin` + `manifest.json`,
tokenized with `--tokenizer`. This does **not exist yet** — build it after
choosing the vocab: tokenize the `../data` sources with the chosen tokenizer and
pack per the `sources.py` weights (val.bin written per-source so `val/<src>/bpb`
works). Then `bpb.measure` caches bytes/token on first run.

## Run (once the mix is built)

```bash
# smoke run (8k / ~9M, 500 steps)
uv run python projects/tiny-llm/gpt/train.py

# full 2B-token pass: ~2e9 / global_token_count steps (~30.5k @ 65536)
uv run python projects/tiny-llm/gpt/train.py --max-steps 30500

# bigger model / different vocab
uv run python projects/tiny-llm/gpt/train.py --arch base --tokenizer /mnt/ai/data/tiny-llm/tokenizer/16k
```

Defaults: seq_len 1024, global 65536 tok/step, muP LRs (muon 0.013 / adamw 0.006),
tied embedding, compile on, CCE off (marginal at tiny vocab). wandb project
`tiny-llm-pretrain`; runs under `/mnt/ai/runs/tiny-llm/gpt/<run-name>/`.

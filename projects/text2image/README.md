# text2image

General-domain text-to-image data for the full T2I pipeline (DiT). Sibling of the
domain-specific `celeba_afhq/*` projects; uses the open-domain
[**jackyhate/text-to-image-2M**](https://hf.co/datasets/jackyhate/text-to-image-2M)
corpus (MIT, ~2.35M image+dense-caption pairs in webdataset shards).

All data lives under `/mnt/ai` (set `export HF_HOME=/mnt/ai/data/hf` so the HF
download cache stays off the root disk).

## Why this dataset

Of the current captioned T2I datasets, text-to-image-2M is the best fit for a general,
images-included, MIT-licensed corpus at workable scale. Its captions are **dense and
grounded** (good for prompt-following — the DALL·E 3 finding) but **single-register**
(measured: word-count p10/50/90 = 20/50/77, no short/tag forms). The DataModule fixes
that at train time with randomized-length sampling (see below).

It's **shard-incremental**: each shard is 50k images / 7.5 GB. Start with a few shards,
add more later with no code change.

## 1. Prepare a compact local base (`prepare.py`)

We train at 64/128/256, never 512, so we downsize once and store small. `prepare.py`
streams each shard: download 512px → resize longest side to `--max-size` (default 256,
LANCZOS) → re-encode JPEG → write `data_<size>/` shard (prompt json preserved) → delete
the raw shard. **Measured: 7.48 GB → 1.26 GB per shard (~83% smaller).**

```bash
export HF_HOME=/mnt/ai/data/hf
# recommended 200k-image subset (~5 GB at 256px)
uv run python projects/text2image/prepare.py --shards 0-3
# scale up later (full set ~59 GB at 256px)
uv run python projects/text2image/prepare.py --shards 4-46
```

Resumable (skips shards already compacted). Train at any size ≤ `--max-size`; going
larger needs a re-run with a bigger `--max-size`.

## 2. Train-time loader (`datamodule.py`)

`T2I2MDataModule` streams the compact shards (no `webdataset` dependency; rank/worker
sharding so every sample is seen once), decodes + center-crops to a square
`--image-size`, and yields `(image float32 CHW [0,1], caption)`.

**Caption = randomized length** (arXiv:2506.16679): each step the dense prompt is served
as **long** (full, 40%), **short** (first sentence, 30%), or **tag** (short head, 30%),
with **10% dropout** to `""` for classifier-free guidance. This is the single-caption
analogue of `celeba_afhq/caption/sampling.py`; offline tag/terse augments can be added
later without changing the loader.

```python
from projects.text2image.datamodule import T2I2MDataModule
dm = T2I2MDataModule(image_size=128, batch_size=64)   # data_dir defaults to /mnt/ai/...
dm.setup(); loader = dm.train_dataloader()

# inspect
uv run python projects/text2image/datamodule.py --image-size 128 --batch-size 8
```

## Status

- Compact base built at 256px for shards 0–3 (200k images, ~5 GB) under
  `/mnt/ai/data/text-to-image-2M/data_256/`.
- DataModule validated: `(B,3,128,128)` float32 batches; length tiers 40/30/30 + 10% dropout.
- Next: wire `T2I2MDataModule` into the DiT trainer; optionally an offline LLM augment for
  high-quality tag/terse caption variants.

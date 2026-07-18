# tiny-llm / data

Data mixture for a **5–20M param** general-language + chat model. The dataset
research behind these choices is in [`DATASETS.md`](./DATASETS.md).

## Composition (pretrain, 2B-token budget)

At 5–20M params the binding constraint is **distribution width, not volume** — so
the blend is dominated by narrow, clean synthetic prose, with knowledge / web
registers layered in at small weights. Chat is a **separate SFT phase**, not
blended into pretrain.

| Source | Weight | Role |
|---|---:|---|
| `tinystories` (TinyStoriesV2-GPT4) | **50%** | fluency backbone — coherent simple English |
| `tiny-strange-textbooks` | 22% | knowledge / expository |
| `finephrase-tutorial` | 9% | explanatory register (chat-adjacent) |
| `finephrase-faq` | 9% | Q&A register |
| `fineweb-edu` (sample-10BT) | 10% | natural-web grounding |
| `smol-smoltalk` | *SFT-only* | chat SFT (trimmed SmolTalk for <1B models) |

**Budget:** `TARGET_TOKENS = 2B` (see `sources.py`). Recommended sweet spot over
the 1–10BT range — ~1 clean epoch of TinyStories + <1 epoch of everything else,
no forced repetition. To scale up, raise the budget and re-stage larger slices of
the *unlimited* sources (finephrase / fineweb-edu); let TinyStories' **share**
fall so its epoch count stays ≤~3.

## Layout

```
sources.py     registry: per-source weight, HF repo, slice, est. tokens, budget
download.py    stage raw parquet shards -> /mnt/ai/data/tiny-llm/raw/<key>/
DATASETS.md    research reference on tiny-LM datasets
```

Raw data is staged under `/mnt/ai/data/tiny-llm/raw/<key>/` (parquet +
`manifest.json` per source). **Nothing is tokenized yet** — this project trains
its own tokenizer; tokenization + packing come after.

## Usage

```bash
uv run python sources.py            # print the mixture plan (weights, targets, repeat)
uv run python download.py           # stage all sources' raw parquet
uv run python download.py fineweb-edu   # re-stage a single source
```

## Notes

- `tiny-strange-textbooks` is **gated** upstream — accept the terms on its HF page;
  it streams fine with `HF_TOKEN` set.
- `finephrase` configs are **>1 TB each** — we stage only a bounded slice
  (first N shards); never pull a whole config.
- `est_tokens` in the manifests is a crude `bytes/≈4` estimate. Real per-source
  token counts (and thus exact mixture packing) are measured once the tokenizer
  exists.
- Licenses vary — TinyStories cdla-sharing-1.0; tiny-strange-textbooks /
  smol-smoltalk apache-2.0; finephrase / fineweb-edu ODC-BY. See `sources.py`.
```

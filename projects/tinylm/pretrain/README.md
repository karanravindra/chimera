# tinylm / pretrain

Pretrains a ~6M-param GPT (dim 384, 12 heads, 6 layers, ReLU² MLP, RoPE + QK-norm,
tied embeddings, 16k BPE vocab) on a blended text mixture (per-run composition logged
in Results), with the vocab trained on a blended sample of that run's sources. Packed
at seq-len 512 with FlexAttention causal+document masking and per-document RoPE
positions, Cut Cross Entropy, and Muon+AdamW.

## Layout

- `model.py` — the GPT (project-local on purpose: per-doc RoPE position reset, no muP —
  diverges from `chimera.models.gpt`; candidate for the library once the unified LLM
  redo picks the canonical GPT)
- `train.py` — raw PyTorch training loop; saves the checkpoint to
  `/mnt/ai/runs/tinylm/pretrain/chimera_gpt6m.pt`
- `main.ipynb` — analysis only: loads the checkpoint, mask visualization, samples,
  zero-shot benchmarks (`chimera.evals`), train-step profile

## Run

```sh
cd projects/tinylm/pretrain
uv run python train.py
```

## Datasets

Index of source `id`s used in the `mix` column below (`chimera.data` module → HF repo).
Add a row here when a new source gets an id.

| id  | dataset                   | module                           | HF repo                                       |
|-----|---------------------------|----------------------------------|-----------------------------------------------|
| tt  | Tiny Textbooks            | `TinyTextbooksDataModule`        | `nampdn-ai/tiny-textbooks`                    |
| str | Tiny Strange Textbooks    | `TinyStrangeTextbooksDataModule` | `nampdn-ai/tiny-strange-textbooks`            |
| fw  | FineWeb-Edu (sample-10BT) | `FineWebEduTextDataModule`       | `HuggingFaceFW/fineweb-edu`                   |
| ts  | TinyStories v2            | `TinyStoriesV2DataModule`        | `noanabeshima/TinyStoriesV2`                  |
| wt  | tiny-webtext              | `TinyWebTextDataModule`          | `nampdn-ai/tiny-webtext`                      |
| cos | Cosmopedia v2             | `CosmopediaV2DataModule`         | `HuggingFaceTB/smollm-corpus` (cosmopedia-v2) |

`cos` is wired but not yet in a logged run.

## Results

One row per run — an append-only log as we iterate mixtures. Zero-shot `lm_eval`
scores (%); headline metric per task: `acc` for blimp & lambada_openai, `acc_norm` for
piqa / sciq / arc_easy. **5k steps unless the row notes otherwise** (~65k tokens/step:
batch 128 × seq 512). Best real run bolded per task; `gpt2` is a reference ceiling
(~20x params), `chance` the floor.

`mix` = per-source share of the training pool (sampling weight = per-source token cap);
source `id`s are defined in Datasets above.

| run    | steps | mix                       | blimp     | lambada   | piqa      | sciq      | arc_easy  |
|--------|-------|---------------------------|-----------|-----------|-----------|-----------|-----------|
| 5-way  | 5k    | tt30 str25 fw20 ts15 wt10 | **67.94** | **16.11** | 56.42     | **55.30** | **34.89** |
| tt+ts  | 1k    | tt50 ts50                 | 65.03     | 12.59     | 56.53     | 54.70     | 31.99     |
| tt     | 1k    | tt100                     | 63.72     | 6.95      | **56.96** | 55.10     | 33.63     |
| ts     | 1k    | ts100                     | 62.93     | 10.87     | 52.34     | 27.40     | 26.94     |
| chance | —     | —                         | 50.0      | 0.0       | 50.0      | 25.0      | 25.0      |
| gpt2   | —     | — (124M ref)              | 82.29     | 32.16     | 62.62     | 64.40     | 39.52     |

5-way stderr: blimp 0.16, lambada 0.51, piqa 1.16, sciq 1.57, arc_easy 0.98.

Notes: the 1k-iter rows are the original ablation (TinyStories v2 + Tiny Textbooks
only) — Tiny Textbooks dominates every task but LAMBADA (long-range narrative, where
TinyStories helps); the 50/50 lands between. The 5-way tops the 50/50 everywhere but a
PIQA wash — confounded by 5x the steps and the retrained tokenizer, not the mix alone.

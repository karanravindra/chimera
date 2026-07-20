# tinylm / pretrain

Pretrains a ~6M-param GPT (dim 384, 12 heads, 6 layers, ReLU² MLP, RoPE + QK-norm,
tied embeddings, 16k BPE vocab) on a blended text mixture (per-run composition logged
in Results), with the vocab trained on a blended sample of that run's sources. Packed
at seq-len 512 with FlexAttention causal+document masking and per-document RoPE
positions, Cut Cross Entropy, and Muon+AdamW.

## Layout

- `model.py` — the GPT, project-local on purpose: per-doc RoPE position reset, VWN
  routing, and no muP; it is the canonical model for this experiment
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

| id  | dataset                     | module                           | HF repo                                       |
| --- | --------------------------- | -------------------------------- | --------------------------------------------- |
| tt  | Tiny Textbooks              | `TinyTextbooksDataModule`        | `nampdn-ai/tiny-textbooks`                    |
| str | Tiny Strange Textbooks      | `TinyStrangeTextbooksDataModule` | `nampdn-ai/tiny-strange-textbooks`            |
| fw  | FineWeb-Edu (sample-10BT)   | `FineWebEduTextDataModule`       | `HuggingFaceFW/fineweb-edu`                   |
| ts  | TinyStories v2              | `TinyStoriesV2DataModule`        | `noanabeshima/TinyStoriesV2`                  |
| wt  | tiny-webtext                | `TinyWebTextDataModule`          | `nampdn-ai/tiny-webtext`                      |
| cos | Cosmopedia v2               | `CosmopediaV2DataModule`         | `HuggingFaceTB/smollm-corpus` (cosmopedia-v2) |
| gq  | GooAQ (Q:/A: pairs)         | `GooAQDataModule`                | `sentence-transformers/gooaq`                 |
| sq  | SQuAD-as-text (passage+QA)  | `SQuADTextDataModule`            | `rajpurkar/squad`                             |
| doc | local documents (always-on) | `LocalDocumentsDataModule`       | `projects/tinylm/documents/*.md`              |

`cos` is the current best textbook source (see Results) — beats `str` on blimp + lambada.

## Results

One row per run — an append-only log as we iterate mixtures. Zero-shot `lm_eval`
scores (%); headline metric per task: `acc` for blimp & lambada_openai, `acc_norm` for
piqa / sciq / arc_easy. **5k steps unless the row notes otherwise** (~65k tokens/step:
batch 128 × seq 512). Best real run bolded per task; `gpt2` is a reference ceiling
(~20x params), `chance` the floor.

`mix` = per-source share of the training pool (sampling weight = per-source token cap);
source `id`s are defined in Datasets above.

| run     | steps | mix                               | blimp     | lambada   | piqa      | sciq      | arc_easy  |
| ------- | ----- | --------------------------------- | --------- | --------- | --------- | --------- | --------- |
| vwn+mhc | 5k    | curric + VWN(2,3) 1.5× + mHC-Lite | **71.01** | **18.36** | 56.31     | 67.80     | **35.35** |
| curric  | 5k    | qa-mix, sc30, cosine LR, cos20→40 | 69.86     | 17.47     | 56.47     | **68.10** | 35.23     |
| sc30    | 5k    | qa-mix + logit softcap 30         | 69.18     | 16.86     | 55.93     | 67.70     | 34.93     |
| qa-mix  | 5k    | cos30 fw34 ts30 gq5 sq1 +doc      | 69.53     | 17.27     | 57.24     | 67.40     | 34.76     |
| cos     | 5k    | cos30 fw40 ts30                   | 70.09     | 16.94     | 55.44     | 54.50     | 33.96     |
| 3-way   | 5k    | str30 fw40 ts30                   | 68.66     | 15.54     | 56.37     | 54.80     | 34.55     |
| 4-way   | 5k    | tt30 str30 fw25 ts15              | 67.63     | 16.01     | **57.29** | 55.80     | 34.34     |
| 5-way   | 5k    | tt30 str25 fw20 ts15 wt10         | 67.94     | 16.11     | 56.42     | 55.30     | 34.89     |
| tt+ts   | 5k    | tt50 ts50                         | 65.03     | 12.59     | 56.53     | 54.70     | 31.99     |
| tt      | 5k    | tt100                             | 63.72     | 6.95      | 56.96     | 55.10     | 33.63     |
| ts      | 5k    | ts100                             | 62.93     | 10.87     | 52.34     | 27.40     | 26.94     |
| chance  | —     | —                                 | 50.0      | 0.0       | 50.0      | 25.0      | 25.0      |
| gpt2    | —     | — (124M ref)                      | 82.29     | 32.16     | 62.62     | 64.40     | 39.52     |

5-way stderr: blimp 0.16, lambada 0.51, piqa 1.16, sciq 1.57, arc_easy 0.98.

Notes: all rows are 5k steps (matched). The knowledge/reasoning tasks (piqa/sciq/arc)
sit within noise across every mix including tt-alone — capacity-bound at 6M, not
data-bound. The trainable axes are blimp (grammar) and lambada (long-range), driven
mainly by FineWeb. As the textbook source, `cos` (Cosmopedia v2) beats `str` on blimp
(+1.4) and lambada (+1.4) and ties the rest — the current best mix. `wt` (tiny-webtext)
added nothing (4-way ≈ 5-way). Cross-mix comparisons are still confounded by each run's
retrained tokenizer — to be standardized under a pinned vocab. In-training curve (cos
run): blimp/arc/piqa plateau by ~4.5k while lambada/sciq + val_bpb are still rising at
5k, so a modest tail remains past 5k for those two.

qa-mix (2026-07-19): adds QA-FORMAT sources to the cos mix — `gq` (closed-book
`Question:/Answer:` pairs, 5%) and `sq` (passage + its Q/A pairs as one doc, 1% ≈ the
full corpus) — plus the always-on `doc` source (`projects/tinylm/documents/`, ~0.95M
tokens = 200 copies, excluded from the mixture tokenizer). Verdict: sciq +12.9 over
`cos` (67.40, above the gpt2 ref) — sciq is question-formatted, so format practice
transfers; lambada + piqa best-in-table; blimp −0.6 (noise-adjacent). Probes: both
documents fully memorized (~109 exposures each, doc ppl 1.02; `DEMO` prompt reproduces
either file verbatim); `Question:/Answer:` elicits direct answers — grounded/extractive
answers are often correct ("What color is Tom's ball?" → "red"), closed-book answers
are fluent but circular (capacity, not format). Run log:
`/mnt/ai/runs/tinylm/pretrain/train_qa-mix_2026-07-19.log`; prior checkpoint preserved
as `chimera_gpt6m_pre-qa-mix.pt`.

sc30 (2026-07-19): qa-mix + Gemma-2-style final-logit soft-capping (30) in training +
eval (CCE `softcap` + capped forward). Verdict: wash — every benchmark within noise,
val_loss −0.03, bpb +0.05. Inference-only capping on the uncapped model strictly HURTS
(monotonic with tighter caps; raw logits reach ~40 so cap 30 saturates); it's a
training-time stabilizer, not a quality lever at 6M with QK-norm already present.
Kept on (harmless); `LOGIT_SOFTCAP=None` disables.

curric (2026-07-19): sc30 + warmup(250)+cosine LR (→0.1x) + two-phase data curriculum —
same per-source totals, reordered so phase 2 (steps 2500-5000) is cosmopedia-dominant
(cos 20%→40%, fw 38→30, ts 36→24). First clearly positive intervention since qa-mix:
best-in-table lambada/sciq/arc + best val_loss 3.154 / bpb 0.844; sciq 68.1 is +3.7
over gpt2. Anneal did NOT erase early/rare data: documents still ppl 1.02 (verbatim
recall), QA format intact. Confounded pair (schedule + curriculum changed together) —
isolate before crediting either alone. Trained-doc coverage audit (sciq/arc wrong
answers vs training text): 96% of missed sciq facts WERE in training (cosmopedia
provides ~90% of coverage) — misses are capacity, not data absence.

vwn+mhc (2026-07-20): curric config + Virtual Width Networks (residual state at 1.5×
virtual width, `VWN_M=2 VWN_N=3`; attn/MLP stay at dim 384) with mHC-Lite carry routing
(`carry_mode="mhc_lite"`, the model default — the square n×n persistent carry map is a
convex combination over a fixed permutation basis; read/write maps stay the rectangular
dynamic GHC maps). Best model to date: best-in-table blimp 71.01 (+1.2 over curric),
lambada 18.36 (+0.9), arc 35.35, and best val_bpb 0.830 (curric 0.844); piqa/sciq tie
within noise. BPB descent is clean and monotone (1.024→0.916→0.857→0.836→0.830, no
forgetting). So the ~1.5× effective width buys real grammar/long-range gains even at 6M.
Not yet isolated: plain VWN (`carry_mode="ghc"`) has NOT been run as a baseline, so
mHC-Lite's contribution vs plain VWN is unmeasured — this row credits the pair. Cost:
the virtual-width machinery (VWN + routing) is a flat +16% step time (profiled 141→168
ms/step single-microbatch, VWN(2,3) vs no-VWN(1,1), both on the mHC-Lite model). Kept —
the quality pays for it. Optimization pass (see below) confirmed the step is otherwise
compute-bound: CCE is 31% of the step and mandatory (plain materialized CE OOMs on the
16GB card; CCE peaks 6.4GB), and `.item()` sync / CUDA graphs / block-mask caching are
all ≤3% dead ends. Eval cadence trimmed this run (val 500→1000, bench 1500→2500) to cut
the ~13% eval overhead. Run log:
`/mnt/ai/runs/tinylm/pretrain/train_vwn_2026-07-20_0008.log`.

## TODO

- **Expand the continued-pretraining sources for the assistant target.** Add filtered
  Dolma Stack Exchange for natural question/explanation structure and a
  Wikipedia/Wikibooks slice for clean expository and long-document text. During late
  pretraining, introduce only 1–3% high-rated English OASST1 or filtered UltraChat
  rendered as full-token ChatML; SFT remains responsible for assistant-only behavior.
  Mix by tokens, audit overlap with evaluation data, and retain per-source validation
  metrics. See the project-level [training roadmap](../README.md#training-roadmap).
- **Add an 8192-token context-extension stage.** Length-bucket long FineWeb/reference
  documents and complete grounded conversations, train with batches of roughly 8–16,
  and retain short-context examples to measure and prevent capability regression.
- **Broaden the zero-shot benchmark suite.** Add `hellaswag` (`acc_norm`) for
  contextual continuation/commonsense, `boolq` for passage comprehension, and
  `winogrande` for coreference/commonsense. These are all loglikelihood-ranking tasks
  and fit the existing `ChimeraLM` adapter. Use a fixed deterministic HellaSwag subset
  for cheap in-training curves and the full validation set for final checkpoints.
- **Add prompt-robustness diagnostics.** Score the same SciQ, ARC-Easy, and BoolQ
  examples under 2–3 semantically equivalent prompt templates; report mean accuracy
  and the range across templates. The qa-mix SciQ jump shows that the current headline
  score partly measures exposure to `Question:/Answer:` formatting, so this should
  separate format transfer from knowledge/reasoning gains.
- **Add external held-out LM evaluation.** Implement `ChimeraLM.loglikelihood_rolling`
  and periodically report WikiText-2 bits per byte (plus its standard perplexity
  metrics) on final checkpoints. Keep the existing fixed Tiny Textbooks BPB for fast
  in-training comparisons.
- **Defer poorly calibrated additions for now.** MMLU and ARC-Challenge will likely be
  dominated by chance noise at 6M parameters; OpenBookQA overlaps SciQ/ARC-Easy; GSM8K
  needs generation support; and TruthfulQA is a poor fit for the model size and
  512-token context.
- **Standardize the tokenizer before trusting cross-mix rankings (circle back).**
  Every Results row retrained its own 16k vocab on that run's mixture, so mix-to-mix
  deltas — including the `cos`-vs-`str` blimp/lambada gap that currently crowns `cos` —
  are confounded by tokenizer differences, not just the data. Fix: pin ONE shared vocab
  (train it once on a fixed broad sample; add a `tokenizer_path=` to
  `ConcatTextDataModule` to load-and-share instead of retraining per source-set), then
  re-run the key mixes under it to confirm the rankings hold. Until then, treat the
  table's cross-mix comparisons as provisional.

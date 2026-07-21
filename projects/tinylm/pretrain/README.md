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

| run       | steps | mix                               | blimp     | lambada   | piqa      | sciq      | arc_easy  |
| -------   | ----- | --------------------------------- | --------- | --------- | --------- | --------- | --------- |
| tok16k_4b | 61k   | 4BT true-mix (cos42 fw43 ts11 gq3 sq1), 16k vocab, plain | **71.29** | **19.62** | **58.32** | **69.10** | **37.71** |
| vwn+mhc   | 5k    | curric + VWN(2,3) 1.5× + mHC-Lite | 71.01     | 18.36     | 56.31     | 67.80     | 35.35     |
| curric    | 5k    | qa-mix, sc30, cosine LR, cos20→40 | 69.86     | 17.47     | 56.47     | 68.10     | 35.23     |
| sc30      | 5k    | qa-mix + logit softcap 30         | 69.18     | 16.86     | 55.93     | 67.70     | 34.93     |
| qa-mix    | 5k    | cos30 fw34 ts30 gq5 sq1 +doc      | 69.53     | 17.27     | 57.24     | 67.40     | 34.76     |
| cos       | 5k    | cos30 fw40 ts30                   | 70.09     | 16.94     | 55.44     | 54.50     | 33.96     |
| 3-way     | 5k    | str30 fw40 ts30                   | 68.66     | 15.54     | 56.37     | 54.80     | 34.55     |
| 4-way     | 5k    | tt30 str30 fw25 ts15              | 67.63     | 16.01     | 57.29     | 55.80     | 34.34     |
| 5-way     | 5k    | tt30 str25 fw20 ts15 wt10         | 67.94     | 16.11     | 56.42     | 55.30     | 34.89     |
| tt+ts     | 5k    | tt50 ts50                         | 65.03     | 12.59     | 56.53     | 54.70     | 31.99     |
| tt        | 5k    | tt100                             | 63.72     | 6.95      | 56.96     | 55.10     | 33.63     |
| ts        | 5k    | ts100                             | 62.93     | 10.87     | 52.34     | 27.40     | 26.94     |
| chance    | —     | —                                 | 50.0      | 0.0       | 50.0      | 25.0      | 25.0      |
| gpt2      | —     | — (124M ref)                      | 82.29     | 32.16     | 62.62     | 64.40     | 39.52     |

5-way stderr: blimp 0.16, lambada 0.51, piqa 1.16, sciq 1.57, arc_easy 0.98.

gpt2 val_bpb reference: **0.5932**, scored by `bpb_gpt2.py` on the exact same fixed
held-out (tiny-textbooks test, 500 docs) and byte-normalized formula as every run's
`val_bpb` above — tokenizer-agnostic, so it's directly comparable across gpt2's BPE and
our own vocabs despite the different token counts. Script is self-contained (repo-tracked
held-out text at `eval_data/bpb_heldout.txt`, CPU-only) — runnable on any machine.

tok16k_4b (2026-07-21): the first full 4BT base run and the new best model — a
**plain baseline** (no VWN, no mHC-Lite, no looping) on the pinned 16k vocab, 4.0B
tokens over 61,035 steps at seq-512 with the true-4B mix (TinyStories capped at its
~0.44B ceiling; cosmopedia/fineweb absorb the rest, ~single-pass). Best-in-table on
every task and **best val_bpb 0.757** (vwn+mhc 0.830, 1BT-16k 0.781) — clean monotone
descent, no forgetting. So 4B tokens + a pinned vocab beat the 1BT architecture tricks
outright, and without VWN's +16% step cost. Caveat: this row crosses regimes (61k steps
+ pinned vocab), so it is NOT a like-for-like entry in the 5k mix-ablation cohort below
— read the bold as "best model", not "best mix at 5k". This checkpoint
(`chimera_gpt6m_tok16k_4b.pt`) is the 512 base the context-extension stages resume from.

Notes: the 5k rows are step-matched (tok16k_4b excepted, see above). The knowledge/reasoning tasks (piqa/sciq/arc)
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

## 512-token training plan

The final 512-token base should preserve the language mixture that already works while
adding a bounded technical slice. Sample every share by tokens, not document count:

| source family                 | token share |
| ----------------------------- | ----------: |
| FineWeb-Edu                   |         30% |
| Cosmopedia v2                 |         25% |
| TinyStories v2                |         20% |
| GooAQ + SQuAD                 |          5% |
| FineMath-4+                   |          6% |
| Tiny Math Textbooks           |          2% |
| Python-Edu                    |          6% |
| filtered The Stack Smol       |          2% |
| filtered Tiny Codes           |          1% |
| validated JSON/XML/tool forms |          3% |

This gives 80% general language and QA, 8% math, and 12% code or structured data.
Within QA, start with GooAQ 4% and SQuAD 1%. FineMath remains the primary math source;
Tiny Math Textbooks is a small, manually audited synthetic supplement. Python-Edu is
the primary code source. Filter The Stack Smol to useful assistant-facing formats such
as Python, Shell, SQL, JavaScript, TypeScript, HTML, Markdown, and Dockerfile rather
than sampling its 30 languages uniformly. Tiny Codes is also synthetic and
template-prone, so retain only parseable, deduplicated Python, Bash, SQL, JavaScript,
and TypeScript examples.

The structured-data slice should contain valid JSON/JSONL, JSON Schema paired with
instances, function signatures paired with argument objects, API examples, and a
smaller amount of XML, HTML, and YAML. Validate every generated or transformed record
with a real parser. Raw code and serialization formats only teach syntax; they do not
teach when to call a tool. Keep NL2Bash and request → tool call → result → grounded
response trajectories for SFT/RLVR, rendered in one canonical tool-call grammar.

Train the initial 4B-token candidate as a curriculum:

| phase | token budget | approximate steps | general/QA | math | code/data | LR policy       |
| ----- | -----------: | ----------------: | ---------: | ---: | --------: | --------------- |
| 1     |         2.0B |            30,518 |        88% |   4% |        8% | warmup + stable |
| 2     |         1.6B |            24,414 |        72% |  12% |       16% | stable          |
| 3     |         0.4B |             6,104 |        72% |  12% |       16% | decay           |

These step counts assume the current effective batch of `128 × 512 = 65,536` tokens.
Across all three phases, the realized mixture is approximately the 80/8/12 target
above. Save and fully evaluate checkpoints at 0.5B, 1B, 2B, and 4B tokens. Preserve a
pre-decay checkpoint at 3.6B so training can continue without reversing a completed
cooldown.

Treat 10B tokens (approximately 152,588 steps) as a maximum extension budget, not the
default run. Continue beyond 4B only if held-out BPB and at least one real capability
metric improve outside noise. Stop when the latest doubling produces less than roughly
1–2% relative BPB improvement and no convincing gain in grammar, grounded QA, math,
code parsing, or arbitrary-prompt probes. If the run is extended, branch from the
pre-decay checkpoint, preserve the overall 80/8/12 mixture, and reserve roughly the
final 10% of the eventual run for learning-rate decay.

Train the pinned tokenizer on the same domain proportions, including the math, code,
and structured-data slices. Because math is now an explicit target, compare the
current `split_digits=False` tokenizer with a digit-splitting candidate rather than
assuming the existing choice still wins. Report held-out BPB separately for prose,
math, Python, JSON, and XML, and add deterministic `ast.parse`, JSON/XML parse-rate,
elementary arithmetic exact-match, and later tool-schema-validity diagnostics.

## Context expansion route

Expand the trained context progressively from the broad 512-token base through 2,048,
4,096, and 8,192 tokens. Each phase mixes broad short examples for capability retention
with genuinely long, coherent documents for long-distance attention training:

| phase context |   broad short data |    long coherent data |
| ------------- | -----------------: | --------------------: |
| 2,048         | 30–40% below 1,024 | 60–70% at 1,024–2,048 |
| 4,096         | 25–30% below 2,048 | 70–75% at 2,048–4,096 |
| 8,192         | 20–30% below 4,096 | 70–80% at 4,096–8,192 |

Sample these shares by tokens, not document count. Broad short data retains the normal
FineWeb, Cosmopedia, stories, QA, and conversational registers. Long data must contain
mutually visible tokens from coherent FineWeb pages, Wikipedia sections, books, Stack
Exchange threads, grounded passages, or complete conversations. Packing unrelated
short documents into an 8k tensor does not train 8k dependencies because document
masking resets attention and positions at every EOS boundary.

Do not discard source documents longer than 8,192 tokens. Treat them as reservoirs of
contiguous windows: sample randomized or overlapping offsets across epochs, cap windows
per source document so a few books cannot dominate, and avoid always selecting the
beginning. A window taken from the middle of a document does not receive a false BOS;
include EOS only when it contains the real document ending.

Positions remain window- and document-relative. Every standalone 8k window uses
`pos_ids=0..8191`, even when its source offset was much larger. Pure RoPE depends on
relative displacement, inference requests also start at zero, and the current
`build_block_mask_and_pos` already derives packed-document positions from EOS markers.
Do not store `(token_id, pos_id)` pairs in the token stream. Store token IDs plus
document boundaries—or compact `(document_id, start_offset, length)` metadata—and
derive position IDs when batching. Absolute source-document offsets would only be
needed for a future architecture whose inference positions continue beyond 8,192.

The fixed, non-overlapping flat-stream `TokenDataset` is sufficient for the current
base runs but not the final context curriculum. Add a document-aware window dataset
that resamples contiguous offsets each epoch and returns `context_length + 1` tokens
for shifted next-token targets. Maintain approximately the same effective token batch
through physical batches of 8–16 plus gradient accumulation.

Validate every phase separately by length band—for example `val/bpb_0_512`,
`val/bpb_512_2k`, `val/bpb_2k_4k`, and `val/bpb_4k_8k`—and retain the short benchmark
suite to detect regression before advancing to the next context length.

### Implementation

The machinery is built (data layer + train wiring; no model change — RoPE is computed
on the fly, so the same 512 weights forward at any length):

- `chimera.data.WindowSampledDataset` — random single-document contiguous windows from
  the flat stream (boundaries recovered from inline EOS). Windows are window-relative in
  position (`build_block_mask_and_pos` gives `0..N-1` for a single-doc window), carry no
  false BOS mid-document, and cap windows-per-document so a few long docs can't dominate.
  This is the only data that trains long-range attention; packed short docs don't (doc
  masking resets at every EOS).
- `chimera.data.ContextMixDataModule` — blends a broad **short** pool (packed
  FineWeb/Cosmopedia/stories/QA) with a **long** pool (Wikipedia + Stack Exchange + a
  long-FineWeb slice, window-sampled) by token share via a `WeightedRandomSampler`. Both
  pools pin the **same frozen tokenizer**.
- `bpb_banded.py` — two long-range capability signals (the short benchmark suite only
  guards against short-context regression; it can't measure long-context *gain*):
  - **length-banded BPB** (`val/bpb_512/2k/4k/8k`) scoring the same long held-out at
    increasing context widths (reuses `bpb_gpt2.score`'s double-counting-safe rolling
    window); shows whether widening the context actually lowers bpb.
  - **retrieval probe** (`probe/recall_128…8k`, `retrieval_probe`) — bigram induction:
    plant a random `(A, B)` pair, gap it by `d` tokens, re-present `A`, and measure how
    often the top prediction is `B`. Chance ≈ 1/vocab, content-agnostic, pure forward
    pass — a recall curve holding out to distance `d` localizes attention reach that
    banded BPB alone can't. Advance a stage only if the capability signal beats the
    previous width outside noise; if it flatlines, the 6M model is context-saturated —
    stop rather than pay the next (4×-attention) stage.

Each stage is one `train.py` run resuming from the prior checkpoint (`TINYLM_INIT_CKPT`),
selected by `TINYLM_CTX_STAGE` (`2k`/`4k`/`8k`). `TINYLM_SEQ_LEN` sets the context; keep
tokens/step ≈ 65k by dropping `TINYLM_PHYS_BATCH` and raising `TINYLM_GRAD_ACCUM`
(2k→16×2, 4k→8×2, 8k→8×1). Short/long shares default to 35/65, 27/73, 25/75.

```sh
# 2k stage, resuming from the 512 base; 4k then resumes from ctx2k, 8k from ctx4k
TINYLM_CTX_STAGE=2k TINYLM_SEQ_LEN=2048 TINYLM_PHYS_BATCH=16 TINYLM_GRAD_ACCUM=2 \
  TINYLM_INIT_CKPT=/mnt/ai/runs/tinylm/pretrain/chimera_gpt6m.pt \
  TINYLM_TOKENIZER_PATH=../data/tokenizers/16k/tokenizer.json \
  TINYLM_MAX_STEPS=... TINYLM_VAL_EVERY_MT=100 TINYLM_BENCH_EVERY_MT=500 \
  TINYLM_RUN_TAG=ctx2k uv run python train.py
```

Correctness is covered by `tests/test_window_dataset.py` (CPU: boundary recovery,
single-doc windows, window-relative positions, per-epoch resampling, mixing shares).

### Results (2026-07-21)

Three stages, each resuming the prior checkpoint (2k from the 4BT `tok16k_4b` base; 8k
was jumped directly from 2k, 4k filled in after) for **500MT** at a low continuation LR
(muon 0.004) with the two-pool short/long mix. The gate metric is **banded BPB** — the
same long held-out (Wikipedia) scored at increasing context widths; a stage trained only
to length L collapses when scored beyond L, and training at L repairs it. Retrieval probe
(bigram induction) stayed at chance for every stage — uninformative at 6M, ignore it.

Banded BPB by scoring context width (lower = better; **bold** = best per width):

| model | 512 | 2k | 4k | 8k | short benches (blimp/lam/piqa/sciq/arc) |
| --- | ---: | ---: | ---: | ---: | --- |
| tok16k_4b (512 base) | 1.388 | 2.502 | — | — | 71.3 / 19.6 / 58.3 / 69.1 / 37.7 |
| ctx2k | **1.297** | **1.292** | 1.611 | 2.812 | 72.1 / 20.9 / 57.7 / 73.2 / 38.1 |
| ctx4k | 1.310 | 1.301 | **1.364** | 2.041 | 71.7 / 19.7 / 57.4 / 72.1 / 37.7 |
| ctx8k | 1.331 | 1.329 | 1.391 | **1.586** | 71.6 / 18.9 / 56.4 / 70.7 / 37.0 |

Verdict: the extension pipeline works — each stage repairs the collapse at its own length
(base is catastrophic past 512: 2.50 @2k; ctx2k fixes ≤2k but cliffs at 4k/8k; ctx4k
fixes 4k and halves the 8k gap 2.81→2.04; ctx8k flattens 8k to 1.59) and the cliff moves
one rung right per stage. **But no genuine long-range gain**: within every stage, bpb
still *rises* with context (e.g. ctx8k 1.33→1.59) — the model learns to *tolerate* long
context, not *exploit* it. Cost: short-context erodes monotonically (512-band 1.297→1.331;
benches peak at ctx2k then decline), so each doubling trades a little short-context ability
for "doesn't break at length L". Consistent with the 6M capacity ceiling seen throughout —
real long-range use is a model-scale lever, not a data one. Long pool is also thin at 8k
(~9k windows / ~75M unique long tokens; Stack Exchange has no ≥8k docs), so 8k long windows
repeat ~5×; enrich the long pool before any serious 8k run. Run logs:
`/mnt/ai/runs/tinylm/pretrain/run_ctx{2k,4k,8k}_2026-07-21_*.log`; checkpoints
`chimera_gpt6m_ctx{2k,4k,8k}.pt`.

## TODO

- **Expand the continued-pretraining sources for the assistant target.** Add filtered
  Dolma Stack Exchange for natural question/explanation structure and a
  Wikipedia/Wikibooks slice for clean expository and long-document text. During late
  pretraining, introduce only 1–3% high-rated English OASST1 or filtered UltraChat
  rendered as full-token ChatML; SFT remains responsible for assistant-only behavior.
  Mix by tokens, audit overlap with evaluation data, and retain per-source validation
  metrics. See the project-level [training roadmap](../README.md#training-roadmap).
- **Implement the context-expansion route.** Add the document-aware randomized window
  dataset, length-bucket validation, and the progressive 2k → 4k → 8k stages described
  in [Context expansion route](#context-expansion-route).
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
- **Standardize the tokenizer before trusting cross-mix rankings.** Follow the
  project-level [tokenizer plan](../README.md#tokenizer-plan), pin the selected artifact
  through `ConcatTextDataModule.tokenizer_path`, and rerun the key mixtures. Until
  then, every cross-mixture result remains confounded by its independently trained
  vocabulary and should be treated as provisional.

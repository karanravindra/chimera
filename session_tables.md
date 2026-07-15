# Tables & Comparisons — full session log

_All 35 markdown tables/comparisons produced across the conversation, in chronological order. Extracted from the session transcript by a worker fork._

---

## 1. Undertrained-checkpoint benchmark scores (starting point)

> ## The evidence it's undertrained

| task | metric | value | chance |
|---|---|---|---|
| lambada_openai | ppl | **525,851** | — |
| lambada_openai | acc | 0.02 | 0.0 |
| arc_challenge | acc_norm | 22.6 | 25.0 |
| hellaswag | acc | 25.7 | 25.0 |
| piqa | acc | 53.2 | 50.0 |
| winogrande | acc | 51.5 | 50.0 |
| sciq | acc | 26.2 | 25.0 |

## 2. Which benchmarks can move at 34–200M scale (capability vs loss ceiling)

> This is the most important finding for you — **it's a model-scale ceiling, not a loss you can train past**:

| Task | Chance | Verdict at 34–200M | Anchor (well-trained ~130M) |
|---|---|---|---|
| **LAMBADA** (acc) | ~0% | ✅ **Earliest mover — your best signal** | ppl 16–38 → acc 33–44% |
| **PIQA** | 50% | ✅ moves (acc_norm) | ~64% |
| **HellaSwag** | 25% | ✅ moves (acc_norm), modestly | ~35% |
| **ARC-Easy** | 25% | ✅ moves | ~48% |
| **SciQ** | 25% | ✅ moves | high |
| **ARC-Challenge** | 25% | ❌ **stuck at ~25% for ALL sub-1B models** | Mamba-790M only 29.5% |
| **WinoGrande** | 50% | ❌ **stays ~50% at this scale** | flat |

## 3. BPB↔val-loss conversion, corrected bytes/token (3.49 not 4.5)

> The assumed 4.5 was off by 22%, in the direction that makes things *worse*. Your `mix_1B` is code-heavy (28% Python + 6% bash + 6% JSON + 20% math), and code tokenizes into many short tokens (whitespace, punctuation, symbols); combined with the 64k vocab, you only get 3.49 UTF-8 bytes per token. That flows straight into every conversion:

| Quantity | With assumed 4.5 B/tok | **Measured 3.49 B/tok** |
|---|---|---|
| **Current** val/loss 4.408 nats → BPB | 1.41 | **1.82** |
| BPB 0.9 → val/loss | 2.81 nats | **2.18 nats** |
| BPB 1.0 → val/loss | 3.12 nats | **2.42 nats** |
| BPB 1.1 → val/loss | 3.43 nats | **2.66 nats** |

## 4. Data-source audit: bytes/token, Haiku's verdict vs actual (decoded)

> The raw table is trustworthy, but Haiku's core frame — *"low bytes/token = bad tokenizer fit = drop it"* — is a misread. **Low bytes/token is intrinsic to code, math, and JSON** (they're dense in short tokens: operators, indentation, digits, braces). It measures content type, not data quality or a broken tokenizer. I spot-checked every source it flagged:

| Source | Wt | B/tok | Haiku said | **Actual (verified by decoding samples)** |
|---|---|---|---|---|
| **tool-toucan** | 0.10 | 2.88 | 🚨 corrupted, DROP | ✅ **Confirmed problem** — see below |
| tool-apigen-mt | 0.01 | 3.34 | DROP (33.8% top8) | ❌ **Clean** — legit multi-turn agentic tool-use; high top8share is just JSON+chat structure |
| openmath-cot | 0.10 | 2.84 | downweight (low diversity) | ❌ **Clean & high-quality** — genuine math chain-of-thought; low vocab/bytes is inherent to math reasoning |
| stackv2-bash | 0.06 | 2.65 | downweight | ⚠️ Fine; low B/tok is inherent to shell syntax (one sampled doc was repetitive, but that's per-doc) |

## 5. Escaped-unicode contamination per source (tool-toucan bug)

> The JSON string values were never unescaped — unicode appears as literal `\u65f6` (6 ASCII chars) instead of the actual character (时), and newlines as literal `\n`. It's Chinese web-article metadata stored as escaped JSON. Prevalence across 20 strided windows per source:

| Source | Windows w/ escaped unicode | `\u` per 1k chars |
|---|---|---|
| **tool-toucan** | **11/20** | **9.05** |
| tool-apigen-mt | 0/20 | 0.00 |
| tool-hermes | 1/20 | 0.01 |
| tool-toolace | 1/20 | 0.00 |

## 6. Benchmark-bug investigation: what's true / retracted

> ## What's actually true now

| Claim | Status |
|---|---|
| LAMBADA perplexity metric is computed correctly | ✅ confirmed (reproduced on your checkpoint) |
| 525k is a real property of a 500-step 34M model | ✅ genuine |
| "Worse than uniform → harness bug" | ❌ false alarm (multi-token vs single-token comparison) |
| BOS bug explains the bad scores | ❌ retracted — <0.4% effect |
| `bench.py:134` BOS/`add_special_tokens` mismatch exists | ✅ real, but **cosmetic** — worth fixing for cleanliness, not urgent |

## 7. Eval/benchmark phase timing breakdown

> Short — well under a minute total. From the run log (`bvs24uftt`):

| Phase | Time |
|---|---|
| `trainer.test()` (76 val batches) | ~6s |
| lm-eval: tokenizing 69,719 inputs | ~18s |
| lm-eval: loglikelihood (all 7 tasks) | ~18s |
| lm-eval: perplexity bootstrap (100 iters) | ~9s |
| **Total eval/benchmark phase** | **~50s** |

## 8. tool-toucan unicode fix results (before/after)

> The toucan fix worked cleanly:

| metric | old | **new** | change |
|---|---|---|---|
| escaped `\u` per 1k chars | 9.05 | **0.00** | eliminated |
| bytes/token | 2.882 | **3.128** | +8.5% (real chars now, not `\uXXXX`) |
| docs in same 1.7B budget | 233,472 | **258,560** | +25k docs (free diversity) |

## 9. Data-scaling law fit L(D)=E+A·D^-α (1BT baseline curve)

> Fitting `L(D) = E + A·D^(−α)`:

| fit | R² | α | **E (∞-token floor)** | **end-of-run (~981M)** |
|---|---|---|---|---|
| all 18 pts | 0.991 | 0.70 | 2.42 nats (BPB 0.998) | 2.56 nats (BPB 1.06) |
| drop 2 warmup pts | 0.99982 | 0.15 | **0.88 nats (BPB 0.36)** | 2.46 nats (BPB 1.02) |

## 10. Metric glossary: val/loss vs bpt vs bpb

> Right — two different metrics, confusingly-named:

| what you see | value | what it is |
|---|---|---|
| `val/loss` | 2.582 nats | cross-entropy per token (natural log) |
| **`val/bpt`** (dashboard) | **3.72** | bits per **token** = `loss / ln2` |
| **BPB** (what I've been quoting) | **1.07** | bits per **byte** = `bpt / 3.49 bytes-per-token` |

## 11. SFT time estimates by epoch count (sft_1B, 905M sup tokens)

> Got it — `sft_1B` has **905M supervised tokens**. Same 34M model and throughput (~3.86 it/s at 2048 ctx, 65536 tokens/step), so:

| SFT scope | steps | **est. time** |
|---|---|---|
| smoke (default `--max-steps 500`) | 500 | ~2 min |
| **1 full epoch** (905M tok, `--max-steps -1`) | ~13,800 | **~60 min** |
| 2 epochs | ~27,600 | ~2 hr |
| 3 epochs | ~41,400 | ~3 hr |

## 12. 1BT baseline results vs 500-step smoke (predictions hit)

> The baseline finished with **excellent** results — and the eval used the pretokenize cache (`loaded pretokenized eval inputs`). The predictions held up remarkably well:

| task | 500-step smoke | **1BT baseline** | predicted? |
|---|---|---|---|
| val/loss · bpt · **bpb** | 4.41 · 6.36 · 1.82 | **2.56 · 3.69 · 1.06** | ✅ ~BPB 1.0 target hit |
| lambada ppl | 525,851 | **6,184** (↓85×) | ✅ collapsed |
| lambada acc | 0.02% | **7.55%** | ✅ off zero |
| sciq acc | 26.2 | **54.3** | ✅ moved |
| arc_easy acc | 28.2 | **36.95** | ✅ moved |
| piqa acc | 53.2 | **57.45** | ✅ moved |
| hellaswag acc | 25.7 | 26.3 | weak (small model) |
| arc_challenge / winogrande | chance | **still chance** | ✅ scale-gated, as predicted |

## 13. CCE_AUTOTUNE crash repro (all-ignored batch)

> Fully confirmed the mechanism:

| CCE_AUTOTUNE | batch | result |
|---|---|---|
| **1** (on) | normal (B>0) | ✅ OK |
| **1** (on) | all-ignored (B=0 supervised) | ❌ **crash** |
| **0** (off) | all-ignored | ✅ OK (fixed config) |

## 14. SFT mix weights (openmath/tool-heavy, pre-UltraChat)

> Compare to the pretrain curve (4.0 → 2.55, dramatic). The SFT curve is essentially flat because **every SFT source was already in the pretrain mix**:

| source | pretrain `mix_1B` | SFT `sft_1B` |
|---|---|---|
| openmath-cot | 0.10 | **0.40** |
| tool-toucan | 0.10 | **0.40** |
| tool-xlam/pythonic/toolace/apigen/hermes | 0.01 each | 0.04 each |

## 15. sft_1B mix — no chat data (the gap)

> Short answer: **the `sft_1B` mix has no UltraChat — or any general-chat data at all.** It's math + tool-calling only. Here's the actual composition (`mix_sft/sft_1B/manifest.json`):

| source | weight |
|---|---|
| openmath-cot | 0.40 |
| tool-toucan | 0.40 |
| tool-xlam / pythonic / toolace / apigen-mt / hermes | 0.04 each |

## 16. Proposed chat-dominant SFT mix (with UltraChat)

> Chat-dominant it is:

| source | weight |
|---|---|
| UltraChat | 0.50 |
| openmath-cot | 0.20 |
| tool-toucan | 0.15 |
| tool-xlam / pythonic / toolace / apigen-mt / hermes | 0.03 each |

## 17. ctx-512 run design vs 1BT baseline (knob diff)

> **The run:** identical to the 1BT baseline (`8xnbxqs8`) in every way except sequence length.

| knob | baseline `8xnbxqs8` | proposed ctx-512 run |
|---|---|---|
| seq_len | 2048 | **512** |
| global tokens/step | 65536 | 65536 (batch 32→**128**) |
| token budget | ~1BT | ~1BT (same ~15.3k steps) |
| mix | mix_1B | mix_1B (unchanged) |
| everything else | — | unchanged |

## 18. ctx-512 vs baseline loss-noise / stability

> It's genuinely noisy, but the data says it's **batch-composition noise, not training instability** — and there's a clean way to prove it:

| | train/loss 2nd-half std | step-to-step std | **val/loss std** (fixed set) |
|---|---|---|---|
| baseline (2048) | 0.222 | 0.311 | **0.025** |
| ctx-512 | 0.139 | 0.199 | **0.045** |

## 19. Throughput: ctx512 vs 2048 baseline (+14%)

> **Throughput — ctx512 wins clearly:**

| metric | baseline (2048) | ctx512 | 
|---|---|---|
| tokens/sec | 240.9k | **275.3k** (+14%) |
| steps/sec | 3.68 | **4.20** |
| wall-clock for 1BT | 67.8 min | **~59 min** |

## 20. Val loss at matched tokens — ctx512 vs 2048 (confounded)

> **Val loss at matched tokens (raw — confounded, read with care):**

| tokens | baseline loss | ctx512 loss | Δ |
|---|---|---|---|
| 65M | 3.290 | **3.175** | −0.115 |
| 262M | **2.814** | 2.824 | +0.010 |
| 524M | **2.618** | 2.670 | +0.053 |
| 688M | **2.567** | 2.623 | +0.056 |

## 21. Benchmark results at ~980M: ctx512 vs 2048

> This is a decisive result. **Both runs hit ~980M tokens (equal budget), and ctx512 did it in 14% less wall-clock — yet ctx512 matches or beats the 2048 baseline on essentially every benchmark that moves:**

| task (metric) | baseline 2048 | ctx512 |
|---|---|---|
| lambada ppl ↓ | 6184 | **3172** |
| lambada acc ↑ | 7.55 | **10.38** |
| sciq acc ↑ | 54.30 | **60.30** |
| arc_easy acc ↑ | 36.95 | **39.44** |
| arc_easy acc_norm ↑ | 34.60 | **36.11** |
| piqa acc ↑ | **57.45** | 56.26 |
| piqa acc_norm ↑ | **55.11** | 54.73 |
| hellaswag acc_norm ↑ | 26.15 | **26.79** |
| winogrande acc ↑ | 49.96 | **50.75** |
| arc_challenge acc_norm ↑ | **22.27** | 21.08 |

## 22. Matched-context eval: ctx512 vs 2048 model (bpb)

> Feeding **both models the exact same 512-token windows** (spread across all of val — I caught and fixed a coverage bug where val.bin is source-ordered, so my first pass only saw a prefix):

| context given | ctx2048 model (bpb) | ctx512 model (bpb) |
|---|---|---|
| matched 512 tokens | 1.132 | **1.077** |
| each model's *best* usable context (same data) | 1.025 @2048 ctx | **1.008 @512 ctx** |

## 23. muP LR Bayesian sweep results (top ranks)

> All 10 have a val/loss, but run 9 (`muon0.0088/adamw0.0083`) is the SIGTERM'd one — its 3.7565 is a step-1500 value, not a finished 3000-step result (note its `test` is blank). So **9 clean runs**. Here's the ranking:

| rank | muon_lr | adamw_lr | val/loss | val/bpb |
|---|---|---|---|---|
| **1** | **0.01329** | **0.00603** | **3.2135** | **1.3279** |
| 2 | 0.00813 | 0.00415 | 3.2231 | 1.3319 |
| 3 | 0.01033 | 0.01301 | 3.2370 | 1.3376 |
| 4 | 0.01860 | 0.00270 | 3.2831 | 1.3566 |
| 5 | 0.00999 | 0.00352 | 3.3074 | 1.3667 |
| 6 | 0.00477 | 0.00580 | 3.3194 | 1.3716 |
| 7 | 0.01717 | 0.00212 | 3.3380 | 1.3793 |
| 8 | 0.02882 | 0.00280 | 3.3759 | 1.3950 |
| 10 | 0.02053 | 0.01993 | 3.8727 | 1.6003 |
| ~~9~~ | ~~0.00877~~ | ~~0.00833~~ | ~~3.7565~~ | (interrupted) |

## 24. Width×depth coord-check val/loss (transfer verification)

> All 8 configs trained cleanly at the **fixed** swept LR (muon 0.013 / adamw 0.006). val/loss @ 65M tokens:

| width | depth 6 | depth 12 | depth gain |
|---|---|---|---|
| 64 | 3.7753 | 3.7552 | −0.020 |
| 128 | 3.5811 | 3.4426 | −0.138 |
| 256 | 3.3455 | 3.2786 | −0.067 |
| 384 | 3.1830 | **3.0962** | −0.087 |

## 25. Loss-vs-wall Pareto (coord-check subset)

> At the fixed 65M-token budget, wall times ranged 2:40 → 7:19. The Pareto front (loss vs wall) and iso-time interpolation change the pick:

| arch | val/loss | wall | note |
|---|---|---|---|
| 384-12-1-**12** | **3.096** | 7:19 | best loss, slowest |
| 384-12-1-**6** | 3.183 | **5:27** | **wall sweet spot** |
| 256-8-1-12 | 3.279 | 5:33 | **dominated** ✗ |
| 256-8-1-6 | 3.346 | 4:25 | |

## 26. Param counts + FLOPs per config (iso-FLOP check)

> Here are the param counts — and the answer is a firm **no, not iso-FLOP**:

| arch | params | FLOPs (6·N·D) @ 65.5M tok |
|---|---|---|
| 64-2-1-6 | 4.39M | 1.7 PFLOP |
| 64-2-1-12 | 4.67M | 1.8 PFLOP |
| 128-4-1-6 | 9.28M | 3.6 PFLOP |
| 128-4-1-12 | 10.31M | 4.1 PFLOP |
| 256-8-1-6 | 20.52M | 8.1 PFLOP |
| 256-8-1-12 | 24.56M | 9.7 PFLOP |
| 384-12-1-4 | 30.73M | 12.1 PFLOP |
| 384-12-1-6 | 33.74M | 13.3 PFLOP |
| 384-12-1-12 | 42.74M | 16.8 PFLOP |

## 27. Iso-FLOP winners by compute budget

> When you re-slice the same trajectories by **compute spent** instead of tokens, the shape preference reverses:

| iso-FLOP budget | winner | runner-up |
|---|---|---|
| 4 PFLOP | **128-4-1-12** (3.445) | 256-8-1-6 (3.716) |
| 8 PFLOP | **256-8-1-12** (3.318) | 256-8-1-6 (3.347) |

## 28. Iso-FLOP depth ladder at width 512 (params/tokens/steps)

> Since C = 6·N·D is fixed, higher depth → more params/token → fewer tokens. Each row below hits **8 PFLOP**:

| arch | params | tokens (D) | steps @64k/step |
|---|---|---|---|
| 512-16-1-4 | 43.6M | 30.6M | 467 |
| 512-16-1-6 | 48.9M | 27.3M | 416 |
| 512-16-1-8 | 54.2M | 24.6M | 375 |
| 512-16-1-12 | 64.9M | 20.6M | 314 |
| 512-16-1-16 | 75.5M | 17.7M | 270 |
| 512-16-1-20 | 86.1M | 15.5M | 236 |

## 29. Combined loss-vs-wall Pareto (coord-check + wide/shallow)

> All 4 ran under the cap (biggest was 640-20-1-4 at 7:42). Combined Pareto:

| arch | val/loss | val/bpb | wall | verdict |
|---|---|---|---|---|
| 384-12-1-12 | **3.0962** | 1.2794 | 7:19 | Pareto (loss floor) |
| **512-16-1-6** | 3.1229 | 1.2904 | **7:05** | **Pareto — new sweet spot** |
| 640-20-1-4 | 3.1737 | 1.3114 | 7:42 | dominated ✗ |
| 384-12-1-6 | 3.1830 | 1.3153 | 5:27 | Pareto |
| 512-16-1-4 | 3.2115 | 1.3271 | 6:12 | dominated ✗ |
| 384-12-1-4 | 3.3263 | 1.3745 | 4:58 | Pareto |

## 30. Corrected tied-param counts + embedding fraction

> Corrected the param counts (the model **ties** the embedding/lm_head, and vocab 64402 is huge — so these are embedding-dominated at small width). Real tied totals:

| arch | total params | emb table | emb fraction |
|---|---|---|---|
| 256-8-1-6 | 20.5M | 16.5M | 80% |
| 384-12-1-6 | 33.7M | 24.7M | 73% |
| 512-16-1-6 | 48.9M | 33.0M | 67% |
| 768-24-1-6 | 85.2M | 49.5M | 58% |
| 1024-32-1-6 | 129.3M | 65.9M | 51% |

## 31. Proposed family (initial, large=1024-32-1-6)

> The curve's two hard signals — **depth 6 is wall-optimal, and width is the paying axis (still climbing at 512)** — point to a clean geometric ladder: double the width each tier, hold depth 6, head_dim 32, GQA n_kv=1, same swept muP LR (0.013/0.006, transfer verified).

| tier | arch | params | status | val/loss @65M | wall |
|---|---|---|---|---|---|
| **small** | 256-8-1-6 | 20.5M | ✅ verified Pareto | 3.346 | 4:25 |
| **base** | 512-16-1-6 | 48.9M | ✅ verified — the star of this curve | **3.123** | 7:05 |
| **large** | 1024-32-1-6 | 129M | ⚠️ extrapolated along winning axis | — | — |

## 32. LOCKED chimera GPT family (small/base/large)

> ## The chimera GPT family

| tier | `--arch` | dims (W-H-K-L) | params | basis |
|---|---|---|---|---|
| **small** | `small` | 256-8-1-6 | 20.5M | verified Pareto point |
| **base** | `base` | 512-16-1-6 | 48.9M | verified curve winner (val/loss 3.123) |
| **large** | `large` | 1024-32-1-8 | 150M | width-ladder extrapolation + depth for long horizons |

## 33. Projected val/loss @1BT + Chinchilla status

> Calibrated to the **real ctx512 1BT run** (`cqf5yqt7`, same depth-6/width-384/seq-512): fitting its 29-point curve gives **L(D) = 2.15 + 349·D⁻⁰·³²³** (b=0.323, textbook Chinchilla range), and its fit at 65M (3.19) matches the coord-check anchor (3.183) — so the projection rests on measured data-scaling, not a blind guess. The model-size term is D-independent (Chinchilla separability), so each model's 65M anchor shifts along the same curve.

| model | params | val/loss @1BT | bpt | **bpb** | vs 65M | Chinchilla @1BT |
|---|---|---|---|---|---|---|
| small `256-8-1-6` | 20.5M | 2.77 | 4.00 | 1.144 | −0.57 | over-trained (2.4×) |
| **base `512-16-1-6`** | 48.9M | **2.55** | **3.67** | **1.052** | −0.57 | **✅ optimal (1.0×)** |
| large `1024-32-1-8` | 150M | **2.35** | **3.40** | **0.973** | −0.57 | under-trained (0.33×) |

## 34. Projected 1BT wall time per model

> Using the **measured training throughput** from the coord-check runs (tok/s at seq 512, stripping the eval overhead that inflated the 65M wall):

| model | params | tok/s (seq 512) | **1BT wall** | source |
|---|---|---|---|---|
| small `256-8-1-6` | 20.5M | 337k | **~49 min** | measured |
| base `512-16-1-6` | 48.9M | 192k | **~87 min** (1h27m) | measured |
| large `1024-32-1-8` | 150M | ~53k | **~5.2 h** (±1h) | **estimated** |

## 35. 1BT wall × loss × Chinchilla tradeoff

> Pairing this with the loss projection makes the picture sharp:

| model | 1BT wall | bpb @1BT | Chinchilla @1BT | verdict |
|---|---|---|---|---|
| small | ~49 min | 1.144 | over-trained | cheap, saturated |
| **base** | ~87 min | **1.052** | **optimal** | **best value** |
| large | ~5.2 h | 0.973 | under-trained (0.33×) | best loss, but 3.5× the wall for a model 1BT can't finish |

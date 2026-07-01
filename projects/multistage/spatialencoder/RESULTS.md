# Stage 1 Autoencoder — rFID Improvement Journal

Append-only log. Goal: minimize **val_rfid** (reconstruction FID between AFHQ eval
images and their AE reconstructions) with an eye on **training efficiency** (GPU
util / MFU). Hardware: single RTX 5070 Ti (16 GB, sm_120, Blackwell).

## Setup / baseline facts (entry 0)

- Dataset: AFHQ 128x128, 14630 train / 1500 test, batch 32.
- Model: `ConvAutoEncoder` (DC-AE style), downsample 8 -> 16x16 latent, latent_dim 16.
  Current default config (== "base" in metrics.csv): base_channels=16,
  dim_per_block=(16,32,64), layers_per_block=(1,2,4). Compression ~12x.
- Training: 5 epochs, AdamW lr=1e-3 constant fused, MSE loss, bf16 + channels_last,
  torch.compile reduce-overhead. LPIPS_WEIGHT currently 0.
- **Best result so far (from prior metrics.csv):** `lpips-alex-0.1` -> val_rfid **15.15**
  at epoch 5 (MSE + 0.1*LPIPS(alex)). Plain MSE "base" -> 17.18. LPIPS clearly helps rFID.
- rFID is the FID(2048) between real eval images and recons, normalize=True, fp32 inception.

### Plan
1. Time one 5-epoch run to learn the per-window compute budget (5-epoch runs look
   cheap; if so, spend the window on more epochs + better schedule).
2. Re-establish LPIPS=0.1 + cosine LR w/ warmup as a strong, cheap baseline.
3. Iterate on: capacity, latent, perceptual weight, schedule, GAN (later).

---

## Entry 1 — 2026-06-30 ~23:55 — Refactor + Experiment 1 launched

**What I did:** Refactored `stage1.py` to be CLI-configurable (no more editing
constants): flags for `--epochs --base-channels --dim-mult --layers-per-block
--latent-dim --lpips-weight --lr --warmup-frac --min-lr-frac --eval-every
--batch-size`. Added a **linear-warmup + cosine-decay LR schedule** (per-step) and
best-rFID tracking. Eval/FID can now run every N epochs to spend more of the window
on training. Kept the efficient path intact (bf16, channels_last, compile
reduce-overhead, on-device metric accumulation).

**Timing baseline:** plain 5-epoch run (bc=16, MSE-only) = 2m15s wall, of which
~60s is import+compile+dataset-load overhead, ~15s/epoch after. So per 5-min window
I can train ~12-14 epochs. Throughput ~80 batch/s = 2560 img/s; model is tiny
(288K params) → ~2% MFU, launch/memory-bound. **Implication: adding capacity
improves both fidelity and MFU** — the GPU is mostly idle on the small model.

**Prior best (old metrics.csv):** `lpips-alex-0.1` val_rfid **15.15** (bc=16, MSE+0.1
LPIPS, 5ep constant LR). Plain MSE bc=16 = 17.18.

**Experiment 1 (`v1-bc32-lpips`, running):** base_channels 16→32 (params 288K→1.14M,
4x), LPIPS 0.1, cosine+warmup, 14 epochs, eval-every 2, lr 1e-3. Tests capacity +
schedule + longer-training combined vs prior 15.15. Compression unchanged (12x,
latent 16ch@16x16 — treated as the fixed multistage interface).

### Experiment queue (planned)
- Ablate: bc=32 with constant LR vs cosine (isolate schedule gain).
- Push capacity: bc=64, more decoder res layers (decoder-heavy helps recon).
- LPIPS weight sweep at best capacity (0.05 / 0.1 / 0.2).
- Try VGG LPIPS net (often better perceptual quality than alex) for final eval.
- Consider lightweight GAN/adversarial term later (largest rFID lever, more complex).

---

## Entry 2 — 2026-07-01 ~00:05 — Experiment 1 result: NEW BEST 14.07

**`v1-bc32-lpips`** (bc=32, LPIPS 0.1, cosine+warmup, 14 ep): val_rfid trajectory
2→23.64, 4→16.47, 6→**15.00** (already < prior best), 8→14.35, 10→14.11, 12→**14.07**
(best), 14→14.14 (tiny uptick = mild overfit/noise near LR floor).

**Result: best val_rfid 14.07 vs prior best 15.15 (~7% better).** Drivers: cosine
decay (big back-half gains, 16.5→14.1 as LR fell 8.7e-4→1e-4) + 4x capacity + longer
horizon. Curve flattens hard after epoch 10 → bc=32 saturates ~14.0 with this recipe.

**Learning:** at fixed 12x compression, schedule + longer training did most of the
work; bc=16→32 alone is modest. To break the 14.0 plateau I need either more capacity
or a stronger perceptual/structural signal, not just more epochs.

**Experiment 2 launched (`v2-bc64-lpips`):** bc=64 (~4.5M params), 16 ep, eval-every 3,
min-lr-frac 0.08. Tests whether more capacity breaks the plateau (and raises MFU on
the idle GPU). Next in queue if capacity saturates: LPIPS-VGG, then decoder-heavy
layers, then a light adversarial term.

---

## Entry 3 — 2026-07-01 ~00:20 — Experiment 2: bc=64 BREAKS THE PLATEAU → 11.41

**`v2-bc64-lpips`** (bc=64 ~4.5M params, LPIPS 0.1 alex, cosine+warmup, 16 ep,
eval-every 3): rFID 3→14.30, 6→12.31, 9→11.64, 12→**11.41 (best)**, 15→11.47, 16→11.44.

**Result: best val_rfid 11.41 — a ~25% improvement over prior best (15.15) and 19%
over bc=32 (14.07).** More capacity both converges FASTER (12.31 at epoch 6, beating
bc=32's whole run) and reaches a much lower floor. Saturates ~11.4 after epoch 12.

**Efficiency / MFU:** bc=64 flips the GPU from launch-bound to **compute-bound: 94-97%
util, ~227W/300W** (bc=32/bc=16 sat mostly idle at ~2% MFU). Throughput ~20.5 batch/s
= 656 img/s (4x slower than bc=32's 2560, but the GPU is now actually working). Great
rFID-per-dollar since it also converges in fewer epochs. Peak mem ~2.75GB — lots of
headroom on the 16GB card for bigger batch or capacity.

**Learning:** at fixed 12x compression, encoder/decoder CAPACITY is the dominant lever
so far. The GPU was massively underutilized before; scaling the model is nearly free.

**Experiment 3 launched (`v3-bc64-vgg`):** bc=64 + LPIPS **VGG** (vs alex), 14 ep.
Tests whether the VGG perceptual net (better FID correlation, sharper textures) beats
alex at the strong capacity. Two research subagents (architecture + loss/training SOTA)
are running in parallel to inform experiment 4+.

---

## Entry 4 — 2026-07-01 ~00:35 — Research synthesized + pipeline upgrades

Two parallel research agents returned (architecture + loss/training SOTA). Key findings:

**Architecture (ViTok arXiv:2501.09755):** at a FIXED bottleneck, total latent floats
(=4096 here) caps rFID; among remaining levers, **decoder capacity is dominant and
encoder size is ~uncorrelated with rFID**. → make the decoder heavier than the encoder
(free win at fixed params). DC-AE residual shortcuts + GroupNorm already present (good).
Optional: drop sigmoid head (saturates high-freq), 1 linear-attn block at 16x16.

**Loss/training (VQGAN, FFL, cloneofsimo, SD-VAE):** (1) **VGG-LPIPS >> AlexNet** for
rFID; (2) **L1 > MSE** pixel term (sharper, free); (3) **Focal Frequency Loss** cheap
high-freq booster (great for AFHQ fur); (4) EMA cheap polish; (5) adversarial = biggest
lever but unstable in a cold short run — only worth it if training ACCUMULATES across
windows via checkpoint resume.

**Confirmed live:** `v3-bc64-vgg` (VGG-LPIPS @ 0.1, symmetric bc64): epoch 3→10.90,
6→**8.57** — already crushing bc64-alex's best (11.41). VGG is the single biggest lever
found so far. (Note: old data showed raising *alex* weight hurt rFID; the win is the
VGG *net*, not the weight — so I keep weight ~0.1.)

**Pipeline changes made (for exp4+):**
- `ConvAutoEncoder` now supports `dec_layers_per_block` → asymmetric decoder-heavy design.
- stage1.py: `--pixel-loss {mse,l1}`, `--ffl-weight` (Focal Frequency Loss impl added),
  `--lpips-net`, `--dec-layers-per-block`, `--resume` (model+optimizer, for accumulation).
- **Switched to a WALL-CLOCK time budget** (`--time-budget`, default 270s) instead of a
  fixed epoch count (per user request). Throughput is calibrated over steps 20–70 (past
  compile) to set the cosine-decay horizon so LR hits its floor exactly at the budget;
  `--epochs` is now just a safety cap. Warmup is now absolute (`--warmup-steps`, 100).

### Next: exp4 = bc64 + VGG + decoder-heavy (isolate ViTok's top finding), then L1, FFL.

---

## Entry 5 — 2026-07-01 ~01:00 — VGG champion, convergence projections, DC-AE recipe

**exp3 `v3-bc64-vgg` final:** best val_rfid **7.587** (bc64, VGG-LPIPS 0.1, MSE, symmetric,
14 ep). VGG-LPIPS is the single biggest lever: 11.41 (alex) → 7.59 (vgg), ~34% better.

**Timing reality (answering "how long are runs taking"):** fixed-epoch runs were badly
overshooting 5 min — bc32+alex ~4min, bc64+alex ~7.5min, bc64+vgg ~10-11min. Steady-state:
bc32+alex ~10s/ep (1408 img/s), bc64+alex ~22s/ep (656), bc64+vgg ~43s/ep (336). VGG's
frozen forward ~doubles step cost; each bc doubling ~halves throughput. → switched all runs
to `--time-budget` (wall-clock) with throughput-calibrated cosine horizon.

**Convergence projection (exp fit rFID(e)=floor+A·e^(-e/τ)):** τ≈1.5-2.6 epochs for all
configs → every run is within ~0.1 of its asymptote by epoch ~12. **Longer training buys
almost nothing.** Projected floors: bc32+alex 14.14, bc64+alex 11.40, bc64+vgg **7.53**.
The floor is set by architecture+loss, NOT run length → fast iteration is correct; short
runs already reveal a config's floor.

**DC-AE deep-dive (arXiv:2410.10733):** 3-phase recipe — (1) low-res full train L1+LPIPS
no GAN, (2) high-res *latent adaptation* (freeze all but encoder-head+decoder-input),
(3) low-res *local refinement* freeze all but decoder-head + PatchGAN. Uses **L1 not MSE**,
constant LR per phase. GAN is the big lever (their FID 3.82→0.69) but decoupled to a final
phase — out of scope for a 10-min cap. My residual shortcuts match theirs (confirmed).
For 128px final: pretrain 64px (70-80% budget) → 128px adapt (10-15%) → 64px GAN (10-15%).

**10-min-budget plan:** validated new code path with a cheap smoke test, then spent the
budget on ONE definitive run — **exp4 `v4-dcae-dech-l1-vgg`**: bc64 + VGG + **L1** +
**decoder-heavy** (enc 1,1,2 / dec 2,3,5), 128px, 360s budget, eval-every 2. Tests the
DC-AE recipe (L1 + ViTok decoder-heavy) on the real metric vs the 7.53 champion. Will fit
its curve to project the floor.

---

## Entry 6 — 2026-07-01 ~01:12 — exp4 result + 10-min cap reached (STOP)

**exp4 `v4-dcae-dech-l1-vgg`** (bc64, VGG 0.1, **L1**, **decoder-heavy** enc(1,1,2)/dec(2,3,5),
128px, 360s budget → 7 epochs): rFID 2→9.83, 4→8.62, 6→8.51, 7→**8.50** (converged for its
schedule). BETTER pixel fidelity than champion (val PSNR 30.9 vs 30.5, val MSE 8.1e-4 vs 8.9e-4)
but HIGHER rFID (8.50 vs 7.59).

**Why NOT a clean win — schedule confound:** exp4's 360s budget gave a ~6.5-epoch cosine, so LR
decayed to floor 2x faster than v3's 14-epoch cosine. At MATCHED epochs exp4 was actually ahead
(e4=8.62 vs v3 e6=8.57, and lower MSE), so the architecture/L1 change looks neutral-to-positive
— but the short schedule capped its rFID higher. Can't declare decoder-heavy+L1 a win/loss yet.

**Core finding (the real lesson): time-budget × cosine interaction.** A shorter wall-clock budget
forces faster LR decay → lands at a HIGHER rFID floor. VGG needs ~10min@128px to reach ~7.5; in a
5-6min budget @128px it only reaches ~8.5. So "5-min-per-run" and "VGG's best rFID" conflict at
128px.

**The efficiency unlock (next):** DC-AE's low-res pretraining. At 64px, steps are ~4x cheaper →
fit ~14 VGG epochs into ~2.5 min → reach the 7.x regime, then a short 128px adaptation. This is
how to get champion-level rFID inside a 5-min budget. Implemented `--image-size` + `--resume`
for exactly this. NOT run yet (10-min cap hit).

### Leaderboard (val_rfid, 128px, 12x compression)
| run | config | best rFID | ~wall |
|-----|--------|-----------|-------|
| v3-bc64-vgg | bc64, VGG0.1, MSE, symmetric, 14ep | **7.59** | ~10min |
| v4-dcae | bc64, VGG0.1, L1, dec-heavy, 7ep (360s) | 8.50 | ~6min |
| v2-bc64-lpips | bc64, alex0.1, MSE, 16ep | 11.41 | ~7.5min |
| v1-bc32-lpips | bc32, alex0.1, MSE, 14ep | 14.07 | ~4min |
| (prior best) | bc16, alex0.1, MSE, 5ep | 15.15 | — |

### Next experiments (when budget resumes)
1. **64px→128px progressive** (DC-AE): VGG L1 @64px to floor, resume @128px adapt. Target ≤7.5 in <5min.
2. Clean ablation of decoder-heavy & L1 at MATCHED schedule length.
3. EMA (cheap ~0.1-0.2 rFID). 4. Phase-3 PatchGAN refinement (biggest lever) once checkpoint accumulates.

---

## Entry 7 — 2026-07-01 ~01:30 — Full DC-AE 3-phase pipeline implemented + research workflow

**Implemented `projects/multistage/spatialencoder/dcae.py`** — the complete DC-AE decoupled recipe:
- **PatchGAN discriminator** (`NLayerDiscriminator`, 3-layer, spectral-norm + GroupNorm,
  batch-size-independent & stable on small AFHQ).
- **Hinge GAN loss** + **taming-transformers adaptive weight** (lambda = ratio of recon vs
  GAN gradient norms at the decoder's last layer) + GAN warmup + ramp.
- **Selective freezing**: phase 2 trains only "middle" (to_latent/from_latent + last enc
  block + first dec block); phase 3 trains only decoder head + output head.
- **3-phase orchestration** in one process:
  - P1: 64px, full model, L1+LPIPS(VGG), no GAN, **compiled** (reduce-overhead), ~240s.
  - P2: 128px latent adaptation, middle-only, L1+LPIPS, eager, ~90s.
  - P3: 64px local refinement, decoder-head-only, L1+LPIPS+**PatchGAN**, eager, betas(0.5,0.9), ~150s.
- rFID/PSNR/LPIPS **always evaluated at 128px** (comparable to prior runs) → outputs/dcae_metrics.csv.
- Per-phase wall-clock budgets with throughput-calibrated cosine LR; best.pt saved on rFID improve.
- Decoder-heavy by default (enc 1,1,2 / dec 2,3,5), bc64, VGG-LPIPS.

**Research workflow launched (background):** scout → 6 parallel topic researchers →
synthesizer that appends cited entries (what/why/how/link) to repo-root **RESEARCH.md**
(append-only, collaboratively maintained). Topics: adversarial losses, small-data GAN
stabilization, progressive/latent adaptation, EMA/optimizer, perceptual/frequency losses,
recent tokenizers.

Currently: smoke-testing the full pipeline (tiny budgets) to validate the GAN double-backward
+ adaptive weight before the real run.

---

## Entry 8 — 2026-07-01 ~01:45 — Research done, gradient/activation diagnostics on real images

**Research workflow finished:** wrote repo-root `RESEARCH.md` — 20 unique papers across 6
themes (Adversarial, Small-data GAN, Progressive/latent adaptation, EMA/optimizer,
Perceptual/frequency, Recent tokenizers). Top "not-yet-using" levers flagged: (1) DC-AE
decoupled 3-phase freeze (now implemented), (2) small-data GAN stabilization — DiffAugment
/ LeCam / R1+R2 to stop the discriminator overfitting ~15k AFHQ imgs (my phase-3 GAN
currently lacks this — candidate upgrade), (3) frequency/texture loss (FFL/wavelet) or
DISTS instead of LPIPS for fur/whisker detail.

**Diagnostics (`diagnostics.py`) on champion v3 (real AFHQ, fp32):** loss 0.00756,
**mse 7.41e-4, PSNR 31.30dB, LPIPS-VGG 0.068**.
- **Latent healthy:** 16ch@16x16, 0/16 dead channels, per-channel std 0.75–1.37, mean≈0,
  absmax 6.3 → latent capacity is well-utilized, not the obvious bottleneck.
- **Latent gradient tiny:** norm 4.3e-4, per-elem RMS **1.7e-6** → very weak signal into
  the encoder; suggests decoder dominates learning (consistent with ViTok "encoder size
  uncorrelated"). Handed to a subagent for deep analysis.
- Activations clean: no dead units (all %~0 = 0.0), GroupNorm std≈1 (a few outliers to 17-18).

Fixed a torchinfo bf16 crash in dcae.py (summarize fp32 before casting). Re-running the
full 3-phase smoke test; gradient/activation analysis subagent running in parallel.

---

## Entry 9 — 2026-07-01 ~02:00 — Full DC-AE run launched with diagnostic-driven arch fixes

**Gradient/activation analysis subagent verdict:** model is HEALTHY (no vanishing/exploding
grads, no dead units — my earlier "tiny latent grad" worry was a non-issue: loss is just
flat w.r.t. the code). rFID is a **capacity-allocation** problem. Top levers:
1. **Decoder capacity-starved at high res** — champion's symmetric (1,2,4) put 4 resblocks
   at the LOWEST-res decoder stage and only 1 at the highest (128px, where texture/rFID is
   made) — backwards! Fix: decoder-heavy weighted to high-res.
2. **Head + sigmoid** — the single 3-ch head conv does ~5x the relative work (grad/weight
   3.2e-2 vs 2-6e-3 interior) and sigmoid throttles gradient ~55x on saturated pixels
   (fur highlights, eye whites). Fix: refinement head.
3. **GroupNorm gammas frozen at init** (pre-norm) — add LayerScale + zero-init last conv.

**Model upgrades (opt-in flags, backward-compatible for other projects):** `ResBlock` gains
`layer_scale` (LayerScale gain 0.1 on residual) + `zero_init` (identity-init); `ConvAutoEncoder`
gains `refine_head` (conv+GN+SiLU before final proj). Verified default build unchanged.

**Launched `v5-dcae-full`** — the complete DC-AE pipeline + all arch fixes:
- bc64, enc (1,2,2) light / **dec (4,3,2) heavy@hi-res**, VGG-LPIPS 0.1, +layer_scale
  +zero_init_res +refine_head. Params 3.70M (enc 916K / **dec 2.73M**, 3x decoder-heavy).
- Phase1 64px full/compiled 240s → Phase2 128px latent-adapt 90s → Phase3 64px
  decoder-head+PatchGAN 150s. rFID always eval'd @128px. Champion to beat: **7.59**.

---

## Entry 10 — 2026-07-01 ~02:20 — Full DC-AE run result: 8.37 (GAN regressed)

**`v5-dcae-full`** trajectory (rFID @128px):
- Phase1 (64px full, compiled, 240s, ~29.9 steps/s = 3x faster than 128px): 249→**9.17**.
  Low-res pretraining reached a strong perceptual regime fast, but 128px eval carries a
  resolution-mismatch penalty (trained@64, eval@128).
- Phase2 (128px latent adaptation, middle-layers, 90s): 9.17→**8.37**. Adaptation closed
  ~half the res gap, as designed.
- Phase3 (64px decoder-head + PatchGAN, 150s): **8.37→10.3 (REGRESSED)**. The naive GAN
  HURT rFID — the small-data instability the research flagged (~15k imgs; needs DiffAugment
  /LeCam), compounded by refining@64px while evaluating@128px.

**Run best = 8.37 (phase2). Does NOT beat the champion 7.59.** best.pt correctly holds the
phase-2 weights (phase3 never improved on it).

**Honest diagnosis — two confounds, not an arch failure:**
1. **Resolution gap:** the champion got ~14 epochs of *direct 128px* VGG training; v5 spent
   most of its budget at 64px + only 90s adapting at 128px. Less target-res training.
2. **GAN on small data regressed** without DiffAugment/LeCam and at a mismatched res.

The decoder-heavy + refine-head + LayerScale arch changes are still untested cleanly (they
were bundled with the res-schedule change). 

**Next: isolate the architecture.** Run the improved arch the CHAMPION way — pure 128px,
VGG, no GAN, long single phase — to test whether decoder-heavy + refine-head actually beats
7.59 without the res-gap/GAN confounds. Then, separately, fix phase-3 GAN (DiffAugment +
LeCam + lower weight, or run @128px) before re-enabling it.

---

## Entry 11 — 2026-07-01 ~02:35 — GAN stabilizers added; v6 arch-isolation running

**Added phase-3 GAN stabilizers to dcae.py** (for the small-data regression seen in v5):
`--diffaug` (DiffAugment: differentiable translation+cutout on D inputs, no color aug to
keep faithful colors) and `--gan-weight` (cap/scale on the adaptive GAN weight, default 0.5).
Ready for a stabilized phase-3 retry once the arch is validated.

**Launched `v6-arch128`** — improved arch (bc64, enc(1,2,2)/dec(4,3,2) heavy@hi-res,
+layer_scale +zero_init_res +refine_head), trained the CHAMPION way: pure 128px, VGG 0.1,
no GAN, single 420s phase. Isolates architecture vs the 7.59 champion (removes v5's res-gap
+ GAN confounds). Note: decoder-heavy is ~25% slower (7.9 vs 10.5 steps/s) → ~6 epochs in
420s vs champion's 14, so v6 trades training length for capacity. Trajectory so far:
step214→25.1, 440→13.5, 663→10.8 (LR still high; back-half decay pending).

---

## Entry 12 — 2026-07-01 ~02:55 — v6 arch verdict + v7 champion-GAN refinement

**`v6-arch128` result: 9.79 @128px — the improved arch UNDERPERFORMS the champion (7.59).**
It plateaued hard (~9.8 from step 1335 on). Diagnosis: the arch changes NET-HURT under a
fixed wall-clock budget:
- `zero_init_res` makes every resblock start as identity → slow to "wake up"; with a heavy
  decoder + short (~6-epoch) schedule they never fully activate.
- decoder-heavy@hi-res is ~25% slower (7.9 vs 10.5 steps/s) → fewer epochs → compressed
  cosine → higher floor (the same time-budget×cosine confound as exp4).
Even vs exp4 (8.5, decoder-heavy + L1, no zero-init) v6 was worse → `zero_init_res` +
over-heavy hi-res decoder are the likely culprits.

**KEY EFFICIENCY LESSON:** under a fixed wall-clock budget, a fast simple arch that gets
MANY epochs beats a "better" heavy arch that gets few. The champion (bc64, VGG, symmetric,
14 epochs @128px, 7.59) is the efficiency frontier. Fancy arch changes need long training
to pay off and are net-negative when they cost epochs.

**Launched `v7-champ-gan`** — the one realistic shot at beating 7.59: resume the champion
(7.59) and run DC-AE phase-3 GAN done RIGHT — at **128px** (target res, not v5's broken
64px), decoder-head-only, **+DiffAugment** (small-data stabilizer) + **gentle gan-weight
0.5**, lr 5e-5, 180s. If the stabilized GAN sharpens without regressing, new best; else
champion stands. Original v3 last.pt preserved as champion regardless.

---

## Entry 13 — 2026-07-01 ~03:15 — 🏆 NEW BEST 6.54 via stabilized GAN refinement

**`v7-champ-gan`: rFID 6.539 @128px — NEW BEST.** Resumed the champion (7.59) and ran a
DC-AE phase-3 GAN done RIGHT: 128px (target res), decoder-head-only, DiffAugment + gentle
adaptive gan-weight 0.5, lr 5e-5, ~180s. Monotonic, STABLE improvement: 7.59 → 6.87 (step
199) → 6.69 → 6.63 → 6.57 → 6.54 (converged). PSNR 30.77 (champion 31.30 — the GAN trades a
little pixel PSNR for perceptual/FID gain, exactly as expected). No regression, no
instability — the opposite of v5's naive 64px GAN.

**What made the GAN work this time (vs v5's regression):**
1. Run at **128px** (target/eval res) not 64px — no train/eval resolution mismatch.
2. **DiffAugment** on D inputs — stops the discriminator memorizing ~15k AFHQ imgs.
3. **Gentle gan-weight (0.5x the adaptive weight)** + warmup + ramp — GAN nudges, doesn't dominate.
4. Applied to a **well-trained AE** (the 7.59 champion) — refinement, not from-scratch.

### FINAL LEADERBOARD (val rFID @128px, 12x compression)
| run | rFID | note |
|-----|------|------|
| **v7-champ-gan** | **6.54** | champion + stabilized 128px DiffAugment GAN — BEST |
| v3-bc64-vgg | 7.59 | champion: bc64, VGG-LPIPS, 14ep @128px |
| v5-dcae-full | 8.37 | full 3-phase; phase3 naive GAN regressed |
| v6-arch128 | 9.79 | decoder-heavy+refine+zero-init — net-hurt under time budget |
| v2-bc64 | 11.41 | bc64 + alex-LPIPS |
| v1-bc32 | 14.07 | bc32 + alex-LPIPS |
| prior best | 15.15 | bc16, alex, 5ep (starting point) |

**Overall: 15.15 → 6.54 = 57% rFID reduction.** Ranked lever impact: VGG-LPIPS (biggest)
> capacity bc16→64 > stabilized GAN refinement (7.59→6.54) > LR schedule. Decoder-heavy /
zero-init / low-res-pretrain did NOT help under a fixed wall-clock budget (cost epochs).


## Entry 14 — 2026-07-01 ~04:00 — 256px target + tooling standardization + Sweep-2 research

**1. v3/v7 re-run at 256px target (64px low-res).** Made `--eval-res` configurable in
`dcae.py` (rFID + phase-2 now run at the target res, default still 128). Ran the full
3-phase pipeline `v3v7-256` at eval-res 256, p1-res 64, GAN at 256px (`--p3-res 256`):

| phase | what | res | best rFID |
|-------|------|-----|-----------|
| phase1 | low-res full pretrain | 64 | 6.52 |
| phase2 | **v3-at-256** (base AE) | 256 | **5.87** |
| phase3 | **v7-at-256** (+GAN) | 256 | **5.69** |

Both beat their 128px counterparts (v3 7.59→5.87, v7 6.54→5.69). **Caveat/confound:** at
fixed 16ch/8×, a 256px input yields a **32×32** latent (E=16,384) vs 128px's 16×16
(E=4,096) — same 12× compression ratio but **4× the latent floats**, so most of the gain is
"bigger E," exactly the ViTok E=L×c scaling law, NOT that 256px is intrinsically easier.
The high-res GAN helped only 5.87→5.69 (~0.18) and non-monotonically (decoder-head
re-warmup: first eval 6.06 > phase2's 5.87, then recovered) — much weaker than the 128px
GAN's ~1.05 drop, because the stronger base AE leaves less HF headroom.

**2. Tooling standardized.** Added `src/chimera/utils/logtee.py` (`tee_to_logfile`): both
`stage1.py` and `dcae.py` now auto-mirror stdout+stderr to a per-run log, flushed live
(`tail -f`-friendly, incl. tqdm) with a timestamped command header — no manual redirects.
Fixed the crash the 256 run hit (`global EVAL_RES` declared after argparse read it → moved
to top of `main()`). NOTE: consolidating logs INTO `outputs/<run>/` and pruning non-v3/v7
runs is in progress this session.

**3. Sweep-2 research (8-lane workflow) → RESEARCH.md.** Target rFID<2. Load-bearing
verdict: the barrier is NOT latent capacity (we're at 2× the E papers need for <2) nor
conv-vs-ViT — it's **decoder + adversarial + too-short training**. Ordered plan: EMA +
honest re-measure on ≥10k pairs → **R3GAN (relativistic + R1+R2) GAN done right** →
full-decoder GAN @target → DINOv2 feature distillation → generative (diffusion/residual)
decoder. Cheapest credible path to <2 builds on the existing dcae.py phase-3.

## Entry 15 — 2026-07-01 ~04:05 — GAN at LOWER resolution + R3GAN loss (in progress)

**Decision (user directive): the phase-3 GAN should run at the LOWER resolution** (per the
DC-AE paper, phase-3 refines at the low res, not the target). My `v3v7-256` run had
overridden this to 256px; the `dcae.py` default is correctly `--p3-res 64`. Tension with
Entry 13 (128px work found target-res GAN helped) is acknowledged — but that regression was
naive-GAN instability, not resolution per se; low-res GAN is ~16× cheaper (more steps) and
should win with modern stabilizers. **Testing empirically.**

- **`v7-lowgan-256` (LAUNCHED):** identical to `v3v7-256` but `--p3-res 64` (low-res GAN),
  hinge loss, eval 256. Clean resolution A/B vs v3v7-256's 5.69 (high-res GAN). Result TBD.
- **R3GAN loss added to `dcae.py`** (`--gan-loss r3gan`, new default; `--r1-gamma`):
  relativistic pairing loss (element-wise on paired PatchGAN maps) + R1 (real) + R2 (fake)
  zero-centered gradient penalties, per arXiv:2501.05441. Rework of the phase-3 branch to
  use it is in progress; next run will be v7 with low-res + R3GAN.

## Entry 16 — 2026-07-01 ~04:30 — Low-res GAN WINS; R3GAN launched; eval-res clarified

**`v7-lowgan-256` result: low-res (64px) GAN beats high-res (256px) GAN.**

| run | GAN train-res | phase1 | phase2 (base) | phase3 (GAN) | within-run GAN Δ |
|-----|---------------|--------|---------------|--------------|------------------|
| v3v7-256 | 256px | 6.52 | 5.87 | 5.69 | −0.18 |
| **v7-lowgan-256** | **64px** | 6.44 | 5.76 | **5.49** | **−0.27** |

Low-res GAN wins on BOTH the absolute best (**5.49 < 5.69**) and the within-run improvement
(−0.27 vs −0.18), while being ~16× cheaper per step (more GAN steps in the same budget). Its
phase3 trajectory is also cleanly monotonic (5.76→5.63→5.61→5.56→**5.49**), unlike the
high-res GAN's non-monotonic re-warmup. **Confirms the user directive: the DC-AE phase-3 GAN
should refine at the LOWER resolution** (matches the DC-AE paper). This UPDATES the Entry-13
128px finding ("GAN at target res"): that was naive-GAN instability, not a resolution rule —
with DiffAugment + gentle adaptive weight, low-res refinement is both better and cheaper.

**Important measurement note (was a point of confusion): `val_rfid` is ALWAYS measured at
`--eval-res` (256px here) for EVERY phase**, including the 64px-trained phase1/phase3 (model
is fully conv; eval feeds full 256px images). The `res64` in phase3 log lines is `train_res`,
NOT the eval res. So 5.49 is a genuine 256px reconstruction rFID — the low-res-GAN win is
real, not a resolution artifact.

**`v7-r3gan-256` LAUNCHED:** identical low-res sizing + `--gan-loss r3gan --r1-gamma 1.0`
(relativistic + R1/R2). Baseline to beat = 5.49. Watching phase-3 early evals for R1/R2
double-backward stability. Next: latent-dim sweep {8,16,32} @128px on stage1.py.

## Entry 17 — 2026-07-01 ~04:55 — R3GAN REGRESSED (as dropped-in); hinge low-res still best

**Checkpoint reuse applied** (user directive): resumed the shared `v7-lowgan-256/phase2.pt`
base (5.755) and ran phase-3 ONLY (`--p1-time 0 --p2-time 0`), skipping ~560s of redundant
retraining → faster AND a clean A/B (same base, only hinge→R3GAN differs). phaseN.pt now
saves `config`; `--resume` also falls back to CLI arch flags when a ckpt lacks it.

**Result: R3GAN made it WORSE, monotonically, from the same base:**
| phase-3 GAN loss | base | trajectory | best |
|------------------|------|-----------|------|
| hinge (low-res)  | 5.755 | 5.76→5.63→5.61→5.56→5.49 | **5.49** ✓ |
| r3gan (low-res)  | 5.755 | 6.07→6.13→6.17→6.28→6.19 | 6.07 ✗ |

R1/R2 double-backward was numerically STABLE (no NaN/collapse) — it just degraded
reconstruction. So R3GAN is not a drop-in win here.

**Likely cause (to fix before re-judging R3GAN):** the relativistic difference is taken
ELEMENT-WISE on the PatchGAN logit MAPS, but DiffAugment applies INDEPENDENT random
translations to real vs fake → the two maps are spatially MISALIGNED, so the pairing
compares mismatched patches (hinge is immune — it never differences real against fake).
Secondary suspect: hinge-tuned `gan-weight 0.5` + short warmup/ramp is too aggressive for
R3GAN's gradient on an already-good base (it hurt from the very first eval).

**Fixes to try (if we revisit R3GAN):** (1) reduce each patch map to a per-SAMPLE scalar
(mean over H×W) BEFORE the relativistic difference — proper RpGAN pairing, alignment-robust;
(2) share the SAME aug transform between real and fake; (3) gentler `gan-weight` + longer
warmup. Deferred — hinge low-res (5.49) stands as champion; moved on to the latent sweep.

**LEADERBOARD @256px (rFID):** v7-lowgan-256 **5.49** (hinge, low-res GAN) > v3v7-256 5.69
(hinge, high-res GAN) > v7-r3gan-256 6.07 (r3gan, low-res — regressed). All measured at 256px.

## Entry 18 — 2026-07-01 ~05:20 — R3GAN fixed but STILL regresses; hinge stays champion; misc

**Latent-4 quick test (`latsweep-c4`, base AE @128px, stage1):** 4ch@16×16 = E=1024, 48×
compression. Got only ~3 epochs in 150s (torch.compile ate the budget) → best 31.0, NOT
converged. But at MATCHED epochs c4(31)≫c8(19@ep3): halving latent 8→4 ~doubles early rFID.
So the low end IS capacity-limited — the rFID-vs-E curve has a steep knee below 16ch. (The
{8,16,32} sweep was cancelled mid-c8 per user; only the c4 spot-check was run.)

**R3GAN fixed (scalar-pairing) — `v7-r3gan-fix`, phase3-only from shared 5.755 base:**
| phase-3 loss | first eval | best | shape |
|--------------|-----------|------|-------|
| hinge low-res | 5.76 | **5.49** | monotonic ↓ |
| r3gan broken (element-wise on aug-misaligned maps) | 6.07 | 6.07 | drifts ↑ |
| r3gan fixed (per-sample scalar RpGAN pairing) | 5.92 | 5.92 | drifts ↑ (→6.2) |

The fix (relativistic diff on per-sample mean-pooled critic scores, robust to DiffAugment
misalignment) improved r3gan 6.07→5.92 but it STILL degrades the base from the first eval and
drifts worse — not a warmup artifact. Generator loss oscillates hard (3.4↔7.7).

**Verdict — why hinge beats R3GAN for RECONSTRUCTION refinement:** hinge SATURATES (zero
gradient once recon scores "real enough" → gentle, bounded, stops pushing); R3GAN's
relativistic softplus never saturates and pushes the decoder to out-score the real image,
which for faithful reconstruction is a misspecified objective (rewards texture hallucination
that lifts D-score but hurts FID). R3GAN is a GENERATION baseline; it doesn't transfer to
decoder-head recon polish here. **Dropping R3GAN; hinge low-res (5.49) remains champion.**
Code kept (`--gan-loss {hinge,r3gan}`, hinge is safer; consider making hinge the default again).

**Tooling:** added `--log-secs` heartbeat to dcae.py (default 10s: step/lr/loss/it-s between
the 45s evals — "log more often"); logs+outputs consolidated per-run; checkpoint reuse now
standard for phase-3 A/Bs (`--resume phase2.pt --p1-time 0 --p2-time 0`).

**Next:** DC-AE architecture-mimicry agent running (EfficientViT linear-attention blocks +
faithful config) — the more promising structural direction to raise the fidelity ceiling.

## Entry 19 — 2026-07-01 ~06:00 — DC-AE mimicry: attention added; iso-param attn HURT; more tests

**DC-AE study (agent) verdict:** DC-AE = our exact residual-conv AE + **EfficientViT
linear-attention (LiteMLA + GLUMBConv, RMSNorm)** in the DEEP (low-res, high-channel) stages;
their residual-autoencoding shortcut matches ours. Their sub-1 rFID is an ImageNet-scale,
~320M-param, GAN-trained number. Implemented non-breaking in `src/chimera/models/autoencoder.py`:
`RMSNorm2d`, `LiteMLA`, `GLUMBConv`, `EfficientViTBlock`, opt-in `ConvAutoEncoder(attn_stages=...)`,
and a `DCAE(...)` factory. Wired `--attn-stages/--attn-dim` into dcae.py; removed the hardcoded
8× `DOWNSAMPLE` assert so 4-stage (f16) configs run.

**A/B #1 — does DC-AE attention help at iso-param? NO (`v7-attn-256`).** Champion recipe +
attention in the deepest stage (4.37M vs 4.55M params — nearly iso), same pipeline/eval:
| | phase1 | phase2 base | phase3 GAN |
|--|--------|-------------|------------|
| conv (v7-lowgan-256) | 6.44 | 5.76 | **5.49** |
| +attn (v7-attn-256)  | 7.09 | 6.20 | 6.31 |
Attention HURT by ~0.8 across all phases. Consistent with Sweep-2's verdict that conv-vs-ViT
is NOT our barrier: at our small scale + short (~19-epoch) budget, linear attention (replacing
conv resblocks at iso-param) is just harder to train and adds no value where global structure
isn't the bottleneck. DC-AE's attention pays off at high compression + huge training, not here.

**Still queued:** `dcae-f16-256` — the FULL f16 mimic (4 stages/16× spatial, latent 32, attn
in 2 deepest stages, 15M params, E=8192@256px = HALF the champion's E). Different test: more
capacity + higher compression + attention together. Kept at lpips 0.1 (per user).

**RUNNING now — `v7-lpips05-256`:** champion recipe with **lpips-weight 0.1→0.5** (LPIPS is our
biggest lever; ViTok uses 1.0). Clean A/B vs 5.49, full pipeline. Result TBD.

**Champion still: `v7-lowgan-256` = 5.49 @256px** (hinge, low-res GAN, lpips 0.1).

## Entry 20 — 2026-07-01 ~07:00 — LPIPS weight is a BIG lever; DC-AE high-compression mimics fail; 10-min cap

**LPIPS weight was under-tuned at 0.1 the whole time.** Bumping it cut base-AE rFID hugely
(phase1 @256px eval): 0.1→**6.44**, 0.5→**5.16** (still dropping when stopped). Default
`--lpips-weight` changed 0.1→**1.0**. (Tradeoff: PSNR drops ~0.5dB — expected, rFID rewards it.)

**Process:** user set a HARD **10-minute/run cap** (no time extensions without approval — I
wrongly doubled a phase-1 budget once). Standard budget now p1 240 / p2 60 / p3 120 = 420s.
See memory [[run-time-budget-approval]].

**DC-AE high-compression mimics — both FAIL vs the 5.49 champion (all lpips 1.0, 10-min):**
| run | arch | E@256 | best rFID | note |
|-----|------|-------|-----------|------|
| champion v7-lowgan-256 | conv 8×, c16 | 16384 | **5.49** | lpips 0.1 |
| c8-full-256 | conv 8×, c8 | 8192 | 6.79 | phase3 GAN regressed 6.79→8.18 |
| dcae-f16-256 | attn 16×, c32 | 8192 | 7.24 | phase3 GAN regressed 7.24→9.14 |
| v7-attn-256 | attn 8×, c16 (iso-param) | 16384 | 6.31 | attention hurt |

Conclusions: (1) **E/capacity matters** — halving E (16384→8192) costs ~1.3 rFID. (2) At equal
E=8192, **plain conv/8× (6.79) beats DC-AE attention/16× (7.24)** — the DC-AE arch's value
(attention, high spatial compression) does NOT transfer to our small-data/short-budget/low-res
regime. Confirms Sweep-2: our barrier is capacity+GAN+training-budget, not conv-vs-ViT. f32
(96×) skipped as near-certain failure.

**NEW ISSUE — LPIPS 1.0 breaks the phase-3 GAN.** Both c8 & f16 regressed in the GAN phase.
Cause: adaptive GAN weight = ‖∇rec‖/‖∇gan‖; lpips 1.0 inflates ‖∇rec‖ → GAN pushed ~10× harder
than tuned → destabilizes. FIX (todo): lower `--gan-weight` when lpips is high, OR keep high
lpips only in phases 1-2. The 5.49 champion's GAN worked because lpips was 0.1.

**Throughput bench (live c8 read):** phase1 64px compiled ~65 steps/s (~1040 img/s) but GPU only
78% util, 3.9/16GB — small model under-feeds GPU at 64px. Full-testset 256px eval every 45s
stalls training ~8-10s each = ~15-20% of the 10-min budget on eval. Speedup levers (todo):
`--eval-secs 45→90`, `--batch-size 16→48`, optional subsampled mid-run eval.

## Entry 20 — 2026-07-01 — SD-VAE external baseline: beats us @256px with 4× fewer latent floats

**What/why:** measured an off-the-shelf **Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`)**
on AFHQ as an external reference point, using the EXACT dcae.py rFID protocol (AFHQ test,
batch 32 drop_last, `FID(feature=2048, normalize=True)` fp32, real vs recon clamped [0,1]).
New script `projects/multistage/spatialencoder/eval_sdvae.py` (loads the VAE via `diffusers`, recon =
`decode(encode(x).latent_dist.mode())`, input scaled to [-1,1]). Added `diffusers` dep.

| eval res | SD-VAE rFID | PSNR | LPIPS-vgg | latent | latent floats | compression |
|----------|-------------|------|-----------|--------|---------------|-------------|
| **256px** | **3.93** | 27.80 | 0.119 | 4ch@32×32 | 4,096 | 48× |
| 128px | 10.82 | 24.96 | 0.140 | 4ch@16×16 | 1,024 | 48× |

**vs our leaderboard:** @256px SD-VAE **3.93 beats our champion 5.49** — while using **4× FEWER
latent floats** (4,096 vs our 16,384) and **4× harder compression** (48× vs 12×). @128px SD-VAE
LOSES (10.82 vs 6.54): 128px is far below its 512px native res and its latent collapses to 1,024
floats there.

**Iso-float read (ViTok E=L·c):** SD-VAE @256 (E=4,096) 3.93 vs OUR AE @128 (E=4,096) 6.54 —
same float budget, SD-VAE's decoder wins by ~2.6 rFID. **Confirms the Sweep-2 verdict: our
barrier is decoder + adversarial + training scale, NOT latent capacity.** SD-VAE trades pixel
fidelity for it though — PSNR 27.8 vs our ~30-31, LPIPS 0.119 vs our 0.068: its heavily
adversarial decoder hallucinates plausible sharp texture (great for FID) rather than
reconstructing faithfully.

**Caveats:** not compression-matched (48× vs 12×), and SD-VAE has massive pretraining vs our
~15k-image AFHQ runs — it's a strong-baseline reference, not a controlled A/B. Actionable
takeaway: the gap is closed by a stronger/longer-trained adversarial decoder, not more latent
channels. (Not run: SDXL VAE, usually stronger at high res — candidate next baseline.)

## Entry 22 — 2026-07-01 ~07:35 — GOAL reframed to PARETO (better rFID AND better compression)

**LPIPS 1.0 is a massive lever.** With latent 32 + lpips 1.0, phase-1 alone hit
**val_rfid 3.045 @256px** (PSNR 33.8, LPIPS 0.059) — crushing the <3.5 target and SD-VAE's 3.93.
BUT latent 32 @ 8× = only **6× compression** (E=32768), WORSE than the champion's 12×. So it's a
rFID win bought with more latent floats — **rejected**: the goal is now explicitly Pareto —
**better rFID AND better compression** (no enlarging the latent to cheat).

**Reference frontier (rFID @256px vs compression):**
| config | latent | compression | E@256 | rFID | lpips |
|--------|--------|-------------|-------|------|-------|
| latent-32 (REJECTED, worse compression) | 32@8× | 6× | 32768 | 3.05 | 1.0 |
| **champion** v7-lowgan-256 | 16@8× | **12×** | 16384 | **5.49** | 0.1 |
| SD-VAE (external ref) | 4@8× | **48×** | 4096 | **3.93** | — |
| c8 (lpips1) | 8@8× | 24× | 8192 | 6.79 | 1.0 |
| f16 (attn) | 32@16× | 24× | 8192 | 7.24 | 1.0 |

To Pareto-beat the champion: rFID < 5.49 at compression > 12×. The lpips-1.0 lever (which took
latent-32 to 3.05) is the hope — apply it at HIGHER compression than the champion.

**RUNNING `goal-c12-lp1`:** latent 12 @ 8× = **16× compression** (E=12288, better than champion's
12×) + lpips 1.0 + gentle GAN (gan-weight 0.05, since lpips 1.0 was blowing up the adaptive GAN).
Target: rFID < 5.49 → a genuine Pareto win (better on BOTH axes). Result TBD.

**Infra note:** project moved to `projects/multistage/spatialencoder/`; a run mid-move crashed on
the stale metrics-CSV path (not a model bug). All new launches use the new path. Speedups now
standard: `--batch-size 32` (GPU 99% util vs 78%@bs16), `--eval-secs 90`. Hard 10-min/run cap.

## Entry 23 — 2026-07-01 ~07:50 — latent-12 near-miss; depthwise convs added; deep+narrow attempt

**`goal-c12-lp1` (latent 12 @ 16×, lpips 1.0):** phase1 7.34 → **phase2 5.69** → phase3 6.54
(GAN REGRESSED again, even at gan-weight 0.05 — confirms LPIPS 1.0 breaks the adaptive GAN;
dropping the GAN phase for high-lpips runs). Best **5.69 @ 16× compression** — a near-miss:
better compression than champion (16×>12×) but rFID slightly WORSE (5.69 > 5.49). Not yet Pareto.

**Architecture lever added — depthwise-separable convs** (`ResBlock(depthwise=True)`, plumbed
through DCDown/UpBlock + ConvAutoEncoder + `dcae.py --depthwise`). ~8× cheaper per conv → buy
depth. E.g. enc(2,3,6)/dec(6,4,3) depthwise @ base64 = 2.21M (vs dense champion 4.55M @1,2,4);
@ base32 = **0.57M**. Lets us go deeper AND run more steps in the 10-min budget.

**RUNNING `goal-dw-c12`:** stacking user's levers to break 5.49 at 16× compression — latent 12,
**base_channels 32** (more steps), **depthwise deep** enc(2,3,6)/dec(6,4,3), **higher LR** (p1
2e-3, p2 5e-4), lpips 1.0, base-only (no GAN). Bet: many more steps + depth offset the narrower
width. Risk: 0.57M may underfit. Target: rFID < 5.49 @ 16× = Pareto win. Result TBD.

## Entry 24 — 2026-07-01 ~08:00 — DECISION: v7-lowgan-256 is the FINAL (until further notice)

**Moving forward with `v7-lowgan-256` as the final stage-1 model until further notice.** All other
experiments STOPPED.

- **Final: `v7-lowgan-256` — rFID 5.49 @256px**, 12× compression (latent 16ch @ 8×), hinge low-res
  DiffAugment GAN, lpips 0.1. Checkpoint: `outputs/v7-lowgan-256/best.pt`. (128px equivalent = 6.54.)

**Why stop here (Pareto goal not met):** the goal was better rFID AND better compression than this
champion. Findings: LPIPS 1.0 is a big rFID lever (latent-32 hit 3.05) but only by spending more
latent floats (6× compression — worse). At HIGHER compression than the champion, the E-penalty
roughly cancels the LPIPS gain (latent-12 @16× = 5.69, a hair worse than 5.49). Depthwise convs
don't buy throughput on this GPU (memory-bound: 40 vs 65 steps/s). Net: our arch's rFID-vs-
compression frontier is ~fixed and the champion sits on it; a strict Pareto win needs a real
frontier-shifter — a GAN that survives high LPIPS (currently regresses), longer training (we're
budget-capped at ~19 epochs vs the 50-125 the literature uses), or a distilled/diffusion decoder —
none feasible in the 30-min budget. These are the documented next steps if we revisit.

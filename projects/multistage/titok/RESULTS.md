# Stage-2 Latent TiTok — rFID Journal

**GOAL: end-to-end rFID < 7.** Append-only log; updated as runs complete.

## Setup
- Stage-2 TiTok 1D tokenizer over the FROZEN stage-1 AE's latents (`titok.py`). Default AE =
  `spatialencoder/outputs/v7-lowgan-256/best.pt` (256px, 8x downsample → latent `(16,32,32)`).
- **end-to-end rFID** = FID(2048) between real 256px images and `AE.decode(tokenizer(AE.encode(img)))`.
- **ae_rfid = 5.53** — the frozen AE alone (`decode(encode)`), the *ceiling*. Token cost = rFID − 5.53.
  So the goal (<7) means the token bottleneck may add at most ~1.5 rFID.
- Objective: latent MSE only (no LPIPS, per request). Optimizer: Muon (2D weights) + AdamW aux.
- Deterministic (seed 0, `use_deterministic_algorithms(warn_only=True)`; flash SDPA on — its
  backward is non-deterministic so not bit-exact, but attention is not the bottleneck).

## Throughput (why runs are now fast)
Per-step cost was the **frozen AE conv-encode at 256px, rerun every step** — NOT attention
(patch 2 with 9× the tokens ran ~as fast as patch 4; flash SDPA already active). Fix: **cache all
normalized train latents once** (AFHQ has no random train aug → deterministic → exact) and train
the ViT purely on cached-latent minibatches. **~8 → ~51 it/s (~6×)**, batch 64 ≈ 3.3k img/s.
Default budget now **300 s (5-min quick tests)**; total wall (cache + compile + eval) < ~7 min.
Heartbeat logs it/s every 10 s to catch regressions. Auto-stops when val_rfid < `--target-rfid` (7).

## The binding constraint (hypothesis)
At the default K=32 tokens × 16 dim = **512 floats**, the tokenizer must reconstruct the
16384-float latent (**32× compression on top of the AE**). Patch size (p2/p4) changes the ViT's
internal resolution but NOT this bottleneck. So the dominant lever toward <7 is almost certainly
**token budget (K, latent_dim)** and **training length/model size**, not patch size. Plan: confirm
with the p2 baseline, then sweep K (32 → 64 → 128 → 256) and latent_dim; research (below) informs.

## Results
| run | patch | K×dim (E) | steps | it/s | best rFID | token cost (−5.53) | latMSE | PSNR | notes |
|-----|:-----:|:-----:|------:|:----:|----------:|-------------------:|-------:|-----:|-------|
| vit-tiny-p4 | 4 | 32×16 (512) | 3250 | ~51 | 80.87 | +75.34 | 0.475 | 21.1 | pre-speedup (480s), undertrained |
| vit-tiny-p2 | 2 | 32×16 (512) | 6280 | ~26 | **63.67** | +58.15 | 0.475 | 20.8 | cached+deterministic 300s; still descending |

**Patch size:** p2 (256 patch tokens) beats p4 (64) — 63.7 vs 80.9 — preserving latent spatial
resolution helps reconstruction. Cost: ~26 vs ~51 it/s (post-caching, attention is now the per-step
cost, ∝ tokens²). Both are **E=512-bound**, far from the goal.

## Log
- **Entry 0 — baseline + speedup.** p4 → rFID 80.9 (undertrained). Diagnosed AE-encode bottleneck
  → latent caching (~6×, 8→51 it/s @ patch4). Reverted to deterministic; added rFID auto-stop.
- **Entry 1 — p2 baseline + research.** p2 (patch 2) → **63.7**, better than p4 but plateauing on
  the E=512 bottleneck. Parallel literature research (below) says this is expected.

## Research findings (parallel agent; arXiv IDs) — REVISES THE PLAN
- **ViTok scaling law (2501.09755):** rFID is governed by **E = tokens × dim = total bottleneck
  floats**, ~independent of how you split tokens-vs-width, encoder size, or FLOPs. Our **E=512**
  (32×16) sits where the curve predicts **~5–6 *added* rFID even at best** (with LPIPS + GAN + big
  decoder) → end-to-end ~8–12. **So <7 is essentially unreachable at E=512.**
- **Continuous ≫ VQ at small K** (TA-TiTok 2501.07730: 32 tok → 2.56 continuous vs 7.72 VQ). We're
  continuous ✓. Per-channel latent normalization ✓ (we do this).
- **#1 lever = LPIPS through the frozen (but differentiable) decoder** (ViTok Finding 5; MSE-only
  6.2→0.9 with LPIPS, 2507.09984). Backprops into the tokenizer without unfreezing the AE.
  **Conflicts with the current no-LPIPS constraint.**
- **Decoder dominates rFID at fixed E; encoder ~irrelevant** (ViTok Finding 3) → make the decoder
  bigger than the encoder.
- **Feasibility verdict:** to clear <7 (≤~1.5 over the 5.53 ceiling) need **E ≈ 2×–4×** (e.g.
  **K=64×dim32 or K=32×dim64, E≈2048**) *and* ideally LPIPS+GAN. Pure latent-MSE alone likely
  plateaus above 7 at any E. Alt route at low E: a generative (flow/diffusion) latent decoder
  (FlexTok 2502.13967) instead of MSE.

## Revised plan (pending user call on LPIPS)
1. Bump the bottleneck E — the biggest no-LPIPS-compatible lever: `--num-latent-tokens 64
   --latent-dim 32` (E=2048), patch 2, decoder-heavy. Predicted end-to-end ~6–8.
2. If still short and LPIPS is permitted: add LPIPS-through-frozen-decoder (+ later a PatchGAN
   fine-tune stage) — the literature-backed route to actually land <7.

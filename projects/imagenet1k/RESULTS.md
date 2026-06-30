# ImageNet-1k DataModule — Throughput Optimization Log

Append-only log. Goal: make `chimera.data.imagenet.ImageNetDataModule` (streaming
webdataset shards) as fast as possible. Step 1 toward using it downstream.

Benchmark box: **8 CPU cores**, **RTX 5070 Ti (16 GB, Blackwell sm_120)**, torchvision
0.27.1 (+cu130, nvjpeg `decode_jpeg` available), PIL 12.2.0 (stock, not SIMD), no DALI.
Data: 28 prepared val shards under `/mnt/ai/data/imagenet/imagenet_256/val` (~21.9k
images, 256px longest side JPEG q95). Background full-download paused (resumable) so it
doesn't steal CPU from the benchmark.

Note: datamodule lives at `src/chimera/data/imagenet.py` (reusable, exported from
`chimera.data`); this project dir holds the benchmark harness + this log.

---

## Session 1 — start

**Strategy (global, not local tweaks).** The pipeline per sample is: untar bytes →
PIL JPEG decode → resize shortest-side → center-crop → `to_tensor` (float32). On an
8-core box the JPEG **decode** dominates and CPU is the hard ceiling. The GPU is idle.
So the structural levers, in order of expected payoff:

1. **num_workers sweep** — find the CPU saturation point (baseline tuning).
2. **uint8 transfer + GPU cast** — workers emit uint8 CHW, not float32; cast/normalize
   on GPU. Cuts ~4× H2D bytes and per-sample CPU `div`.
3. **GPU JPEG decode (nvjpeg)** — workers just untar + ship raw `.jpg` bytes (nearly
   free); `decode_jpeg(device='cuda')` + GPU resize/crop. Moves the decode bottleneck
   off the 8 CPU lanes onto the idle GPU. Expected biggest win.

Plan: establish baseline worker sweep, then layer (2) and (3), keep the winner, fold
into the datamodule as an option. Measuring steady-state img/s (warmup batches skipped).

## Session 1 — ~5 min: baseline sweep (size=224, bs=256, 48 measured batches)

| variant | best img/s | @workers | vs pil |
|---------|-----------:|:--------:|:------:|
| pil (current, float32) | 1268 | 8 | 1.00× |
| uint8 (CPU decode, GPU cast) | 1948 | 6 | 1.54× |
| gpu (nvjpeg decode+resize) | 2087 | 2 | 1.65× |

Full grid (img/s): pil 2→709 4→1022 6→1210 8→1269; uint8 2→1089 4→1648 6→1948
8→1720; gpu 2→2087 4→2069 6→1801 8→1791.

**Reading.** (1) `pil` scales with workers → hard CPU-bound, 8 cores is the ceiling.
(2) `uint8` (emit uint8 CHW, normalize on GPU) is a clean +54% — less per-sample CPU
(`pil_to_tensor` vs `to_tensor` float div) and ~4× less H2D. Peaks at 6 workers; 8
contends with the main proc. (3) `gpu` (workers only untar+ship raw bytes, nvjpeg
decode + GPU resize/crop) is fastest at **2 workers** and *decreases* with more → decode
is off-CPU, the remaining serial cost is GPU-side (decode + per-image resize loop).

**Caveat for downstream use:** `gpu` decode competes with the model for GPU cycles
during real training; `uint8` keeps the GPU free. Will report both.

**Next:** optimize the gpu path (per-image resize loop, batch size), and confirm uint8
as the no-GPU-contention winner.

## Session 1 — ~15 min: pivot to real target (size=128, **bf16**)

User clarified: train at **128px, bf16**. Re-ran at that operating point.

| variant (size=128, bf16 out) | best img/s | @workers |
|------------------------------|-----------:|:--------:|
| pil float32→bf16 | 2230 | 6 |
| uint8→GPU bf16 cast | 2850 | 6 |
| uint8 + libjpeg `draft()` | 2890 | 6 |
| gpu nvjpeg decode+resize | 1908 | 2 |

**nvjpeg lost its edge at 128**: output is cheap, so the serial GPU resize-loop +
per-batch sync dominate — and it would contend with the model's GPU anyway. `draft()`
gave only +1.4% (most images fall below 128 on the short side at half-scale, so libjpeg
declines the 1/2 reduction). Winner so far: **uint8→GPU-bf16, CPU decode, GPU stays free.**

## Session 1 — ~22 min: THE structural win — store at train resolution

Decode cost is fixed by the **stored** JPEG size (256px), not the 128 target. Transcoded
8 shards to 128px (177MB → 69MB, 39%) and re-benchmarked uint8→bf16:

| source shards | w=4 | w=6 | w=8 |
|---------------|----:|----:|----:|
| 256px | 2420 | 2779 | 2104 |
| **128px** | 3996 | 5455 | **5700** |

**2.05× from matching stored↔train resolution**, and it now scales to all 8 cores
(decode got cheap enough that workers stop saturating). **5700 img/s = 4.5× the original
PIL/256px baseline (1268).** Disk also drops to 39%.

**Recommendation (folding into the datamodule):**
1. Prepare ImageNet at `--max-size 128` (or ~144 for random-crop headroom) for 128px
   training — biggest single win, do this for the full download.
2. DataModule emits **uint8** from workers; cast to **bf16 [0,1] on the GPU** in
   `on_after_batch_transfer` (mirrors `data/base.py`). Keeps the 8 cores on decode only
   and the GPU free for the model.
3. Default **num_workers=8**, pin_memory, persistent_workers, prefetch_factor=4.

## Session 1 — ~28 min: folded into the datamodule + end-to-end confirm

Applied to `src/chimera/data/imagenet.py`:
- `_to_square` now emits **uint8** (`pil_to_tensor`) instead of float32.
- New `on_after_batch_transfer` casts uint8 → **bf16 [0,1]** on the GPU (mirrors
  `data/base.py`). Output contract unchanged (still [0,1]), cast just moved off-CPU.
- Defaults: `max_size=128, image_size=128, num_workers=8`, `prefetch_factor=4`,
  `pin_memory`, `persistent_workers`, train `drop_last=True`.

**End-to-end through `ImageNetDataModule` (128px shards, bf16): 7678 img/s** — confirms
the win in the real loader (faster than the isolated 5700 once the page cache is warm and
H2D overlaps). Loader yields `(256,3,128,128) uint8`; after GPU cast `bf16`, range [0,1]. ✓

### Summary — what made it fast (global levers, ranked)
1. **Store at train resolution** (256px→128px shards): ~2.0× and disk →39%. Decode cost
   is set by stored JPEG size; this is the dominant lever.
2. **uint8 emit + GPU bf16 cast** (vs float32 in workers): ~1.5×, GPU stays free.
3. **num_workers=8 + prefetch**: saturates the 8 cores once decode is cheap.

Net: **1268 → 7678 img/s (~6×)** vs the original PIL/256px/float32 baseline at 128px.
nvjpeg GPU-decode was a strong lever at 224 (1.65×) but loses at 128 and would contend
with the model's GPU — kept as a documented option, not the default.

### Follow-ups (next session)
- Full set is being (re)prepared at `--max-size 128`. If random-crop augmentation is
  wanted later, prepare at ~144 instead (small speed/disk cost) for crop headroom.
- nvjpeg path could be revisited for >=224 training (batch the resize to kill the loop).

## Session 1 — follow-up: store at 144px for crop headroom

User opted for `--max-size 144` (train 128) so the train split can random-crop with real
spatial headroom instead of just center-cropping. Implemented in `data/imagenet.py`:
- **Train split**: `RandomResizedCrop`→128 (`scale=(0.65,1.0)`, `ratio=(3/4,4/3)`; mild
  scale because the 144px source is small). Verified crops differ across passes.
- **Val/test**: deterministic resize-shortest-side + center-crop→128 (unchanged).
- Defaults now `max_size=144, image_size=128`; prepare `--max-size` default → 144.

Throughput @144px source (val, center-crop, uint8→bf16, 8 workers): **4700 img/s**
(vs 7678 at 128px source — 144 decodes ~20% more pixels). Still ~3.7× the original
PIL/256/float32 baseline and far above what a 128px model step consumes. Disk for 144px
≈ ~14 GB train + ~0.5 GB val.

Full set re-downloading at 144px. `scale`/`ratio` are constructor args if you want to
tune the train aug. (Stale partial `imagenet_128`/`imagenet_256` dirs removed.)

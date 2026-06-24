---
name: optimize-training-throughput
description: Profile and optimize the throughput (img/s, steps/s) of a PyTorch model or training step. Use when asked to speed up training, make a model train faster, find the training bottleneck, benchmark a training step, or diagnose whether training is GPU-bound, CPU/launch-bound, or dataloader-bound.
---

Optimize the throughput of a PyTorch training step on a single GPU. You drive it with
`.claude/skills/optimize-training-throughput/throughput_probe.py` — wrap the target's one training
iteration in a zero-arg `step()` closure, and the probe classifies the bottleneck and measures
img/s. **The first move is always to classify the bound — never optimize before you have.**

All paths below are relative to the repo root (`/root/Code/chimera`). Run Python with `uv run python`.

## Prerequisites

Torch + CUDA, already in this repo's env (`uv`). Verify:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

No `apt-get` needed for the probe (it's pure torch). `nsys`/`ncu` exist at `/usr/local/cuda/bin/`
but are **not** the primary tool here — see Gotchas for why.

## Run (agent path) — classify the bound first

The driver is both a library and a CLI. Self-contained demos prove it works on any step:

```bash
cd /root/Code/chimera/.claude/skills/optimize-training-throughput
# self-contained conv-AE on synthetic data (no repo deps) — eager vs compiled:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python throughput_probe.py --demo synthetic --compile off --steps 40 --warmup 15
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python throughput_probe.py --demo synthetic --compile reduce-overhead --steps 40 --warmup 20
# the real chimera celeba_afhq autoencoder step (needs its REPA cache; see that project's run path):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python throughput_probe.py --demo autoencoder --compile reduce-overhead --steps 40 --warmup 20
```

To profile **your** model, import the library and pass a `step()` closure over a fixed on-device batch:

```bash
uv run python - <<'PY'
import torch
from throughput_probe import measure, dataloader_ceiling   # run from the skill dir, or add it to sys.path
dev = torch.device("cuda")
net = torch.nn.Conv2d(3, 32, 3, 1, 1).to(dev); opt = torch.optim.AdamW(net.parameters(), foreach=True)
x = torch.rand(64, 3, 128, 128, device=dev)
def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = net(x).square().mean()
    loss.backward(); opt.step()
measure(step, batch_size=64, steps=30, warmup=10)   # -> img/s + VERDICT
def get(): return x                                  # closure that yields a device batch (no compute)
dataloader_ceiling(get, batch_size=64, steps=30, warmup=10)
PY
```

`measure()` reports three timings and a **VERDICT**:
- **GPU-event ≈ async-wall → GPU-BOUND.** The stream is busy the whole step. Compile/launch tricks
  won't help; attack what the kernels *do* (precision, memory traffic) or shrink the model. Then
  check `roofline` MFU: GPU-bound + single-digit MFU = **memory-bandwidth bound** (norm/activation
  heavy) — expected for conv autoencoders, little compute headroom.
- **GPU-event ≪ async-wall → CPU/LAUNCH-BOUND.** The GPU starves between kernels. Fuse/compile,
  cudagraph, cut kernel count, remove eager Python from the step.
- **real-step img/s ≈ `dataloader_ceiling` → DATALOADER-BOUND.** More workers, in-memory cache,
  lighter transforms, or precompute frozen-network targets offline (see Playbook).

## Playbook — the levers, cheapest first (all measured this session)

Re-run `measure()` after each change; keep only what moves img/s. The chimera AE went **712 → 988
img/s (+39%)** via, in order:

1. **`channels_last`** on the model + inputs (`model.to(memory_format=torch.channels_last)`, inputs
   `.to(memory_format=...)`). NHWC convs hit faster cuDNN kernels. Free, numerically identical.
2. **`foreach=True` AdamW** (NOT `fused=True` — see Gotchas). Batches the optimizer into a few kernels.
3. **bf16-native frozen networks.** A frozen perceptual/feature net (LPIPS, DINO) under autocast runs
   fp32 weights with a bf16 cast inserted at *every* op boundary — hundreds of no-op `bfloat16_copy`
   kernels/step. Cast the frozen net to bf16 once and call it with `autocast(enabled=False)` + bf16
   inputs. Numerically equivalent (autocast already ran it bf16).
4. **Precompute frozen-network targets offline.** If augmentation is deterministic (or absent), a
   frozen target net (e.g. DINO for a REPA/distill loss) computes the *same* output every epoch —
   run it once, memmap the results, load with the batch. Removes the net from the loop entirely
   (its GPU time *and* its launches). Biggest single win here (+19%).
5. **`torch.compile`** — `reduce-overhead` (cudagraphs) or `default`. For a launch-bound step, fuse
   the **whole** step (model fwd + losses + projector) into ONE compiled function, not per-module
   islands, to avoid cudagraph↔eager seams. Inline any compile-hostile loss (see Gotchas).
6. **Only if still short and quality is negotiable** (these change the trained result — confirm with
   the user): shrink model width, lower resolution, drop/decimate auxiliary losses, fp8.

## Gotchas (the battle scars — these cost hours)

- **nsys "GPU busy %" lies for cudagraph/async workloads.** Computing busy as a union of kernel
  intervals over per-step NVTX windows under-counts asynchronously-executed cudagraph kernels (the
  CPU races ahead; kernels run outside the window). This reported a GPU-bound step as "65% idle /
  launch-bound" and sent a whole investigation the wrong way. **Trust CUDA events vs wall** (what the
  probe does) or batch-scaling, not NVTX-window busy%.
- **`fused=True` AdamW is incompatible with Lightning gradient clipping** ("fused optimizer does
  internal unscaling" RuntimeError). It only fires on a *clipped* run, so throughput benchmarks
  (which don't clip) hide it — a real run then crashes. Use `foreach=True`.
- **torchmetrics LPIPS is compile-hostile.** Its `_LPIPS.forward` returns a NamedTuple and calls the
  backbone on two tensors of differing `requires_grad` → graph break + endless recompile. Reimplement
  the forward inline (scaling layer → per-slice channel-normalized squared feature diffs → 1x1 lin
  heads → spatial mean → sum) reusing the frozen submodules. Validate against the original.
- **Pure bf16 ≠ faster than autocast** for a memory-bound conv net (measured 1124 < 1154 img/s) —
  compile already handles the dtype conversions. Don't assume; measure.
- **Bigger batch doesn't help a bandwidth-saturated step** — img/s was flat 64→128 then OOM. Batch
  tuning only helps when you're launch/latency-bound with spare bandwidth.
- **cudagraph (reduce-overhead) reuses output memory.** Any tensor you keep past the next
  `cudagraph_mark_step_begin()` (e.g. scalars handed to a logger for epoch-end reduction) must be
  `.detach().clone()`d, or you read stale data.
- **`torch.compiler.cudagraph_mark_step_begin()`** must be called once at the top of each step for
  `reduce-overhead`/`max-autotune`. The probe and demos do this.

## Troubleshooting

- `CUDA out of memory` at larger batch: lower `--batch-size`; the probe holds one fixed batch + grads.
- Recompile storm / `graph break` spam: run with `TORCH_LOGS="graph_breaks,recompiles"` to find the
  offending op; inline or move it out of the compiled region.
- `measure()` shows wildly varying numbers: raise `--warmup` (compiled paths need ≥15–20 to finish
  tracing + cudagraph capture before timing).
- autoencoder demo `AssertionError: run repa_cache.py first`: that demo needs the precomputed REPA
  cache; use `--demo synthetic` to exercise the probe with no repo data.

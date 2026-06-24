"""Throughput probe: classify a training step as GPU-bound / CPU-launch-bound / dataloader-bound.

This is the load-bearing diagnostic for optimizing training throughput. The trap it avoids:
nsys "GPU busy %" computed as a union of kernel intervals over per-step NVTX windows UNDER-counts
asynchronously-executed cudagraph kernels (the CPU races ahead, kernels run outside the window) --
it can report a GPU-bound step as "65% idle / launch-bound" and send you optimizing the wrong thing.
The reliable signal is CUDA events vs wall time: if GPU-event time == wall, the GPU stream is busy
the whole step -> GPU-bound; if event << wall, the CPU can't feed it -> launch/CPU-bound.

Usage as a library (the intended path -- wrap YOUR training step in a zero-arg `step()` closure):

    from throughput_probe import measure, dataloader_ceiling, roofline
    def step():                      # one full train iter on a FIXED on-device batch
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x); loss = loss_fn(loss, x)
        loss.backward(); opt.step()
    measure(step, batch_size=64)     # -> prints verdict + img/s, returns dict

Self-contained demos (no repo deps; what this file runs as a CLI):

    python throughput_probe.py --demo synthetic --compile reduce-overhead
    python throughput_probe.py --demo autoencoder   # the chimera celeba_afhq AE, real worked example
"""

from __future__ import annotations

import argparse
import time


def measure(step, *, batch_size: int, steps: int = 50, warmup: int = 25, label: str = "") -> dict:
    """Run `step` (a zero-arg closure doing one full train iteration on a fixed, already-on-device
    batch) and classify the bottleneck. Returns a dict; also prints a one-line verdict.

    Three timings, all over the same `step`:
      - async wall : CPU loops issuing steps, ONE sync at the end (real steady-state throughput).
      - GPU event  : a CUDA event pair around each step, summed (time the GPU stream is busy).
      - synced wall: sync after every step (CPU and GPU can't overlap -> upper bound).
    GPU-event ~= async-wall  => GPU-BOUND (the stream is busy the whole step; compile/launch tricks
    won't help -- attack the compute/memory the kernels do, or shrink the model).
    GPU-event << async-wall  => CPU/LAUNCH-BOUND (GPU starves between kernels; fuse/compile/cudagraph,
    cut kernel count, remove eager Python in the step)."""
    import torch

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    async_wall = (time.perf_counter() - t0) / steps

    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
    for s, e in evs:
        s.record(); step(); e.record()
    torch.cuda.synchronize()
    gpu = sum(s.elapsed_time(e) for s, e in evs) / steps / 1e3

    t0 = time.perf_counter()
    for _ in range(steps):
        step(); torch.cuda.synchronize()
    synced_wall = (time.perf_counter() - t0) / steps

    gpu_bound = gpu >= 0.85 * async_wall
    verdict = "GPU-BOUND" if gpu_bound else "CPU/LAUNCH-BOUND"
    img_s = batch_size / async_wall
    print(f"\n[{label or 'step'}]  {img_s:,.0f} img/s   ({async_wall*1e3:.2f} ms/step)")
    print(f"  GPU-event time : {gpu*1e3:6.2f} ms/step   <- GPU stream busy")
    print(f"  async wall     : {async_wall*1e3:6.2f} ms/step   <- steady-state throughput basis")
    print(f"  synced wall    : {synced_wall*1e3:6.2f} ms/step")
    print(f"  VERDICT: {verdict}  (GPU-event is {100*gpu/async_wall:.0f}% of wall; "
          f"{'attack kernel compute/memory or shrink model' if gpu_bound else 'fuse/compile to cut launch overhead'})")
    return dict(img_s=img_s, gpu_ms=gpu * 1e3, async_ms=async_wall * 1e3,
                synced_ms=synced_wall * 1e3, gpu_bound=gpu_bound, verdict=verdict)


def dataloader_ceiling(get_to_gpu, *, batch_size: int, steps: int = 50, warmup: int = 10) -> float:
    """Throughput ceiling of the input pipeline alone: `get_to_gpu()` pulls the next batch and moves
    it to the device, NO compute. If your real step's img/s ~= this ceiling, you're DATALOADER-BOUND
    (more workers / in-memory cache / lighter transforms). If it's far below, the loader isn't the
    limit. Returns img/s."""
    import torch

    for _ in range(warmup):
        get_to_gpu()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        get_to_gpu()
    torch.cuda.synchronize()
    ceil = batch_size * steps / (time.perf_counter() - t0)
    print(f"  dataloader-only ceiling: {ceil:,.0f} img/s")
    return ceil


def roofline(model, example_input, *, batch_size: int, peak_tflops: float) -> None:
    """Forward FLOPs of one batch (FlopCounterMode counts conv/matmul-class ops only) -> step FLOPs
    (~3x fwd) -> MFU vs the GPU's bf16 peak. GPU-BOUND + low MFU (single-digit %) = MEMORY-bandwidth
    bound (norm/activation/elementwise heavy, few big matmuls) -- expected for conv autoencoders;
    don't chase compute tricks. VERIFY peak_tflops for your card (e.g. RTX 5070 Ti ~177 bf16)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        model(example_input)
    fwd = fc.get_total_flops()
    step_flops = 3 * fwd
    print(f"  roofline: fwd {fwd/1e9:.1f} GFLOP -> step ~{step_flops/1e9:.1f} GFLOP (3x); "
          f"peak {peak_tflops:.0f} TFLOPS bf16 (VERIFY)")
    return step_flops


# ----------------------------------------------------------------------------- demos

def build_synthetic_step(compile_mode: str, batch_size: int):
    """A self-contained conv-autoencoder train step on synthetic data -- no repo deps. Mirrors the
    memory-bound conv case (GroupNorm/SiLU heavy). Returns (step, batch_size, model, x, peak)."""
    import torch
    from torch import nn

    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    def down(ci, co):  # halve spatial
        return nn.Sequential(nn.Conv2d(ci, co, 3, 2, 1), nn.GroupNorm(min(32, co), co), nn.SiLU())

    def up(ci, co):  # double spatial (upsample + stride-1 conv)
        return nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(ci, co, 3, 1, 1),
                             nn.GroupNorm(min(32, co), co), nn.SiLU())

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(down(3, 32), down(32, 64), down(64, 128))  # 128->16
            self.dec = nn.Sequential(up(128, 64), up(64, 32), up(32, 16),       # 16->128
                                     nn.Conv2d(16, 3, 3, 1, 1), nn.Sigmoid())

        def forward(self, x):
            return self.dec(self.enc(x))

    model = AE().to(dev, memory_format=torch.channels_last).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    x = torch.rand(batch_size, 3, 128, 128, device=dev).to(memory_format=torch.channels_last)
    fwd = model
    cuda_graphs = compile_mode in ("reduce-overhead", "max-autotune")
    if compile_mode != "off":
        fwd = torch.compile(model, mode=compile_mode)

    def step():
        if cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon = fwd(x)
            loss = torch.nn.functional.mse_loss(recon, x)
        loss.backward()
        opt.step()

    return step, batch_size, model, x


def build_autoencoder_step(compile_mode: str, batch_size: int):
    """The REAL chimera celeba_afhq autoencoder step (the worked example this skill was built from).
    Requires the precomputed REPA cache (see that project's run path). Returns (step, batch_size)."""
    import os
    import sys

    import torch
    import torch.nn.functional as F

    here = os.path.dirname(os.path.abspath(__file__))
    ae_dir = os.path.normpath(os.path.join(here, "..", "..", "..", "projects", "celeba_afhq", "autoencoder"))
    sys.path.insert(0, ae_dir)
    from repa_cache import cache_exists, repa_paths  # noqa: E402
    from train import LitAutoEncoder, build_datamodule, build_model_config  # noqa: E402

    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    assert cache_exists("/mnt/ai/data", 128, "facebook/dinov2-small", 224), "run repa_cache.py first"
    paths = repa_paths("/mnt/ai/data", 128, "facebook/dinov2-small", 224)
    m = LitAutoEncoder(build_model_config(32), image_size=128, lpips_net="squeeze",
                       lpips_weight=0.1, repa_weight=0.5, repa_model="facebook/dinov2-small",
                       repa_dino_size=224)
    m.repa_precomputed = True
    m.to(dev).train(); m._ensure_metrics()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, foreach=True)
    fwd = torch.compile(m._forward_loss, mode=compile_mode) if compile_mode != "off" else m._forward_loss
    cuda_graphs = compile_mode in ("reduce-overhead", "max-autotune")
    dm = build_datamodule(data_dir="/mnt/ai/data", image_size=128, batch_size=batch_size,
                          num_workers=7, repa_paths=paths)
    dm.drop_last = True; dm.prepare_data(); dm.setup("fit")
    b = next(iter(dm.train_dataloader()))
    x = b[0].to(dev).float().to(memory_format=torch.channels_last)
    tgt = b[2].to(dev)

    def step():
        if cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, *_ = fwd(x, tgt)
        loss.backward()
        opt.step()

    return step, batch_size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", choices=["synthetic", "autoencoder"], default="synthetic")
    p.add_argument("--compile", default="reduce-overhead",
                   choices=["off", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--peak-tflops", type=float, default=177.0, help="GPU bf16 peak for MFU (VERIFY)")
    args = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  demo={args.demo}  compile={args.compile}  batch={args.batch_size}")

    if args.demo == "synthetic":
        step, bs, model, x = build_synthetic_step(args.compile, args.batch_size)
        measure(step, batch_size=bs, steps=args.steps, warmup=args.warmup, label=f"synthetic/{args.compile}")
        roofline(model, x, batch_size=bs, peak_tflops=args.peak_tflops)
    else:
        step, bs = build_autoencoder_step(args.compile, args.batch_size)
        measure(step, batch_size=bs, steps=args.steps, warmup=args.warmup, label=f"autoencoder/{args.compile}")


if __name__ == "__main__":
    main()

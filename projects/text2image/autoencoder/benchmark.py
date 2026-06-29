"""Profile the CelebA-HQ+AFHQ autoencoder training step: throughput + bottleneck.

Reuses the *real* model/datamodule/loss construction from ``train.py`` (``build_model_config``,
``build_datamodule``, ``LitAutoEncoder``, ``compile_model``) so it measures the actual training
step -- fp32 input + bf16-mixed autocast, MSE + LPIPS, AdamW -- not a reimplementation.

What it reports:
  1. Steady-state images/sec for the full step (forward + LPIPS + backward + optimizer),
     warmup excluded, timed with torch.cuda.synchronize() around the loop.
  2. Per-stage breakdown (eager only): dataloader+H2D / forward / LPIPS / backward / optimizer,
     plus a starvation test -- real loader vs a cached constant batch vs dataloader-only.
  3. Roofline: model FLOPs/step (FlopCounterMode) vs the GPU's bf16 peak -> MFU%, and the
     dataloader-only throughput ceiling; states which ceiling we hit.
  4. Compiled graph: with --compile, dumps graph breaks + inductor output_code to a file.

Examples
--------
    uv run python projects/text2image/autoencoder/benchmark.py --image-size 128 --batch-size 16 \
        --num-workers 4 --compile off --steps 30 --warmup 8
    uv run python projects/text2image/autoencoder/benchmark.py --compile reduce-overhead --steps 40
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from chimera.models import ConvAutoEncoder
from chimera.utils.experiment import CUDA_GRAPH_COMPILE_MODES, compile_model

# Reuse the exact construction from the training script (co-located; same dir on sys.path).
from train import LATENT_CHANNELS, LitAutoEncoder, build_datamodule, build_model_config

# RTX 5070 Ti (Blackwell, sm_120) bf16 dense w/ fp32 accumulate -- a rough estimate ONLY.
# Override with --peak-tflops using your card's actual spec; MFU is meaningless otherwise.
DEFAULT_PEAK_TFLOPS = 88.0


def cycle(loader):
    """Yield batches forever, restarting the loader when a pass ends."""
    while True:
        for batch in loader:
            yield batch


def move(batch, device):
    """Move a batch to the device (H2D; async only if pinned). Handles both (images, labels)
    and the precomputed-REPA (images, labels, repa_target) layout."""
    return tuple(t.to(device, non_blocking=True) for t in batch)


def make_step_fn(module, opt, lpips, lpips_weight, repa_weight, device, *, mark_step=False):
    """The canonical training step on an already-on-device batch -- mirrors
    LitAutoEncoder._forward_loss + training_step (minus the Trainer-bound self.log).

    Crucial fidelity detail: the real code casts ``images.float()`` BEFORE the model, then
    runs bf16-mixed autocast inside -- so the model sees fp32 input, not the loader's bf16.
    With ``repa_weight > 0`` the model returns the latent too and the frozen-DINOv2 REPA term
    is added, exactly as in training.

    ``mark_step`` marks a new CUDA-graph iteration each step (required by cudagraph-trees under
    reduce-overhead/max-autotune so a replay may reuse the previous step's output memory)."""
    model = module.model

    def step(gpu_batch):
        if mark_step:
            torch.compiler.cudagraph_mark_step_begin()
        images = gpu_batch[0]
        repa_target = gpu_batch[2] if len(gpu_batch) > 2 else None  # precomputed REPA target
        x = images.float().to(memory_format=torch.channels_last)  # NHWC, matches train.py
        assert x.dtype == torch.float32  # guard the fp32-input + autocast path
        with torch.autocast(device.type, dtype=torch.bfloat16):
            if repa_weight > 0:
                recon, z = model(x, return_latent=True)
            else:
                recon, z = model(x), None
            loss = F.mse_loss(recon, x) + lpips_weight * lpips(recon, x)
            if repa_weight > 0:
                loss = loss + repa_weight * module._repa(z, x, repa_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    return step


def time_loop(iter_fn, steps, warmup, *, per_step=True):
    """Run warmup (discarded) then `steps` timed iterations. Wall time is measured with one
    synchronize() before and after the loop (captures CPU-side dataloader stalls too); the
    optional per-step CUDA events capture GPU-side step time (async, synced once at the end)."""
    for _ in range(warmup):
        iter_fn()
    torch.cuda.synchronize()
    evs = (
        [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(steps)
        ]
        if per_step
        else None
    )
    t0 = time.perf_counter()
    for i in range(steps):
        if evs:
            evs[i][0].record()
        iter_fn()
        if evs:
            evs[i][1].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gpu_ms = [s.elapsed_time(e) for s, e in evs] if evs else []
    return wall, gpu_ms


def measure_per_stage_eager(loader_iter, step_inputs, steps, device):
    """Serialized per-substage timing on the REAL loader (eager only). A synchronize() after
    each substage means the reported times are an upper-bound decomposition whose sum exceeds
    the fused end-to-end time -- it shows where the work is, not an additive budget."""
    module, opt, lpips, w, repa_w = step_inputs
    model = module.model
    acc = dict(
        dataloader=0.0, forward=0.0, lpips=0.0, repa=0.0, backward=0.0, optimizer=0.0
    )

    def tick():
        torch.cuda.synchronize()
        return time.perf_counter()

    for _ in range(steps):
        t = tick()
        gpu_batch = move(next(loader_iter), device)
        images = gpu_batch[0]
        repa_target = gpu_batch[2] if len(gpu_batch) > 2 else None
        x = images.float().to(memory_format=torch.channels_last)  # NHWC, matches train.py
        acc["dataloader"] += tick() - t

        t = tick()
        z = None
        with torch.autocast(device.type, dtype=torch.bfloat16):
            if repa_w > 0:
                recon, z = model(x, return_latent=True)
            else:
                recon = model(x)
        acc["forward"] += tick() - t

        t = tick()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            lp = lpips(recon, x)
            loss = (
                F.mse_loss(recon, x) + w * lp
            )  # mse is negligible; folded into the LPIPS stage
        acc["lpips"] += tick() - t

        t = tick()
        if repa_w > 0:  # (frozen DINOv2 forward when not precomputed) + projector + cosine
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = loss + repa_w * module._repa(z, x, repa_target)
        acc["repa"] += tick() - t

        t = tick()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        acc["backward"] += tick() - t

        t = tick()
        opt.step()
        acc["optimizer"] += tick() - t

    return {k: v / steps * 1e3 for k, v in acc.items()}  # mean ms per stage


def count_forward_flops(model_config, image_size, batch_size, device):
    """Forward FLOPs of one batch through a FRESH eager fp32 ConvAutoEncoder (FlopCounterMode
    bypasses autocast/compiled regions, so count uncompiled & fp32). Counts conv/matmul-class
    ops only -- GroupNorm/SiLU/pixel-shuffle are not counted, so this is a lower bound."""
    m = ConvAutoEncoder(**model_config).to(device).eval()
    x = torch.randn(
        batch_size, model_config["input_dim"], image_size, image_size, device=device
    )
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        m(x)
    flops = fc.get_total_flops()
    del m, x
    torch.cuda.empty_cache()
    return flops


def dump_compile_artifacts(
    model_config, image_size, batch_size, mode, device, out_path
):
    """Write graph-break summary + inductor output_code for the compiled forward to a file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = ConvAutoEncoder(**model_config).to(device).train()
    x = torch.randn(
        batch_size, model_config["input_dim"], image_size, image_size, device=device
    )
    cuda_graphs = mode in CUDA_GRAPH_COMPILE_MODES
    summary = {}
    with contextlib.suppress(Exception):
        expl = torch._dynamo.explain(m)(
            x
        )  # structured break info, before compiling in place
        summary = {
            "graph_count": expl.graph_count,
            "graph_break_count": expl.graph_break_count,
            "break_reasons": [str(r) for r in getattr(expl, "break_reasons", [])],
        }
    with (
        open(out_path, "w") as f,
        contextlib.redirect_stderr(f),
        contextlib.redirect_stdout(f),
    ):
        torch._logging.set_logs(output_code=True, graph_breaks=True, recompiles=True)
        try:
            m.compile(mode=mode)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                for _ in range(3):  # trigger trace + autotune + (cuda)graph capture
                    if cuda_graphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    m(x).sum().backward()
            torch.cuda.synchronize()
        except (
            Exception
        ) as e:  # the dump is auxiliary -- never let it sink the benchmark
            summary["dump_error"] = str(e).splitlines()[0][:160]
        finally:
            torch._logging.set_logs()  # reset handlers
    del m, x
    torch.cuda.empty_cache()
    return summary


def fmt_thru(name, imgs_per_sec, gpu_ms=None):
    extra = f"  (GPU step {statistics.median(gpu_ms):6.1f} ms median)" if gpu_ms else ""
    return f"  {name:<22} {imgs_per_sec:8.1f} img/s{extra}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=7)
    p.add_argument(
        "--compile",
        default="off",
        choices=["off", "default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument(
        "--steps", type=int, default=50, help="timed steps (warmup excluded)"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="discarded warmup steps (>=3 for compile)",
    )
    p.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_PEAK_TFLOPS,
        help="GPU bf16 peak TFLOPS for MFU -- VERIFY for your card; default is an estimate",
    )
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lpips-net", choices=["vgg", "alex", "squeeze"], default="alex")
    p.add_argument("--lpips-weight", type=float, default=0.1)
    p.add_argument(
        "--repa-weight",
        type=float,
        default=0.5,
        help="REPA latent-alignment weight; 0 disables the frozen DINOv2 forward",
    )
    p.add_argument("--repa-model", default="facebook/dinov2-small")
    p.add_argument("--repa-dino-size", type=int, default=224)
    p.add_argument(
        "--repa-cache",
        choices=["auto", "off"],
        default="auto",
        help="'auto' uses precomputed DINOv2 targets if present (no in-loop DINOv2); 'off' runs DINOv2",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("benchmark requires a CUDA GPU")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")  # match build_trainer
    cuda_graphs = args.compile in CUDA_GRAPH_COMPILE_MODES

    # --- build the real module/datamodule/optimizer (reused from train.py) -------------------
    model_config = build_model_config(args.base_channels)
    module = LitAutoEncoder(
        model_config,
        image_size=args.image_size,
        lr=args.lr,
        lpips_weight=args.lpips_weight,
        lpips_net=args.lpips_net,
        repa_weight=args.repa_weight,
        repa_model=args.repa_model,
        repa_dino_size=args.repa_dino_size,
    )
    # Use precomputed REPA targets when available (matches train.py's default path): skip the
    # in-loop DINOv2 and feed cached targets through the batch. Must be set before _ensure_metrics.
    repa_paths = None
    if args.repa_weight > 0 and args.repa_cache != "off":
        from repa_cache import cache_exists
        from repa_cache import repa_paths as _repa_paths

        ck = dict(data_dir=args.data_dir, image_size=args.image_size,
                  repa_model=args.repa_model, repa_dino_size=args.repa_dino_size)
        if cache_exists(**ck):
            repa_paths = _repa_paths(**ck)
            module.repa_precomputed = True
            print("[repa-cache] benchmarking the precomputed-target path (no in-loop DINOv2)")
    module.to(device).train()
    module._ensure_metrics()  # builds LPIPS (+ FID unused here, + DINOv2 when REPA is on)
    # Mirror training's LPIPS exactly: LitAutoEncoder._lpips calls the net directly rather than
    # the stateful metric (whose forward appends to an unbounded `all_scores` list that would
    # inflate the very per-step timing this benchmark measures). Reusing it keeps the benchmark
    # and the real training step from drifting apart.
    lpips = module._lpips
    opt = torch.optim.AdamW(
        module.parameters(), lr=args.lr, foreach=True
    )  # mirrors configure_optimizers (foreach; fused is incompatible with Lightning grad-clip)
    if args.compile != "off":
        compile_model(module, args.compile)

    datamodule = build_datamodule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        repa_paths=repa_paths,
    )
    datamodule.drop_last = (
        cuda_graphs  # static shape for CUDA graphs (mirrors run_training)
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    step = make_step_fn(
        module,
        opt,
        lpips,
        args.lpips_weight,
        args.repa_weight,
        device,
        mark_step=cuda_graphs,
    )
    bs, steps, warmup = (
        args.batch_size,
        args.steps,
        max(args.warmup, 3 if cuda_graphs else 0),
    )
    thru = lambda wall: bs * steps / wall  # noqa: E731

    # --- 1+2. throughput: real loader / cached compute / dataloader-only ---------------------
    real_iter = cycle(loader)
    real_wall, real_gpu = time_loop(
        lambda: step(move(next(real_iter), device)), steps, warmup
    )

    cached = move(next(cycle(loader)), device)  # one constant batch, kept on GPU
    cached_wall, cached_gpu = time_loop(lambda: step(cached), steps, warmup)

    dl_iter = cycle(loader)
    dl_wall, _ = time_loop(
        lambda: move(next(dl_iter), device), steps, warmup, per_step=False
    )

    real_thru, cached_thru, dl_thru = thru(real_wall), thru(cached_wall), thru(dl_wall)

    # --- 3. FLOPs / roofline -----------------------------------------------------------------
    fwd_flops = count_forward_flops(model_config, args.image_size, bs, device)
    step_flops = 3 * fwd_flops  # 1 fwd + ~2 bwd
    peak = args.peak_tflops * 1e12
    mfu_real = 100 * step_flops * (real_thru / bs) / peak
    mfu_cached = 100 * step_flops * (cached_thru / bs) / peak

    # --- per-stage breakdown (eager only) ----------------------------------------------------
    stages = None
    if args.compile == "off":
        stages = measure_per_stage_eager(
            cycle(loader),
            (module, opt, lpips, args.lpips_weight, args.repa_weight),
            max(steps // 2, 5),
            device,
        )

    # --- 4. compiled graph dump --------------------------------------------------------------
    dump_path, dump_summary = None, None
    if args.compile != "off":
        dump_path = (
            Path(__file__).parent
            / "outputs"
            / "benchmark"
            / f"compile_{args.compile}.log"
        )
        dump_summary = dump_compile_artifacts(
            model_config, args.image_size, bs, args.compile, device, dump_path
        )

    # --- starvation verdict ------------------------------------------------------------------
    starved = real_thru < 0.9 * cached_thru and real_thru <= 1.15 * dl_thru
    bound = "DATALOADER-BOUND (GPU starved)" if starved else "COMPUTE-BOUND"

    # ============================ REPORT =====================================================
    print("\n" + "=" * 78)
    print("  CelebA-HQ+AFHQ autoencoder — training-step benchmark")
    print("=" * 78)
    print(
        f"  image={args.image_size}  batch={bs}  workers={args.num_workers}  "
        f"compile={args.compile}  steps={steps}  warmup={warmup}"
    )
    print(
        f"  base_channels={args.base_channels}  lpips_net={args.lpips_net}  "
        f"latent={LATENT_CHANNELS}x{args.image_size // 8}x{args.image_size // 8}"
    )
    repa_desc = (
        f"{args.repa_model} @ {args.repa_dino_size}px (w={args.repa_weight})"
        if args.repa_weight > 0
        else "off"
    )
    print(f"  repa: {repa_desc}")
    print("  precision: fp32 input + bf16-mixed autocast (matches train.py)")

    print("\n  THROUGHPUT")
    print(fmt_thru("real loader (step)", real_thru, real_gpu))
    print(fmt_thru("cached batch (compute)", cached_thru, cached_gpu))
    print(fmt_thru("dataloader-only", dl_thru))
    print(f"\n  VERDICT: {bound}")
    print(
        f"           real={real_thru:.0f}  cached-compute={cached_thru:.0f}  "
        f"dataloader-only={dl_thru:.0f} img/s"
    )

    if stages is not None:
        total = sum(stages.values())
        print(
            "\n  PER-STAGE (eager, serialized — sum > end-to-end; shows where work is)"
        )
        for k in ("dataloader", "forward", "lpips", "repa", "backward", "optimizer"):
            print(f"    {k:<12} {stages[k]:7.2f} ms  ({100 * stages[k] / total:4.1f}%)")
    elif args.compile != "off":
        print(
            "\n  PER-STAGE: skipped (can't sync inside a CUDA graph; run --compile off for it)"
        )

    print("\n  ROOFLINE  (AE conv/matmul FLOPs only; excludes LPIPS + REPA/DINOv2; lower bound)")
    print(
        f"    fwd {fwd_flops / 1e9:.1f} GFLOP  ->  step ~{step_flops / 1e9:.1f} GFLOP (3x)"
    )
    print(f"    peak {args.peak_tflops:.0f} TFLOPS bf16 (VERIFY for your GPU)")
    print(
        f"    MFU: {mfu_real:.1f}% at real throughput, {mfu_cached:.1f}% compute-bound ceiling"
    )

    if dump_summary is not None:
        print("\n  COMPILED GRAPH")
        print(
            f"    graphs={dump_summary.get('graph_count', '?')}  "
            f"breaks={dump_summary.get('graph_break_count', '?')}"
        )
        for r in dump_summary.get("break_reasons", [])[:3]:
            print(f"      break: {r[:90]}")
    if dump_path is not None:
        print(f"    fusion/output_code dump -> {dump_path}")

    # --- one-paragraph diagnosis -------------------------------------------------------------
    print("\n  DIAGNOSIS")
    print("  " + "-" * 76)
    if starved:
        msg = (
            f"The GPU is starved: real throughput ({real_thru:.0f} img/s) tracks the "
            f"dataloader-only ceiling ({dl_thru:.0f} img/s) and sits well below the "
            f"cached-compute rate ({cached_thru:.0f} img/s) — time goes to feeding data, "
            f"not the model. Highest-leverage fix: relieve the input pipeline — raise "
            f"--num-workers (currently {args.num_workers}), keep in_memory=True (avoid --mmap), "
            f"and ensure the materialized cache exists so no per-epoch resize happens."
        )
    else:
        lp = (
            f" LPIPS is {stages['lpips'] / sum(stages.values()) * 100:.0f}% of the step."
            if stages
            else ""
        )
        fix = (
            "enable torch.compile (currently off) for kernel fusion"
            if args.compile == "off"
            else "switch to channels_last memory format and/or grow the batch to raise utilization"
        )
        msg = (
            f"Compute-bound: real ({real_thru:.0f} img/s) ≈ cached-compute "
            f"({cached_thru:.0f} img/s), so the dataloader keeps up.{lp} MFU is "
            f"{mfu_cached:.1f}% of the {args.peak_tflops:.0f}-TFLOP roofline — a conv "
            f"autoencoder is GroupNorm/SiLU/pixel-shuffle heavy (memory-bound, few matmuls), "
            f"so low MFU is expected. Highest-leverage fix: {fix}."
        )
    # wrap to ~76 cols
    words, line = msg.split(), "  "
    for w in words:
        if len(line) + len(w) + 1 > 78:
            print(line)
            line = "  "
        line += w + " "
    print(line.rstrip())
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

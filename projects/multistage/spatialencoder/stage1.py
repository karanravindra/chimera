"""DEPRECATED — do NOT use for new experiments. Use `dcae.py` for ALL multistage stage-1
runs so every result goes through one comparable pipeline. `dcae.py` subsumes this
single-phase base-AE trainer: for a base-AE-only run set `--p2-time 0 --p3-time 0`
(add `--p1-res 128` to train at the target resolution instead of the low-res phase).
Kept only as the reference reproduction of the original 128px champion recipe.
"""

import argparse
import csv
import math
import time
from pathlib import Path

from tqdm import tqdm
import torch
from torchinfo import summary
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from chimera.data import AFHQDataModule
from chimera.models import ConvAutoEncoder
from chimera.utils.logtee import tee_to_logfile
from chimera.utils.seed import seed_everything


SEED = 0
DOWNSAMPLE = 8
LATENT_DIM = 16  # fixed interface for the multistage pipeline: 16ch @ 16x16 (12x compression)
LPIPS_NET = "alex"

METRIC_FIELDS = (
    "run",
    "epoch",
    "train_loss",
    "train_lpips",
    "train_psnr",
    "val_loss",
    "val_lpips",
    "val_psnr",
    "val_rfid",
)


RUN_WIDTH = 14  # left-justified; everything else is right-justified
EPOCH_WIDTH = 5
NUM_WIDTH = 10  # min width for numeric columns (header name wins if longer)


def _column_width(field: str) -> int:
    if field == "run":
        return RUN_WIDTH
    if field == "epoch":
        return EPOCH_WIDTH
    return max(len(field), NUM_WIDTH)


def _format_row(metrics: dict | None) -> str:
    """Render one aligned, ``", "``-separated row (header row if ``metrics`` is None)."""
    cells = []
    for field in METRIC_FIELDS:
        width = _column_width(field)
        if metrics is None:
            text = field
        elif field in ("run", "epoch"):
            text = str(metrics[field])
        else:
            value = metrics.get(field, "")
            text = "" if value == "" or value is None else f"{value:.6f}"
        cells.append(text.ljust(width) if field == "run" else text.rjust(width))
    return ", ".join(cells)


def log_metrics(csv_path: Path, metrics: dict) -> None:
    """Append one aligned row of train/val metrics to a CSV, writing the header once.

    Rows from every run share one CSV, distinguished by the leading ``run`` column.
    Columns are space-padded so the file is readable as a plain table while staying
    valid CSV (``", "`` separated; strip cells when parsing)."""
    write_header = not csv_path.exists()
    if not write_header:
        # Guard against silently appending rows under a stale header: if the
        # schema (METRIC_FIELDS) changed since the file was created, the columns
        # would no longer line up. Fail loudly instead of corrupting the CSV.
        with open(csv_path, newline="") as f:
            existing_header = [c.strip() for c in next(csv.reader(f), [])]
        if existing_header != list(METRIC_FIELDS):
            raise ValueError(
                f"{csv_path} header {existing_header} does not match current "
                f"schema {list(METRIC_FIELDS)}; remove or migrate the file first."
            )
    with open(csv_path, "a", newline="") as f:
        if write_header:
            f.write(_format_row(None) + "\n")
        f.write(_format_row(metrics) + "\n")


def cosine_warmup_lr(step: int, total_steps: int, base_lr: float,
                     warmup_steps: int, min_lr_frac: float) -> float:
    """Linear warmup (absolute step count) to base_lr, then cosine decay to
    min_lr_frac*base_lr. Progress is clamped to 1.0 so overshooting the planned
    horizon (time-budgeted runs) holds LR at the floor instead of oscillating."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="name for this run; tags its metrics rows and artifacts")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="max epochs (safety cap); use --time-budget to bound runs")
    parser.add_argument("--time-budget", type=float, default=270.0,
                        help="wall-clock seconds of training before stopping (excludes "
                             "compile/import). Overshoots by at most one partial epoch.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128,
                        help="train/eval resolution (128 is final; use 32/64 for cheap "
                             "low-res pretraining, then --resume at 128)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lpips-weight", type=float, default=0.1)
    parser.add_argument("--lpips-net", type=str, default=LPIPS_NET, choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--dim-mult", type=str, default="1,2,4",
                        help="per-block channel multipliers over base-channels")
    parser.add_argument("--layers-per-block", type=str, default="1,2,4")
    parser.add_argument("--dec-layers-per-block", type=str, default=None,
                        help="decoder res-blocks per block; defaults to encoder's (symmetric)")
    parser.add_argument("--pixel-loss", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--ffl-weight", type=float, default=0.0,
                        help="focal frequency loss weight (0 disables)")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint (last.pt) to resume model+optimizer from")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="run FID/val eval every N epochs (always on the last)")
    args = parser.parse_args()

    BASE_CHANNELS = args.base_channels
    LATENT_DIM = args.latent_dim
    LPIPS_WEIGHT = args.lpips_weight
    DIM_PER_BLOCK = tuple(BASE_CHANNELS * int(m) for m in args.dim_mult.split(","))
    LAYERS_PER_BLOCK = tuple(int(x) for x in args.layers_per_block.split(","))
    DEC_LAYERS_PER_BLOCK = (
        tuple(int(x) for x in args.dec_layers_per_block.split(","))
        if args.dec_layers_per_block else LAYERS_PER_BLOCK
    )
    assert len(DIM_PER_BLOCK) == len(LAYERS_PER_BLOCK)
    assert DOWNSAMPLE == 2 ** len(DIM_PER_BLOCK)

    seed_everything(SEED, deterministic=True)

    OUTPUT_DIR = Path(__file__).parent / "outputs"
    # Per-run images/checkpoints live in their own subdir so runs don't clobber each
    # other; metrics from all runs share one CSV, keyed by the `run` column.
    RUN_DIR = OUTPUT_DIR / args.run
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = tee_to_logfile(RUN_DIR / f"{args.run}.log")
    print(f"logging to {log_path}", flush=True)

    dm = AFHQDataModule(batch_size=args.batch_size, image_size=args.image_size, num_workers=7)
    dm.prepare_data()
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    model = ConvAutoEncoder(
        input_dim=3,
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        dim_per_block=DIM_PER_BLOCK,
        layers_per_block=LAYERS_PER_BLOCK,
        dec_layers_per_block=DEC_LAYERS_PER_BLOCK,
    )
    S = args.image_size
    summary(model, input_size=(1, 3, S, S), depth=2)

    print(
        f"Compression ratio: {(3 * S * S) / (LATENT_DIM * (S // DOWNSAMPLE) ** 2):.2f}x "
        f"(latent {LATENT_DIM}ch @ {S // DOWNSAMPLE}x{S // DOWNSAMPLE}, image {S}px)"
    )

    resume_ckpt = None
    resume_step = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_ckpt["model"])
        resume_step = resume_ckpt.get("global_step", 0)
        print(f"Resumed model weights from {args.resume} (prev global_step={resume_step})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        print("Restored optimizer state")
    total_steps = args.epochs * len(train_loader)
    criterion = torch.nn.L1Loss() if args.pixel_loss == "l1" else torch.nn.MSELoss()
    mse_metric = torch.nn.MSELoss()  # always track MSE for a comparable PSNR

    def focal_frequency_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Frequency Loss (Jiang et al., ICCV'21): weight the per-frequency
        squared spectral error by its own magnitude, so hard-to-synthesize (usually
        high) frequencies dominate. Recovers fine texture (AFHQ fur) that pixel/LPIPS
        losses blur. Computed in fp32 (rFFT) for numerical stability."""
        fr = torch.fft.rfft2(recon.float(), norm="ortho")
        ft = torch.fft.rfft2(target.float(), norm="ortho")
        diff = fr - ft
        dist = diff.real**2 + diff.imag**2
        weight = dist.detach() ** 1.0  # focal weight = squared spectral distance
        weight = weight / (weight.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        return (weight * dist).mean()
    # channels_last: hurts in eager for these small channel counts, but lets torch.compile
    # emit faster fused NHWC kernels (measured win once compiled).
    model = model.to("cuda", dtype=torch.bfloat16)
    model = model.to(memory_format=torch.channels_last)

    # The step is GPU-bound but memory-bandwidth bound (~4% MFU): dozens of tiny
    # conv/groupnorm/silu/pixel-shuffle kernels. reduce-overhead (cudagraphs) fuses them
    # and removes per-kernel launch overhead -> ~2360 -> ~3600 img/s (+52%) on this card.
    # Compile only the hot training path; eval/grid stay eager so FID/LPIPS never read a
    # reused cudagraph output buffer.
    compiled_model = torch.compile(model, mode="reduce-overhead")

    # rFID = FID between originals and their reconstructions, accumulated over the
    # whole eval set each epoch. Inception runs in fp32; normalize=True expects [0,1].
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to("cuda")

    # LPIPS perceptual loss: frozen AlexNet, fp32, eval-only. normalize=True expects
    # [0,1] (the AE head is sigmoid'd). The net is added to the MSE loss as a
    # weighted perceptual term and logged per epoch.
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=args.lpips_net, normalize=True)
    lpips.requires_grad_(False)  # input still gets gradients; net stays frozen
    lpips = lpips.to("cuda").eval()

    def lpips_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-batch LPIPS (differentiable w.r.t. ``recon``) via the underlying net.

        Calling the metric's ``forward`` would append every batch's score to an
        internal list that's never read or reset, so per-step cost climbs over the
        run; calling ``lpips.net`` directly gives the same value statelessly. Run in
        fp32 since the AlexNet weights are fp32. Call sites gate this on a nonzero
        LPIPS_WEIGHT, so the frozen net never runs when the perceptual term is off."""
        return lpips.net(recon.float(), target.float(), normalize=True).mean()

    metrics_path = OUTPUT_DIR / "metrics.csv"

    global_step = 0
    best_rfid = float("inf")
    # Time-budgeted training: `total_steps` (from the epoch cap) is only the initial
    # cosine horizon; once throughput is measured past the compile warmup, we recompute
    # `planned_total` so the schedule decays to its floor exactly at the time budget.
    planned_total = total_steps
    train_start = None      # wall-clock of the first step (excludes import/compile)
    calib_t0 = None
    CALIB_START, CALIB_END = 20, 70
    stop = False
    for epoch in range(args.epochs):
        model.train()

        # Accumulate metric sums on-device and read them back only periodically: a
        # per-step .item() would sync the host every iteration and stall the
        # reduce-overhead cudagraph pipeline (the CPU could no longer queue the next
        # step's replay while the GPU runs the current one).
        train_loss_sum = torch.zeros((), device="cuda")
        train_lpips_sum = torch.zeros((), device="cuda")
        train_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for step_idx, batch in enumerate(pbar):
            x, _ = batch

            images = x.to(
                "cuda", dtype=torch.bfloat16, memory_format=torch.channels_last
            )

            # Calibrate the cosine horizon to the wall-clock budget. Measure steady-state
            # throughput over steps [CALIB_START, CALIB_END] (past the compile warmup),
            # then set planned_total so LR reaches its floor right at --time-budget.
            now = time.perf_counter()
            if train_start is None:
                train_start = now
            if global_step == CALIB_START:
                calib_t0 = now
            elif global_step == CALIB_END and calib_t0 is not None:
                sps = (CALIB_END - CALIB_START) / (now - calib_t0)
                remaining = args.time_budget - (now - train_start)
                planned_total = global_step + max(1, int(sps * remaining))
                print(f"[calib] {sps:.1f} steps/s -> planned_total={planned_total} "
                      f"(~{planned_total / sps:.0f}s train)")

            lr = cosine_warmup_lr(global_step, planned_total, args.lr,
                                  args.warmup_steps, args.min_lr_frac)
            for g in optimizer.param_groups:
                g["lr"] = lr
            global_step += 1

            # reduce-overhead replays a captured cudagraph; mark the step so it knows
            # the previous step's buffers are free to reuse.
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()
            outputs = compiled_model(images)
            pixel = criterion(outputs, images)  # l1 or mse per --pixel-loss
            loss = pixel
            if LPIPS_WEIGHT:  # skip the frozen perceptual net entirely when disabled
                lpips_val = lpips_loss(outputs, images)
                loss = loss + LPIPS_WEIGHT * lpips_val
            if args.ffl_weight:
                loss = loss + args.ffl_weight * focal_frequency_loss(outputs, images)
            loss.backward()
            optimizer.step()

            # always track MSE (not the pixel/combined loss) so PSNR stays a pure,
            # comparable reconstruction metric. .detach() so the running sums don't
            # retain the autograd graph.
            bs = images.size(0)
            train_loss_sum += mse_metric(outputs, images).detach() * bs
            if LPIPS_WEIGHT:
                train_lpips_sum += lpips_val.detach() * bs
            train_count += bs
            if step_idx % 50 == 0:  # refresh the bar without an every-step host sync
                m = (train_loss_sum / train_count).item()
                pbar.set_postfix({"mse": m, "psnr": 10 * math.log10(1 / m)})

            if args.time_budget and (now - train_start) >= args.time_budget:
                stop = True  # wall-clock budget hit; finish this epoch's eval then exit
                break

        train_loss = (train_loss_sum / train_count).item()
        train_lpips = (train_lpips_sum / train_count).item()
        train_psnr = 10 * math.log10(1 / train_loss)

        do_eval = stop or (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        if not do_eval:
            continue

        model.eval()
        fid.reset()
        val_loss_sum, val_lpips_sum, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(
                    "cuda", dtype=torch.bfloat16, memory_format=torch.channels_last
                )
                outputs = model(images)
                val_loss_sum += mse_metric(outputs, images).item() * images.size(0)
                if LPIPS_WEIGHT:
                    val_lpips_sum += lpips_loss(outputs, images).item() * images.size(0)
                val_count += images.size(0)

                fid.update(images.float().clamp(0, 1), real=True)
                fid.update(outputs.float().clamp(0, 1), real=False)
        val_loss = val_loss_sum / val_count
        val_lpips = val_lpips_sum / val_count
        val_psnr = 10 * math.log10(1 / val_loss)
        val_rfid = fid.compute().item()
        best_rfid = min(best_rfid, val_rfid)
        print(f"[{args.run}] epoch {epoch+1} val_rfid={val_rfid:.3f} (best {best_rfid:.3f}) lr={lr:.2e}")

        log_metrics(
            metrics_path,
            {
                "run": args.run,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_lpips": train_lpips,
                "train_psnr": train_psnr,
                "val_loss": val_loss,
                "val_lpips": val_lpips,
                "val_psnr": val_psnr,
                "val_rfid": val_rfid,
            },
        )

        originals = next(iter(test_loader))[0][:8]
        with torch.no_grad():
            model.eval()
            reconstructions = (
                model(
                    originals.to(
                        "cuda", dtype=torch.bfloat16, memory_format=torch.channels_last
                    )
                )
                .float()
                .cpu()
            )
        grid = make_grid(
            torch.cat([originals.float().cpu(), reconstructions], dim=0), nrow=4
        )
        to_pil_image(grid).save(RUN_DIR / f"epoch_{epoch + 1:02d}.png")

        if stop:
            break

    originals = next(iter(test_loader))[0][:8]
    with torch.no_grad():
        model.eval()
        reconstructions = (
            model(
                originals.to(
                    "cuda", dtype=torch.bfloat16, memory_format=torch.channels_last
                )
            )
            .float()
            .cpu()
        )
    grid = make_grid(
        torch.cat([originals.float().cpu(), reconstructions], dim=0), nrow=4
    )
    to_pil_image(grid).save(RUN_DIR / "final.png")

    ckpt_path = RUN_DIR / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "best_rfid": best_rfid,
            "config": {
                "input_dim": 3,
                "latent_dim": LATENT_DIM,
                "base_channels": BASE_CHANNELS,
                "dim_per_block": DIM_PER_BLOCK,
                "layers_per_block": LAYERS_PER_BLOCK,
                "dec_layers_per_block": DEC_LAYERS_PER_BLOCK,
            },
        },
        ckpt_path,
    )
    print(f"Saved model to {ckpt_path} (best_rfid={best_rfid:.3f})")

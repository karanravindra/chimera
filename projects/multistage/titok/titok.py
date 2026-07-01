"""Stage-2 TiTok tokenizer over the stage-1 autoencoder's LATENTS (multistage pipeline).

Stage 1 (``spatialencoder/``) is a ConvAutoEncoder that maps an image to a continuous
**spatial** latent grid at 8x downsampling (16 channels). The default AE is the champion
``v7-lowgan-256``, a **256px** model -- its native/benchmark resolution -- so a 256px image
encodes to a ``(16, 32, 32)`` latent. This script trains a TiTok-style 1D tokenizer (Yu et al.
2024, "An Image is Worth 32 Tokens") as **stage 2** that consumes that latent grid and squeezes
it into ``K`` continuous 1D tokens of width ``latent_dim`` -- a further, order-independent
bottleneck a downstream generator can model as a short sequence.

Pipeline (stage-1 AE is FROZEN throughout -- we only train the tokenizer):

    image --AE.encode--> z (B,16,32,32) --normalize--> zn --TiTok.encode--> tokens (B,K,d)
                                                              --TiTok.decode--> zn_hat
    zn_hat --denormalize--> z_hat --AE.decode--> reconstructed image

Throughput: the frozen AE encode (a 256px conv net) was the dominant per-step cost -- rerun
every step, while the ViT/attention was cheap (patch 2 with 9x the tokens ran ~as fast as patch
4). The fix is to **cache all normalized train latents once** (AFHQ has no random train aug, so
latents are deterministic) and train the ViT purely on cached-latent minibatches -- the AE
encode leaves the hot loop entirely. Plus: cudnn.benchmark autotuning (determinism off), flash
SDPA attention, TF32 matmuls, torch.compile (reduce-overhead), and a ViT-Tiny (embed 192, depth
12, 3 heads) from projects/text2image/titok. ``--patch-size`` sets the patch grid (4 -> 8x8=64
tokens on the 32x32 latent, matching text2image's ViT-Tiny sequence length).

The tokenizer's objective is **latent MSE** on the normalized grid (no LPIPS anywhere -- not in
the loss, not as a metric). Latents are unbounded, so the TiTok decoder runs with
``sigmoid_output=False`` and we standardize the latent per-channel (mean/std estimated once over
the train set, SD-VAE style) so the ViT sees a unit-scale target.

Metrics (outputs/titok_metrics.csv, keyed by run/step), all at the eval resolution:
  * ``val_latmse``  -- normalized-latent reconstruction MSE (the training objective);
  * ``val_rfid``    -- END-TO-END rFID: real images vs AE.decode(tokenizer(AE.encode(image)));
  * ``ae_rfid``     -- reference rFID of the FROZEN AE alone (decode(encode)); the ceiling the
                       tokenizer is measured against -- val_rfid - ae_rfid is the token cost;
  * ``val_psnr``    -- pixel-space PSNR of the end-to-end reconstruction.

Every eval also saves a reconstruction grid (outputs/<run>/recon_step*.png) with three rows --
originals / frozen-AE recon / tokenizer-through-AE recon -- so the token bottleneck's effect on
the image is visible over training (plus a final.png at the end).

Optimizer: Muon on the ViT's 2D hidden matmul weights + AdamW aux on embeddings/norms/biases
(MuonWithAuxAdam), each group scaled by a shared warmup+cosine factor.

Efficiency mirrors dcae.py: bf16 + channels_last, torch.compile (reduce-overhead cudagraphs)
on the tokenizer step, single wall-clock budget with a throughput-calibrated warmup+cosine LR.

Examples
--------
    # train on the champion stage-1 AE's latents (default checkpoint), quick 5-min test
    uv run python projects/multistage/titok/titok.py run1 --time 300

    # coarser bottleneck (16 tokens)
    uv run python projects/multistage/titok/titok.py run2 --num-latent-tokens 16
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchinfo import summary
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from chimera.data import AFHQDataModule
from chimera.models import ConvAutoEncoder, TiTokAutoEncoder
from chimera.optim import MuonWithAuxAdam, muon_adam_param_groups
from chimera.utils.logtee import tee_to_logfile
from chimera.utils.seed import seed_everything


SEED = 0
# The stage-1 champion (v7-lowgan-256): rFID ~5.5, 16ch@16x16 latent at 128px. Frozen here.
DEFAULT_AE_CKPT = (
    Path(__file__).parent.parent / "spatialencoder/outputs/v7-lowgan-256/best.pt"
)


# --------------------------------------------------------------------------------------
# LR schedule (linear warmup -> cosine, horizon calibrated to a wall-clock budget)
# --------------------------------------------------------------------------------------
def cosine_warmup_lr(step, total_steps, base_lr, warmup_steps, min_lr_frac):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


# --------------------------------------------------------------------------------------
# Metrics logging
# --------------------------------------------------------------------------------------
METRIC_FIELDS = ("run", "step", "val_latmse", "val_rfid", "ae_rfid", "val_psnr")


def log_metrics(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(METRIC_FIELDS)
        w.writerow([row.get(k, "") for k in METRIC_FIELDS])


# --------------------------------------------------------------------------------------
# Frozen stage-1 AE
# --------------------------------------------------------------------------------------
def load_ae(ckpt_path: Path, dev: str) -> ConvAutoEncoder:
    """Rebuild the stage-1 ConvAutoEncoder from its checkpoint's own config and freeze it.
    strict load: any arch mismatch surfaces loudly rather than silently training on garbage."""
    ck = torch.load(ckpt_path, map_location="cpu")
    c = ck.get("config", {})
    ae = ConvAutoEncoder(
        input_dim=c.get("input_dim", 3),
        latent_dim=c.get("latent_dim", 16),
        base_channels=c.get("base_channels", 64),
        dim_per_block=tuple(c.get("dim_per_block", (64, 128, 256))),
        layers_per_block=tuple(c.get("layers_per_block", (1, 2, 4))),
        dec_layers_per_block=(tuple(c["dec_layers_per_block"])
                              if c.get("dec_layers_per_block") else None),
        layer_scale=c.get("layer_scale", False),
        refine_head=c.get("refine_head", False),
        attn_stages=tuple(c.get("attn_stages", ())),
        attn_dim=c.get("attn_dim", 32),
    )
    ae.load_state_dict(ck["model"])
    ae.requires_grad_(False)
    ae = ae.to(dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last).eval()
    print(f"Loaded frozen stage-1 AE from {ckpt_path} config={c}", flush=True)
    return ae


@torch.no_grad()
def latent_stats(ae: ConvAutoEncoder, loader, dev: str, max_batches: int = 64):
    """Per-channel mean/std of the AE latent over the train set (SD-VAE-style standardization).
    Accumulated in fp32; returned as bf16 (1,C,1,1) channels_last-friendly buffers."""
    n = 0
    s = s2 = None
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        imgs = batch[0].to(dev, dtype=torch.bfloat16, memory_format=torch.channels_last)
        z = ae.encode(imgs).float()  # (B,C,h,w)
        flat = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1)  # (C, B*h*w)
        s = flat.sum(1) if s is None else s + flat.sum(1)
        s2 = (flat * flat).sum(1) if s2 is None else s2 + (flat * flat).sum(1)
        n += flat.shape[1]
    mean = (s / n)
    std = (s2 / n - mean**2).clamp_min(1e-8).sqrt()
    print(f"latent stats over {n} positions: mean~{mean.mean():.3f} std~{std.mean():.3f}", flush=True)
    shape = (1, -1, 1, 1)
    return (mean.reshape(shape).to(dev, torch.bfloat16),
            std.reshape(shape).to(dev, torch.bfloat16))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--ae-ckpt", type=str, default=str(DEFAULT_AE_CKPT),
                   help="frozen stage-1 ConvAutoEncoder checkpoint whose latents we tokenize")
    p.add_argument("--image-size", type=int, default=256,
                   help="images encoded at this res; latent grid = image_size // 8 (256 -> 32). "
                        "v7-lowgan-256 is a 256px AE -- that is its native/benchmark resolution")
    # TiTok bottleneck: K continuous tokens of width --latent-dim over the latent grid.
    p.add_argument("--num-latent-tokens", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=16, help="width of each 1D token")
    p.add_argument("--patch-size", type=int, default=4,
                   help="patch over the latent grid. Attention is O(tokens^2), so this is the "
                        "dominant speed lever: patch 4 on the 32x32 (256px) latent = an 8x8=64-"
                        "token grid, matching text2image's ViT-Tiny sequence length")
    # ViT-Tiny (embed 192, depth 12, 3 heads) -- matches projects/text2image/titok.
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)  # text2image's ViT-Tiny batch
    p.add_argument("--time", type=float, default=300.0,
                   help="wall-clock training budget in seconds (excludes compile/eval); 300 = a "
                        "quick 5-min test. Total wall (compile + cache + eval) stays under ~7 min")
    # Muon on the ViT's 2D hidden matmul weights + AdamW aux on embeddings/norms/biases.
    p.add_argument("--muon-lr", type=float, default=0.01, help="LR for the Muon (2D weight) group")
    p.add_argument("--adam-lr", type=float, default=3e-4,
                   help="LR for the AdamW aux group (embeddings/norms/biases)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--min-lr-frac", type=float, default=0.05,
                   help="cosine schedule floors each group at this fraction of its base LR")
    p.add_argument("--eval-secs", type=float, default=45.0)
    p.add_argument("--log-secs", type=float, default=10.0)
    p.add_argument("--target-rfid", type=float, default=7.0,
                   help="GOAL: stop as soon as val_rfid drops below this. 0 disables")
    p.add_argument("--resume", type=str, default=None,
                   help="tokenizer checkpoint to resume weights from (same arch)")
    args = p.parse_args()

    # Deterministic run (reproducible). The per-step AE encode is now cached ONCE up front, so we
    # no longer need cudnn autotuning in the hot loop and determinism costs ~nothing. NOTE: flash
    # attention's backward is non-deterministic; use_deterministic_algorithms(warn_only=True) lets
    # it still run (not bit-exact) -- fine here since attention is not the bottleneck.
    seed_everything(SEED, deterministic=True)
    torch.set_float32_matmul_precision("high")   # TF32 matmuls (matches text2image build_trainer)
    torch.backends.cuda.enable_flash_sdp(True)   # ViT attention -> flash SDPA (F.sdpa default)
    dev = "cuda"
    grid_res = args.image_size // 8  # stage-1 AE downsamples 8x (3 stages)

    out_dir = Path(__file__).parent / "outputs"
    run_dir = out_dir / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "titok_metrics.csv"
    log_path = tee_to_logfile(run_dir / f"{args.run}.log")
    print(f"logging to {log_path}", flush=True)

    # ---- data (one resolution: encode + eval both happen at image_size) ----
    dm = AFHQDataModule(batch_size=args.batch_size, image_size=args.image_size, num_workers=7)
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    eval_loader = dm.test_dataloader()

    # ---- frozen stage-1 AE + latent standardization ----
    ae = load_ae(Path(args.ae_ckpt), dev)
    with torch.no_grad():  # probe the true latent geometry from the loaded AE
        probe = ae.encode(torch.zeros(1, 3, args.image_size, args.image_size,
                                      device=dev, dtype=torch.bfloat16))
    latent_ch = probe.shape[1]
    assert tuple(probe.shape[2:]) == (grid_res, grid_res), (
        f"AE latent grid {tuple(probe.shape[2:])} != expected ({grid_res},{grid_res}); "
        f"--image-size {args.image_size} assumes an 8x-downsampling stage-1 AE")
    lat_mean, lat_std = latent_stats(ae, train_loader, dev)

    def normalize(z):
        return (z - lat_mean) / lat_std

    def denormalize(zn):
        return zn * lat_std + lat_mean

    # ---- cache ALL normalized train latents once (the big speedup) ----
    # Diagnosis: per-step cost was dominated by the frozen AE encode (a 256px conv net) rerun
    # every step -- NOT the ViT/attention (patch 2 with 9x the tokens ran ~as fast as patch 4).
    # AFHQDataModule applies no random train augmentation, so latents are deterministic and this
    # cache is exact. Training then samples minibatches of cached latents and runs ONLY the ViT.
    @torch.no_grad()
    def cache_train_latents():
        def enc(x):
            x = x.to(dev, dtype=torch.bfloat16, memory_format=torch.channels_last)
            return normalize(ae.encode(x)).to(torch.bfloat16)
        return torch.cat([enc(b[0]) for b in train_loader]).contiguous(
            memory_format=torch.channels_last)

    t_cache = time.perf_counter()
    latent_cache = cache_train_latents()
    n_cached = latent_cache.shape[0]
    print(f"cached {n_cached} train latents {tuple(latent_cache.shape[1:])} on GPU "
          f"({latent_cache.element_size() * latent_cache.nelement() / 1e6:.0f} MB) in "
          f"{time.perf_counter() - t_cache:.1f}s -- AE encode is now out of the training loop",
          flush=True)

    # ---- TiTok tokenizer over the latent grid (unbounded target -> no sigmoid) ----
    model = TiTokAutoEncoder(
        input_dim=latent_ch, image_size=grid_res, patch_size=args.patch_size,
        num_latent_tokens=args.num_latent_tokens, latent_dim=args.latent_dim,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio, drop_path_rate=args.drop_path, sigmoid_output=False,
    )
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu")["model"])
        print(f"Resumed tokenizer weights from {args.resume}", flush=True)
    summary(model, input_size=(1, latent_ch, grid_res, grid_res), depth=1)
    model = model.to(dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last)

    lat_floats = latent_ch * grid_res * grid_res
    tok_floats = args.num_latent_tokens * args.latent_dim
    img_floats = 3 * args.image_size * args.image_size
    print(f"compression: image {img_floats} -> AE latent {lat_floats} ({img_floats/lat_floats:.0f}x) "
          f"-> {args.num_latent_tokens} tokens x {args.latent_dim} = {tok_floats} "
          f"({lat_floats/tok_floats:.0f}x on latent, {img_floats/tok_floats:.0f}x total)", flush=True)

    # ---- frozen metric net (rFID only; no LPIPS) ----
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)

    def to_dev(x):
        return x.to(dev, dtype=torch.bfloat16, memory_format=torch.channels_last)

    # ---- one-time reference: rFID of the frozen AE alone (the tokenizer's ceiling) ----
    @torch.no_grad()
    def ae_reference_rfid() -> float:
        fid.reset()
        for batch in eval_loader:
            imgs = to_dev(batch[0])
            rec = ae.decode(ae.encode(imgs))
            fid.update(imgs.float().clamp(0, 1), real=True)
            fid.update(rec.float().clamp(0, 1), real=False)
        return fid.compute().item()

    ae_rfid = ae_reference_rfid()
    print(f"[{args.run}] frozen-AE reference rFID = {ae_rfid:.3f} (tokenizer ceiling)", flush=True)

    @torch.no_grad()
    def evaluate(step: int) -> float:
        model.eval()
        fid.reset()
        latmse_sum = mse_sum = cnt = 0
        for batch in eval_loader:
            imgs = to_dev(batch[0])
            zn = normalize(ae.encode(imgs))
            zn_hat = model(zn)
            rec = ae.decode(denormalize(zn_hat))
            latmse_sum += F.mse_loss(zn_hat, zn).item() * imgs.size(0)
            mse_sum += F.mse_loss(rec, imgs).item() * imgs.size(0)
            cnt += imgs.size(0)
            fid.update(imgs.float().clamp(0, 1), real=True)
            fid.update(rec.float().clamp(0, 1), real=False)
        rfid = fid.compute().item()
        latmse = latmse_sum / cnt
        psnr = 10 * math.log10(1 / (mse_sum / cnt))
        log_metrics(metrics_path, {
            "run": args.run, "step": step, "val_latmse": f"{latmse:.5f}",
            "val_rfid": f"{rfid:.4f}", "ae_rfid": f"{ae_rfid:.4f}", "val_psnr": f"{psnr:.4f}",
        })
        print(f"[{args.run}] step {step} latmse={latmse:.4f} val_rfid={rfid:.3f} "
              f"(ae {ae_rfid:.3f}, +{rfid - ae_rfid:.3f}) psnr={psnr:.2f}", flush=True)
        model.train()
        return rfid

    @torch.no_grad()
    def save_grid(tag: str):
        model.eval()
        imgs = to_dev(next(iter(eval_loader))[0][:8])
        ae_rec = ae.decode(ae.encode(imgs)).float().cpu()
        tok_rec = ae.decode(denormalize(model(normalize(ae.encode(imgs))))).float().cpu()
        # rows: original / AE recon / tokenizer recon
        panel = torch.cat([imgs.float().cpu(), ae_rec, tok_rec], 0)
        to_pil_image(make_grid(panel.clamp(0, 1), nrow=8)).save(run_dir / f"{tag}.png")
        model.train()

    def save_ckpt(name: str):
        torch.save({
            "model": model.state_dict(),
            "lat_mean": lat_mean.cpu(), "lat_std": lat_std.cpu(),
            "ae_ckpt": str(args.ae_ckpt),
            "config": {
                "input_dim": latent_ch, "image_size": grid_res, "patch_size": args.patch_size,
                "num_latent_tokens": args.num_latent_tokens, "latent_dim": args.latent_dim,
                "embed_dim": args.embed_dim, "depth": args.depth, "num_heads": args.num_heads,
                "mlp_ratio": args.mlp_ratio,
            },
        }, run_dir / name)

    # ---- time-budgeted training (warmup+cosine, horizon calibrated to --time) ----
    # Muon (2D hidden weights) + AdamW aux (embeddings/norms/biases) behind one .step().
    # The two groups keep their own base LRs; the warmup+cosine schedule scales BOTH by a
    # shared factor each step, so each is floored at min_lr_frac of ITS base LR.
    opt = MuonWithAuxAdam(muon_adam_param_groups(
        model, muon_lr=args.muon_lr, adam_lr=args.adam_lr, weight_decay=args.weight_decay))
    base_lrs = [g["lr"] for g in opt.param_groups]
    step_model = torch.compile(model, mode="reduce-overhead")

    planned_total = 10**9
    train_start = calib_t0 = None
    CS, CE = 20, 70
    gstep = 0
    best = float("inf")
    last_eval_t = last_hb_t = 0.0
    last_hb_step = 0
    done = False
    while not done:
        # sample a minibatch of cached latents (fixed batch size -> static shape for cudagraphs)
        idx = torch.randint(0, n_cached, (args.batch_size,), device=dev)
        zn = latent_cache[idx].contiguous(memory_format=torch.channels_last)

        now = time.perf_counter()
        if train_start is None:
            train_start = last_eval_t = last_hb_t = now
        if gstep == CS:
            calib_t0 = now
        elif gstep == CE and calib_t0 is not None:
            sps = (CE - CS) / (now - calib_t0)
            planned_total = gstep + max(1, int(sps * (args.time - (now - train_start))))
            print(f"[calib] {sps:.1f} steps/s -> planned_total={planned_total}", flush=True)
        lr_frac = cosine_warmup_lr(gstep, planned_total, 1.0, 50, args.min_lr_frac)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_frac

        torch.compiler.cudagraph_mark_step_begin()
        opt.zero_grad(set_to_none=True)
        zn_hat = step_model(zn)
        loss = F.mse_loss(zn_hat, zn)  # latent MSE only (no LPIPS)
        loss.backward()
        opt.step()

        gstep += 1
        now = time.perf_counter()
        if now - last_hb_t >= args.log_secs:
            sps = (gstep - last_hb_step) / max(1e-6, now - last_hb_t)
            print(f"[{args.run}] step {gstep} muon_lr={base_lrs[0] * lr_frac:.2e} "
                  f"loss={loss.item():.4f} ({sps:.1f} it/s)", flush=True)
            last_hb_t, last_hb_step = now, gstep
        if now - last_eval_t >= args.eval_secs:
            r = evaluate(gstep)
            save_grid(f"recon_step{gstep:06d}")  # log recons through the AE each eval
            if r <= best:
                best = r
                save_ckpt("best.pt")
            if args.target_rfid and r < args.target_rfid:
                print(f"[{args.run}] GOAL MET: val_rfid {r:.3f} < {args.target_rfid} -> stopping",
                      flush=True)
                done = True
            last_eval_t = now
        if now - train_start >= args.time:
            done = True

    r = evaluate(gstep)
    if r <= best:
        best = r
        save_ckpt("best.pt")
    save_grid("final")
    save_ckpt("final.pt")
    print(f"\nSUMMARY {args.run}: best_rfid={best:.3f} (frozen-AE ceiling {ae_rfid:.3f}, "
          f"token cost +{best - ae_rfid:.3f})", flush=True)


if __name__ == "__main__":
    main()

"""Full DC-AE (Deep Compression Autoencoder, arXiv:2410.10733) 3-phase training
pipeline for the multistage stage-1 autoencoder on AFHQ.

The DC-AE recipe *decouples* resolution adaptation and the GAN from the main run:

  Phase 1 -- low-res (64px) FULL-model training, L1 + LPIPS, no GAN.
             Bulk of training; cheap steps -> reaches the good perceptual regime fast.
  Phase 2 -- high-res (128px) LATENT ADAPTATION: freeze everything except the
             "middle" layers (encoder head + decoder input) so the latent distribution
             re-aligns for the final resolution. Short.
  Phase 3 -- low-res (64px) LOCAL REFINEMENT: freeze everything except the decoder
             head, add a PatchGAN with hinge loss + the taming-transformers adaptive
             weight. Sharpens local detail; low-res keeps the GAN stable & cheap.

rFID / PSNR / LPIPS are ALWAYS evaluated at the eval/target resolution (--eval-res,
default 128) -- INCLUDING the phase-3 GAN, which per DC-AE refines at the LOWER
(--p3-res, default 64) resolution but is still scored at the target res. So numbers are
comparable across phases and to the single-phase runs. Metrics go to
outputs/dcae_metrics.csv keyed by (run, phase).

Efficiency: bf16 + channels_last throughout; phase 1 uses torch.compile
(reduce-overhead cudagraphs); phases 2/3 run eager (partial-freeze guards + the GAN's
double-backward don't play well with cudagraphs and those phases are short anyway).
Each phase is bounded by a WALL-CLOCK budget with a throughput-calibrated cosine LR.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchinfo import summary
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from chimera.data import AFHQDataModule
from chimera.models import ConvAutoEncoder
from chimera.utils.logtee import tee_to_logfile
from chimera.utils.seed import seed_everything


SEED = 0
LATENT_DIM = 16
EVAL_RES = 128  # rFID always measured here, whatever the training resolution


# --------------------------------------------------------------------------------------
# PatchGAN discriminator (phase 3)
# --------------------------------------------------------------------------------------
class NLayerDiscriminator(nn.Module):
    """PatchGAN (Isola et al. 2017; the taming-transformers/VQGAN discriminator).

    Spectral-norm on every conv for cheap Lipschitz control (more stable than BatchNorm
    on the small AFHQ set and batch-size-independent). Outputs a per-patch logit map;
    hinge loss is applied to the map directly."""

    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers = [sn(nn.Conv2d(in_ch, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
        mult = 1
        for i in range(1, n_layers):
            prev, mult = mult, min(2**i, 8)
            layers += [
                sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1)),
                nn.GroupNorm(min(32, ndf * mult), ndf * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        prev, mult = mult, min(2**n_layers, 8)
        layers += [
            sn(nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1)),
            nn.GroupNorm(min(32, ndf * mult), ndf * mult),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(ndf * mult, 1, 4, 1, 1)),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """DiffAugment (Zhao et al. 2020): differentiable translation + cutout applied to
    BOTH real and fake before the discriminator, so D can't memorize the ~15k-image
    AFHQ set (the main failure mode that makes a small-data GAN hurt rFID). Color aug is
    omitted -- we want faithful reconstruction colors. Same transform family both sides
    keeps the game fair; per-sample randomness is what regularizes D."""
    b, c, h, w = x.shape
    # translation up to 1/8 each way
    shift_h, shift_w = h // 8, w // 8
    tx = torch.randint(-shift_w, shift_w + 1, (b, 1, 1), device=x.device)
    ty = torch.randint(-shift_h, shift_h + 1, (b, 1, 1), device=x.device)
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device),
                            torch.arange(w, device=x.device), indexing="ij")
    gx = (gx.unsqueeze(0) + tx).clamp(0, w - 1)
    gy = (gy.unsqueeze(0) + ty).clamp(0, h - 1)
    idx = (gy * w + gx).view(b, 1, -1).expand(-1, c, -1)
    x = x.reshape(b, c, -1).gather(2, idx).view(b, c, h, w)
    # cutout a random half-size square
    cy = torch.randint(0, h, (b, 1, 1), device=x.device)
    cx = torch.randint(0, w, (b, 1, 1), device=x.device)
    yy = torch.arange(h, device=x.device).view(1, h, 1)
    xx = torch.arange(w, device=x.device).view(1, 1, w)
    mask = ((yy - cy).abs() > h // 4) | ((xx - cx).abs() > w // 4)
    return x * mask.unsqueeze(1).to(x.dtype)


def d_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def g_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


# --- R3GAN (arXiv:2501.05441): relativistic pairing loss + R1/R2 grad penalties ---
# Real and fake are naturally PAIRED in an AE (fake = reconstruction of the real), so the
# relativistic difference is taken element-wise on the paired PatchGAN logit maps. The
# relativistic loss is what gives R3GAN its local-convergence guarantee; R1 (on real) and
# R2 (on fake) are the zero-centered gradient penalties that keep D from overfitting the
# ~15k-image set -- R2 specifically stops the fake-side gradient explosion that hinge/R1-only
# suffer, which is what lets the GAN run long/strong on small data without collapse.
def r3gan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator wants real to outscore fake: minimize softplus(fake - real).

    The relativistic difference is taken on per-SAMPLE scalar critic scores (mean over the
    patch map), NOT element-wise on the maps: DiffAugment applies INDEPENDENT random shifts
    to real vs fake, so the maps are spatially misaligned and an element-wise real-vs-fake
    difference compares mismatched patches (this made the first r3gan run regress). Reducing
    to a per-image scalar first restores correct RpGAN pairing (real_i vs its recon_i)."""
    real = real_logits.flatten(1).mean(1)
    fake = fake_logits.flatten(1).mean(1)
    return F.softplus(fake - real).mean()


def r3gan_g_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator wants fake to outscore real: minimize softplus(real - fake), on per-sample
    scalar scores (see r3gan_d_loss). real_logits are a detached baseline (disc frozen);
    grad flows only through fake."""
    real = real_logits.flatten(1).mean(1)
    fake = fake_logits.flatten(1).mean(1)
    return F.softplus(real - fake).mean()


def grad_penalty(logits: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Zero-centered gradient penalty E[||d logits / d inputs||^2] (R1 on real / R2 on fake).
    `inputs` must be a leaf with requires_grad=True; uses create_graph for the outer backward."""
    g = torch.autograd.grad(logits.sum(), inputs, create_graph=True)[0]
    return g.pow(2).flatten(1).sum(1).mean()


def adaptive_gan_weight(rec_loss: torch.Tensor, g_loss: torch.Tensor,
                        last_layer: torch.Tensor) -> torch.Tensor:
    """taming-transformers adaptive weight: balance the GAN gradient against the
    reconstruction gradient at the decoder's last layer, so the GAN never overwhelms
    reconstruction. lambda = ||grad_last(L_rec)|| / (||grad_last(L_gan)|| + eps)."""
    rec_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_grad = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    w = rec_grad.norm() / (g_grad.norm() + 1e-4)
    return w.clamp(0.0, 1e4).detach()


# --------------------------------------------------------------------------------------
# LR schedule (linear warmup -> cosine, horizon calibrated to a wall-clock budget)
# --------------------------------------------------------------------------------------
def cosine_warmup_lr(step, total_steps, base_lr, warmup_steps, min_lr_frac):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


# --------------------------------------------------------------------------------------
# Selective freezing (which submodules train in each phase)
# --------------------------------------------------------------------------------------
def set_trainable(model: ConvAutoEncoder, prefixes: tuple[str, ...] | None) -> int:
    """Enable grad only on params whose name starts with one of `prefixes`
    (None = all trainable). Returns the trainable param count."""
    n = 0
    for name, p in model.named_parameters():
        on = prefixes is None or any(name.startswith(pref) for pref in prefixes)
        p.requires_grad_(on)
        if on:
            n += p.numel()
    return n


def middle_prefixes(model: ConvAutoEncoder) -> tuple[str, ...]:
    """Phase-2 'middle': bottleneck 1x1s + last encoder block + first decoder block."""
    return ("to_latent", "from_latent",
            f"encoder.{len(model.encoder) - 1}", "decoder.0")


def decoder_head_prefixes(model: ConvAutoEncoder) -> tuple[str, ...]:
    """Phase-3 'decoder head': last decoder block + output head."""
    return (f"decoder.{len(model.decoder) - 1}", "head")


# --------------------------------------------------------------------------------------
# Metrics logging
# --------------------------------------------------------------------------------------
METRIC_FIELDS = ("run", "phase", "step", "train_res", "val_rfid", "val_psnr", "val_lpips")


def log_metrics(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(METRIC_FIELDS)
        w.writerow([row.get(k, "") for k in METRIC_FIELDS])


def main():
    global EVAL_RES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    p.add_argument("--dim-mult", type=str, default="1,2,4")
    p.add_argument("--layers-per-block", type=str, default="1,2,2")       # encoder (light)
    # decoder heavy, weighted to HIGH RES: index 0 -> highest-res decoder block (the
    # stage that paints fine texture / sets rFID). Champion's symmetric (1,2,4) put its
    # 4 resblocks at the LOWEST-res decoder stage and only 1 at the highest -- backwards.
    p.add_argument("--dec-layers-per-block", type=str, default="4,3,2")   # decoder (heavy@hi-res)
    p.add_argument("--layer-scale", action="store_true")
    p.add_argument("--zero-init-res", action="store_true")
    p.add_argument("--refine-head", action="store_true")
    p.add_argument("--attn-stages", type=str, default="",
                   help="DC-AE mimicry: comma-sep stage indices (0=highest-res) to use "
                        "EfficientViT linear-attention blocks instead of conv ResBlocks; "
                        "DC-AE puts attention in the DEEP stages, e.g. '2' for a 3-stage net")
    p.add_argument("--attn-dim", type=int, default=32)
    p.add_argument("--depthwise", action="store_true",
                   help="depthwise-separable convs in ResBlocks (~8x cheaper -> go deeper)")
    p.add_argument("--lpips-net", type=str, default="vgg", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--lpips-weight", type=float, default=1.0)  # higher LPIPS >> 0.1 for rFID
    p.add_argument("--batch-size", type=int, default=32)
    # per-phase wall-clock budgets (seconds of training, excluding compile/eval)
    p.add_argument("--p1-time", type=float, default=240.0)
    p.add_argument("--p2-time", type=float, default=90.0)
    p.add_argument("--p3-time", type=float, default=150.0)
    p.add_argument("--p1-res", type=int, default=64)
    p.add_argument("--p3-res", type=int, default=64)
    p.add_argument("--eval-res", type=int, default=EVAL_RES,
                   help="final/target resolution: rFID measured here; phase-2 trains here")
    p.add_argument("--p1-lr", type=float, default=1e-3)
    p.add_argument("--p2-lr", type=float, default=2.5e-4)
    p.add_argument("--p3-lr", type=float, default=1e-4)
    p.add_argument("--gan-warmup-steps", type=int, default=50)
    p.add_argument("--diffaug", action="store_true",
                   help="DiffAugment on discriminator inputs (small-data stabilizer)")
    p.add_argument("--gan-weight", type=float, default=0.5,
                   help="cap/scale on the adaptive GAN weight (lower = gentler GAN)")
    p.add_argument("--gan-loss", type=str, default="hinge", choices=["hinge", "r3gan"],
                   help="phase-3 adversarial loss. hinge (default) = taming/VQGAN hinge; it "
                        "beat r3gan for decoder-head recon refinement on AFHQ (r3gan's "
                        "non-saturating relativistic objective regressed rFID, see RESULTS.md).")
    p.add_argument("--r1-gamma", type=float, default=1.0,
                   help="weight on the R1+R2 zero-centered gradient penalties (r3gan only)")
    p.add_argument("--eval-secs", type=float, default=45.0,
                   help="min wall-seconds between in-phase evals")
    p.add_argument("--log-secs", type=float, default=10.0,
                   help="min wall-seconds between lightweight train-loss heartbeat logs")
    p.add_argument("--epochs", type=int, default=0,
                   help="if >0, run phase-1 & phase-2 for N EPOCHS each (overrides their "
                        "wall-clock budget); cosine LR horizon = N*steps_per_epoch")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    seed_everything(SEED, deterministic=True)
    dev = "cuda"
    EVAL_RES = args.eval_res  # rFID + phase-2 target resolution (default 128)

    base = args.base_channels
    dim_per_block = tuple(base * int(m) for m in args.dim_mult.split(","))
    enc_layers = tuple(int(x) for x in args.layers_per_block.split(","))
    dec_layers = tuple(int(x) for x in args.dec_layers_per_block.split(","))
    attn_stages = tuple(int(x) for x in args.attn_stages.split(",")) if args.attn_stages else ()
    spatial_factor = 2 ** len(dim_per_block)  # 8x for 3 stages, 16x for 4 (DC-AE f16)

    out_dir = Path(__file__).parent / "outputs"
    run_dir = out_dir / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "dcae_metrics.csv"
    log_path = tee_to_logfile(run_dir / f"{args.run}.log")
    print(f"logging to {log_path}", flush=True)

    # ---- data: one datamodule per training resolution + a fixed 128px eval loader ----
    dms: dict[int, AFHQDataModule] = {}

    def loader(res: int, train: bool):
        if res not in dms:
            dm = AFHQDataModule(batch_size=args.batch_size, image_size=res, num_workers=7)
            dm.prepare_data()
            dm.setup("fit")
            dms[res] = dm
        dm = dms[res]
        return dm.train_dataloader() if train else dm.test_dataloader()

    eval_loader = loader(EVAL_RES, train=False)

    # ---- model + discriminator ----
    if args.resume:
        # Build to MATCH the checkpoint's own config so refinement runs on the exact
        # trained arch (e.g. the symmetric champion), not the CLI arch defaults.
        ck = torch.load(args.resume, map_location="cpu")
        c = ck.get("config", {})
        model = ConvAutoEncoder(
            input_dim=c.get("input_dim", 3),
            latent_dim=c.get("latent_dim", args.latent_dim),
            base_channels=c.get("base_channels", base),
            dim_per_block=tuple(c.get("dim_per_block", dim_per_block)),
            layers_per_block=tuple(c.get("layers_per_block", enc_layers)),
            dec_layers_per_block=(tuple(c["dec_layers_per_block"])
                                  if c.get("dec_layers_per_block") else None),
            layer_scale=c.get("layer_scale", False), zero_init_res=False,
            refine_head=c.get("refine_head", False),
            attn_stages=tuple(c.get("attn_stages", ())), attn_dim=c.get("attn_dim", 32),
        )
        model.load_state_dict(ck["model"])
        print(f"Resumed AE from {args.resume} with its own config {c}")
    else:
        model = ConvAutoEncoder(
            input_dim=3, latent_dim=args.latent_dim, base_channels=base,
            dim_per_block=dim_per_block, layers_per_block=enc_layers,
            dec_layers_per_block=dec_layers,
            layer_scale=args.layer_scale, zero_init_res=args.zero_init_res,
            refine_head=args.refine_head, depthwise=args.depthwise,
            attn_stages=attn_stages, attn_dim=args.attn_dim,
        )
    summary(model, input_size=(1, 3, args.p1_res, args.p1_res), depth=1)  # fp32, pre-cast
    model = model.to(dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    disc = NLayerDiscriminator().to(dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last)

    # ---- frozen metric nets ----
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=args.lpips_net, normalize=True)
    lpips.requires_grad_(False)
    lpips = lpips.to(dev).eval()

    def lpips_loss(recon, target):
        return lpips.net(recon.float(), target.float(), normalize=True).mean()

    def to_dev(x, res):
        return x.to(dev, dtype=torch.bfloat16, memory_format=torch.channels_last)

    @torch.no_grad()
    def evaluate(phase_name: str, step: int, train_res: int) -> float:
        model.eval()
        fid.reset()
        loss_sum = lp_sum = cnt = 0
        for batch in eval_loader:
            imgs = to_dev(batch[0], EVAL_RES)
            rec = model(imgs)
            loss_sum += F.mse_loss(rec, imgs).item() * imgs.size(0)
            lp_sum += lpips_loss(rec, imgs).item() * imgs.size(0)
            cnt += imgs.size(0)
            fid.update(imgs.float().clamp(0, 1), real=True)
            fid.update(rec.float().clamp(0, 1), real=False)
        rfid = fid.compute().item()
        psnr = 10 * math.log10(1 / (loss_sum / cnt))
        lp = lp_sum / cnt
        log_metrics(metrics_path, {
            "run": args.run, "phase": phase_name, "step": step, "train_res": train_res,
            "val_rfid": f"{rfid:.4f}", "val_psnr": f"{psnr:.4f}", "val_lpips": f"{lp:.5f}",
        })
        print(f"[{args.run}/{phase_name}] step {step} res{train_res} "
              f"val_rfid={rfid:.3f} psnr={psnr:.2f} lpips={lp:.4f}", flush=True)
        model.train()
        return rfid

    # save a recon grid
    @torch.no_grad()
    def save_grid(tag: str):
        model.eval()
        originals = next(iter(eval_loader))[0][:8]
        rec = model(to_dev(originals, EVAL_RES)).float().cpu()
        grid = make_grid(torch.cat([originals.float().cpu(), rec], 0), nrow=4)
        to_pil_image(grid).save(run_dir / f"{tag}.png")
        model.train()

    # ---- generic phase runner ----
    def run_phase(phase_name, *, res, time_budget, base_lr, min_lr_frac,
                  trainable_prefixes, use_gan, use_compile, betas=(0.9, 0.999), epochs=0):
        if epochs <= 0 and time_budget <= 0:
            print(f"=== {phase_name}: SKIPPED (budget<=0) ===", flush=True)
            return float("inf")
        n_train = set_trainable(model, trainable_prefixes)
        mode = f"{epochs} epochs" if epochs > 0 else f"budget={time_budget}s"
        print(f"\n=== {phase_name}: res={res} {mode} "
              f"trainable={n_train/1e3:.0f}K gan={use_gan} compile={use_compile} ===", flush=True)
        train_loader = loader(res, train=True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=base_lr, betas=betas, fused=True)
        d_opt = (torch.optim.AdamW(disc.parameters(), lr=base_lr, betas=(0.5, 0.9), fused=True)
                 if use_gan else None)
        step_model = torch.compile(model, mode="reduce-overhead") if use_compile else model
        # decoder's last conv weight, for the adaptive GAN weight (head may be a refine
        # Sequential ending in a Conv2d, or a bare Conv2d).
        last_layer = (model.head[-1].weight if isinstance(model.head, nn.Sequential)
                      else model.head.weight)

        steps_per_epoch = len(train_loader)
        planned_total = epochs * steps_per_epoch if epochs > 0 else 10**9
        train_start = calib_t0 = None
        CS, CE = 20, 70
        gstep = 0
        epoch = 0
        best = float("inf")
        last_eval_t = 0.0
        last_hb_t = last_hb_step = 0
        while True:
            done = False
            for batch in train_loader:
                imgs = to_dev(batch[0], res)
                now = time.perf_counter()
                if train_start is None:
                    train_start = last_eval_t = last_hb_t = now
                if epochs <= 0 and gstep == CS:
                    calib_t0 = now
                elif epochs <= 0 and gstep == CE and calib_t0 is not None:
                    sps = (CE - CS) / (now - calib_t0)
                    planned_total = gstep + max(1, int(sps * (time_budget - (now - train_start))))
                    print(f"[calib/{phase_name}] {sps:.1f} steps/s -> planned_total={planned_total}", flush=True)
                lr = cosine_warmup_lr(gstep, planned_total, base_lr, 50, min_lr_frac)
                for g in opt.param_groups:
                    g["lr"] = lr

                if use_compile:
                    torch.compiler.cudagraph_mark_step_begin()

                if not use_gan:
                    opt.zero_grad(set_to_none=True)
                    rec = step_model(imgs)
                    loss = F.l1_loss(rec, imgs) + args.lpips_weight * lpips_loss(rec, imgs)
                    loss.backward()
                    opt.step()
                else:
                    # ---- discriminator step (disc trainable, AE detached) ----
                    rec = model(imgs)
                    aug = diff_augment if args.diffaug else (lambda t: t)
                    disc.requires_grad_(True)
                    d_opt.zero_grad(set_to_none=True)
                    if args.gan_loss == "r3gan":
                        # leaf inputs so R1 (real) / R2 (fake) can differentiate D wrt inputs
                        real_in = (imgs.detach() * 2 - 1).requires_grad_(True)
                        fake_in = (rec.detach() * 2 - 1).requires_grad_(True)
                        real_logits = disc(aug(real_in))
                        fake_logits = disc(aug(fake_in))
                        d_loss = r3gan_d_loss(real_logits, fake_logits)
                        if args.r1_gamma > 0:
                            gp = (grad_penalty(real_logits, real_in)
                                  + grad_penalty(fake_logits, fake_in))
                            d_loss = d_loss + 0.5 * args.r1_gamma * gp
                    else:
                        d_loss = d_hinge_loss(disc(aug(imgs.detach() * 2 - 1)),
                                              disc(aug(rec.detach() * 2 - 1)))
                    d_loss.backward()
                    d_opt.step()
                    # ---- generator step (disc frozen so grads flow THROUGH it to rec
                    #      without touching disc params) ----
                    disc.requires_grad_(False)
                    opt.zero_grad(set_to_none=True)
                    rec_loss = F.l1_loss(rec, imgs) + args.lpips_weight * lpips_loss(rec, imgs)
                    if args.gan_loss == "r3gan":
                        # detached real baseline for the relativistic term (disc frozen +
                        # imgs detached -> no grad to generator); grad flows only via fake.
                        real_base = disc(aug(imgs.detach() * 2 - 1))
                        g_loss = r3gan_g_loss(real_base, disc(aug(rec * 2 - 1)))
                    else:
                        g_loss = g_hinge_loss(disc(aug(rec * 2 - 1)))
                    if gstep >= args.gan_warmup_steps:
                        w = adaptive_gan_weight(rec_loss, g_loss, last_layer) * args.gan_weight
                        ramp = min(1.0, (gstep - args.gan_warmup_steps) / 100.0)
                        loss = rec_loss + ramp * w * g_loss
                    else:
                        loss = rec_loss
                    loss.backward()
                    opt.step()

                gstep += 1
                now = time.perf_counter()
                if now - last_hb_t >= args.log_secs:
                    sps = (gstep - last_hb_step) / max(1e-6, now - last_hb_t)
                    print(f"[{args.run}/{phase_name}] step {gstep} lr={lr:.2e} "
                          f"loss={loss.item():.4f} ({sps:.1f} it/s)", flush=True)
                    last_hb_t, last_hb_step = now, gstep
                if now - last_eval_t >= args.eval_secs:
                    r = evaluate(phase_name, gstep, res)
                    best = min(best, r)
                    if r <= best:
                        torch.save({"model": model.state_dict(), "phase": phase_name,
                                    "config": {"base_channels": base, "latent_dim": args.latent_dim,
                                               "dim_per_block": dim_per_block,
                                               "layers_per_block": enc_layers,
                                               "dec_layers_per_block": dec_layers}},
                                   run_dir / "best.pt")
                    last_eval_t = now
                if epochs <= 0 and now - train_start >= time_budget:
                    done = True
                    break
            epoch += 1
            if epochs > 0:
                print(f"[{args.run}/{phase_name}] epoch {epoch}/{epochs} (step {gstep})", flush=True)
                if epoch >= epochs:
                    done = True
            if done:
                break
        r = evaluate(phase_name, gstep, res)
        best = min(best, r)
        save_grid(phase_name)
        torch.save({"model": model.state_dict(), "phase": phase_name,
                    "config": {"base_channels": base, "latent_dim": args.latent_dim,
                               "dim_per_block": dim_per_block,
                               "layers_per_block": enc_layers,
                               "dec_layers_per_block": dec_layers,
                               "attn_stages": attn_stages, "attn_dim": args.attn_dim}},
                   run_dir / f"{phase_name}.pt")
        print(f"=== {phase_name} done: best_rfid={best:.3f} ===", flush=True)
        return best

    # ---- run the three phases ----
    b1 = run_phase("phase1", res=args.p1_res, time_budget=args.p1_time, base_lr=args.p1_lr,
                   min_lr_frac=0.05, trainable_prefixes=None, use_gan=False, use_compile=True,
                   epochs=args.epochs)
    b2 = run_phase("phase2", res=EVAL_RES, time_budget=args.p2_time, base_lr=args.p2_lr,
                   min_lr_frac=0.05, trainable_prefixes=middle_prefixes(model),
                   use_gan=False, use_compile=False, epochs=args.epochs)
    b3 = run_phase("phase3", res=args.p3_res, time_budget=args.p3_time, base_lr=args.p3_lr,
                   min_lr_frac=0.1, trainable_prefixes=decoder_head_prefixes(model),
                   use_gan=True, use_compile=False, betas=(0.5, 0.9))

    print(f"\nSUMMARY {args.run}: phase1={b1:.3f} phase2={b2:.3f} phase3={b3:.3f}")
    torch.save({"model": model.state_dict()}, run_dir / "final.pt")


if __name__ == "__main__":
    main()

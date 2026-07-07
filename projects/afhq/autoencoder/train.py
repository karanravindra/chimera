"""Train PetPaletteAE, a DC-AE-style conv autoencoder, on AFHQ animal faces.

    uv run python projects/afhq/autoencoder/train.py

This is a **three-phase curriculum** (one in-memory model, weights carry over):

1. Phase 1 -- L1 + LPIPS @ 64x64, train the *full* model (``--phase1-epochs``).
2. Phase 2 -- L1 + LPIPS @ 256x256, train *only* the latent bottleneck
   (``to_latent`` / ``from_latent`` 1x1 convs) so it adapts to high resolution
   (``--phase2-epochs``).
3. Phase 3 -- L1 + LPIPS + PatchGAN @ 64x64, train *only* the output head
   (``out_from_channels``) to sharpen texture (``--phase3-epochs``).

Validation is always at 256x256 (the model is fully convolutional) and reports
L1, LPIPS, PSNR, SSIM, and reconstruction-FID (rFID).

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/afhq/autoencoder``).
Each phase writes a stable, overwritten checkpoint
(``checkpoints/ae_phase{1,2,3}.ckpt``); ``ae_phase3.ckpt`` is the final AE a
downstream project loads. ``main.ipynb`` loads that checkpoint for analysis only.
"""

import argparse
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import AFHQDataModule
from chimera.models import PatchGANDiscriminator, PetPaletteAE
from chimera.modules import AdversarialAutoencoderModule, AutoencoderModule
from chimera.optim import LinearWarmupCosineAnnealingLR
from chimera.utils import build_run_loggers

# PetPaletteAE config is fixed (the notebook hardcoded it, not a swept knob).
AE_LATENT_DIM = 4
AE_BASE_CHANNELS = 32
AE_FSQ_LEVELS = [8, 5, 5, 5]

# Phase-2 fine-tunes the latent bottleneck; phase-3 the output head. The rest of
# the AE (params + BatchNorm running stats) is frozen by AutoencoderModule /
# AdversarialAutoencoderModule via ``train_only``.
TRAIN_ONLY_P2 = ["to_latent", "from_latent"]
TRAIN_ONLY_P3 = ["out_from_channels"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/afhq/autoencoder")
    # Reproducibility: seed all RNGs (incl. dataloader workers + the train/val
    # split, so the 64px and 256px datamodules share split indices). Pair with
    # Trainer(deterministic=True) below for deterministic CUDA kernels too.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="afhq-autoencoder")
    p.add_argument("--wandb-offline", action="store_true")

    # LPIPS weight is shared by all three phases; 1.0 is the recipe's top rFID
    # lever (do not drop it for the AE, unlike the latent-tokenizer runs).
    p.add_argument("--lpips-weight", type=float, default=1.0)

    # Phase 1 -- full model @ 64x64.
    p.add_argument("--phase1-epochs", type=int, default=10)
    p.add_argument("--phase1-lr", type=float, default=1e-3)
    p.add_argument("--phase1-batch-size", type=int, default=64)
    p.add_argument("--phase1-warmup-steps", type=int, default=100)

    # Phase 2 -- bottleneck only @ 256x256 (also supplies the 256px val set).
    p.add_argument("--phase2-epochs", type=int, default=15)
    p.add_argument("--phase2-lr", type=float, default=1e-3)
    p.add_argument("--phase2-batch-size", type=int, default=32)
    p.add_argument("--phase2-warmup-steps", type=int, default=100)

    # Phase 3 -- output head only @ 64x64, adversarial (PatchGAN, hinge + DiffAug).
    p.add_argument("--phase3-epochs", type=int, default=15)
    p.add_argument("--phase3-lr", type=float, default=2e-4)
    p.add_argument("--phase3-batch-size", type=int, default=64)
    p.add_argument("--phase3-warmup-steps", type=int, default=100)
    p.add_argument("--gan-weight", type=float, default=0.1)
    return p.parse_args()


def make_dm(data_dir, image_size, batch_size, seed):
    """AFHQ datamodule in [0, 1] pixels (sigmoid decoder + data_range=1.0 metrics)."""
    dm = AFHQDataModule(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=8,
        normalize=False,
        seed=seed,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"

    # Loggers are built ONCE and shared across all three phases, so the whole
    # curriculum is a single coherent wandb run (and exactly one run per sweep
    # trial). Lightning's WandbLogger logs the per-phase ``trainer/global_step``
    # as a regular metric and lets wandb auto-increment its internal step, so
    # metrics from every phase are recorded (the global_step x-axis just sawtooths
    # back to 0 at each phase boundary).
    loggers = build_run_loggers(
        run_dir,
        args.wandb_project,
        None,
        args.wandb_offline,
        tags=["petpalette-ae", "3phase-curriculum"],
    )

    # One model instance threads through all three phases, so trained weights
    # carry over in memory (the freezing schemes only change which params get
    # gradients, never the tensors themselves).
    model = PetPaletteAE(
        in_channels=3,
        latent_dim=AE_LATENT_DIM,
        base_channels=AE_BASE_CHANNELS,
        fsq_levels=AE_FSQ_LEVELS,
    )

    # Validation is always at 256x256; the phase-2 datamodule supplies it (and
    # its own train loader). Sharing the seed keeps the 64px/256px splits aligned.
    dm256 = make_dm(args.data_dir, 256, args.phase2_batch_size, args.seed)
    val_loader = dm256.val_dataloader()

    def phase_checkpoint(name):
        # enable_version_counter=False overwrites checkpoints/<name>.ckpt each run
        # instead of appending -v1.ckpt, so the load path stays stable downstream.
        return ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=name,
            monitor="val/loss",
            enable_version_counter=False,
        )

    # ---- Phase 1: L1 + LPIPS @ 64x64, full model ----
    dm64_p1 = make_dm(args.data_dir, 64, args.phase1_batch_size, args.seed)
    train_loader = dm64_p1.train_dataloader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.phase1_lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=args.phase1_warmup_steps,
        n_epochs=args.phase1_epochs,
        train_loader_length=len(train_loader),
    )
    p1 = AutoencoderModule(
        model, optimizer, scheduler, lpips_weight=args.lpips_weight, compute_rfid=True
    )
    ckpt1 = phase_checkpoint("ae_phase1")
    trainer = Trainer(
        max_epochs=args.phase1_epochs,
        precision="bf16-mixed",
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        deterministic=True,
        check_val_every_n_epoch=2,
        logger=loggers,
        callbacks=[ckpt1],
    )
    trainer.fit(p1, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(p1, dataloaders=val_loader)

    # ---- Phase 2: L1 + LPIPS @ 256x256, bottleneck only ----
    train_loader = dm256.train_dataloader()
    params = AutoencoderModule.trainable_params(model, TRAIN_ONLY_P2)
    optimizer = torch.optim.AdamW(params, lr=args.phase2_lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=args.phase2_warmup_steps,
        n_epochs=args.phase2_epochs,
        train_loader_length=len(train_loader),
    )
    p2 = AutoencoderModule(
        model,
        optimizer,
        scheduler,
        lpips_weight=args.lpips_weight,
        compute_rfid=True,
        train_only=TRAIN_ONLY_P2,
    )
    ckpt2 = phase_checkpoint("ae_phase2")
    trainer = Trainer(
        max_epochs=args.phase2_epochs,
        precision="bf16-mixed",
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        deterministic=True,
        logger=loggers,
        callbacks=[ckpt2],
    )
    trainer.fit(p2, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(p2, dataloaders=val_loader)

    # ---- Phase 3: L1 + LPIPS + PatchGAN @ 64x64, output head only ----
    dm64_p3 = make_dm(args.data_dir, 64, args.phase3_batch_size, args.seed)
    train_loader = dm64_p3.train_dataloader()
    discriminator = PatchGANDiscriminator(in_channels=3, base_channels=64, n_layers=3)
    g_params = AdversarialAutoencoderModule.trainable_params(model, TRAIN_ONLY_P3)
    opt_g = torch.optim.AdamW(g_params, lr=args.phase3_lr, betas=(0.5, 0.9))
    opt_d = torch.optim.AdamW(
        discriminator.parameters(), lr=args.phase3_lr, betas=(0.5, 0.9)
    )
    sched_g = LinearWarmupCosineAnnealingLR(
        opt_g,
        warmup_steps=args.phase3_warmup_steps,
        n_epochs=args.phase3_epochs,
        train_loader_length=len(train_loader),
    )
    sched_d = LinearWarmupCosineAnnealingLR(
        opt_d,
        warmup_steps=args.phase3_warmup_steps,
        n_epochs=args.phase3_epochs,
        train_loader_length=len(train_loader),
    )
    p3 = AdversarialAutoencoderModule(
        model,
        discriminator,
        opt_g,
        opt_d,
        sched_g=sched_g,
        sched_d=sched_d,
        lpips_weight=args.lpips_weight,
        gan_weight=args.gan_weight,
        compute_rfid=True,
        train_only=TRAIN_ONLY_P3,
    )
    ckpt3 = phase_checkpoint("ae_phase3")
    # No Trainer gradient clipping: phase 3 is a manual-optimization GAN.
    trainer = Trainer(
        max_epochs=args.phase3_epochs,
        precision="bf16-mixed",
        deterministic=True,
        logger=loggers,
        callbacks=[ckpt3],
    )
    trainer.fit(p3, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(p3, dataloaders=val_loader)

    print("final AE checkpoint:", ckpt_dir / "ae_phase3.ckpt")


if __name__ == "__main__":
    main()

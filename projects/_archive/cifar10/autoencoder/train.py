"""Train CIFARAutoencoder, a small conv autoencoder, on CIFAR-10.

    uv run python projects/cifar10/autoencoder/train.py

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/cifar10/autoencoder``).
The checkpoint filename is fixed (``ae.ckpt``, overwritten each run) so downstream
projects have a stable path to load from. ``main.ipynb`` loads the resulting
checkpoint for analysis only.
"""

import argparse
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import CIFAR10DataModule
from chimera.models import CIFARAutoencoder
from chimera.modules import AutoencoderModule
from chimera.optim import LinearWarmupCosineAnnealingLR
from chimera.utils import build_run_loggers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/cifar10/autoencoder")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    # Reproducibility: seed all RNGs (incl. dataloader workers). Pair with
    # Trainer(deterministic=True) below for deterministic CUDA kernels too.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="cifar10-autoencoder")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--latent-dim", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    # Unnormalized [0, 1] pixels to match the decoder's sigmoid output.
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    model = CIFARAutoencoder(in_channels=3, latent_dim=args.latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        n_epochs=args.epochs,
        train_loader_length=len(dm.train_dataloader()),
    )
    autoencoder_module = AutoencoderModule(model, optimizer, scheduler)

    run_dir = Path(args.run_dir)
    # enable_version_counter=False overwrites checkpoints/ae.ckpt each run instead
    # of appending ae-v1.ckpt, so the load path stays stable for downstream projects.
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="ae",
        monitor="val/loss",
        enable_version_counter=False,
    )
    loggers = build_run_loggers(run_dir, args.wandb_project, None, args.wandb_offline)

    trainer = Trainer(
        max_epochs=args.epochs,
        precision="bf16-mixed",
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        deterministic=True,
        logger=loggers,
        callbacks=[checkpoint],
    )
    trainer.fit(autoencoder_module, datamodule=dm)
    trainer.test(
        autoencoder_module, datamodule=dm, ckpt_path=checkpoint.best_model_path
    )
    print("best checkpoint:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()

"""Train DigitDreamerAE, a small conv autoencoder, on MNIST.

    uv run python projects/mnist/autoencoder/train.py

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/mnist/autoencoder``).
The checkpoint filename is fixed (``ae.ckpt``, overwritten each run) so
``projects/mnist/rectified_flow`` has a stable path to load from.
``main.ipynb`` loads the resulting checkpoint for analysis only.
"""

import argparse
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import MNISTDataModule
from chimera.models import DigitDreamerAE
from chimera.modules import AutoencoderModule
from chimera.optim import LinearWarmupCosineAnnealingLR
from chimera.utils import build_run_loggers

MODEL_VARIANTS = ["tiny", "small", "medium", "large"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/mnist/autoencoder")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--latent-dim", type=int, default=1)
    p.add_argument("--model-variant", choices=MODEL_VARIANTS, default="tiny")
    # seed 42 lands this 1-channel-bottleneck AE in a black-image collapse basin
    # (BatchNorm + L1 + an extreme bottleneck makes "predict all-zero" an easy
    # local optimum); seed 0 trains cleanly to psnr ~23dB. Not a data/precision
    # issue -- verified by reproducing the collapse in fp32 with no grad clipping.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="mnist-autoencoder")
    p.add_argument("--wandb-offline", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    # Unnormalized [0, 1] pixels to match the decoder's sigmoid output.
    dm = MNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=4,
        image_size=32,
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    model = DigitDreamerAE.from_variant(
        args.model_variant, in_channels=1, latent_dim=args.latent_dim
    )
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
    loggers = build_run_loggers(
        run_dir, args.wandb_project, "autoencoder", args.wandb_offline, tags=[args.model_variant]
    )

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
    trainer.test(autoencoder_module, datamodule=dm, ckpt_path=checkpoint.best_model_path)
    print("best checkpoint:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()

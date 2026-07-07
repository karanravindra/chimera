"""Train DigitDreamer (rectified flow) in the latent space of a pretrained
MNIST autoencoder.

Pipeline: load pretrained AE -> precompute + cache latents -> train DigitDreamer.

    uv run python projects/mnist/rectified_flow/train.py

Run ``projects/mnist/autoencoder/train.py`` first so its checkpoint exists.
Checkpoints + logs go to ``--run-dir`` (default
``/mnt/ai/runs/mnist/rectified_flow``); ``main.ipynb`` loads the resulting
checkpoints (AE + DigitDreamer) for analysis and sampling only.
"""

import argparse
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import MNISTLatentDataModule
from chimera.models import DigitDreamer, DigitDreamerAE
from chimera.modules import RectifiedFlowModule
from chimera.optim import LinearWarmupCosineAnnealingLR
from chimera.utils import build_run_loggers

MODEL_VARIANTS = ["tiny", "small", "medium", "large"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/mnist/rectified_flow")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="mnist-rectified-flow")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--model-variant", choices=MODEL_VARIANTS, default="tiny")
    p.add_argument(
        "--ae-ckpt", default="/mnt/ai/runs/mnist/autoencoder/checkpoints/ae.ckpt"
    )
    # Must match whatever --model-variant produced --ae-ckpt (not swept -- pinned
    # to a specific pretrained checkpoint, same reasoning as --latent-channels).
    p.add_argument("--ae-model-variant", choices=MODEL_VARIANTS, default="tiny")
    p.add_argument("--latent-channels", type=int, default=1)
    return p.parse_args()


def load_autoencoder(ckpt_path, latent_channels, model_variant, device):
    ae = DigitDreamerAE.from_variant(model_variant, in_channels=1, latent_dim=latent_channels)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ae_state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    ae.load_state_dict(ae_state)
    return ae.to(device).eval()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ae = load_autoencoder(args.ae_ckpt, args.latent_channels, args.ae_model_variant, device)
    print(f"loaded AE from {args.ae_ckpt} ({sum(p.numel() for p in ae.parameters()):,} params)")

    latent_dm = MNISTLatentDataModule(
        autoencoder=ae,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        device=device,
        image_size=32,
    )
    latent_dm.prepare_data()
    latent_dm.setup("fit")

    digit_dreamer = DigitDreamer.from_variant(
        args.model_variant,
        latent_channels=args.latent_channels,
        latent_size=4,
        patch_size=1,
        n_classes=10,
        n_cond_tokens=4,
        class_dropout_prob=0.1,
    )
    n_params = sum(p.numel() for p in digit_dreamer.parameters())
    print(f"DigitDreamer parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        digit_dreamer.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        n_epochs=args.epochs,
        train_loader_length=len(latent_dm.train_dataloader()),
    )
    rf_module = RectifiedFlowModule(digit_dreamer, optimizer, scheduler)

    run_dir = Path(args.run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="digit_dreamer",
        monitor="val/loss",
        enable_version_counter=False,
    )
    loggers = build_run_loggers(
        run_dir, args.wandb_project, None, args.wandb_offline, tags=[args.model_variant]
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
    trainer.fit(rf_module, datamodule=latent_dm)
    print("best checkpoint:", checkpoint.best_model_path)
    print("latent cache:", latent_dm.cache_path)


if __name__ == "__main__":
    main()

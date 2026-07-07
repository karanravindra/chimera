"""Train DigitNet (LeNet-5) on MNIST.

    uv run python projects/mnist/classifier/train.py

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/mnist/classifier``);
``main.ipynb`` loads the resulting checkpoint for analysis only.
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from chimera.data import MNISTDataModule
from chimera.models import DigitNet
from chimera.modules import ClassifierModule
from chimera.optim import LinearWarmupCosineAnnealingLR
from chimera.utils import build_run_loggers

NUM_CLASSES = 10
MODEL_VARIANTS = ["tiny", "small", "medium", "large"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/mnist/classifier")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="mnist-classifier")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--model-variant", choices=MODEL_VARIANTS, default="tiny")

    return p.parse_args()


def class_weights_from(loader):
    """Inverse-frequency class weights, normalized to average 1."""
    counts = Counter()
    for _, labels in tqdm(loader, desc="Counting training labels"):
        counts.update(labels.tolist())
    counts = torch.tensor([counts[c] for c in range(NUM_CLASSES)], dtype=torch.float)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return weights / weights.mean()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    dm = MNISTDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=4)
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    class_weights = class_weights_from(dm.train_dataloader())

    model = DigitNet.from_variant(args.model_variant, in_channels=1, num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        n_epochs=args.epochs,
        train_loader_length=len(dm.train_dataloader()),
    )
    classifier_module = ClassifierModule(
        model,
        optimizer,
        scheduler,
        class_weights=class_weights,
        num_classes=NUM_CLASSES,
        log_confusion_matrix=True,
    )

    run_dir = Path(args.run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="classifier",
        monitor="val/acc",
        mode="max",
        enable_version_counter=False,
    )
    loggers = build_run_loggers(
        run_dir, args.wandb_project, None, args.wandb_offline, tags=[args.model_variant]
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        precision="bf16-mixed",
        deterministic=True,
        logger=loggers,
        callbacks=[checkpoint],
    )
    trainer.fit(classifier_module, datamodule=dm)
    trainer.test(classifier_module, datamodule=dm, ckpt_path=checkpoint.best_model_path)
    print("best checkpoint:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()

"""Train a CIFAR-style ResNet-18 on CIFAR-10.

    uv run python projects/cifar10/classify/train.py

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/cifar10/classify``);
``main.ipynb`` loads the resulting checkpoint for analysis only.
"""

import argparse
from collections import Counter
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from chimera.data import CIFAR10DataModule
from chimera.models import ResNet
from chimera.modules import ClassifierModule
from chimera.optim import LinearWarmupCosineAnnealingLR
from chimera.utils import build_run_loggers

NUM_CLASSES = 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/cifar10/classify")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="cifar10-classifier")
    p.add_argument("--wandb-offline", action="store_true")
    # project-specific
    p.add_argument("--weight-decay", type=float, default=5e-4)

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

    # Notebook parity: pin_memory=False, num_workers=0 (single-process loading).
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    class_weights = class_weights_from(dm.train_dataloader())

    model = ResNet(in_channels=3, num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
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
        run_dir, args.wandb_project, None, args.wandb_offline
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

"""Train a small CNN+GRU VQA baseline on CLEVR.

    uv run python projects/clevr/vqa/train.py

By default this expects the extracted official dataset at
``/mnt/ai/data/CLEVR_v1.0``. Pass ``--download`` to fetch the full ~18GB
official archive from the CLEVR mirror.

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/clevr/vqa``);
``main.ipynb`` loads the resulting checkpoint for analysis only.
"""

import argparse
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import CLEVRVQADataModule
from chimera.models import CLEVRVQAModel
from chimera.modules import VQAModule
from chimera.optim import LinearWarmupCosineAnnealingLR
from chimera.utils import build_run_loggers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/clevr/vqa")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="clevr-vqa")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--fast-dev-run", action="store_true")

    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-question-len", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--image-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-confusion-matrix", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    dm = CLEVRVQADataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_question_len=args.max_question_len,
        num_workers=args.num_workers,
        pin_memory=True,
        download=args.download,
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")
    print(
        f"question_vocab={dm.vocab_size} "
        f"num_answers={dm.num_answers} "
        f"answers={dm.answer_names}"
    )

    model = CLEVRVQAModel(
        vocab_size=dm.vocab_size,
        num_answers=dm.num_answers,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        image_dim=args.image_dim,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CLEVR VQA parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        n_epochs=args.epochs,
        train_loader_length=len(dm.train_dataloader()),
    )
    vqa_module = VQAModule(
        model,
        optimizer,
        scheduler,
        answer_names=dm.answer_names,
        log_confusion_matrix=args.log_confusion_matrix,
    )

    run_dir = Path(args.run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="vqa",
        monitor="val/acc",
        mode="max",
        enable_version_counter=False,
    )
    loggers = build_run_loggers(run_dir, args.wandb_project, None, args.wandb_offline)

    trainer = Trainer(
        max_epochs=args.epochs,
        precision="bf16-mixed",
        deterministic=True,
        logger=loggers,
        callbacks=[checkpoint],
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(vqa_module, datamodule=dm)
    ckpt_path = checkpoint.best_model_path or None
    trainer.test(vqa_module, datamodule=dm, ckpt_path=ckpt_path)
    print("best checkpoint:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()

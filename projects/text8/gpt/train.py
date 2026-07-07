"""Train a decoder-only GPT (GQA + RoPE + QK-norm) on text8.

text8 is tokenized with the pretrained LiquidAI/LFM2.5-230M subword tokenizer.
Next-token cross-entropy is fused with the lm_head via Cut Cross Entropy, and the
model is optimized with Muon (hidden matrices) + AdamW (embedding/head/1D params).

    uv run python projects/text8/gpt/train.py

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/text8/gpt``);
``main.ipynb`` loads the resulting checkpoint for analysis and sampling only.
"""

import argparse
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import Text8DataModule
from chimera.models import GPT
from chimera.modules import LanguageModelModule
from chimera.optim import LinearWarmupCosineAnnealingLR, Muon, muon_param_groups
from chimera.utils import build_run_loggers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/text8/gpt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    # Muon learning rate for the hidden weight matrices (the primary lr).
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="text8-gpt")
    p.add_argument("--wandb-offline", action="store_true")
    # AdamW learning rate for the embedding / head / 1D params (Muon aux group).
    p.add_argument("--adamw-lr", type=float, default=1e-3)
    # GPT has no from_variant, so its hyperparameters are exposed directly.
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--n-embd", type=int, default=48)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-kv-head", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    dm = Text8DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=0,
        pin_memory=False,
        tokenizer_backend="pretrained",  # LiquidAI/LFM2.5-230M
    )
    dm.prepare_data()
    dm.setup("fit")
    print(f"tokenizer={dm.pretrained_id}  vocab_size={dm.vocab_size}")

    # round to multiple of 32
    rounded_vocab_size = (dm.vocab_size + 31) // 32 * 32

    model = GPT(
        vocab_size=rounded_vocab_size,
        block_size=args.seq_len,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        tie_embedding=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT parameters: {n_params / 1e6:.2f}M")

    optimizer = Muon(
        muon_param_groups(
            model,
            muon_lr=args.lr,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=0.1,
        )
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        n_epochs=args.epochs,
        train_loader_length=len(dm.train_dataloader()),
    )

    model = torch.compile(model, mode="reduce-overhead")
    # use_cce: fuse lm_head + cross-entropy (apple/ml-cross-entropy), no logits materialized
    lm_module = LanguageModelModule(model, optimizer, scheduler, use_cce=True)

    run_dir = Path(args.run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="gpt",
        monitor="val/loss",
        enable_version_counter=False,
    )
    loggers = build_run_loggers(run_dir, args.wandb_project, None, args.wandb_offline)

    trainer = Trainer(
        max_epochs=args.epochs,
        precision="bf16-true",
        gradient_clip_val=1.0,
        deterministic=True,
        logger=loggers,
        callbacks=[checkpoint],
    )
    trainer.fit(lm_module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    trainer.test(lm_module, dataloaders=dm.val_dataloader())
    print("best checkpoint:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()

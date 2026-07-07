"""Pretrain a decoder-only GPT (GQA + RoPE + QK-norm) on FineWeb-Edu.

FineWeb-Edu ``sample-10BT`` tokenized with the pretrained LiquidAI/LFM2.5-230M
subword tokenizer; documents are concatenated with an ``<|endoftext|>`` separator
so the model learns document boundaries. Optimized with Muon (2D hidden weight
matrices) + AdamW (embedding/head/biases/norms) under one LR schedule, with
``torch.compile`` and Cut Cross Entropy (fused lm_head + cross-entropy).

    uv run python projects/fineweb-edu/gpt/train.py

The effective (global) token count per step is ``--global-token-count`` and the
per-sequence length is ``--seq-len``; the micro-batch size is derived as
``global_token_count // seq_len`` (which must divide evenly).

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/fineweb-edu/gpt``);
``main.ipynb`` loads the resulting checkpoint for analysis and text generation only.
"""

import argparse
import os
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import FineWebEduDataModule
from chimera.models import GPT
from chimera.modules import LanguageModelModule
from chimera.optim import LinearWarmupCosineAnnealingLR, Muon, muon_param_groups
from chimera.utils import build_run_loggers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/fineweb-edu/gpt")
    p.add_argument("--epochs", type=int, default=1)
    # Effective tokens per optimizer step; the micro-batch size is derived as
    # global_token_count // seq_len (must divide evenly). Exposed instead of a
    # raw --batch-size so the effective batch is defined in tokens, not sequences.
    p.add_argument("--global-token-count", type=int, default=65536)
    p.add_argument("--seq-len", type=int, default=2048)
    # Cap on max optimizer steps (the notebook runs a 500-step smoke run over the
    # capped token budget); set to -1 to disable and run the full --epochs.
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--max-train-tokens", type=int, default=1_000_000_000)
    # Muon and AdamW each carry their own base LR (see muon_param_groups); both
    # anneal under a single LinearWarmupCosineAnnealingLR.
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adamw-lr", type=float, default=8e-4)
    p.add_argument("--adamw-weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="fineweb-edu-gpt")
    p.add_argument("--wandb-offline", action="store_true")
    # GPT has no from_variant, so its hyperparameters are exposed directly.
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-kv-head", type=int, default=3)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--no-tie-embedding", dest="tie_embedding", action="store_false")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--no-cce", dest="use_cce", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    # datasets + tokenizer caches live on the big volume (as in the notebook)
    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    # Reproducibility: seed all RNGs (incl. dataloader workers). Paired with
    # Trainer(deterministic=True) below for deterministic CUDA kernels too.
    seed_everything(args.seed, workers=True)

    if args.global_token_count % args.seq_len != 0:
        raise ValueError(
            f"GLOBAL_TOKEN_COUNT ({args.global_token_count}) must be divisible "
            f"by SEQ_LEN ({args.seq_len})"
        )
    batch_size = args.global_token_count // args.seq_len
    print(
        f"global tokens/step={args.global_token_count}  seq_len={args.seq_len}  "
        f"-> batch_size={batch_size}"
    )

    dm = FineWebEduDataModule(
        data_dir=args.data_dir,
        name="sample-10BT",
        batch_size=batch_size,
        seq_len=args.seq_len,
        tokenizer_backend="pretrained",  # LiquidAI/LFM2.5-230M
        add_eos=True,  # append <|endoftext|> after each document
        max_train_tokens=args.max_train_tokens,
        num_workers=4,
    )
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"tokenizer={dm.pretrained_id}  vocab_size={dm.vocab_size}")
    print(f"eos_token={dm.eos_token!r}  eos_id={dm.eos_id}")

    model = GPT(
        vocab_size=dm.vocab_size,
        block_size=args.seq_len,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        tie_embedding=args.tie_embedding,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT parameters: {n_params / 1e6:.2f}M")

    # Muon routes the 2D hidden weight matrices (attention + MLP) to Muon and the
    # token embedding, output head, biases, and norm gains to AdamW; both groups
    # share one LR schedule (each anneals from its own base LR).
    optimizer = Muon(
        muon_param_groups(
            model,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=args.adamw_weight_decay,
        )
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=args.warmup_steps,
        n_epochs=args.epochs,
        train_loader_length=len(train_loader),
    )

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    # use_cce: fuse lm_head + cross-entropy (apple/ml-cross-entropy), no logits materialized
    lm_module = LanguageModelModule(model, optimizer, scheduler, use_cce=args.use_cce)

    run_dir = Path(args.run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="gpt",
        monitor="val/loss",
        enable_version_counter=False,
    )
    loggers = build_run_loggers(
        run_dir, args.wandb_project, None, args.wandb_offline
    )

    trainer = Trainer(
        max_steps=args.max_steps,
        max_epochs=args.epochs,
        precision="bf16-true",
        gradient_clip_val=1.0,
        deterministic=True,
        logger=loggers,
        callbacks=[checkpoint],
    )
    # FineWebEduDataModule exposes only train/val loaders (no test split), so the
    # loaders are passed explicitly; validation reuses the val loader as in the
    # notebook (logged under train/*, val/*, and test/* respectively).
    trainer.fit(lm_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lm_module, dataloaders=val_loader)
    print("best checkpoint:", checkpoint.best_model_path)


if __name__ == "__main__":
    main()

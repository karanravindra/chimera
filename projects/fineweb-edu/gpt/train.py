"""Pretrain a decoder-only GPT (GQA + RoPE + QK-norm) on FineWeb-Edu.

FineWeb-Edu ``sample-10BT`` tokenized with the pretrained LiquidAI/LFM2.5-230M
subword tokenizer; documents are concatenated with an ``<|endoftext|>`` separator
so the model learns document boundaries. Optimized with Muon (2D hidden weight
matrices) + AdamW (embedding/head/norms) under one LR schedule, with
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
from bench import DEFAULT_TASKS, flatten_for_wandb, print_table, run_benchmarks


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
    # muTransfer optimum from the emb48 muP sweep: these transfer across width
    # unchanged (Muon spectral-normalized + AdamW on the width-independent
    # embedding/head). Found via a wide 2D LR map, then confirmed at full length
    # (5000 steps): adamw=8e-3 reaches val/loss 5.13 vs 5.33 at the old 1e-3.
    # CAVEAT: adamw=8e-3 is a HOT LR — it needs the cosine schedule to anneal over
    # the ACTUAL run horizon or it diverges late (constant-LR final 5.37; cosine
    # matched to the run 5.13). Keep the scheduler on and its period = run length.
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adamw-lr", type=float, default=4e-3)
    p.add_argument("--adamw-weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eta-min", type=float, default=1e-5)
    # Run validation every N optimizer steps (so we get a val/loss curve during
    # training, not just one point at the end). limit-val-batches caps how many
    # val batches each validation pass uses, keeping periodic validation cheap on
    # the large (1% of corpus) val split.
    p.add_argument("--val-check-interval", type=int, default=500)
    p.add_argument("--limit-val-batches", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="fineweb-edu-gpt")
    # Explicit wandb run name; left unset, wandb assigns a random one. Useful to
    # tag a run with what makes it different from the rest (an ablation, a fix).
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-offline", action="store_true")
    # wandb run tags, comma-separated (e.g. "mup,sweep").
    p.add_argument("--tags", default=None)
    # GPT has no from_variant, so its hyperparameters are exposed directly.
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-kv-head", type=int, default=3)
    p.add_argument("--n-layer", type=int, default=6)
    # Compact override of the four model dims as "n_embd-n_head-n_kv_head-n_layer"
    # (e.g. "48-2-1-3"). Lets a single wandb sweep vary width by sweeping ONE
    # coupled parameter (the dims are not independent, so a grid over each would
    # produce invalid combinations). When set, overrides --n-embd/--n-head/etc.
    p.add_argument("--arch", default=None)
    # muP (Maximal Update Parameterization): tune muon-lr/adamw-lr (+ the mults
    # below) at a small proxy width, then transfer them to a large model by only
    # changing --n-embd/--n-head (keep head_dim fixed). At n_embd == mup-base-width
    # this reduces to the original GPT-2-style init.
    p.add_argument("--mup-base-width", type=int, default=256)
    p.add_argument("--mup-base-std", type=float, default=0.02)
    p.add_argument("--mup-input-mult", type=float, default=1.0)
    p.add_argument("--mup-output-mult", type=float, default=1.0)
    # Disable the LR scheduler entirely -> constant LR (no warmup/cosine). Cleaner
    # for muP LR sweeps, where a schedule would confound which LR value matters.
    p.add_argument("--no-scheduler", dest="use_scheduler", action="store_false")
    # Recompute block activations in backward instead of storing them: lets wide
    # models fit on limited VRAM without shrinking tokens/step (~1 extra fwd cost).
    p.add_argument("--grad-checkpoint", dest="gradient_checkpointing", action="store_true")
    p.add_argument("--no-tie-embedding", dest="tie_embedding", action="store_false")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--no-cce", dest="use_cce", action="store_false")
    # Skip the final trainer.test() pass (redundant with periodic validation when
    # a sweep only needs to rank runs on val/loss; saves a full val-loader pass).
    p.add_argument("--no-test", dest="run_test", action="store_false")
    # Zero-shot benchmark suite (bench.py) run after trainer.test() and logged
    # to wandb under test/<task>/<metric>; a sweep can skip it to save time.
    p.add_argument("--eval-tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--eval-batch-tokens", type=int, default=32768)
    p.add_argument("--no-eval", dest="run_eval", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    if args.arch:
        args.n_embd, args.n_head, args.n_kv_head, args.n_layer = (
            int(x) for x in args.arch.split("-")
        )
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
        mup_base_width=args.mup_base_width,
        mup_base_std=args.mup_base_std,
        mup_input_mult=args.mup_input_mult,
        mup_output_mult=args.mup_output_mult,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT parameters: {n_params / 1e6:.2f}M")

    # Muon routes the 2D hidden weight matrices (attention + MLP) to Muon and the
    # token embedding, output head, and norm gains to AdamW; both groups
    # share one LR schedule (each anneals from its own base LR).
    optimizer = Muon(
        muon_param_groups(
            model,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=args.adamw_weight_decay,
        )
    )
    if args.use_scheduler:
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=args.warmup_steps,
            n_epochs=args.epochs,
            train_loader_length=len(train_loader),
            eta_min=args.eta_min,
            # Anneal over the actual horizon: when --max-steps caps the run short
            # of a full epoch, the cosine must cool down by max_steps, not by the
            # (never-reached) epoch length — else the LR is still hot at the cutoff.
            max_steps=args.max_steps,
        )
    else:
        scheduler = None  # constant LR

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
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    loggers = build_run_loggers(
        run_dir, args.wandb_project, args.run_name, args.wandb_offline, tags=tags
    )
    # so main.ipynb can pull model hyperparams (n_embd, seq_len, ...) for ANY past
    # run straight from wandb config instead of hardcoding them per checkpoint.
    loggers[1].log_hyperparams(vars(args))

    trainer = Trainer(
        max_steps=args.max_steps,
        max_epochs=args.epochs,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
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
    if args.run_test:
        trainer.test(lm_module, dataloaders=val_loader)
    print("best checkpoint:", checkpoint.best_model_path)

    if args.run_eval:
        eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
        if eval_tasks:
            # Lightning leaves the model on CPU after fit/test (see chimera memory
            # lightning-cpu-after-test); torch.compile wraps model in _orig_mod, and
            # eval batch shapes vary per request, so score with the raw eager module.
            eval_model = getattr(model, "_orig_mod", model)
            # torch.compile(mode="reduce-overhead") pins several GiB of CUDA-graph
            # private-pool memory sized for the training step shape; switching to
            # the eager _orig_mod doesn't release it, so eval's own (larger,
            # variable-shape) batches OOM against that dead memory. Reset dynamo's
            # cudagraph trees to reclaim it before eval allocates anything.
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            eval_device = "cuda" if torch.cuda.is_available() else "cpu"
            eval_model.to(eval_device)
            results = run_benchmarks(
                eval_model,
                dm.tokenizer,
                eval_tasks,
                block_size=args.seq_len,
                batch_tokens=args.eval_batch_tokens,
                device=eval_device,
            )
            print_table(results)
            loggers[1].log_metrics(flatten_for_wandb(results), step=trainer.global_step)


if __name__ == "__main__":
    main()

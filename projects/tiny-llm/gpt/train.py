"""Pretrain a tiny (5-20M param) decoder-only GPT on the tiny-llm mixture.

Dense GQA + RoPE + QK-norm + ReLU² MLP, muP-parameterized (LRs transfer across
width), Muon (hidden matrices) + AdamW (embedding/head/norms) under one LR
schedule, torch.compile. Trains on a pre-packed mixture built from the tiny-llm
sources (TinyStoriesV2 / tiny-strange-textbooks / finephrase / fineweb-edu; see
../data), tokenized with one of the project's own 4k/8k/16k BPE tokenizers.

Evals (hand-rolled via the Lightning val/test phase, per the design decision):
the mix serves one val loader per source, so we log the headline
**``val/<src>/bpb``** — bits-per-byte per source, each normalized by that
source's own bytes/token (bpb is the tokenizer-invariant metric, so it is the
only one comparable across the 4k/8k/16k tokenizers and across sources). Plus
aggregate ``val/loss`` (checkpoint monitor) and ``val/bpb``. No bpt (redundant
with loss). Downstream benchmarks (BLiMP / LAMBADA / generative judge) are
deferred — added later as extra test-phase evals.

Prerequisite: the packed mix must exist at
``/mnt/ai/data/tiny-llm/mix/<mix>/{train,val}.bin`` (+ manifest.json), tokenized
with ``--tokenizer``. Build it after choosing the vocab (tokenize the sources +
pack per the sources.py weights).

    uv run python projects/tiny-llm/gpt/train.py               # 8k / ~9M smoke run
    uv run python projects/tiny-llm/gpt/train.py --arch base   # ~18M
"""

import argparse
import os
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import MixtureDataModule
from chimera.models import GPT
from chimera.optim import LinearWarmupCosineAnnealingLR, Muon, muon_param_groups
from chimera.utils import EMACallback, ProgressPrinter, TokenAxisCallback, build_run_loggers

from module import TinyLMModule
from bpb import measure as measure_bpb
from bench import DEFAULT_TASKS, flatten_for_wandb, print_table, run_benchmarks


# Tiny muP family (W-H-K-L): head_dim=32, GQA n_kv=1, depth-6 (wall-optimal). Keeps
# head_dim fixed + scales width so the swept muP LR (muon 0.013 / adamw 0.006)
# transfers. Param counts below are for the 8k vocab (tied embedding); they scale
# with vocab (4k smaller, 16k larger) — actual count is printed at startup.
ARCH_PRESETS = {
    "tiny": "256-8-2-6",    # ~6M  @ 8k
    "small": "320-10-2-6",  # ~9M  @ 8k  (default — the ~10M / 8k first config)
    "base": "448-14-2-6",   # ~18M @ 8k
}


# Fixed prompts sampled at test time and logged to wandb (a "test/generations"
# table) — for a tiny model, eyeballing coherence catches what metrics miss. One
# per register in the mix: story, expository, FAQ-style Q&A, procedural.
SAMPLE_PROMPTS = [
    "Once upon a time, there was a little",
    "The sun is a star that",
    "Question: Why do birds fly south in the winter?\nAnswer:",
    "Here is how to plant a seed. First,",
]


def log_generations(model, tokenizer, device, wandb_logger,
                    prompts=SAMPLE_PROMPTS, max_new_tokens=80, temperature=0.8,
                    repetition_penalty=1.3, min_p=0.05):
    """Sample from fixed prompts and log a wandb table (+ print).

    Uses repetition_penalty + min_p by default: plain temperature sampling on a
    tiny/under-trained model degenerates into ``x x x …`` repetition loops; these
    two knobs remove them (see gpt.generate)."""
    import wandb

    model.eval()
    rows = []
    for p in prompts:
        ids = tokenizer._tok.encode(p, add_special_tokens=False).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature,
                             repetition_penalty=repetition_penalty, min_p=min_p)
        cont = tokenizer.decode(out[0].tolist()[len(ids):])
        rows.append([p, cont])
    wandb_logger.experiment.log(
        {"test/generations": wandb.Table(columns=["prompt", "generation"], data=rows)}
    )
    print("\n=== sample generations ===")
    for p, c in rows:
        print(f"> {p!r}\n  {c!r}\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/tiny-llm/gpt")
    p.add_argument("--mix", default="tiny_2B_8k", help="mixture name under tiny-llm/mix/")
    p.add_argument("--tokenizer", default="/mnt/ai/data/tiny-llm/tokenizer/8k",
                   help="local tokenizer dir (4k/8k/16k) or HF id; MUST match the "
                        "tokenizer the mix was packed with.")
    p.add_argument("--vocab-tag", default=None,
                   help="4k/8k/16k: convenience that sets BOTH --tokenizer and --mix "
                        "to the matched pair (tokenizer/<tag> + tiny_2B_<tag>), so a "
                        "vocab sweep can't desync them. Overrides --tokenizer/--mix.")
    p.add_argument("--epochs", type=int, default=1)
    # Effective tokens per optimizer step; micro-batch = global_token_count // seq_len.
    p.add_argument("--global-token-count", type=int, default=65536 // 32)
    # tiny-llm docs are short (TinyStories ~200 tok); ctx-512/1024 beats 2048 at
    # equal tokens and is faster (see ctx-len memory). 1024 balances the longer
    # textbook/web sources + LAMBADA-style coherence.
    p.add_argument("--seq-len", type=int, default=512)
    # Default is a smoke run; a full 2B-token pass is ~2e9 / global_token_count
    # steps (~30.5k at 65536). Raise --max-steps for a real run (mind the run-time
    # budget). -1 disables the cap and runs the full --epochs.
    p.add_argument("--max-steps", type=int, default=1000)
    # muP-transferable LRs (swept optima; transfer across width unchanged).
    p.add_argument("--muon-lr", type=float, default=0.013)
    p.add_argument("--adamw-lr", type=float, default=6e-3)
    p.add_argument("--adamw-weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eta-min", type=float, default=1e-5)
    p.add_argument("--val-check-interval", type=int, default=500)
    p.add_argument("--limit-val-batches", type=int, default=200)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="tiny-llm-pretrain")
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--tags", default=None)
    # Model dims — default is the "small" (~9M @ 8k) config; --arch overrides.
    p.add_argument("--n-embd", type=int, default=320)
    p.add_argument("--n-head", type=int, default=10)
    p.add_argument("--n-kv-head", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--arch", default="tiny", help="preset (tiny/small/base) or 'W-H-K-L'")
    # muP: base width 256 (keep head_dim fixed when scaling width to transfer LRs).
    p.add_argument("--mup-base-width", type=int, default=256)
    p.add_argument("--mup-base-std", type=float, default=0.02)
    p.add_argument("--mup-input-mult", type=float, default=1.0)
    p.add_argument("--mup-output-mult", type=float, default=1.0)
    # Tied embedding saves params (embedding is a big fraction of a tiny model) —
    # on by default; --no-tie-embedding to separate input/output matrices.
    p.add_argument("--no-tie-embedding", dest="tie_embedding", action="store_false")
    # Document masking: pack docs but block cross-document attention (block-diagonal
    # causal via flex_attention) + ignore the loss at eos boundaries. On by default
    # for packed pretraining; --no-doc-masking reverts to plain causal over the window.
    p.add_argument("--no-doc-masking", dest="doc_masking", action="store_false")
    # EMA of the weights (warmed-up decay); val/test + downstream eval run on the
    # averaged model. On by default; --no-ema to train/eval on raw weights only.
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-warmup-steps", type=int, default=100,
                   help="decay-ramp constant: d = min(decay, (1+n)/(warmup+n))")
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--no-scheduler", dest="use_scheduler", action="store_false")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    # CCE (fused lm_head+CE) barely helps at tiny vocab (logits are cheap), so it is
    # OFF by default here; --cce to enable.
    p.add_argument("--cce", dest="use_cce", action="store_true")
    # Final trainer.test() pass -> test/<src>/bpb on the held-out val windows.
    p.add_argument("--no-test", dest="run_test", action="store_false")
    # Zero-shot downstream benchmarks (bench.py, lm-eval-harness) run after
    # trainer.test() and logged to wandb under test/<task>/<metric>. Default set is
    # tuned for 5-20M (BLiMP is the signal-bearing one; PIQA/SciQ/ARC near-chance).
    p.add_argument("--eval-tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--eval-batch-tokens", type=int, default=32768)
    p.add_argument("--eval-limit", type=int, default=None,
                   help="cap examples per benchmark task (smoke); None = full")
    p.add_argument("--no-eval", dest="run_eval", action="store_false")
    # Log sample generations (test/generations table) at the end of the test phase.
    p.add_argument("--no-gen", dest="gen_samples", action="store_false")
    return p.parse_args()


def use_ema(args) -> bool:
    # EMA is on unless disabled (--no-ema) or decay <= 0 (the sweepable "off"
    # sentinel: a grid can't express a store_false flag, so --ema-decay 0 disables).
    return args.use_ema and args.ema_decay > 0


def default_run_name(args) -> str:
    tok = Path(args.tokenizer).name  # e.g. "8k"
    ema = f"ema{args.ema_decay}" if use_ema(args) else "noema"
    return (f"gpt-{args.n_embd}-{args.n_head}-{args.n_kv_head}-{args.n_layer}"
            f"-{tok}-seq{args.seq_len}-{ema}-{args.mix}-muon{args.muon_lr}-steps{args.max_steps}")


def main():
    args = parse_args()
    if args.vocab_tag:
        args.tokenizer = f"/mnt/ai/data/tiny-llm/tokenizer/{args.vocab_tag}"
        args.mix = f"tiny_2B_{args.vocab_tag}"
    if args.arch:
        arch = ARCH_PRESETS.get(args.arch, args.arch)
        args.n_embd, args.n_head, args.n_kv_head, args.n_layer = (
            int(x) for x in arch.split("-")
        )
    args.run_name = args.run_name or default_run_name(args)
    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    seed_everything(args.seed, workers=True)

    if args.global_token_count % args.seq_len != 0:
        raise ValueError(
            f"--global-token-count ({args.global_token_count}) must be divisible "
            f"by --seq-len ({args.seq_len})"
        )
    batch_size = args.global_token_count // args.seq_len
    print(f"global tokens/step={args.global_token_count}  seq_len={args.seq_len}"
          f"  -> batch_size={batch_size}")

    dm = MixtureDataModule(
        data_dir=args.data_dir,
        mix_name=args.mix,
        batch_size=batch_size,
        seq_len=args.seq_len,
        pretrained_id=args.tokenizer,
        num_workers=7,
        root_subdir="tiny-llm",  # mixes live under /mnt/ai/data/tiny-llm/mix/
    )
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"mix={args.mix}  tokenizer={dm.pretrained_id}  vocab_size={dm.vocab_size}")
    if dm.manifest:
        srcs = ", ".join(f"{r['key']}:{r['renorm_weight']:.2f}"
                         for r in dm.manifest["sources"])
        print(f"mix sources -> {srcs}")

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
        doc_mask_eos_id=dm.eos_id if args.doc_masking else None,
    )
    if args.doc_masking:
        print(f"document masking ON (eos_id={dm.eos_id}): intra-doc attention + boundary loss mask")
    n_params = sum(p.numel() for p in model.parameters())
    emb = dm.vocab_size * args.n_embd
    print(f"GPT parameters: {n_params/1e6:.2f}M  "
          f"(embedding {emb/1e6:.2f}M = {100*emb/n_params:.0f}%)")

    optimizer = Muon(
        muon_param_groups(
            model,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=args.adamw_weight_decay,
            adamw_name_keywords=("emb", "head", "gate"),
        )
    )
    scheduler = (
        LinearWarmupCosineAnnealingLR(
            optimizer, warmup_steps=args.warmup_steps, n_epochs=args.epochs,
            train_loader_length=len(train_loader), eta_min=args.eta_min,
            max_steps=args.max_steps,
        )
        if args.use_scheduler else None
    )

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    # per-source + aggregate bytes/token for tokenizer-invariant bpb logging.
    agg_bpt, src_bpt = measure_bpb(args.tokenizer, args.mix, data_dir=args.data_dir)
    print(f"bytes/token: aggregate={agg_bpt:.4f}  per-source={src_bpt}")
    lm_module = TinyLMModule(
        model, optimizer, scheduler, use_cce=args.use_cce,
        bytes_per_token=agg_bpt, source_bpt=src_bpt,
        doc_boundary_eos_id=dm.eos_id if args.doc_masking else None,
    )

    run_dir = Path(args.run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / args.run_name / "checkpoints",
        filename="gpt", monitor="val/loss", enable_version_counter=False,
    )
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    loggers = build_run_loggers(
        run_dir, args.wandb_project, args.run_name, args.wandb_offline, tags=tags
    )
    loggers[1].log_hyperparams(vars(args))

    callbacks = [
        checkpoint,
        TokenAxisCallback(args.global_token_count),
        ProgressPrinter(args.print_every, args.global_token_count),
    ]
    if use_ema(args):
        print(f"EMA ON: decay={args.ema_decay}  warmup_steps={args.ema_warmup_steps} "
              f"(val/test + downstream eval use the averaged weights)")
        callbacks.append(EMACallback(decay=args.ema_decay,
                                     warmup_steps=args.ema_warmup_steps))
    else:
        print("EMA OFF")

    trainer = Trainer(
        max_steps=args.max_steps,
        max_epochs=args.epochs,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        precision="bf16-true",
        gradient_clip_val=1.0,
        deterministic=True,
        logger=loggers,
        callbacks=callbacks,
    )
    # loaders passed explicitly, so hand the module the per-source names for
    # val/<src>/bpb (the module can't reach the datamodule this way).
    lm_module.val_source_names = dm.val_source_names
    trainer.fit(lm_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if args.run_test:
        trainer.test(lm_module, dataloaders=val_loader)
    print("best checkpoint:", checkpoint.best_model_path)

    # Test-phase extras: sample generations (test/generations table) + downstream
    # zero-shot benchmarks (BLiMP / LAMBADA / PIQA / SciQ / ARC-Easy) logged under
    # test/<task>/<metric>, alongside the test/<src>/bpb from trainer.test().
    if args.gen_samples or args.run_eval:
        # Lightning leaves the model on CPU after fit/test; torch.compile wraps it
        # in _orig_mod and eval batch shapes vary, so use the raw eager module.
        # reduce-overhead pins a CUDA-graph pool sized for the train shape — reset
        # dynamo + empty_cache so eval's larger batches don't OOM against dead
        # memory (see chimera cudagraph-eval memory).
        eval_model = getattr(model, "_orig_mod", model)
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_model.to(eval_device)

        if args.gen_samples:
            log_generations(eval_model, dm.tokenizer, eval_device, loggers[1])

        if args.run_eval:
            eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
            if eval_tasks:
                results = run_benchmarks(
                    eval_model, dm.tokenizer, eval_tasks,
                    block_size=args.seq_len, batch_tokens=args.eval_batch_tokens,
                    device=eval_device, limit=args.eval_limit,
                )
                # headline metric per task only (BLiMP aggregate, not its 67
                # subtasks; one metric each) — keeps the wandb test/ namespace small.
                print_table(results, tasks=eval_tasks)
                loggers[1].log_metrics(
                    flatten_for_wandb(results, tasks=eval_tasks), step=trainer.global_step
                )


if __name__ == "__main__":
    main()

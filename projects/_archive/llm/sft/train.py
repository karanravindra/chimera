"""Supervised finetuning (SFT) of the FineWeb-Edu-pretrained GPT on UltraChat.

Loads a base GPT checkpoint pretrained by ``projects/fineweb-edu/gpt/train.py``
(``--init-ckpt``; omit it for a randomly initialized model, useful only for
smoke tests) and finetunes it on UltraChat 200k rendered in ChatML with the same
LiquidAI/LFM2.5-230M tokenizer used for pretraining. Conversations are packed
into a flat stream like pretraining; the loss is masked to assistant tokens
(everything else is ``-100``, ignored by both the plain CE and the Cut Cross
Entropy paths). Optimized with Muon + AdamW under one LR schedule, with
``torch.compile`` and CCE, exactly as in pretraining but at lower base LRs.

The checkpoint stores no model hyperparameters, so the architecture flags
(``--arch``, ``--use-mla``, ``--use-moe``, ``--use-attn-res``, ...) must match
the base run; ``load_state_dict(strict=True)`` surfaces any mismatch immediately.

    uv run python projects/fineweb-edu/sft/train.py \\
        --init-ckpt /mnt/ai/runs/fineweb-edu/gpt/checkpoints/gpt.ckpt

Checkpoints + logs go to ``--run-dir`` (default ``/mnt/ai/runs/fineweb-edu/sft``);
unlike the pretrain script, each run checkpoints under its own run-name
subdirectory so runs never overwrite each other. ``main.ipynb`` loads the
resulting checkpoint for chat-style generation.
"""

import argparse
import os
from pathlib import Path

# Masked SFT produces batches whose supervised-token count varies (and can be ~0
# for an all-prompt/tool-result window). Cut Cross Entropy's Triton autotune
# perf-model derives throughput from that count and divides by it -> a
# ZeroDivisionError on such a batch (pretrain never hits this: it's unmasked).
# Disable CCE autotune (fixed heuristic config) BEFORE importing the model, whose
# import does os.environ.setdefault("CCE_AUTOTUNE", "1").
os.environ.setdefault("CCE_AUTOTUNE", "0")

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from chimera.data import MixtureDataModule
from chimera.models import GPT
from chimera.modules import LanguageModelModule
from chimera.optim import LinearWarmupCosineAnnealingLR, Muon, muon_param_groups
from chimera.utils import TokenAxisCallback, build_run_loggers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/llm/sft")
    # SFT mixture built by build_mixture.py --sft (masked ids)
    p.add_argument(
        "--mix", default="sft_smoke", help="SFT mixture name under llm-mix/mix_sft/"
    )
    p.add_argument(
        "--tokenizer",
        default="LiquidAI/LFM2.5-230M",
        help="tokenizer: HF hub id or local path (train_tokenizer.py output). "
        "Must match the one the SFT mix was tokenized with.",
    )
    # Base checkpoint from the pretrain stage. None = random init (smoke tests
    # only -- an un-pretrained model produces gibberish chat).
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--epochs", type=int, default=1)
    # Effective tokens per optimizer step; micro-batch = global_token_count // seq_len.
    p.add_argument("--global-token-count", type=int, default=65536)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--max-train-tokens", type=int, default=100_000_000)
    p.add_argument("--max-val-tokens", type=int, default=2_000_000)
    # Finetuning LRs: same Muon/AdamW split as pretraining but ~5x lower base
    # LRs (untuned starting points -- sweep before trusting them).
    p.add_argument("--muon-lr", type=float, default=2e-3)
    p.add_argument("--adamw-lr", type=float, default=8e-4)
    p.add_argument("--adamw-weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--eta-min", type=float, default=1e-5)
    p.add_argument("--val-check-interval", type=int, default=250)
    p.add_argument("--limit-val-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="llm-sft")
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--tags", default=None)
    # Model architecture: MUST match the --init-ckpt base run (the checkpoint
    # stores no hyperparameters). Defaults mirror the pretrain script.
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-kv-head", type=int, default=3)
    p.add_argument("--n-layer", type=int, default=6)
    # Compact "n_embd-n_head-n_kv_head-n_layer" override, as in pretraining.
    p.add_argument("--arch", default=None)
    p.add_argument("--mup-base-width", type=int, default=256)
    p.add_argument("--mup-base-std", type=float, default=0.02)
    p.add_argument("--mup-input-mult", type=float, default=1.0)
    p.add_argument("--mup-output-mult", type=float, default=1.0)
    p.add_argument("--use-attn-res", action="store_true")
    p.add_argument("--attn-res-n-blocks", type=int, default=8)
    p.add_argument("--use-mla", action="store_true")
    p.add_argument("--kv-lora-rank", type=int, default=128)
    p.add_argument("--q-lora-rank", type=int, default=0)
    p.add_argument("--qk-nope-head-dim", type=int, default=64)
    p.add_argument("--qk-rope-head-dim", type=int, default=32)
    p.add_argument("--v-head-dim", type=int, default=64)
    p.add_argument("--use-moe", action="store_true")
    p.add_argument("--n-routed-experts", type=int, default=8)
    p.add_argument("--n-shared-experts", type=int, default=1)
    p.add_argument("--n-activated-experts", type=int, default=2)
    p.add_argument("--moe-inter-dim", type=int, default=None)
    p.add_argument("--route-scale", type=float, default=1.0)
    p.add_argument("--bias-update-speed", type=float, default=0.001)
    p.add_argument("--no-scheduler", dest="use_scheduler", action="store_false")
    p.add_argument(
        "--grad-checkpoint", dest="gradient_checkpointing", action="store_true"
    )
    p.add_argument("--no-tie-embedding", dest="tie_embedding", action="store_false")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--no-cce", dest="use_cce", action="store_false")
    p.add_argument("--no-test", dest="run_test", action="store_false")
    # Sample chat generations printed + logged after training (a quick
    # qualitative check; the zero-shot bench suite is a base-model measure and
    # doesn't apply here).
    p.add_argument(
        "--sample-prompts",
        default="What is photosynthesis?|Write a haiku about the ocean.",
    )
    p.add_argument("--sample-max-new-tokens", type=int, default=128)
    p.add_argument("--no-sample", dest="run_sample", action="store_false")
    return p.parse_args()


def default_run_name(args) -> str:
    """Derive a descriptive wandb run name from the swept hyperparameters."""
    parts = [f"sft-{args.n_embd}-{args.n_head}-{args.n_kv_head}-{args.n_layer}"]
    if args.use_attn_res:
        parts.append(f"attnres{args.attn_res_n_blocks}")
    if args.use_mla:
        parts.append(f"mla-kv{args.kv_lora_rank}")
    if args.use_moe:
        parts.append(f"moe{args.n_activated_experts}of{args.n_routed_experts}")
    if args.init_ckpt is None:
        parts.append("scratch")
    parts.append(f"muon{args.muon_lr}-adamw{args.adamw_lr}")
    parts.append(f"steps{args.max_steps}")
    return "-".join(parts)


def load_base_checkpoint(model: GPT, ckpt_path: str) -> None:
    """Load pretrain-stage weights into ``model``.

    Pretrain checkpoints are Lightning checkpoints of ``LanguageModelModule``
    whose keys are prefixed ``model.`` (and additionally ``_orig_mod.`` when the
    run used torch.compile); both prefixes are stripped. ``strict=True`` so an
    architecture-flag mismatch fails loudly instead of silently skipping weights.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ("model.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)
    print(f"loaded base checkpoint {ckpt_path} (global_step={ckpt.get('global_step')})")


@torch.no_grad()
def sample_chats(model, dm, prompts, max_new_tokens, device):
    """Generate a reply for each prompt via the ChatML template; returns texts."""
    model.eval().to(device)
    replies = []
    for prompt in prompts:
        ids = dm.render_prompt([{"role": "user", "content": prompt}])
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens)[0, len(ids) :].tolist()
        if dm.im_end_id in out:  # trim at the assistant's end-of-turn token
            out = out[: out.index(dm.im_end_id)]
        replies.append(dm.decode(out))
    return replies


def main():
    args = parse_args()
    if args.arch:
        args.n_embd, args.n_head, args.n_kv_head, args.n_layer = (
            int(x) for x in args.arch.split("-")
        )
    args.run_name = args.run_name or default_run_name(args)
    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
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

    dm = MixtureDataModule(
        data_dir=args.data_dir,
        mix_name=args.mix,
        sft=True,  # masked ChatML mixture (supervise assistant/tool-call turns)
        batch_size=batch_size,
        seq_len=args.seq_len,
        pretrained_id=args.tokenizer,
        num_workers=7,
    )
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(
        f"sft mixture={args.mix}  tokenizer={dm.pretrained_id}  vocab_size={dm.vocab_size}"
    )
    if dm.manifest:
        srcs = ", ".join(
            f"{r['key']}:{r['renorm_weight']:.2f}" for r in dm.manifest["sources"]
        )
        print(f"sft sources -> {srcs}")

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
        use_attn_res=args.use_attn_res,
        attn_res_n_blocks=args.attn_res_n_blocks,
        use_mla=args.use_mla,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        use_moe=args.use_moe,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
        moe_inter_dim=args.moe_inter_dim,
        route_scale=args.route_scale,
        bias_update_speed=args.bias_update_speed,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT parameters: {n_params / 1e6:.2f}M")

    if args.init_ckpt is not None:
        load_base_checkpoint(model, args.init_ckpt)
    else:
        print(
            "[warn] no --init-ckpt: SFT starts from RANDOM weights. This only "
            "makes sense for smoke-testing the pipeline."
        )

    optimizer = Muon(
        muon_param_groups(
            model,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            adamw_weight_decay=args.adamw_weight_decay,
            adamw_name_keywords=("emb", "head", "gate"),
        )
    )
    if args.use_scheduler:
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=args.warmup_steps,
            n_epochs=args.epochs,
            train_loader_length=len(train_loader),
            eta_min=args.eta_min,
            max_steps=args.max_steps,
        )
    else:
        scheduler = None  # constant LR

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    # bytes-per-supervised-token for this SFT mix: enables tokenizer-independent
    # bpb logging alongside bpt. SFT loss averages over supervised (assistant)
    # tokens only, so bpb is normalized by supervised bytes/token (mask-aware).
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpt"))
    from bpb import bytes_per_token_cached

    bytes_per_token = bytes_per_token_cached(
        args.tokenizer, args.mix, data_dir=args.data_dir, sft=True
    )
    print(f"bytes/supervised-token={bytes_per_token:.4f} (<stage>/bpb = bpt / this)")
    lm_module = LanguageModelModule(
        model,
        optimizer,
        scheduler,
        use_cce=args.use_cce,
        bytes_per_token=bytes_per_token,
    )

    # Per-run checkpoint directory: the pretrain script's shared
    # checkpoints/gpt.ckpt let every run overwrite the last (which is how the
    # full-size base checkpoint was lost) -- here each run keeps its own.
    run_dir = Path(args.run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / args.run_name / "checkpoints",
        filename="sft",
        monitor="val/loss",
        enable_version_counter=False,
    )
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    loggers = build_run_loggers(
        run_dir, args.wandb_project, args.run_name, args.wandb_offline, tags=tags
    )
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
        callbacks=[checkpoint, TokenAxisCallback(args.global_token_count)],
    )
    trainer.fit(lm_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if args.run_test:
        trainer.test(lm_module, dataloaders=val_loader)
    print("best checkpoint:", checkpoint.best_model_path)

    if args.run_sample:
        prompts = [s.strip() for s in args.sample_prompts.split("|") if s.strip()]
        if prompts:
            # Lightning leaves the model on CPU after fit/test; generation uses
            # the eager module (compiled graphs are shaped for training batches).
            sample_model = getattr(model, "_orig_mod", model)
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            replies = sample_chats(
                sample_model, dm, prompts, args.sample_max_new_tokens, device
            )
            for prompt, reply in zip(prompts, replies):
                print(f"\n>>> {prompt}\n{reply}")
            import wandb

            loggers[1].experiment.log(
                {
                    "samples": wandb.Table(
                        columns=["prompt", "reply"], data=list(zip(prompts, replies))
                    )
                }
            )


if __name__ == "__main__":
    main()

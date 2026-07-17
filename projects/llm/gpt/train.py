"""Pretrain a decoder-only GPT (GQA + RoPE + QK-norm) on FineWeb-Edu.

FineWeb-Edu ``sample-10BT`` tokenized with the pretrained LiquidAI/LFM2.5-230M
subword tokenizer; documents are concatenated with an ``<|endoftext|>`` separator
so the model learns document boundaries. Optimized with Muon (2D/batched hidden
weight matrices) + AdamW (embedding/head/norms/router) under one LR schedule,
with ``torch.compile`` and Cut Cross Entropy (fused lm_head + cross-entropy).

The model can additionally enable DeepSeek-V3 style Multi-head Latent
Attention (``--use-mla``, low-rank KV/Q compression, smaller KV cache) and
DeepSeek MoE (``--use-moe``, fine-grained routed experts + shared expert(s),
aux-loss-free load balancing) in place of the dense GQA/MLP:

    uv run python projects/fineweb-edu/gpt/train.py --use-mla --use-moe

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
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from chimera.data import MixtureDataModule
from chimera.models import GPT
from chimera.modules import LanguageModelModule
from chimera.optim import LinearWarmupCosineAnnealingLR, Muon, muon_param_groups
from chimera.utils import ProgressPrinter, TokenAxisCallback, build_run_loggers
from bench import DEFAULT_TASKS, flatten_for_wandb, print_table, run_benchmarks


# Locked muP model family (W-H-K-L, head_dim=32, GQA n_kv=1, depth 6 wall-optimal).
# Ladder doubles width; the swept muP LR (muon 0.013 / adamw 0.006) transfers to all.
# Sizes: small 20.5M / base 48.9M / large 150M (tied embedding; vocab 64402).
ARCH_PRESETS = {
    "small": "256-8-1-6",    # 20.5M — smallest verified point off the loss floor
    "base": "512-16-1-6",    # 48.9M — Pareto winner; near the loss floor, below depth-12's wall
    "large": "1024-32-1-8",  # 150M — width ladder + a touch of depth (aspect 128) for long horizons
}


class DynamicUntie(Callback):
    """Untie the (initially tied) embedding into a separate output head partway
    through training — modded-nanogpt's "untie embed/lm_head at 2/3 of training".

    Tied early = fewer params / better data-efficiency / less optimizer memory
    while the model learns basic structure; untied late = input and output
    representations specialize for the final loss. The fork copies the embedding's
    Adam moments into the fresh head so there is no cold restart, and touches
    neither ``tok_emb``'s tensor identity nor the compiled forward graph (the head
    lives only in the eager CCE call), so no recompile is triggered.
    """

    def __init__(self, split_step: int):
        # `| 1` -> an odd step, matching modded-nanogpt (keeps Adam bias-correction
        # landing cleanly relative to the mirrored/forked state).
        self.split_step = max(1, int(split_step)) | 1
        self._done = False

    def on_train_batch_end(self, trainer, pl_module, *args):
        if self._done or trainer.global_step < self.split_step:
            return
        raw = getattr(pl_module.model, "_orig_mod", pl_module.model)
        forked = raw.untie_head()
        if forked is not None:
            emb_w, head_w = forked
            trainer.optimizers[0].add_adamw_param(head_w, copy_state_from=emb_w)
            print(f"[untie] forked lm_head from tied embedding at step "
                  f"{trainer.global_step} (split_step={self.split_step})")
        self._done = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--run-dir", default="/mnt/ai/runs/llm/gpt")
    # which pre-packed mixture to train on (built by build_mixture.py)
    p.add_argument("--mix", default="mix_1B", help="mixture name under llm-mix/mix/")
    p.add_argument("--tokenizer", default="LiquidAI/LFM2.5-230M",
                   help="tokenizer: HF hub id or local path (train_tokenizer.py output). "
                        "Must match the one the mix was tokenized with.")
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
    # muTransfer LRs from the emb48 muP sweep: these transfer across width
    # unchanged (Muon spectral-normalized + AdamW on the width-independent
    # embedding/head). adamw=8e-3 reaches a marginally lower val/loss (5.13 vs
    # 5.16 here) but is a HOT LR: gpt-no-bias-48emb-long (adamw=8e-3) shows a
    # real mid-run instability — train/loss climbs from ~5.11 (step ~2k) back
    # up to ~5.38 (step ~5.5k) while LR is still >=0.007, before the cosine
    # schedule anneals it back down by the end of the run. gpt-no-bias-48emb-
    # a0.004 (this default) has no such hump — monotonically decreasing the
    # whole run. Keep the scheduler on and its period = run length regardless.
    p.add_argument("--muon-lr", type=float, default=0.013)   # swept optimum (muP LR sweep 60lc74lr, 9M proxy/200M tok): 0.0133
    p.add_argument("--adamw-lr", type=float, default=6e-3)    # swept optimum: 0.00603 (broad basin muon 8e-3..1.8e-2 x adamw 3e-3..6e-3)
    p.add_argument("--adamw-weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eta-min", type=float, default=1e-5)
    # Run validation every N optimizer steps (so we get a val/loss curve during
    # training, not just one point at the end). limit-val-batches caps how many
    # val batches each validation pass uses, keeping periodic validation cheap on
    # the large (1% of corpus) val split.
    p.add_argument("--val-check-interval", type=int, default=500)
    p.add_argument("--limit-val-batches", type=int, default=250)
    # stdout progress cadence (throughput + metrics); per-stage timing is always on
    p.add_argument("--print-every", type=int, default=500,
                   help="print step/throughput/metrics every N steps (0 disables)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="llm-pretrain")
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
    # Accepts either a compact "W-H-K-L" string or a named preset from the locked
    # muP family (see ARCH_PRESETS): small/base/large. Family found via the muP LR
    # sweep + width×depth coord-check (sweeps 60lc74lr/adlxq011/2ndqvuzp): depth-6
    # is wall-optimal and width is the paying axis, so the ladder doubles width at
    # head_dim=32 / GQA n_kv=1; the same swept LR (0.013/0.006) transfers to all.
    p.add_argument("--arch", default=None)
    # muP (Maximal Update Parameterization): tune muon-lr/adamw-lr (+ the mults
    # below) at a small proxy width, then transfer them to a large model by only
    # changing --n-embd/--n-head (keep head_dim fixed). At n_embd == mup-base-width
    # this reduces to the original GPT-2-style init.
    p.add_argument("--mup-base-width", type=int, default=256)
    p.add_argument("--mup-base-std", type=float, default=0.02)
    p.add_argument("--mup-input-mult", type=float, default=1.0)
    p.add_argument("--mup-output-mult", type=float, default=1.0)
    # Attention Residuals (MoonshotAI): replace the plain additive residual with
    # learned softmax attention over depth. --attn-res-n-blocks == --n-layer is
    # "Full AttnRes" (attends over every layer's output); smaller values are
    # "Block AttnRes" (attends over block-level summaries, O(N*d) memory).
    p.add_argument("--use-attn-res", action="store_true",
        help="Enable (Block) Attention Residuals in place of plain additive "
             "residual connections (MoonshotAI AttnRes). Default off.")
    p.add_argument("--attn-res-n-blocks", type=int, default=8,
        help="Number of blocks to partition the network into for Block AttnRes "
             "(must evenly divide --n-layer). Set equal to --n-layer for Full "
             "AttnRes (attention over every individual layer's output).")
    # DeepSeek-V3 Multi-head Latent Attention: compresses K/V (and optionally Q)
    # into a low-rank latent, shrinking the KV cache. --n-kv-head is ignored
    # when this is on.
    p.add_argument("--use-mla", action="store_true",
        help="Replace grouped-query attention with Multi-head Latent Attention "
             "(DeepSeek-V3). Default off.")
    p.add_argument("--kv-lora-rank", type=int, default=128)
    p.add_argument("--q-lora-rank", type=int, default=0,
        help="0 disables Q compression (direct q_proj); only pays off at much "
             "larger width than this repo trains at.")
    p.add_argument("--qk-nope-head-dim", type=int, default=64)
    p.add_argument("--qk-rope-head-dim", type=int, default=32)
    p.add_argument("--v-head-dim", type=int, default=64)
    # DeepSeek-V3 MoE: fine-grained routed experts (top-k) + always-on shared
    # expert(s), aux-loss-free load balancing (bias buffer, no gradient-based
    # balance loss). Expert FFN compute uses torch.nn.functional.grouped_mm on
    # CUDA+bf16 (falls back to an eager per-expert loop otherwise).
    p.add_argument("--use-moe", action="store_true",
        help="Replace the dense MLP with DeepSeek MoE (aux-loss-free). Default off.")
    p.add_argument("--n-routed-experts", type=int, default=8)
    p.add_argument("--n-shared-experts", type=int, default=1)
    p.add_argument("--n-activated-experts", type=int, default=2)
    p.add_argument("--moe-inter-dim", type=int, default=None,
        help="Per-expert FFN hidden dim; defaults to n_embd // 2 (fine-grained: "
             "much narrower than the dense 4x FFN) when unset.")
    p.add_argument("--route-scale", type=float, default=1.0)
    p.add_argument("--bias-update-speed", type=float, default=0.001,
        help="Step size for the aux-loss-free per-expert routing bias. Note: "
             "double-applies under --grad-checkpoint (the gate runs twice per "
             "step under activation-checkpoint recompute) -- halve this or avoid "
             "combining the two flags until a call-count guard is added.")
    # Disable the LR scheduler entirely -> constant LR (no warmup/cosine). Cleaner
    # for muP LR sweeps, where a schedule would confound which LR value matters.
    p.add_argument("--no-scheduler", dest="use_scheduler", action="store_false")
    # Recompute block activations in backward instead of storing them: lets wide
    # models fit on limited VRAM without shrinking tokens/step (~1 extra fwd cost).
    p.add_argument("--grad-checkpoint", dest="gradient_checkpointing", action="store_true")
    # Pure-causal attention uses SDPA/FlashAttention by default (faster than
    # flex_attention at small head_dim); flex is still used for decode and any
    # custom/sparse mask. This disables the SDPA fast path (always flex).
    p.add_argument("--no-flash-attn", dest="use_flash_attn", action="store_false")
    # Sparse attention: causal sliding window + always-visible global prefix
    # tokens (attention sinks), via flex_attention block masks that prune
    # fully-masked KV blocks. Unset = dense causal.
    p.add_argument("--attn-window", type=int, default=None,
        help="Sliding-window size in tokens (e.g. 256). Unset or 0 = dense "
             "causal (0 lets a wandb sweep grid include the dense baseline).")
    p.add_argument("--attn-global-tokens", type=int, default=16,
        help="Number of prefix tokens every query may attend to regardless of "
             "the window (only used with --attn-window).")
    p.add_argument("--no-tie-embedding", dest="tie_embedding", action="store_false")
    p.add_argument("--untie-at-frac", type=float, default=None,
                   help="modded-nanogpt dynamic untie: start tied, fork lm_head "
                        "into a separate param at this fraction of --max-steps "
                        "(e.g. 0.667). Requires tied embedding; None disables it.")
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


def default_run_name(args) -> str:
    """Derive a descriptive wandb run name from the swept hyperparameters.

    Used when --run-name isn't passed explicitly (e.g. every run in a wandb
    sweep grid) so runs stay identifiable in the dashboard instead of falling
    back to wandb's random auto-generated names (e.g. "eager-sweep-1").
    """
    parts = [f"gpt-{args.n_embd}-{args.n_head}-{args.n_kv_head}-{args.n_layer}"]
    if args.use_attn_res:
        parts.append(f"attnres{args.attn_res_n_blocks}")
    if args.attn_window is not None:
        parts.append(f"swa{args.attn_window}g{args.attn_global_tokens}")
    if args.use_mla:
        parts.append(f"mla-kv{args.kv_lora_rank}")
    if args.use_moe:
        parts.append(f"moe{args.n_activated_experts}of{args.n_routed_experts}")
    parts.append(f"muon{args.muon_lr}-adamw{args.adamw_lr}")
    parts.append(f"steps{args.max_steps}")
    return "-".join(parts)


def main():
    args = parse_args()
    if args.arch:
        arch = ARCH_PRESETS.get(args.arch, args.arch)  # named preset -> "W-H-K-L"
        args.n_embd, args.n_head, args.n_kv_head, args.n_layer = (
            int(x) for x in arch.split("-")
        )
    if args.attn_window == 0:
        args.attn_window = None  # sweep-grid sentinel for the dense baseline
    args.run_name = args.run_name or default_run_name(args)
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

    if args.use_moe and args.compile:
        # Measured (mfu_bench, base 512-16-1-6): compiled MoE runs at ~74% MFU vs
        # ~60% eager -- compile fuses the routing elementwise/norm kernels and is a
        # clear win, so keep it on. The data-dependent expert offs are handled
        # without per-step recompiles in steady state; only drop to --no-compile if
        # you actually hit compile errors.
        print("[info] --compile with --use-moe: keeping compile on (measured "
              "~+14 MFU points over eager).")

    dm = MixtureDataModule(
        data_dir=args.data_dir,
        mix_name=args.mix,  # pre-packed mixture built by build_mixture.py
        batch_size=batch_size,
        seq_len=args.seq_len,
        pretrained_id=args.tokenizer,
        num_workers=7,
    )
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"mixture={args.mix}  tokenizer={dm.pretrained_id}  vocab_size={dm.vocab_size}")
    print(f"eos_token={dm.eos_token!r}  eos_id={dm.eos_id}")
    if dm.manifest:
        srcs = ", ".join(f"{r['key']}:{r['renorm_weight']:.2f}" for r in dm.manifest["sources"])
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
        gradient_checkpointing=args.gradient_checkpointing,
        use_flash_attn=args.use_flash_attn,
        attn_window=args.attn_window,
        attn_global_tokens=args.attn_global_tokens,
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

    # Muon routes 2D/batched hidden weight matrices (attention + MLP/experts) to
    # Muon and the token embedding, output head, norm gains, and MoE router/gate
    # weights to AdamW; both groups share one LR schedule (each anneals from its
    # own base LR). "gate" is always included -- harmless (matches nothing) when
    # --use-moe is off.
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
            # Anneal over the actual horizon: when --max-steps caps the run short
            # of a full epoch, the cosine must cool down by max_steps, not by the
            # (never-reached) epoch length — else the LR is still hot at the cutoff.
            max_steps=args.max_steps,
        )
    else:
        scheduler = None  # constant LR

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
    # bytes-per-token for this (tokenizer, mix): enables tokenizer-independent bpb
    # logging alongside bpt. Measured+cached once by bpb.py (git-tracked cache).
    from bpb import bytes_per_token_cached

    bytes_per_token = bytes_per_token_cached(
        args.tokenizer, args.mix, data_dir=args.data_dir
    )
    print(f"bytes/token={bytes_per_token:.4f} (val/*bpb = bpt / bytes_per_token)")
    # use_cce: fuse lm_head + cross-entropy (apple/ml-cross-entropy), no logits materialized
    lm_module = LanguageModelModule(
        model, optimizer, scheduler, use_cce=args.use_cce,
        bytes_per_token=bytes_per_token,
    )

    run_dir = Path(args.run_dir)
    # Per-run checkpoint directory: a shared checkpoints/gpt.ckpt let every run
    # (esp. sweep runs) overwrite the last -- which is how the full-size base
    # checkpoint was lost. Same pattern as the sft project.
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / args.run_name / "checkpoints",
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
        callbacks=[
            checkpoint,
            TokenAxisCallback(args.global_token_count),
            ProgressPrinter(args.print_every, args.global_token_count),
        ]
        + (
            [DynamicUntie(int(args.untie_at_frac * args.max_steps))]
            if args.untie_at_frac and args.tie_embedding
            else []
        ),
    )
    if args.untie_at_frac and not args.tie_embedding:
        print("[warn] --untie-at-frac ignored: embedding is already untied "
              "(--no-tie-embedding). Dynamic untie needs a tied start.")
    # FineWebEduDataModule exposes only train/val loaders (no test split), so the
    # loaders are passed explicitly; validation reuses the val loader as in the
    # notebook (logged under train/*, val/*, and test/* respectively).
    # When the mix has >=2 sources, val_loader is a list (one loader per source);
    # hand the source names to the module so it logs per-dataset val_<src>/* curves
    # (grokking) alongside the aggregate val/* (the module can't reach the
    # datamodule since loaders are passed explicitly).
    lm_module.val_source_names = dm.val_source_names
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

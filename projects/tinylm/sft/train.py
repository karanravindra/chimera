"""SFT the pretrained tinylm GPT (~6M params) on a simple-QA chat mixture.

Same rails as ../pretrain (shared chimera.modules.LanguageModelModule: packed
FlexAttention doc-masking + CCE + Muon/AdamW under one warmup-cosine schedule),
but the stream is ChatML conversations with loss only on assistant tokens (labels
are -100 elsewhere; CCE's ignore_index). Starts from the pretrain checkpoint and
keeps its tokenizer (chat special tokens were reserved in the vocab from day one).

Run from this directory:

    uv run python train.py            # full-FT (Muon)
    USE_LORA=1 uv run python train.py # LoRA (AdamW on low-rank deltas)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("CCE_AUTOTUNE", "1")
os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

import lightning.pytorch as pl
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.insert(0, str(Path(__file__).parent.parent / "pretrain"))
from model import GPT  # noqa: E402  (the pretrain model, single source of truth)

from chimera.data.text import (  # noqa: E402
    MixtureSource,
    TextDataModule,
    TextMixtureSpec,
    TokenizerSpec,
)
from chimera.data.text.chat_template import render  # noqa: E402
from chimera.modules import LanguageModelModule  # noqa: E402
from chimera.optim import LinearWarmupCosineAnnealingLR, Muon, muon_param_groups  # noqa: E402
from chimera.utils import ProgressPrinter, TokenAxisCallback, build_run_loggers  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# architecture — MUST match the pretrain checkpoint. SEQ_LEN is env-driven so we
# can SFT a context-extended base (ctx2k/4k/8k) at its native length; the model
# has no baked context limit (RoPE on the fly), so only the data-window length
# and batch change.
SEQ_LEN = int(os.environ.get("TINYLM_SEQ_LEN", "512"))
DIM = 384
N_HEADS = 12
MLP_MULT = 3
N_LAYERS = 6
LOGIT_SOFTCAP = 30.0  # base model was trained under this cap; keep it

# optimization — gentler than pretrain: the model is converged, SFT reshapes.
# Env-driven so an unstable config (e.g. ctx8k SFT: seq8192 + small batch
# diverged at 0.005, loss->ln(V)) can drop to a lower peak LR.
MUON_LR = float(os.environ.get("TINYLM_MUON_LR", "0.005"))
ADAMW_LR = float(os.environ.get("TINYLM_ADAMW_LR", "2.5e-4"))

# LoRA mode: freeze the base, train low-rank deltas on every Linear (qkv/out/
# fc1/fc2; embeddings + tied head stay frozen). Merged back before saving, so
# the checkpoint stays architecture-identical to full-FT runs.
USE_LORA = os.environ.get("USE_LORA", "0") == "1"  # USE_LORA=1 uv run python train.py
LORA_R = 16
LORA_ALPHA = 32.0
LORA_LR = 1e-3  # LoRA convention: well above the full-FT AdamW lr
# On-device batch; env-driven so long-context SFT (2k/4k/8k) can shrink it to fit
# (SFT has no grad accum). ~65k tokens/batch at 512; drop proportionally for ctx.
BATCH_SIZE = int(os.environ.get("TINYLM_BATCH_SIZE", "128"))
MAX_TRAIN_STEPS = 700
VALIDATE_EVERY_N_STEPS = 100
N_EPOCHS = 2  # small pool; MAX_TRAIN_STEPS is the real cap
WARMUP_STEPS = 50
FINAL_LR_FRAC = 0.1
SEED = int(os.environ.get("TINYLM_SEED", "1234"))

# the pinned 16k pretrain vocab (chat specials reserved at fixed ids); SFT never
# retrains it. MUST match the base checkpoint's vocab — the current base
# (tok16k_4b) was trained on this frozen tokenizer, not the old per-mixture one.
TOKENIZER_PATH = os.environ.get(
    "TINYLM_TOKENIZER_PATH",
    "/root/Code/chimera/projects/tinylm/data/tokenizers/16k/tokenizer.json",
)
# The 4BT plain base (best pretrain checkpoint, pinned-vocab). Override via env.
BASE_CHECKPOINT = os.environ.get(
    "TINYLM_BASE_CKPT", "/mnt/ai/runs/tinylm/pretrain/chimera_gpt6m_tok16k_4b.pt"
)

RUN_DIR = Path("/mnt/ai/runs/tinylm/sft")
# Optional tag so SFT runs off different bases (e.g. context stages) save under
# distinct names instead of clobbering the base-SFT checkpoint.
_SFT_TAG = os.environ.get("TINYLM_RUN_TAG", "").strip()
_SFT_TAG = f"_{_SFT_TAG}" if _SFT_TAG else ""
_LORA_TAG = "_lora" if USE_LORA else ""
CHECKPOINT_PATH = RUN_DIR / f"chimera_gpt6m_sft{_LORA_TAG}{_SFT_TAG}.pt"

GENERATION_PROMPTS = [
    [{"role": "user", "content": "What color is the sky?"}],
    [{"role": "user", "content": "Hi! How are you today?"}],
    [
        {
            "role": "user",
            "content": (
                "Tom has a red ball and a blue kite. He plays with them in the "
                "park every Sunday with his sister Amy.\n\nWhat color is Tom's ball?"
            ),
        }
    ],
]


def make_datamodule() -> TextDataModule:
    """The simple-QA chat mixture: bulk closed-book QA + grounded QA + chat style."""
    # Phase-2 mix: the grounded-QA core (CoQA+QuAC lead, SQuAD/GooAQ moderate)
    # PLUS instruction breadth — No Robots (summarize/rewrite/extract/classify,
    # the transformation-instruction target) and a light SODA sample (social
    # dialog; synthetic, capped low to avoid style imprint). OASST1 (-> preference
    # tuning) and Tulu (needs source-filtering) deferred. Shares via max_train_tokens.
    sources = (
        MixtureSource("coqa.sft", max_train_tokens=15_000_000, max_val_tokens=250_000),
        MixtureSource("quac.sft", max_train_tokens=12_000_000, max_val_tokens=250_000),
        MixtureSource(
            "no-robots.sft", max_train_tokens=8_000_000, max_val_tokens=250_000
        ),
        MixtureSource("squad.sft", max_train_tokens=6_000_000, max_val_tokens=250_000),
        MixtureSource("soda.sft", max_train_tokens=4_000_000, max_val_tokens=250_000),
        MixtureSource("gooaq.sft", max_train_tokens=3_000_000, max_val_tokens=250_000),
        MixtureSource(
            "smoltalk.everyday.sft", max_train_tokens=None, max_val_tokens=250_000
        ),
    )
    dm = TextDataModule(
        TextMixtureSpec(
            sources=sources,
            tokenizer=TokenizerSpec.pinned(TOKENIZER_PATH),
            add_bos=True,
        ),
        data_dir="/mnt/ai/data",
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


def make_model(vocab_size: int) -> GPT:
    return GPT(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        dim=DIM,
        n_heads=N_HEADS,
        mlp_mult=MLP_MULT,
        n_layers=N_LAYERS,
        logit_softcap=LOGIT_SOFTCAP,
        # plain baseline (VWN off) — MUST match the tok16k_4b base checkpoint
        vwn_m=1,
        vwn_n=1,
    )


@torch.no_grad()
def print_generations(model, tokenizer, bos_id: int, eos_id: int, im_end_id: int):
    """Greedy continuation, sliced in TOKEN space (char-slicing the decoded text
    is wrong: decode(encode(prompt)) need not equal the prompt string)."""
    net = getattr(model, "_orig_mod", model)
    net.eval()
    device = next(net.parameters()).device
    for messages in GENERATION_PROMPTS:
        prompt = render(messages, add_generation_prompt=True)
        ids = tokenizer._tok.encode(prompt, add_special_tokens=False).ids
        x = torch.tensor([[bos_id] + ids], device=device)
        out: list[int] = []
        for _ in range(80):
            nxt = int(net(x)[0, -1].argmax())
            if nxt in (eos_id, im_end_id):
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        print(
            f"\n>>> {messages[-1]['content'][:80]}\n{tokenizer._tok.decode(out).strip()!r}"
        )
    net.train()


def train():
    pl.seed_everything(SEED, workers=True)
    dm = make_datamodule()
    tok = dm.tokenizer
    eos_id, bos_id = dm.eos_id, dm.bos_id
    im_end_id = tok._tok.token_to_id("<|im_end|>")

    mix = dm.source_train_tokens
    total = sum(mix.values())
    print(
        "sft mix: " + "  ".join(f"{k}={v:,} ({v / total:.0%})" for k, v in mix.items())
    )
    print(f"train tokens={total:,}")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = make_model(tok.vocab_size)
    state = torch.load(BASE_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    print(f"loaded base checkpoint: {BASE_CHECKPOINT}")

    if USE_LORA:
        from chimera.models.lora import apply_lora

        lora_params = apply_lora(model, r=LORA_R, alpha=LORA_ALPHA)
        n_lora = sum(p.numel() for p in lora_params)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA r={LORA_R}: {n_lora:,} trainable / {n_total:,} ({n_lora / n_total:.1%})"
        )

    if DEVICE == "cuda":
        model = torch.compile(model)

    if USE_LORA:
        # only the LoRA A/B params; base weights keep requires_grad=True (the
        # flex_attn wrapper breaks on no-grad q/k/v) but are never stepped — the
        # module zeros grads model-wide (zero_grad_all_params) so they don't stale.
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=LORA_LR,
            weight_decay=0.0,  # don't decay low-rank deltas toward zero
        )
    else:
        optimizer = Muon(muon_param_groups(model, muon_lr=MUON_LR, adamw_lr=ADAMW_LR))
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_TRAIN_STEPS,
        final_lr_frac=FINAL_LR_FRAC,
    )

    module = LanguageModelModule(
        model,
        optimizer,
        scheduler,
        use_cce=(DEVICE == "cuda"),
        logit_softcap=LOGIT_SOFTCAP,
        eos_id=eos_id,
        # assistant-only supervision is already in the labels (-100 elsewhere);
        # no doc-boundary target mask needed.
        zero_grad_all_params=USE_LORA,
    )

    tokens_per_step = BATCH_SIZE * SEQ_LEN
    run_name = CHECKPOINT_PATH.stem
    ckpt = ModelCheckpoint(
        dirpath=RUN_DIR / run_name / "checkpoints",
        filename="gpt",
        monitor="val/loss",
        save_last=True,
        enable_version_counter=False,
    )
    trainer = Trainer(
        max_steps=MAX_TRAIN_STEPS,
        max_epochs=N_EPOCHS,
        precision="bf16-true" if DEVICE == "cuda" else "32-true",
        val_check_interval=VALIDATE_EVERY_N_STEPS,  # SFT has no grad accum (accum=1)
        num_sanity_val_steps=0,
        enable_progress_bar=False,  # ProgressPrinter owns stdout
        logger=build_run_loggers(RUN_DIR, "tinylm-sft", run_name),
        callbacks=[
            ckpt,
            ProgressPrinter(
                print_every=max(1, VALIDATE_EVERY_N_STEPS // 10),
                tokens_per_step=tokens_per_step,
            ),
            TokenAxisCallback(tokens_per_step),
        ],
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    net = getattr(module.model, "_orig_mod", module.model)
    if USE_LORA:
        from chimera.models.lora import merge_lora

        net = merge_lora(net)  # fold deltas back -> base-architecture state_dict
    torch.save(net.state_dict(), CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")
    print(f"best (val/loss) checkpoint: {ckpt.best_model_path}")

    net = net.to(DEVICE)
    print_generations(net, tok, bos_id, eos_id, im_end_id)


if __name__ == "__main__":
    train()

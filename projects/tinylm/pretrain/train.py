"""Pretrain the tinylm GPT (~6M params) on a 5-way text mixture.

Sources (sampling weight = per-source max_train_tokens): tiny-textbooks 30 /
tiny-strange-textbooks 25 / fineweb-edu 20 / tinystories-v2 15 / tiny-webtext 10.
The 16k BPE vocab is trained on a round-robin sample of all five, so it
compresses the whole mixture rather than the owner's register alone.

Raw PyTorch loop (deliberately not on the chimera.train Lightning rails):
FlexAttention causal+document masking with per-doc RoPE positions, Cut Cross
Entropy, Muon+AdamW, torch.compile. Run from this directory:

    uv run python train.py

Saves the final checkpoint to /mnt/ai/runs/tinylm/pretrain/. main.ipynb is
analysis-only and loads that checkpoint.
"""

import math
import os
from itertools import islice
from pathlib import Path

os.environ.setdefault("CCE_AUTOTUNE", "1")
os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")  # datasets cache

import torch
import torch.nn as nn
from cut_cross_entropy import linear_cross_entropy
from model import GPT
from torchinfo import summary
from tqdm import tqdm

from chimera.data import (
    ConcatTextDataModule,
    FineWebEduTextDataModule,
    TinyStoriesV2DataModule,
    TinyStrangeTextbooksDataModule,
    TinyTextbooksDataModule,
    TinyWebTextDataModule,
)
from chimera.models.attention import build_block_mask_and_pos
from chimera.optim import Muon, muon_param_groups

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
USE_CCE = DEVICE == "cuda"

LN2 = math.log(2.0)

# Bits-per-byte held-out: a FIXED text (tiny-textbooks test split, cached to disk
# so the raw bytes are identical across runs) scored as tokenizer-agnostic BPB —
# the metric to compare runs regardless of what tokenizer/vocab they used.
BPB_HELDOUT_PATH = Path("/mnt/ai/data/tinylm/bpb_heldout.txt")
BPB_HELDOUT_DOCS = 500
BPB_BATCH_WINDOWS = 32

# model
SEQ_LEN = 512
DIM = 384
N_HEADS = 12
MLP_MULT = 3
N_LAYERS = 6

# optimization
MUON_LR = 0.02
ADAMW_LR = 1e-3
BATCH_SIZE = 128
MAX_TRAIN_STEPS = 5000
VALIDATE_EVERY_N_STEPS = 500
N_EPOCHS = 1

RUN_DIR = Path("/mnt/ai/runs/tinylm/pretrain")
CHECKPOINT_PATH = RUN_DIR / "chimera_gpt6m.pt"


def make_datamodule() -> ConcatTextDataModule:
    # Balanced 5-way mix (per-source max_train_tokens = sampling weight over the
    # concatenated stream). ~600M-token pool; the run is step-capped so the model
    # sees ~328M of it (pool > seen => negligible repetition). The 16k vocab is
    # trained on a round-robin 1GB-char sample of ALL sources (200M/source —
    # saturating region for 16k, cheap now that training streams a doc iterator)
    # via train_tokenizer_on_mixture rather than any single register.
    DATA_DIR = "/mnt/ai/data"
    VAL_TOKENS = 500_000
    dm = ConcatTextDataModule(
        [
            # owner: carries the canonical vocab_size / backend / doc convention
            # (tokenizer is trained on the blend below, not on this source alone)
            TinyTextbooksDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                vocab_size=16_384,
                max_train_tokens=180_000_000,
                max_val_tokens=VAL_TOKENS,
            ),
            TinyStrangeTextbooksDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=150_000_000,
                max_val_tokens=VAL_TOKENS,
            ),
            FineWebEduTextDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=120_000_000,
                max_val_tokens=VAL_TOKENS,
            ),
            TinyStoriesV2DataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=90_000_000,
                max_val_tokens=VAL_TOKENS,
            ),
            TinyWebTextDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=60_000_000,
                max_val_tokens=VAL_TOKENS,
            ),
        ],
        batch_size=BATCH_SIZE,
        train_tokenizer_on_mixture=True,
        tokenizer_sample_chars=1_000_000_000,
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
    )


def make_optimizer(model: GPT) -> Muon:
    return Muon(muon_param_groups(model, muon_lr=MUON_LR, adamw_lr=ADAMW_LR))


def compute_loss(model, x, y, eos_id: int, vocab_size: int):
    if USE_CCE:
        # FlexAttention block mask (causal + document) + per-document RoPE position
        # ids — rebuilt per batch. Note: the smaller last validation batch triggers a
        # one-time torch.compile recompile on the first val pass, then it's cached.
        block_mask, pos_ids = build_block_mask_and_pos(x, eos_id)
        hidden = model(x, return_hidden=True, block_mask=block_mask, pos_ids=pos_ids)
        weight = getattr(model, "_orig_mod", model).token_emb.weight  # tied lm_head
        return linear_cross_entropy(hidden, weight, y)
    logits = model(x)  # CPU fallback: plain causal, no doc masking / flex
    return nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))


@torch.no_grad()
def evaluate(model, dm) -> float:
    """Mean validation loss over the full val set."""
    model.eval()
    total, n = 0.0, 0
    for val_x, val_y in dm.val_dataloader():
        total += compute_loss(
            model, val_x.to(DEVICE), val_y.to(DEVICE), dm.eos_id, dm.vocab_size
        ).item()
        n += 1
    model.train()
    return total / n


def load_bpb_heldout() -> tuple[str, int]:
    """Fixed held-out text + its UTF-8 byte count, for tokenizer-agnostic BPB.

    Cached to disk on first use so every run (any tokenizer) scores the exact
    same bytes. Textbook-register (tiny-textbooks *test* split — disjoint from
    the train stream, so no leakage); swap the source here to change the yardstick.
    """
    if not BPB_HELDOUT_PATH.exists():
        from datasets import load_dataset

        ds = load_dataset(
            "nampdn-ai/tiny-textbooks",
            split="test",
            cache_dir="/mnt/ai/data/hf_cache",
        )
        n = min(BPB_HELDOUT_DOCS, len(ds))
        text = "\n\n".join(ds[i]["textbook"] for i in range(n))
        BPB_HELDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        BPB_HELDOUT_PATH.write_text(text, encoding="utf-8")
    text = BPB_HELDOUT_PATH.read_text(encoding="utf-8")
    return text, len(text.encode("utf-8"))


def prepare_bpb(tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize the held-out once into fixed-size (x, y) windows for BPB.

    Non-overlapping ``SEQ_LEN`` windows (every target predicted exactly once, no
    BOS/EOS), tail padded with ``ignore_index=-100`` and rounded up to a whole
    number of ``BPB_BATCH_WINDOWS`` batches so the compiled forward sees a single
    fixed shape.
    """
    text, n_bytes = load_bpb_heldout()
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    xs, ys = [], []
    for i in range(0, len(ids) - 1, SEQ_LEN):
        xc, yc = ids[i : i + SEQ_LEN], ids[i + 1 : i + 1 + SEQ_LEN]
        if yc.numel() == 0:
            break
        x = torch.zeros(SEQ_LEN, dtype=torch.long)
        y = torch.full((SEQ_LEN,), -100, dtype=torch.long)
        x[: xc.numel()], y[: yc.numel()] = xc, yc
        xs.append(x)
        ys.append(y)
    X, Y = torch.stack(xs), torch.stack(ys)
    pad = (-len(X)) % BPB_BATCH_WINDOWS
    if pad:  # pad with all-ignored rows -> every batch is exactly BPB_BATCH_WINDOWS
        X = torch.cat([X, torch.zeros(pad, SEQ_LEN, dtype=torch.long)])
        Y = torch.cat([Y, torch.full((pad, SEQ_LEN), -100, dtype=torch.long)])
    return X, Y, n_bytes


@torch.no_grad()
def bits_per_byte(model, X: torch.Tensor, Y: torch.Tensor, n_bytes: int) -> float:
    """Total next-token NLL over the held-out (nats), normalized to bits/byte.

    Plain causal (no doc masking), contiguous RoPE. Tokenizer-invariant: the
    summed token-NLL of a text is ~constant under retokenization, and dividing by
    bytes (fixed) rather than tokens (tokenizer-dependent) cancels the vocab out.
    """
    model.eval()
    weight = getattr(model, "_orig_mod", model).token_emb.weight  # tied lm_head
    total_nll = 0.0
    for b in range(0, len(X), BPB_BATCH_WINDOWS):
        x = X[b : b + BPB_BATCH_WINDOWS].to(DEVICE)
        y = Y[b : b + BPB_BATCH_WINDOWS].to(DEVICE)
        if USE_CCE:
            hidden = model(x, return_hidden=True)
            total_nll += linear_cross_entropy(
                hidden, weight, y, reduction="sum", ignore_index=-100
            ).item()
        else:
            logits = model(x)
            total_nll += nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
                ignore_index=-100,
            ).item()
    model.train()
    return total_nll / n_bytes / LN2


def print_model_stats(model: GPT, global_batch_size: int):
    # verbose=0 + explicit print → exactly one table (avoids a duplicate when the
    # returned ModelStatistics is auto-displayed under a REPL/notebook kernel).
    print(
        summary(
            model,
            input_size=(1, SEQ_LEN),
            dtypes=[torch.long],
            device="cpu",
            depth=2,
            verbose=0,
        )
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # position is supplied by RoPE (no learned pos_emb), so the only embedding
    # table is the tied token embedding / output head.
    embedding_params = model.token_emb.weight.numel()
    non_embedding_params = total_params - embedding_params

    print(f"Total Embedding Parameters: {embedding_params:,}")
    print(f"Total Non-Embedding Parameters: {non_embedding_params:,}")
    print(f"Embedding Parameter Ratio: {embedding_params / total_params:.2%}")
    print(f"Chinchilla Training Tokens (x20): {total_params * 20:,}")
    print(f"OpenCPM-5 Training Tokens (x100): {total_params * 100:,}")
    print(f"Max Train Tokens: {MAX_TRAIN_STEPS * global_batch_size:,}")
    print("=" * 90)


def train():
    dm = make_datamodule()

    x, y = next(iter(dm.train_dataloader()))
    train_tokens = len(dm.train_dataset.data)
    val_tokens = len(dm.val_dataset.data)
    global_batch_size = x.numel()

    print(f"vocab_size={dm.vocab_size}")
    print(
        "train mix: "
        + "  ".join(
            f"{k}={v:,} ({v / train_tokens:.0%})"
            for k, v in dm.source_train_tokens.items()
        )
    )
    print(
        f"train tokens={train_tokens:,}  val tokens={val_tokens:,}  "
        f"total={train_tokens + val_tokens:,}"
    )
    print(f"train batch: x={x.shape}, y={y.shape}")
    print(f"global batch size={global_batch_size:,}")

    model = make_model(dm.vocab_size)
    optimizer = make_optimizer(model)
    print_model_stats(model, global_batch_size)

    model.to(DEVICE, dtype=DTYPE)
    if DEVICE == "cuda":
        model = torch.compile(
            model
        )  # fuse rope/act/residuals; ~1.8x step with FlexAttn

    # tokenizer-agnostic BPB yardstick (fixed held-out, tokenized once for this run)
    bpb_X, bpb_Y, bpb_bytes = prepare_bpb(dm.tokenizer)
    print(f"bpb held-out: {len(bpb_X)} windows over {bpb_bytes:,} bytes")

    global_step = 1
    for epoch in range(N_EPOCHS):
        model.train()
        # cap the epoch at MAX_TRAIN_STEPS batches (the pool is far larger) so the
        # bar tracks the real run length instead of the full ~9k-batch dataloader.
        pbar = tqdm(
            islice(dm.train_dataloader(), MAX_TRAIN_STEPS),
            desc=f"Epoch {epoch + 1}/{N_EPOCHS}",
            total=MAX_TRAIN_STEPS,
            dynamic_ncols=True,
        )

        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = compute_loss(model, x, y, dm.eos_id, dm.vocab_size)
            loss.backward()
            optimizer.step()

            step_loss = loss.detach().item()
            step_perplexity = math.exp(min(step_loss, 10))

            global_step += 1

            if global_step % VALIDATE_EVERY_N_STEPS == 0:
                val_loss = evaluate(model, dm)
                val_perplexity = math.exp(min(val_loss, 100))
                val_bpb = bits_per_byte(model, bpb_X, bpb_Y, bpb_bytes)
                print(
                    f"\nStep {global_step}: train_loss={step_loss:.4f}, train_perplexity={step_perplexity:.2f}, val_loss={val_loss:.4f}, val_perplexity={val_perplexity:.2f}, val_bpb={val_bpb:.4f}"
                )

            # val metrics get their own `Step N:` line; keep the bar to live-changing fields
            pbar.set_postfix(
                step=global_step,
                tokens=f"{global_step * global_batch_size / 1e6:.1f}M",
                loss=f"{step_loss:.4f}",
                ppl=f"{step_perplexity:.2f}",
            )

            if global_step >= MAX_TRAIN_STEPS:
                break

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # Unwrap torch.compile so the checkpoint has clean (no _orig_mod.) keys.
    torch.save(getattr(model, "_orig_mod", model).state_dict(), CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()

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
from itertools import chain, islice
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
    CosmopediaV2DataModule,
    FineWebEduTextDataModule,
    GooAQDataModule,
    LocalDocumentsDataModule,
    SQuADTextDataModule,
    TinyStoriesV2DataModule,
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
# Warmup + cosine LR schedule (None disables -> constant LR). Late-run decay is
# the anti-forgetting lever: what erases early-seen data (documents, early mix
# samples) is late HIGH-lr updates; annealing to FINAL_LR_FRAC consolidates.
LR_SCHEDULE = "warmup-cosine"
WARMUP_STEPS = 250
FINAL_LR_FRAC = 0.1

# Two-phase data curriculum: same per-source TOTALS as the flat mix (so every ids
# cache stays valid), but reordered so cosmopedia dominates the tail of training
# (paired with the LR anneal: what the model sees last under a decaying LR is what
# it consolidates). Ratios are per-phase shares (each sums to 100); their mean must
# equal the flat mix ratios. None disables (single flat-shuffled pool).
CURRICULUM_PHASE_FRAC = 0.5  # fraction of MAX_TRAIN_STEPS in phase 1
RATIOS_PHASE1 = {
    "cosmopedia-v2": 20,
    "fineweb-edu": 38,
    "tinystories-v2": 36,
    "gooaq": 5,
    "squad": 1,
}
RATIOS_PHASE2 = {
    "cosmopedia-v2": 40,  # cos-dominant tail
    "fineweb-edu": 30,
    "tinystories-v2": 24,
    "gooaq": 5,
    "squad": 1,
}
# Final-logit soft-capping (cap*tanh(logits/cap)) during training + eval; None = off.
# Inference-only capping strictly hurt (raw logits reach ~40, cap 30 saturates them);
# this tests the Gemma-2-style TRAIN-time variant. Attention already has QK-norm.
LOGIT_SOFTCAP = 30.0
ADAMW_LR = 1e-3
BATCH_SIZE = 128
MAX_TRAIN_STEPS = 5000
VALIDATE_EVERY_N_STEPS = 500
BENCH_EVERY_N_STEPS = 1500  # in-training benchmark curve (0/None to disable)
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
    TRAIN_TOKENS = 600_000_000
    # projects/tinylm/documents/*.md: always in the mix, outside the ratios.
    # The files are tiny, so they ride along repeated (~a few hundred exposures
    # over the run) rather than as a ratio share; excluded from the mixture
    # tokenizer so the existing vocab + ids caches stay valid.
    DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
    DOCUMENTS_REPEAT = 200

    ratios = {
        "tiny-textbooks": 0,
        "cosmopedia-v2": 30,  # str→cos ablation (vs the logged 3-way str30 fw40 ts30)
        "fineweb-edu": 34,
        "tinystories-v2": 30,
        "tiny-webtext": 0,
        "gooaq": 5,  # closed-book Question:/Answer: format signal
        # grounded passage + Question:/Answer: format signal; the full corpus
        # is only ~6M tokens, so 1% ~= all of SQuAD (a bigger share can't fill)
        "squad": 1,
    }
    total = sum(ratios.values())
    if total != 100:
        raise ValueError(f"train mix ratios must sum to 100, got {total}")

    ratios = {k: v / total for k, v in ratios.items()}  # normalize to sum=1
    print(
        "train mix: "
        + "  ".join(
            f"{k}={int(r * TRAIN_TOKENS):,} ({r:.0%})" for k, r in ratios.items()
        )
    )

    dm = ConcatTextDataModule(
        [
            # owner: carries the canonical vocab_size / backend / doc convention
            # (tokenizer is trained on the blend below, not on this source alone)
            TinyTextbooksDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                vocab_size=16_384,
                max_train_tokens=int(ratios["tiny-textbooks"] * TRAIN_TOKENS),
                max_val_tokens=VAL_TOKENS,
            ),
            CosmopediaV2DataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=int(ratios["cosmopedia-v2"] * TRAIN_TOKENS),
                max_val_tokens=VAL_TOKENS,
            ),
            FineWebEduTextDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=int(ratios["fineweb-edu"] * TRAIN_TOKENS),
                max_val_tokens=VAL_TOKENS,
            ),
            TinyStoriesV2DataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=int(ratios["tinystories-v2"] * TRAIN_TOKENS),
                max_val_tokens=VAL_TOKENS,
            ),
            TinyWebTextDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=int(ratios["tiny-webtext"] * TRAIN_TOKENS),
                max_val_tokens=VAL_TOKENS,
            ),
            GooAQDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=int(ratios["gooaq"] * TRAIN_TOKENS),
                max_val_tokens=VAL_TOKENS,
            ),
            SQuADTextDataModule(
                data_dir=DATA_DIR,
                add_bos=True,
                max_train_tokens=int(ratios["squad"] * TRAIN_TOKENS),
                max_val_tokens=VAL_TOKENS,
            ),
            LocalDocumentsDataModule(
                doc_dir=str(DOCUMENTS_DIR),
                repeat=DOCUMENTS_REPEAT,
                data_dir=DATA_DIR,
                add_bos=True,
            ),
        ],
        batch_size=BATCH_SIZE,
        train_tokenizer_on_mixture=True,
        tokenizer_sample_chars=1_000_000_000,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


def make_curriculum_loaders(dm: ConcatTextDataModule):
    """Split each source's token stream into phase-1/phase-2 chunks per the phase
    ratios and return one shuffled DataLoader per phase. Slicing each source at
    r1/(r1+r2) keeps per-source totals (and ids caches) identical to the flat mix —
    only the ORDER across phases changes. `documents` (and any source not in the
    ratio dicts) is split evenly so it stays present throughout."""
    from torch.utils.data import DataLoader

    from chimera.data._text import TokenDataset

    phase1, phase2 = [], []
    for name, sub in zip(dm.source_names, dm.datamodules):
        data = sub.train_dataset.data
        r1 = RATIOS_PHASE1.get(name)
        r2 = RATIOS_PHASE2.get(name)
        frac = 0.5 if not (r1 or r2) else r1 / (r1 + r2)
        cut = int(len(data) * frac)
        phase1.append(data[:cut])
        phase2.append(data[cut:])

    for label, parts in (("phase1", phase1), ("phase2", phase2)):
        total = sum(len(p) for p in parts)
        mix = "  ".join(
            f"{n}={len(p) / total:.0%}" for n, p in zip(dm.source_names, parts)
        )
        print(f"curriculum {label}: {total:,} tokens  ({mix})")

    def loader(parts):
        return DataLoader(
            TokenDataset(torch.cat(parts), dm.seq_len),
            batch_size=dm.batch_size,
            shuffle=True,
            num_workers=dm.num_workers,
            pin_memory=dm.pin_memory,
            drop_last=True,
        )

    return loader(phase1), loader(phase2)


def make_model(vocab_size: int) -> GPT:
    return GPT(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        dim=DIM,
        n_heads=N_HEADS,
        mlp_mult=MLP_MULT,
        n_layers=N_LAYERS,
        logit_softcap=LOGIT_SOFTCAP,
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
        return linear_cross_entropy(hidden, weight, y, softcap=LOGIT_SOFTCAP)
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


@torch.no_grad()
def run_benchmarks(model, dm, step=None):
    """Zero-shot lm-eval over the standard task set.

    With ``step`` given, prints a compact one-line checkpoint row — call it every
    BENCH_EVERY_N_STEPS to get the in-training benchmark *curve* (is blimp/lambada
    still climbing, or plateaued?). At the end (``step=None``) prints the full
    table + a copy-paste README Results row. Scored on the UNCOMPILED net (varied
    eval shapes would otherwise thrash torch.compile). Mirrors main.ipynb's eval cell."""
    from chimera.evals import CHANCE, GPT2_SMALL, TASKS, ChimeraLM, headline, run_eval

    net = getattr(model, "_orig_mod", model)
    net.eval()
    lm = ChimeraLM(
        net,
        dm.tokenizer,
        eot_id=dm.eos_id,
        bos_id=dm.bos_id,
        block_size=net.seq_len,
        device=DEVICE,
        batch_tokens=131_072,  # tiny model -> big batches, fewer kernel launches
    )
    results = run_eval(lm, TASKS)

    order = ["blimp", "lambada_openai", "piqa", "sciq", "arc_easy"]
    metrics = {}  # task -> (metric_name, value as %)
    for task in TASKS:
        name, val, _ = headline(results[task])
        metrics[task] = (name, val if ("perplex" in name or "bits_per_byte" in name) else val * 100)
    row = " | ".join(f"{t}={metrics[t][1]:.2f}" for t in order if t in metrics)

    if step is not None:
        print(f"\n[bench @ step {step}] {row}")
    else:
        print("\n" + "=" * 60)
        print(f"zero-shot benchmarks  ({MAX_TRAIN_STEPS} steps)")
        print(f"{'task':<16}{'metric':<10}{'model':>8}{'chance':>8}{'gpt2':>8}")
        print("-" * 60)
        for task in TASKS:
            name, pct = metrics[task]
            gpt2 = GPT2_SMALL.get(task)
            print(
                f"{task:<16}{name:<10}{pct:>8.2f}{CHANCE.get(task, float('nan')):>8.1f}"
                f"{'-' if gpt2 is None else f'{gpt2:>8.2f}'}"
            )
        print("=" * 60)
        print(f"[bench] {row}")  # copy-paste into the README Results table
    net.train()
    return {t: metrics[t][1] for t in metrics}


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
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_factor(step: int) -> float:
        if LR_SCHEDULE is None:
            return 1.0
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        t = (step - WARMUP_STEPS) / max(1, MAX_TRAIN_STEPS - WARMUP_STEPS)
        return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * t))
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
        if CURRICULUM_PHASE_FRAC is not None:
            p1_steps = int(MAX_TRAIN_STEPS * CURRICULUM_PHASE_FRAC)
            loader1, loader2 = make_curriculum_loaders(dm)
            train_iter = chain(
                islice(loader1, p1_steps),
                islice(loader2, MAX_TRAIN_STEPS - p1_steps),
            )
        else:
            train_iter = islice(dm.train_dataloader(), MAX_TRAIN_STEPS)
        pbar = tqdm(
            train_iter,
            desc=f"Epoch {epoch + 1}/{N_EPOCHS}",
            total=MAX_TRAIN_STEPS,
            dynamic_ncols=True,
        )

        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * lr_factor(global_step)
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

            # in-training benchmark curve (skip the final step; the full table runs
            # after the loop). Adds ~20s/eval — a few points to see the trajectory.
            if (
                BENCH_EVERY_N_STEPS
                and global_step % BENCH_EVERY_N_STEPS == 0
                and global_step < MAX_TRAIN_STEPS
            ):
                run_benchmarks(model, dm, step=global_step)

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

    # Benchmarks run AFTER the checkpoint is on disk, so an eval hiccup can never
    # cost the trained model.
    run_benchmarks(model, dm)


if __name__ == "__main__":
    train()

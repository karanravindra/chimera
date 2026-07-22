"""Pretrain the tinylm GPT (~6M params) on a 5-way text mixture.

Sources (sampling weight = per-source max_train_tokens): tiny-textbooks 30 /
tiny-strange-textbooks 25 / fineweb-edu 20 / tinystories-v2 15 / tiny-webtext 10.
The 16k BPE vocab is trained on a round-robin sample of all five, so it
compresses the whole mixture rather than the owner's register alone.

Raw PyTorch loop (deliberately independent of the archived Lightning harness):
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
# Persist inductor/autotune artifacts: cold max-autotune costs ~4 min, warm ~seconds.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/mnt/ai/data/torchinductor_cache")

import torch
import torch.nn as nn
from cut_cross_entropy import linear_cross_entropy
from model import GPT
from torchinfo import summary
from tqdm import tqdm

from chimera.data import (
    ConcatTextDataModule,
    ContextMixDataModule,
    CosmopediaV2DataModule,
    FineWebEduTextDataModule,
    GooAQDataModule,
    LocalDocumentsDataModule,
    SQuADTextDataModule,
    StackExchangeDataModule,
    TinyStoriesV2DataModule,
    TinyTextbooksDataModule,
    TinyWebTextDataModule,
    WikipediaDataModule,
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
# Context length. 512 base by default; the context-extension stages bump it via
# TINYLM_SEQ_LEN (2048/4096/8192). The model has no baked context limit (RoPE is
# computed on the fly), so the same weights load and forward at any length.
SEQ_LEN = int(os.environ.get("TINYLM_SEQ_LEN", "512"))
DIM = 384
N_HEADS = 12
MLP_MULT = 3
# Looped A/B vs the 6-unique-layer baseline. Matched-params variant (Ouro's
# recipe): all 6 unique blocks applied twice — same params as baseline, 2x
# per-token FLOPs / unrolled depth 12. N_LOOPS=1 recovers the baseline;
# 3x2 (matched FLOPs) was run 2026-07-20, see loop3x2_train.log.
N_LAYERS = 6
N_LOOPS = 1

# optimization. Env-overridable so a continuation stage (context extension) can
# resume a converged checkpoint at a LOWER peak LR — a fresh full-0.02 rewarm
# perturbs the base (the 2k stage spiked short val_bpb 0.757->0.834 before
# recovering); ~0.2x avoids that while still adapting.
MUON_LR = float(os.environ.get("TINYLM_MUON_LR", "0.02"))
# Warmup + cosine decay to FINAL_LR_FRAC (None -> constant LR). The late-run
# decay doubles as the anti-forgetting lever: high-lr updates near the end are
# what erase early-seen data.
LR_SCHEDULE = "warmup-cosine"
WARMUP_STEPS = 250
FINAL_LR_FRAC = 0.1

# Two-phase data curriculum: same per-source TOTALS as the flat mix (so every
# ids cache stays valid), only the order across phases changes — what the model
# sees last under the decaying LR is what it consolidates. Each phase dict sums
# to 100 and their mean must equal the flat mix ratios below. None disables.
CURRICULUM_PHASE_FRAC = 0.5  # fraction of MAX_TRAIN_STEPS in phase 1
RATIOS_PHASE1 = {
    "cosmopedia-v2": 20,
    "fineweb-edu": 38,
    "tinystories-v2": 36,
    "gooaq": 5,
    "squad": 1,
}
RATIOS_PHASE2 = {
    "cosmopedia-v2": 40,
    "fineweb-edu": 30,
    "tinystories-v2": 24,
    "gooaq": 5,
    "squad": 1,
}
# Gemma-2-style final-logit soft-capping (cap*tanh(logits/cap)), applied in
# training AND eval so both see the same distribution; None = off. Must be a
# train-time setting — capping an uncapped model at inference only hurts.
LOGIT_SOFTCAP = 30.0
ADAMW_LR = float(os.environ.get("TINYLM_ADAMW_LR", "1e-3"))
ADAMW_WEIGHT_DECAY = 0.0

# Virtual Width Networks: residual state runs at (VWN_N/VWN_M)*dim while
# attention/MLP stay at dim. (2, 3) = the paper's 1.5x; (1, 1) recovers the
# plain model. Static routing matrices are excluded from weight decay.
VWN_M = int(os.environ.get("TINYLM_VWN_M", "2"))
VWN_N = int(os.environ.get("TINYLM_VWN_N", "3"))
# The VWN state at batch 128 hits VRAM pressure on the 16GB card (-31% tok/s vs
# 2x64); split each global batch into PHYS_BATCH-row microbatches with gradient
# accumulation — same global batch / identical gradients up to fp rounding. The
# long-context stages drop PHYS_BATCH (attention is O(N^2); a genuine 8k doc has
# no block-sparsity to exploit) and raise GRAD_ACCUM to hold tokens/step ~= 65k:
# 512->64x2, 2k->16x2, 4k->8x2, 8k->8x1 (all TINYLM_SEQ_LEN * BATCH_SIZE ~= 65536).
PHYS_BATCH = int(os.environ.get("TINYLM_PHYS_BATCH", "64"))
GRAD_ACCUM_STEPS = int(os.environ.get("TINYLM_GRAD_ACCUM", "2"))
BATCH_SIZE = PHYS_BATCH * GRAD_ACCUM_STEPS  # global batch (split into microbatches)
MAX_TRAIN_STEPS = int(os.environ.get("TINYLM_MAX_STEPS", "5000"))
# Context-extension stage: "base" (flat 512 TokenDataset, unchanged) or
# 2k/4k/8k (ContextMixDataModule: broad-short pool + long-window pool). Each
# non-base stage resumes from the prior checkpoint via TINYLM_INIT_CKPT.
CTX_STAGE = os.environ.get("TINYLM_CTX_STAGE", "base").strip()
# Weights to load before training (the 512 base, or the prior context stage).
INIT_CKPT = os.environ.get("TINYLM_INIT_CKPT", "").strip() or None
# Per-stage short/long token shares (README "Context expansion route").
CTX_STAGE_SHARES = {"2k": (0.35, 0.65), "4k": (0.27, 0.73), "8k": (0.25, 0.75)}
# Banded-BPB eval cadence (millions of tokens); 0/unset -> reuse bench cadence.
BAND_EVAL_EVERY_MTOKENS = float(os.environ.get("TINYLM_BAND_EVAL_EVERY_MT", "0")) or None
# Eval cadence is expressed in TOKENS (env, in millions) and converted to steps
# below once the global batch is known — so a token budget maps to the same
# real cadence regardless of batch/seq. Falls back to the fixed step counts.
VAL_EVERY_MTOKENS = float(os.environ.get("TINYLM_VAL_EVERY_MT", "0")) or None
BENCH_EVERY_MTOKENS = float(os.environ.get("TINYLM_BENCH_EVERY_MT", "0")) or None
VALIDATE_EVERY_N_STEPS = 1000
BENCH_EVERY_N_STEPS = 2500  # in-training benchmark curve (0/None to disable)
N_EPOCHS = 1

RUN_DIR = Path("/mnt/ai/runs/tinylm/pretrain")
# Looped runs save under their own name so the baseline checkpoint survives.
_LOOP_TAG = "" if N_LOOPS == 1 else f"_loop{N_LAYERS}x{N_LOOPS}"
# Ablation runs save under their own tag so named checkpoints survive.
_RUN_TAG = os.environ.get("TINYLM_RUN_TAG", "")
_RUN_TAG = f"_{_RUN_TAG}" if _RUN_TAG else ""
CHECKPOINT_PATH = RUN_DIR / f"chimera_gpt6m{_LOOP_TAG}{_RUN_TAG}.pt"


def make_datamodule() -> ConcatTextDataModule:
    # Per-source max_train_tokens = sampling weight over the concatenated
    # stream. The pool (TRAIN_TOKENS) deliberately exceeds what the step-capped
    # run consumes, so repetition is negligible. The vocab is trained on a
    # round-robin sample of ALL sources (train_tokenizer_on_mixture) rather
    # than any single register.
    DATA_DIR = "/mnt/ai/data"
    VAL_TOKENS = 500_000
    # Pool per source (sampling weight = share of this). Must exceed the tokens
    # the step-capped run actually consumes (MAX_TRAIN_STEPS x global batch) so
    # repetition stays negligible; env-bumped for the multi-BT context runs.
    TRAIN_TOKENS = int(os.environ.get("TINYLM_TRAIN_TOKENS", "600_000_000"))
    # A pinned, frozen tokenizer (the git-tracked suite) removes the vocab as a
    # cross-run confound: every run keys its ids caches on this file's content
    # hash instead of retraining a per-mix vocab. Empty -> train on the mixture.
    TOKENIZER_PATH = os.environ.get("TINYLM_TOKENIZER_PATH", "").strip() or None
    # Any .md files dropped in documents/ are always in the mix, outside the
    # ratios: repeated DOCUMENTS_REPEAT times (tiny files would otherwise be
    # invisible) and excluded from the mixture tokenizer so the vocab + ids
    # caches stay valid. Empty directory => source skipped.
    DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
    DOCUMENTS_REPEAT = 200

    # The true-4B mix: TinyStories is exhausted at ~0.44B tokens, so at a 4B pool
    # its validated 30% share would repeat ~3x. Drop it to its ceiling (~11%) and
    # let cosmopedia/fineweb (effectively unlimited) absorb the rest, so the bulk
    # is single-pass. Also widen the shard windows for those sources (below).
    # Gated on TINYLM_MIX_4B so the small tokenizer-suite runs are untouched.
    MIX_4B = os.environ.get("TINYLM_MIX_4B", "") == "1"
    if MIX_4B:
        ratios = {
            "tiny-textbooks": 0,
            "cosmopedia-v2": 42,
            "fineweb-edu": 43,
            "tinystories-v2": 11,  # ~0.44B unique = whole corpus, no repeat
            "tiny-webtext": 0,
            "gooaq": 3,
            "squad": 1,  # ~all of the small corpus; format signal (repeats)
        }
        # Widen the download/tokenize windows to supply ~3.9B unique tokens.
        # Instance-process-local class-attr overrides (never touches the shared
        # module defaults on disk): cosmopedia 2->7 shards (~1.9B), fineweb
        # 1->4 shards (~2.8B avail), gooaq 1->2 shards (~0.12B, its ceiling).
        CosmopediaV2DataModule.DATA_FILES = [
            f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(7)
        ]
        FineWebEduTextDataModule.DATA_FILES = [
            f"sample/10BT/{i:03d}_00000.parquet" for i in range(4)
        ]
        GooAQDataModule.DATA_FILES = [
            "pair/train-00000-of-00002.parquet",
            "pair/train-00001-of-00002.parquet",
        ]
    else:
        ratios = {
            "tiny-textbooks": 0,
            "cosmopedia-v2": 30,
            "fineweb-edu": 34,
            "tinystories-v2": 30,
            "tiny-webtext": 0,
            "gooaq": 5,  # closed-book Question:/Answer: format signal
            # grounded passage+QA format signal; small corpus, this share ~= all of it
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
            *(
                [
                    LocalDocumentsDataModule(
                        doc_dir=str(DOCUMENTS_DIR),
                        repeat=DOCUMENTS_REPEAT,
                        data_dir=DATA_DIR,
                        add_bos=True,
                    )
                ]
                if any(DOCUMENTS_DIR.glob("*.md"))
                else []
            ),
        ],
        batch_size=BATCH_SIZE,
        # Pin the frozen tokenizer when given; otherwise train one on the mixture.
        train_tokenizer_on_mixture=TOKENIZER_PATH is None,
        tokenizer_sample_chars=1_000_000_000,
        tokenizer_path=TOKENIZER_PATH,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


def make_context_datamodule() -> ContextMixDataModule:
    """Two-pool datamodule for a context-extension stage (2k/4k/8k).

    A broad SHORT pool (packed windows, short-context retention) plus a LONG pool
    of genuinely long documents (Wikipedia + Stack Exchange + a long-FineWeb
    slice) served as single-document random windows — the only data that trains
    long-range attention, since doc masking resets at every EOS. Both pools pin
    the SAME frozen tokenizer so their ids caches share one vocab.
    """
    DATA_DIR = "/mnt/ai/data"
    VAL_TOKENS = 500_000
    ctx = SEQ_LEN
    TOKENIZER_PATH = os.environ.get("TINYLM_TOKENIZER_PATH", "").strip() or None
    assert TOKENIZER_PATH is not None, (
        "context stages require a pinned frozen tokenizer (TINYLM_TOKENIZER_PATH)"
    )
    assert CTX_STAGE in CTX_STAGE_SHARES, (
        f"unknown TINYLM_CTX_STAGE={CTX_STAGE!r}; expected one of {list(CTX_STAGE_SHARES)}"
    )
    short_share, long_share = CTX_STAGE_SHARES[CTX_STAGE]
    # Per-pool token pools (sampling weight = per-source cap). Generous so the
    # step-capped run stays ~single-pass; env-overridable.
    SHORT_TOKENS = int(os.environ.get("TINYLM_SHORT_POOL_TOKENS", "1_500_000_000"))
    LONG_TOKENS = int(os.environ.get("TINYLM_LONG_POOL_TOKENS", "1_500_000_000"))

    def _cap(pool: int, frac: float) -> int:
        return int(pool * frac)

    short_pool = ConcatTextDataModule(
        [
            # owner carries vocab_size/backend/doc convention (vocab is pinned)
            CosmopediaV2DataModule(
                data_dir=DATA_DIR, add_bos=True, seq_len=ctx, vocab_size=16_384,
                max_train_tokens=_cap(SHORT_TOKENS, 0.30), max_val_tokens=VAL_TOKENS,
            ),
            FineWebEduTextDataModule(
                data_dir=DATA_DIR, add_bos=True, seq_len=ctx,
                max_train_tokens=_cap(SHORT_TOKENS, 0.34), max_val_tokens=VAL_TOKENS,
            ),
            TinyStoriesV2DataModule(
                data_dir=DATA_DIR, add_bos=True, seq_len=ctx,
                max_train_tokens=_cap(SHORT_TOKENS, 0.30), max_val_tokens=VAL_TOKENS,
            ),
            GooAQDataModule(
                data_dir=DATA_DIR, add_bos=True, seq_len=ctx,
                max_train_tokens=_cap(SHORT_TOKENS, 0.05), max_val_tokens=VAL_TOKENS,
            ),
            SQuADTextDataModule(
                data_dir=DATA_DIR, add_bos=True, seq_len=ctx,
                max_train_tokens=_cap(SHORT_TOKENS, 0.01), max_val_tokens=VAL_TOKENS,
            ),
        ],
        batch_size=BATCH_SIZE,
        tokenizer_path=TOKENIZER_PATH,
    )
    long_pool = ConcatTextDataModule(
        [
            WikipediaDataModule(
                data_dir=DATA_DIR, add_bos=True, seq_len=ctx, vocab_size=16_384,
                max_train_tokens=_cap(LONG_TOKENS, 0.50), max_val_tokens=VAL_TOKENS,
            ),
            # long-FineWeb slice: short pages get filtered out by min_doc_len,
            # leaving the genuinely long coherent pages as long-range training data
            FineWebEduTextDataModule(
                data_dir=DATA_DIR, add_bos=True, seq_len=ctx,
                max_train_tokens=_cap(LONG_TOKENS, 0.35), max_val_tokens=VAL_TOKENS,
            ),
            StackExchangeDataModule(
                data_dir=DATA_DIR, add_bos=True, seq_len=ctx,
                max_train_tokens=_cap(LONG_TOKENS, 0.15), max_val_tokens=VAL_TOKENS,
            ),
        ],
        batch_size=BATCH_SIZE,
        tokenizer_path=TOKENIZER_PATH,
    )
    dm = ContextMixDataModule(
        short_pool,
        long_pool,
        ctx=ctx,
        short_share=short_share,
        long_share=long_share,
        batch_size=BATCH_SIZE,
        # bound the sampler to exactly this run's length (one item == one ctx row)
        num_samples=MAX_TRAIN_STEPS * BATCH_SIZE,
    )
    dm.prepare_data()
    dm.setup("fit")
    print(
        f"context stage {CTX_STAGE}: ctx={ctx} short/long={short_share:.0%}/{long_share:.0%}  "
        f"short_items={len(dm.short_dataset):,} long_windows={len(dm.long_dataset):,}"
    )
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



def make_model(vocab_size: int, seq_len: int = SEQ_LEN) -> GPT:
    return GPT(
        vocab_size=vocab_size,
        seq_len=seq_len,
        dim=DIM,
        n_heads=N_HEADS,
        mlp_mult=MLP_MULT,
        n_layers=N_LAYERS,
        n_loops=N_LOOPS,
        logit_softcap=LOGIT_SOFTCAP,
        vwn_m=VWN_M,
        vwn_n=VWN_N,
    )


def make_optimizer(model: GPT) -> Muon:
    # VWN routing tensors ("connection") are not hidden weight matrices — keep
    # them off Muon's orthogonalization and on AdamW. The static routing
    # matrices additionally get no weight decay (model.no_weight_decay()).
    groups = muon_param_groups(
        model,
        muon_lr=MUON_LR,
        adamw_lr=ADAMW_LR,
        adamw_weight_decay=ADAMW_WEIGHT_DECAY,
        adamw_name_keywords=("emb", "head", "connection"),
    )
    no_decay = model.no_weight_decay()
    if no_decay and ADAMW_WEIGHT_DECAY:
        _split_no_decay_group(model, groups, no_decay)
    return Muon(groups)


def _split_no_decay_group(model, groups, no_decay_names: set[str]) -> None:
    """Move the named params into a sibling AdamW group with weight_decay=0."""
    no_decay_ids = {id(p) for n, p in model.named_parameters() if n in no_decay_names}
    for group in list(groups):
        if group.get("use_muon"):
            continue
        keep = [p for p in group["params"] if id(p) not in no_decay_ids]
        move = [p for p in group["params"] if id(p) in no_decay_ids]
        if not move:
            continue
        group["params"] = keep
        groups.append({**group, "params": move, "weight_decay": 0.0})


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
    try:
        from chimera.evals import (
            CHANCE,
            GPT2_SMALL,
            TASKS,
            ChimeraLM,
            headline,
            run_eval,
        )
    except ModuleNotFoundError as error:
        if error.name in {"lm_eval", "pandas", "transformers"}:
            print(
                "[bench] skipped: install optional evaluation dependencies with "
                "`uv sync --extra eval`"
            )
            return {}
        raise

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
    is_context = CTX_STAGE != "base"
    dm = make_context_datamodule() if is_context else make_datamodule()

    x, y = next(iter(dm.train_dataloader()))
    global_batch_size = x.numel()
    if is_context:
        # ContextMixDataModule.train_dataset is a ConcatDataset (no flat .data)
        train_tokens = (len(dm.short_dataset) + len(dm.long_dataset)) * dm.ctx
        val_tokens = len(dm.short_pool.val_dataset.data)
    else:
        train_tokens = len(dm.train_dataset.data)
        val_tokens = len(dm.val_dataset.data)

    # token-budget cadence -> steps (fall back to the fixed step constants)
    val_every = (
        max(1, round(VAL_EVERY_MTOKENS * 1e6 / global_batch_size))
        if VAL_EVERY_MTOKENS
        else VALIDATE_EVERY_N_STEPS
    )
    bench_every = (
        max(1, round(BENCH_EVERY_MTOKENS * 1e6 / global_batch_size))
        if BENCH_EVERY_MTOKENS
        else BENCH_EVERY_N_STEPS
    )
    band_every = (
        max(1, round(BAND_EVAL_EVERY_MTOKENS * 1e6 / global_batch_size))
        if BAND_EVAL_EVERY_MTOKENS
        else bench_every
    )
    print(f"cadence: val every {val_every} steps, bench every {bench_every} steps")

    print(f"vocab_size={dm.vocab_size}")
    if not is_context:
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
    # Resume from a prior checkpoint (the 512 base, or the previous context
    # stage) before compile so the state_dict keys are clean. The model has no
    # seq_len-dependent buffers, so 512 weights load into a longer-ctx model
    # unchanged.
    if INIT_CKPT is not None:
        state = torch.load(INIT_CKPT, map_location="cpu")
        model.load_state_dict(state)
        print(f"loaded init weights from {INIT_CKPT}")
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
        # max-autotune is REQUIRED with the fused GHC routing: default-mode
        # inductor picks pathological backward reduction kernels for the
        # broadcast-sum maps (bwd 174ms vs 52ms at batch 64). cudagraphs add
        # nothing here (per-batch block mask defeats capture), so skip them.
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    # tokenizer-agnostic BPB yardstick (fixed held-out, tokenized once for this run)
    bpb_X, bpb_Y, bpb_bytes = prepare_bpb(dm.tokenizer)
    print(f"bpb held-out: {len(bpb_X)} windows over {bpb_bytes:,} bytes")

    global_step = 1
    for epoch in range(N_EPOCHS):
        model.train()
        # cap the epoch at MAX_TRAIN_STEPS batches (the pool is far larger) so
        # the bar tracks the real run length, not the full dataloader.
        if is_context:
            # context stages: single mixed loader (short + long-window pools),
            # no phase curriculum. Resample long-window offsets each epoch.
            dm.set_epoch(epoch)
            train_iter = islice(dm.train_dataloader(), MAX_TRAIN_STEPS)
        elif CURRICULUM_PHASE_FRAC is not None:
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
            # Microbatches are equal-sized, so the mean of half-losses equals the
            # global-batch mean loss and gradients match the unsplit batch.
            micro = x.shape[0] // GRAD_ACCUM_STEPS
            loss = torch.zeros((), device=DEVICE)
            for mb in range(GRAD_ACCUM_STEPS):
                sl = slice(mb * micro, (mb + 1) * micro)
                mb_loss = (
                    compute_loss(model, x[sl], y[sl], dm.eos_id, dm.vocab_size)
                    / GRAD_ACCUM_STEPS
                )
                mb_loss.backward()
                loss += mb_loss.detach()
            optimizer.step()

            step_loss = loss.item()
            step_perplexity = math.exp(min(step_loss, 10))

            global_step += 1

            if global_step % val_every == 0:
                val_loss = evaluate(model, dm)
                val_perplexity = math.exp(min(val_loss, 100))
                val_bpb = bits_per_byte(model, bpb_X, bpb_Y, bpb_bytes)
                print(
                    f"\nStep {global_step}: train_loss={step_loss:.4f}, train_perplexity={step_perplexity:.2f}, val_loss={val_loss:.4f}, val_perplexity={val_perplexity:.2f}, val_bpb={val_bpb:.4f}"
                )
                # length-banded BPB on long held-out docs: does widening the
                # context actually lower bpb (long-range modelling) or flatline?
                if is_context and global_step % band_every == 0:
                    from bpb_banded import (
                        CTX_WIDTHS,
                        PROBE_DISTANCES,
                        retrieval_probe,
                        score_banded,
                    )

                    net = getattr(model, "_orig_mod", model)
                    widths = [w for w in CTX_WIDTHS if w <= SEQ_LEN]
                    bands = score_banded(net, dm.tokenizer, DEVICE, widths=widths)
                    print(
                        "  banded bpb: "
                        + "  ".join(f"{k}={v:.4f}" for k, v in bands.items())
                    )
                    # synthetic long-range recall (bigram induction) up to ctx
                    dists = [d for d in PROBE_DISTANCES if d <= SEQ_LEN]
                    probe = retrieval_probe(net, DEVICE, distances=dists)
                    print(
                        "  retrieval: "
                        + "  ".join(f"{k}={v:.2f}" for k, v in probe.items())
                    )

            # in-training benchmark curve (skip the final step; the full table runs
            # after the loop) — a few cheap points to see the trajectory.
            if (
                bench_every
                and global_step % bench_every == 0
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

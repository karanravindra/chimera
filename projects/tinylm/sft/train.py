"""SFT the pretrained tinylm GPT (~6M params) on a simple-QA chat mixture.

Mirrors ../pretrain/train.py: same model (imported from there), same packed
FlexAttention doc-masking + CCE + Muon/AdamW rails — but the stream is
ChatML conversations with loss only on assistant tokens (labels are -100
elsewhere; CCE's ignore_index). Starts from the pretrain checkpoint and keeps
its tokenizer (chat special tokens were reserved in the vocab from day one).

Run from this directory:

    uv run python train.py
"""

import math
import os
import sys
from itertools import islice
from pathlib import Path

os.environ.setdefault("CCE_AUTOTUNE", "1")
os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

import torch
from cut_cross_entropy import linear_cross_entropy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "pretrain"))
from model import GPT  # noqa: E402  (the pretrain model, single source of truth)

from chimera.data import (  # noqa: E402
    EverydayConversationsDataModule,
    GooAQChatDataModule,
    SQuADChatDataModule,
)
from chimera.data._text import MaskedTokenDataset  # noqa: E402
from chimera.data.chat_template import render  # noqa: E402
from chimera.models.attention import build_block_mask_and_pos  # noqa: E402
from chimera.optim import Muon, muon_param_groups  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# architecture — MUST match the pretrain checkpoint
SEQ_LEN = 512
DIM = 384
N_HEADS = 12
MLP_MULT = 3
N_LAYERS = 6
LOGIT_SOFTCAP = 30.0  # base model was trained under this cap; keep it

# optimization — gentler than pretrain: the model is converged, SFT reshapes
MUON_LR = 0.005
ADAMW_LR = 2.5e-4

# LoRA mode: freeze the base, train low-rank deltas on every Linear (qkv/out/
# fc1/fc2; embeddings + tied head stay frozen). Merged back before saving, so
# the checkpoint stays architecture-identical to full-FT runs.
USE_LORA = os.environ.get("USE_LORA", "0") == "1"  # USE_LORA=1 uv run python train.py
LORA_R = 16
LORA_ALPHA = 32.0
LORA_LR = 1e-3  # LoRA convention: well above the full-FT AdamW lr
BATCH_SIZE = 128
MAX_TRAIN_STEPS = 700
VALIDATE_EVERY_N_STEPS = 100
N_EPOCHS = 2  # small pool; MAX_TRAIN_STEPS is the real cap
WARMUP_STEPS = 50
FINAL_LR_FRAC = 0.1

# the pretrain vocab (chat specials already reserved); SFT never retrains it
TOKENIZER_PATH = "/mnt/ai/data/mixture_tokenizers/tok_hf_v16384_c1000000000_371c2bf05f53.json"
BASE_CHECKPOINT = "/mnt/ai/runs/tinylm/pretrain/chimera_gpt6m.pt"

RUN_DIR = Path("/mnt/ai/runs/tinylm/sft")
CHECKPOINT_PATH = RUN_DIR / (
    "chimera_gpt6m_sft_lora.pt" if USE_LORA else "chimera_gpt6m_sft.pt"
)

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


def make_datamodules() -> list:
    """The simple-QA chat mixture: bulk closed-book QA + grounded QA + chat style."""
    DATA_DIR = "/mnt/ai/data"
    common = dict(
        tokenizer_path=TOKENIZER_PATH,
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        max_val_tokens=250_000,
    )
    dms = [
        GooAQChatDataModule(max_train_tokens=15_000_000, **common),
        SQuADChatDataModule(max_train_tokens=None, **common),
        EverydayConversationsDataModule(max_train_tokens=None, **common),
    ]
    for dm in dms:
        dm.prepare_data()
        dm.setup("fit")
    return dms


def concat_streams(dms) -> tuple[MaskedTokenDataset, MaskedTokenDataset, dict]:
    """One packed train/val stream across sources (each ends on an EOS boundary)."""
    train = MaskedTokenDataset(
        torch.cat([dm.train_dataset.ids for dm in dms]),
        torch.cat([dm.train_dataset.labels for dm in dms]),
        SEQ_LEN,
    )
    val = MaskedTokenDataset(
        torch.cat([dm.val_dataset.ids for dm in dms]),
        torch.cat([dm.val_dataset.labels for dm in dms]),
        SEQ_LEN,
    )
    mix = {dm.name: len(dm.train_dataset.ids) for dm in dms}
    return train, val, mix


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


def compute_loss(model, x, y, eos_id: int):
    block_mask, pos_ids = build_block_mask_and_pos(x, eos_id)
    hidden = model(x, return_hidden=True, block_mask=block_mask, pos_ids=pos_ids)
    weight = getattr(model, "_orig_mod", model).token_emb.weight
    return linear_cross_entropy(hidden, weight, y, softcap=LOGIT_SOFTCAP)


@torch.no_grad()
def evaluate(model, loader, eos_id: int) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        losses.append(compute_loss(model, x, y, eos_id).item())
    model.train()
    return sum(losses) / max(1, len(losses))


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
        print(f"\n>>> {messages[-1]['content'][:80]}\n{tokenizer._tok.decode(out).strip()!r}")
    net.train()


def train():
    dms = make_datamodules()
    tok = dms[0].tokenizer
    eos_id, bos_id = dms[0].eos_id, dms[0].bos_id
    im_end_id = tok._tok.token_to_id("<|im_end|>")

    train_ds, val_ds, mix = concat_streams(dms)
    total = sum(mix.values())
    print("sft mix: " + "  ".join(f"{k}={v:,} ({v / total:.0%})" for k, v in mix.items()))
    print(f"train tokens={total:,}  supervised={int((train_ds.labels != -100).sum()):,}")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    model = make_model(tok.vocab_size)
    state = torch.load(BASE_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    print(f"loaded base checkpoint: {BASE_CHECKPOINT}")

    if USE_LORA:
        from chimera.models.lora import apply_lora

        lora_params = apply_lora(model, r=LORA_R, alpha=LORA_ALPHA)
        n_lora = sum(p.numel() for p in lora_params)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"LoRA r={LORA_R}: {n_lora:,} trainable / {n_total:,} ({n_lora / n_total:.1%})")

    model.to(DEVICE, dtype=DTYPE)
    if DEVICE == "cuda":
        model = torch.compile(model)

    if USE_LORA:
        # only the LoRA A/B params; base weights keep requires_grad=True (the
        # flex_attn wrapper breaks on no-grad q/k/v) but are never stepped
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=LORA_LR,
            weight_decay=0.0,  # don't decay low-rank deltas toward zero
        )
    else:
        optimizer = Muon(muon_param_groups(model, muon_lr=MUON_LR, adamw_lr=ADAMW_LR))
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_factor(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        t = (step - WARMUP_STEPS) / max(1, MAX_TRAIN_STEPS - WARMUP_STEPS)
        return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * t))

    global_step = 0
    for epoch in range(N_EPOCHS):
        steps_left = MAX_TRAIN_STEPS - global_step
        if steps_left <= 0:
            break
        pbar = tqdm(
            islice(train_loader, steps_left),
            desc=f"Epoch {epoch + 1}/{N_EPOCHS}",
            total=min(steps_left, len(train_loader)),
            dynamic_ncols=True,
        )
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * lr_factor(global_step)
            # model-wide (not optimizer-wide): in LoRA mode base weights still
            # produce grads (see apply_lora) which must not accumulate
            model.zero_grad(set_to_none=True)
            loss = compute_loss(model, x, y, eos_id)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % VALIDATE_EVERY_N_STEPS == 0:
                val_loss = evaluate(model, val_loader, eos_id)
                print(
                    f"\nStep {global_step}: train_loss={loss.item():.4f}, "
                    f"val_loss={val_loss:.4f}, val_ppl={math.exp(min(val_loss, 20)):.2f}"
                )
            pbar.set_postfix(step=global_step, loss=f"{loss.item():.4f}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    net = getattr(model, "_orig_mod", model)
    if USE_LORA:
        from chimera.models.lora import merge_lora

        net = merge_lora(net)  # fold deltas back -> base-architecture state_dict
    torch.save(net.state_dict(), CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    net = net.to(DEVICE)
    print_generations(net, tok, bos_id, eos_id, im_end_id)


if __name__ == "__main__":
    train()

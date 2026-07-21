"""Score GPT-2 (small, 124M) bits-per-byte on the tinylm fixed held-out text.

Uses the EXACT same held-out (tiny-textbooks test, 500 docs, cached at
BPB_HELDOUT_PATH) and the same formula as train.py's bits_per_byte: summed
next-token NLL (nats) over the held-out, normalized by UTF-8 bytes (not
tokens) and converted to bits. Normalizing by bytes is what makes bpb
comparable ACROSS tokenizers/vocabs — GPT-2's own BPE encodes the same text
differently than our tokenizers, but the byte count is fixed, so the number is
a fair reference point for the README's gpt2 row (previously only zero-shot
task accuracy, never bpb).

CPU-only by default (won't touch the GPU running a training job); pass
--device cuda to use the GPU if it's free.

Usage:
    uv run python bpb_gpt2.py
    uv run python bpb_gpt2.py --model gpt2-medium --device cuda
"""

import argparse
import math
from pathlib import Path

import torch

# Repo-tracked copy first (self-contained, works on any machine incl. a Mac
# with no /mnt/ai); falls back to the shared cache path used by train.py.
_REPO_HELDOUT = Path(__file__).parent / "eval_data" / "bpb_heldout.txt"
BPB_HELDOUT_PATH = (
    _REPO_HELDOUT
    if _REPO_HELDOUT.exists()
    else Path("/mnt/ai/data/tinylm/bpb_heldout.txt")
)
BPB_HELDOUT_DOCS = 500
LN2 = math.log(2.0)


def load_bpb_heldout() -> tuple[str, int]:
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


@torch.no_grad()
def score(model_name: str, device: str, block_size: int, stride: int) -> float:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    text, n_bytes = load_bpb_heldout()
    ids = tok(text, return_tensors="pt").input_ids[0].to(device)

    # Sliding-window NLL: each position's loss counted once, via overlapping
    # windows that only score the NEW (non-overlapping) suffix per step —
    # standard rolling-loglikelihood scoring for a fixed causal context.
    total_nll = 0.0
    n_scored = 0
    prev_end = 0
    for begin in range(0, ids.numel(), stride):
        end = min(begin + block_size, ids.numel())
        window = ids[begin:end].unsqueeze(0)
        target_len = end - max(begin, prev_end)  # only score the new suffix
        if window.numel() < 2:
            break
        out = model(window)
        logits = out.logits[0, :-1]
        targets = window[0, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits, targets, reduction="none"
        )
        # score only the tail `target_len` predictions (the new, non-overlapping part)
        total_nll += nll[-target_len:].sum().item()
        n_scored += target_len
        prev_end = end
        if end == ids.numel():
            break

    bpb = total_nll / n_bytes / LN2
    print(
        f"{model_name}: n_bytes={n_bytes:,} n_tokens_scored={n_scored:,} "
        f"total_nll_nats={total_nll:.1f} bpb={bpb:.4f}"
    )
    return bpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    args = ap.parse_args()
    score(args.model, args.device, args.block_size, args.stride)


if __name__ == "__main__":
    main()

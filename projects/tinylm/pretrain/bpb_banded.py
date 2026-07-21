"""Length-banded bits-per-byte for the context-extension stages.

The single-band ``bits_per_byte`` in train.py scores non-overlapping 512-token
windows — it cannot tell whether a longer-context model actually *uses* the extra
context. This module scores the SAME fixed set of long held-out documents at
several context widths (512 / 2k / 4k / 8k) and reports one bpb per width:

    val/bpb_512  val/bpb_2k  val/bpb_4k  val/bpb_8k

If widening the window lowers bpb on genuinely long documents, the model is
exploiting long-range dependencies; if it flatlines, it isn't. Only widths
``<= ctx`` of the current stage are meaningful (the others are skipped by the
caller).

Scoring reuses the exact double-counting-safe rolling-loglikelihood loop from
``bpb_gpt2.score``: overlapping windows that only add the newly-advanced suffix's
NLL, so every token position is counted once. Forward is the tinylm GPT with no
document mask and contiguous RoPE (a single coherent document window) — identical
to train.py's ``bits_per_byte`` forward, just at a longer block size. Run on the
UNCOMPILED net so the varied window sizes don't thrash torch.compile.

The held-out is a fixed slice of long Wikipedia articles, cached like
``bpb_heldout.txt`` so the byte denominator is stable and tokenizer-agnostic.
"""

import math
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)

# Fixed long-document held-out. Repo-tracked copy first (portable), else the
# shared /mnt/ai cache; built from Wikipedia if neither exists.
_REPO_HELDOUT = Path(__file__).parent / "eval_data" / "bpb_long_heldout.txt"
_CACHE_HELDOUT = Path("/mnt/ai/data/tinylm/bpb_long_heldout.txt")
# One document per line, blank-line separated on disk -> split on the sentinel.
_DOC_SEP = "\n\x1e\n"  # record separator; never appears in normal text

CTX_WIDTHS = (512, 2048, 4096, 8192)
BAND_NAMES = {512: "val/bpb_512", 2048: "val/bpb_2k", 4096: "val/bpb_4k", 8192: "val/bpb_8k"}
DEFAULT_LONG_DOCS = 200  # articles in the held-out slice

# Retrieval probe: distances (tokens between the two occurrences of the key)
PROBE_DISTANCES = (128, 512, 1024, 2048, 4096, 8192)
PROBE_NAMES = {
    128: "probe/recall_128", 512: "probe/recall_512", 1024: "probe/recall_1k",
    2048: "probe/recall_2k", 4096: "probe/recall_4k", 8192: "probe/recall_8k",
}


def _build_long_heldout(n_docs: int = DEFAULT_LONG_DOCS) -> str:
    """Fixed long-article held-out (Wikipedia val slice), cached to disk."""
    from datasets import load_dataset

    # first shard, tail rows -> disjoint from any train slice taken off the head
    ds = load_dataset(
        "wikimedia/wikipedia",
        data_files=["20231101.en/train-00000-of-00041.parquet"],
        split="train",
        cache_dir="/mnt/ai/data/hf_cache",
        verification_mode="no_checks",
    )
    docs, i = [], len(ds) - 1
    # walk backwards collecting genuinely long articles
    while i >= 0 and len(docs) < n_docs:
        text = ds[i]["text"]
        if len(text) >= 8192:  # rough char floor; token length checked per band
            docs.append(text)
        i -= 1
    return _DOC_SEP.join(docs)


def load_long_heldout() -> list[str]:
    for path in (_REPO_HELDOUT, _CACHE_HELDOUT):
        if path.exists():
            return path.read_text(encoding="utf-8").split(_DOC_SEP)
    text = _build_long_heldout()
    _CACHE_HELDOUT.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_HELDOUT.write_text(text, encoding="utf-8")
    return text.split(_DOC_SEP)


@torch.no_grad()
def _score_doc(net, ids: torch.Tensor, block_size: int, stride: int, device: str) -> tuple[float, int]:
    """Rolling-window summed NLL (nats) for one document; each position once."""
    total_nll, n_scored, prev_end = 0.0, 0, 0
    for begin in range(0, ids.numel(), stride):
        end = min(begin + block_size, ids.numel())
        window = ids[begin:end].unsqueeze(0).to(device)
        if window.numel() < 2:
            break
        target_len = end - max(begin, prev_end)  # only the new (non-overlap) suffix
        logits = net(window)  # plain causal, contiguous RoPE, no doc mask
        logits = logits[0, :-1]
        targets = window[0, 1:]
        nll = F.cross_entropy(logits.float(), targets, reduction="none")
        total_nll += nll[-target_len:].sum().item()
        n_scored += target_len
        prev_end = end
        if end == ids.numel():
            break
    return total_nll, n_scored


@torch.no_grad()
def score_banded(
    net,
    tokenizer,
    device: str,
    widths: Sequence[int] = CTX_WIDTHS,
    docs: Optional[list[str]] = None,
) -> dict[str, float]:
    """bpb per context width over the fixed long held-out.

    For width ``w`` only documents with at least ``w`` tokens are scored (so there
    is genuinely ``w`` tokens of context to model), with a ``w``-token window and
    ``w // 2`` stride. Returns ``{"val/bpb_2k": ..., ...}`` for each requested width.
    """
    was_training = net.training
    net.eval()
    if docs is None:
        docs = load_long_heldout()

    # tokenize once; keep bytes alongside for the denominator
    tok_docs = [(torch.tensor(tokenizer.encode(d), dtype=torch.long), len(d.encode("utf-8"))) for d in docs]

    out: dict[str, float] = {}
    for w in widths:
        total_nll, total_bytes = 0.0, 0
        for ids, n_bytes in tok_docs:
            if ids.numel() < w:  # not long enough to exercise this context width
                continue
            nll, _ = _score_doc(net, ids, block_size=w, stride=w // 2, device=device)
            total_nll += nll
            total_bytes += n_bytes
        if total_bytes:
            out[BAND_NAMES.get(w, f"val/bpb_{w}")] = total_nll / total_bytes / LN2
    if was_training:
        net.train()
    return out


def _induction_batch(
    d: int, n_trials: int, prefix_len: int, vocab_size: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch of bigram-induction sequences and their target tokens.

    Layout per row: ``[prefix filler] A B [gap filler of length d] A`` — the model
    sees the key ``A`` twice, ``d+1`` tokens apart, with the value ``B`` right after
    the first ``A``. The prediction at the final position should be ``B`` iff the
    model retrieves it across the gap. Filler tokens are drawn from a low id range
    and ``A``/``B`` from a disjoint high range, so a random filler token can never
    collide with the key/value and create a spurious match.
    """
    lo, mid = 2, max(3, vocab_size // 2)  # 0,1 are reserved specials (EOS/BOS)
    hi = vocab_size
    seq_len = prefix_len + 2 + d + 1
    seqs = torch.randint(lo, mid, (n_trials, seq_len), generator=gen)
    ab = torch.randint(mid, hi, (n_trials, 2), generator=gen)
    a, b = ab[:, 0], ab[:, 1]
    b = torch.where(b == a, (b + 1 - mid) % (hi - mid) + mid, b)  # ensure A != B
    seqs[:, prefix_len] = a  # first key
    seqs[:, prefix_len + 1] = b  # value
    seqs[:, -1] = a  # second key (query)
    return seqs, b


@torch.no_grad()
def retrieval_probe(
    net,
    device: str,
    distances=PROBE_DISTANCES,
    n_trials: int = 64,
    prefix_len: int = 8,
    micro_batch: int = 16,
    seed: int = 0,
) -> dict[str, float]:
    """Bigram-induction recall accuracy vs key→value distance.

    For each distance, ``n_trials`` sequences plant a random ``(A, B)`` pair, gap it
    by ``d`` tokens, then re-present ``A``; accuracy is the fraction where the model's
    top prediction after the second ``A`` is ``B`` (chance ~= 1/vocab, so any signal
    is real). A recall curve that holds up to distance ``d`` shows attention reaching
    back ``d`` tokens — the long-range capability a bpb-only signal can't localize.
    Forward is plain causal on the uncompiled net (single sequence, no doc mask).
    """
    was_training = net.training
    net.eval()
    vocab_size = net.token_emb.weight.shape[0]
    weight = net.token_emb.weight
    gen = torch.Generator().manual_seed(seed)

    out: dict[str, float] = {}
    for d in distances:
        seqs, targets = _induction_batch(d, n_trials, prefix_len, vocab_size, gen)
        correct = 0
        for b0 in range(0, n_trials, micro_batch):
            xb = seqs[b0 : b0 + micro_batch].to(device)
            tb = targets[b0 : b0 + micro_batch].to(device)
            hidden = net(xb, return_hidden=True)  # (B, N, dim)
            last = hidden[:, -1, :]
            logits = F.linear(last.float(), weight.float())  # (B, V)
            pred = logits.argmax(dim=-1)
            correct += int((pred == tb).sum())
        out[PROBE_NAMES.get(d, f"probe/recall_{d}")] = correct / n_trials
    if was_training:
        net.train()
    return out

"""Bits-per-byte (BPB): cross-tokenizer-comparable loss + a cached converter.

Per-token cross-entropy is tokenizer-dependent -- a larger vocab packs more
bytes into each token, which deflates the per-token loss -- so a raw ``val/loss``
(nats) can't be compared across tokenizers or against published thresholds
(e.g. modded-nanogpt's 3.28, Du et al.'s 2.2). Bits-per-byte normalizes that out:

    BPB = loss_nats / (ln 2 * bytes_per_token)

This is algebraically the corpus-level ``(total bits) / (total UTF-8 bytes)`` as
long as ``loss_nats`` is the token-mean CE and ``bytes_per_token`` is the
*aggregate* ``(total bytes) / (total tokens)`` over the SAME corpus -- not a
mean-of-per-example ratios. So the only quantity worth measuring (and caching)
is ``bytes_per_token`` for a given (tokenizer, mix): we get it by decoding the
packed val stream back to UTF-8 and dividing total bytes by total tokens.

The measurement is cached to a git-tracked JSON (``bpb_cache.json`` next to this
file) keyed by tokenizer id then mix name, so it runs once and is reused across
runs/analyses and by teammates. Usage::

    from bpb import bytes_per_token_cached, nats_to_bpb, bpb_to_nats
    bpt = bytes_per_token_cached("LiquidAI/LFM2.5-230M", "mix_1B")
    bpb = nats_to_bpb(4.408, bpt)              # -> current model's BPB
    target_nats = bpb_to_nats(1.0, bpt)        # -> val/loss to aim for at BPB 1.0

Or from the CLI (measures + caches, then prints the conversion table)::

    uv run python projects/llm/gpt/bpb.py --mix mix_1B --loss 4.408
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_PATH = Path(__file__).with_name("bpb_cache.json")
LN2 = math.log(2.0)


# -- pure conversions ---------------------------------------------------------


def nats_to_bpb(loss_nats: float, bytes_per_token: float) -> float:
    """Per-token cross-entropy (nats) -> bits-per-byte."""
    return loss_nats / (LN2 * bytes_per_token)


def bpb_to_nats(bpb: float, bytes_per_token: float) -> float:
    """Bits-per-byte -> the per-token cross-entropy (nats) it implies. Inverse of
    :func:`nats_to_bpb`; use it to turn a BPB target into a ``val/loss`` target."""
    return bpb * LN2 * bytes_per_token


# -- measurement --------------------------------------------------------------


def measure_bytes_per_token(
    tokenizer,
    bin_path: Path,
    *,
    dtype=np.uint16,
    max_tokens: Optional[int] = None,
    chunk_tokens: int = 1_000_000,
    mask_path: Optional[Path] = None,
) -> dict:
    """Decode the packed token stream at ``bin_path`` and return byte statistics.

    Returns ``{bytes_per_token, total_bytes, total_tokens}``. ``bytes_per_token``
    is the aggregate ``total_bytes / total_tokens`` -- exactly the factor that
    turns this corpus's token-mean loss into corpus-level BPB. Decodes in chunks
    to bound memory; ``max_tokens`` caps the scan (``None`` = whole file, which is
    the right default for a ~1% val split of a 1B mix).

    ``mask_path`` (SFT): a parallel ``uint8`` supervise mask; when given, only the
    supervised tokens are counted, so ``bytes_per_token`` is bytes-per-SUPERVISED-
    token -- the correct BPB normalizer for a masked SFT loss (which averages only
    over supervised/assistant tokens).
    """
    data = np.memmap(str(bin_path), dtype=dtype, mode="r")
    mask = np.memmap(str(mask_path), dtype=np.uint8, mode="r") if mask_path else None
    n = len(data) if max_tokens is None else min(max_tokens, len(data))
    total_bytes = 0
    total_tokens = 0
    for i in range(0, n, chunk_tokens):
        j = min(i + chunk_tokens, n)
        chunk = data[i:j]
        if mask is not None:
            chunk = chunk[mask[i:j].astype(bool)]
        ids = chunk.tolist()
        if not ids:
            continue
        # count UTF-8 bytes of the decoded text; a byte-level BPE decode of the
        # id stream reconstructs the original corpus text (bar chunk-boundary
        # merges, negligible at 1M-token chunks).
        total_bytes += len(tokenizer.decode(ids).encode("utf-8"))
        total_tokens += len(ids)
    return {
        "bytes_per_token": total_bytes / total_tokens,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
    }


# -- cache --------------------------------------------------------------------


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    # sorted keys + trailing newline -> stable diffs in git.
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def bytes_per_token_cached(
    tokenizer_id: str = "LiquidAI/LFM2.5-230M",
    mix_name: str = "mix_1B",
    *,
    data_dir: str = "/mnt/ai/data",
    sft: bool = False,
    split: str = "val",
    cache_path: Path = CACHE_PATH,
    force: bool = False,
    max_tokens: Optional[int] = None,
) -> float:
    """Bytes-per-token for ``(tokenizer_id, mix_name)``, measured once and cached.

    The cache is a git-tracked JSON nested ``{tokenizer_id: {mix_name: stats}}``
    so it holds multiple tokenizers (and multiple mixes per tokenizer). On a miss
    (or ``force``) it loads the tokenizer + packed ``<split>.bin`` and measures.
    """
    cache = _load_cache(cache_path)
    # SFT mixes live in a separate namespace and measure bytes-per-SUPERVISED-token,
    # so key them apart from a pretrain mix that might share a name.
    cache_key = f"sft:{mix_name}" if sft else mix_name
    entry = cache.get(tokenizer_id, {}).get(cache_key)
    if entry is not None and not force:
        return entry["bytes_per_token"]

    # lazy imports: keep pure conversions usable without torch/tokenizers/data.
    from chimera.tokenizers import BPETokenizer

    sub = "mix_sft" if sft else "mix"
    mix_dir = Path(data_dir) / "llm-mix" / sub / mix_name
    bin_path = mix_dir / f"{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"packed stream not found: {bin_path}")
    # SFT: normalize by supervised tokens only (masked loss), if a mask exists.
    mask_path = mix_dir / f"{split}_mask.bin"
    mask_path = mask_path if (sft and mask_path.exists()) else None

    tokenizer = BPETokenizer.from_pretrained(tokenizer_id)
    stats = measure_bytes_per_token(
        tokenizer, bin_path, max_tokens=max_tokens, mask_path=mask_path
    )
    stats["split"] = split
    stats["supervised_only"] = mask_path is not None
    cache.setdefault(tokenizer_id, {})[cache_key] = stats
    _save_cache(cache_path, cache)
    return stats["bytes_per_token"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", default="LiquidAI/LFM2.5-230M")
    p.add_argument("--mix", default="mix_1B")
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--split", default="val")
    p.add_argument("--sft", action="store_true")
    p.add_argument("--force", action="store_true", help="re-measure, ignore cache")
    p.add_argument("--max-tokens", type=int, default=None)
    # a measured/observed val/loss (nats) to convert to BPB; repeatable.
    p.add_argument("--loss", type=float, default=None, help="val/loss in nats -> BPB")
    # BPB targets to invert back into val/loss (nats) targets.
    p.add_argument("--bpb-targets", default="0.9,1.0,1.1")
    args = p.parse_args()

    bpt = bytes_per_token_cached(
        args.tokenizer,
        args.mix,
        data_dir=args.data_dir,
        sft=args.sft,
        split=args.split,
        force=args.force,
        max_tokens=args.max_tokens,
    )
    entry = _load_cache(CACHE_PATH)[args.tokenizer][args.mix]
    print(f"tokenizer={args.tokenizer}  mix={args.mix}  split={entry.get('split')}")
    print(
        f"total_tokens={entry['total_tokens']:,}  total_bytes={entry['total_bytes']:,}"
    )
    print(f"bytes/token = {bpt:.4f}")
    print(f"(cached in {CACHE_PATH})")

    if args.loss is not None:
        b = nats_to_bpb(args.loss, bpt)
        print(
            f"\nval/loss {args.loss} nats = {args.loss / LN2:.3f} bits/token"
            f"  ->  BPB = {b:.4f}"
        )

    targets = [float(x) for x in args.bpb_targets.split(",") if x.strip()]
    if targets:
        print("\nBPB target -> val/loss (nats) to aim for:")
        for t in targets:
            print(
                f"  BPB {t:>4}  ->  {bpb_to_nats(t, bpt):.3f} nats"
                f"  ({bpb_to_nats(t, bpt) / LN2:.3f} bits/token)"
            )


if __name__ == "__main__":
    main()

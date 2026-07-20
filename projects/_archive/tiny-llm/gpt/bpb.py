"""Per-source bits-per-byte normalizers for the tiny-llm mix.

    bpb = loss_nats / (ln2 * bytes_per_token)

``bytes_per_token`` is the aggregate (total UTF-8 bytes) / (total tokens) over a
corpus, and it differs per source (a story tokenizes differently than a textbook)
and per tokenizer (4k vs 16k vocab). So a correct ``val/<src>/bpb`` needs each
source's OWN bytes/token — obtained by decoding that source's val window back to
UTF-8. We also return the mix-wide aggregate (for ``val/bpb``).

Measured once and cached to a git-tracked ``bpb_cache.json`` keyed
``{tokenizer_id: {mix_name: {aggregate, per_source}}}``. Reads the per-source val
windows straight from the packed ``val.bin`` + ``manifest.json`` (val.bin is
written in manifest source order — see build_mixture.py).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

CACHE_PATH = Path(__file__).with_name("bpb_cache.json")
LN2 = math.log(2.0)


def nats_to_bpb(loss_nats: float, bytes_per_token: float) -> float:
    return loss_nats / (LN2 * bytes_per_token)


def _decode_bytes(tokenizer, ids_array, chunk: int = 1_000_000):
    """Decode a uint16 id array to text in chunks; return (bytes, tokens)."""
    total_bytes = total_tokens = 0
    n = len(ids_array)
    for i in range(0, n, chunk):
        ids = ids_array[i : i + chunk].tolist()
        if not ids:
            continue
        total_bytes += len(tokenizer.decode(ids).encode("utf-8"))
        total_tokens += len(ids)
    return total_bytes, total_tokens


def measure(
    tokenizer_id: str,
    mix_name: str,
    *,
    data_dir: str = "/mnt/ai/data",
    root_subdir: str = "tiny-llm",
    split: str = "val",
    force: bool = False,
    cache_path: Path = CACHE_PATH,
):
    """Return ``(aggregate_bpt, {src: bpt})`` for ``(tokenizer_id, mix_name)``.

    Cached; on a miss (or ``force``) loads the tokenizer + packed ``<split>.bin``
    and decodes each per-source val window.
    """
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    entry = cache.get(tokenizer_id, {}).get(mix_name)
    if entry is not None and not force:
        return entry["aggregate"], entry["per_source"]

    from chimera.tokenizers import BPETokenizer

    mix_dir = Path(data_dir) / root_subdir / "mix" / mix_name
    bin_path = mix_dir / f"{split}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"packed stream not found: {bin_path} — build the mix first"
        )
    manifest = json.loads((mix_dir / "manifest.json").read_text())
    tok = BPETokenizer.from_pretrained(tokenizer_id)
    data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")

    per_source, off, agg_b, agg_t = {}, 0, 0, 0
    for r in manifest.get("sources", []):
        n = int(r.get("val_tokens", 0))
        if n > 0:
            b, t = _decode_bytes(tok, data[off : off + n])
            per_source[r["key"]] = b / t
            agg_b, agg_t = agg_b + b, agg_t + t
        off += n
    if agg_t == 0:
        raise ValueError(f"no val tokens found in manifest for {mix_name!r}")
    aggregate = agg_b / agg_t

    cache.setdefault(tokenizer_id, {})[mix_name] = {
        "aggregate": aggregate,
        "per_source": per_source,
        "split": split,
    }
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    return aggregate, per_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", default="/mnt/ai/data/tiny-llm/tokenizer/8k")
    p.add_argument("--mix", default="tiny_2B_8k")
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    agg, per = measure(
        args.tokenizer, args.mix, data_dir=args.data_dir, force=args.force
    )
    print(f"tokenizer={args.tokenizer}  mix={args.mix}")
    print(f"aggregate bytes/token = {agg:.4f}")
    for k, v in per.items():
        print(f"  {k:<24} {v:.4f}")


if __name__ == "__main__":
    main()

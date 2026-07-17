"""Pack a token mixture from the tiny-llm per-source caches into train/val bins.

Reads the ``tok/<tok_tag>/<key>/ids.bin`` caches (tokenize_source.py output),
gives each source its target share of the budget (``weight`` from sources.py),
block-shuffles the pieces so no source sits in one contiguous run, and writes::

    /mnt/ai/data/tiny-llm/mix/<name>/{train.bin, val.bin, manifest.json}

Unlike the llm project's build_mixture (which *caps* a source at its available
tokens), this **oversamples**: when a source's target exceeds its unique supply
(TinyStories at 50% of 2B vs ~0.5B unique), its train region is tiled to hit the
target so the intended ratio is honored (epochs = target / unique). The val split
is always a UNIQUE tail slice (never repeated), written in manifest source order
so MixtureDataModule can window it for ``val/<src>/bpb``.

Usage:
    uv run python build_mixture.py --tok-tag 8k              # -> mix/tiny_2B_8k
    uv run python build_mixture.py --tok-tag 16k --name tiny_2B_16k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sources as S

DTYPE = np.uint16
BLOCK = 1_000_000  # cross-source shuffle granularity
TOK_ROOT = Path("/mnt/ai/data/tiny-llm/tok")
MIX_ROOT = Path("/mnt/ai/data/tiny-llm/mix")


def _tiled_blocks(key: str, n_unique: int, want: int) -> list[tuple[str, int, int]]:
    """Contiguous (key, start, length) slices over [0, n_unique) that tile to
    ``want`` tokens (wraps around when want > n_unique -> oversampling)."""
    blocks, cursor = [], 0
    while cursor < want:
        start = cursor % n_unique
        length = min(BLOCK, n_unique - start, want - cursor)
        blocks.append((key, start, length))
        cursor += length
    return blocks


def build(name: str, tok_tag: str, total: int, val_frac: float, seed: int):
    tokdir = TOK_ROOT / tok_tag
    srcs = [s for s in S.SOURCES if s.weight > 0]
    wsum = sum(s.weight for s in srcs)
    rng = np.random.default_rng(seed)

    # load availability
    avail = {}
    for s in srcs:
        meta_p = tokdir / s.key / "meta.json"
        if not meta_p.exists():
            raise SystemExit(f"missing cache for {s.key!r} at {meta_p} — "
                             f"run tokenize_source.py --tokenizer .../{tok_tag} first")
        avail[s.key] = json.loads(meta_p.read_text())["n_tokens"]

    out_dir = MIX_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_f = open(out_dir / "train.bin", "wb")
    val_f = open(out_dir / "val.bin", "wb")
    caches = {s.key: np.memmap(tokdir / s.key / "ids.bin", dtype=DTYPE, mode="r")
              for s in srcs}

    plan_rows, train_blocks = [], []
    for s in srcs:
        navail = avail[s.key]
        n_val = int(navail * val_frac)
        n_train_unique = navail - n_val
        want_train = int(round(s.weight / wsum * total * (1 - val_frac)))
        # val = unique tail slice, written in source order (for per-source windows)
        np.asarray(caches[s.key][n_train_unique:navail]).tofile(val_f)
        # train = tiled/subsampled from the unique head to hit the target
        train_blocks += _tiled_blocks(s.key, n_train_unique, want_train)
        plan_rows.append({
            "key": s.key, "weight": s.weight,
            "renorm_weight": round(s.weight / wsum, 4),
            "train_tokens": want_train, "val_tokens": n_val,
            "avail_tokens": navail,
            "repeat": round(want_train / max(n_train_unique, 1), 2),
            "oversampled": want_train > n_train_unique,
        })

    # shuffle blocks across sources, then stream out
    order = rng.permutation(len(train_blocks))
    n_train = 0
    for i in order:
        key, start, length = train_blocks[i]
        np.asarray(caches[key][start:start + length]).tofile(train_f)
        n_train += length
    train_f.close()
    val_f.close()

    manifest = {
        "name": name, "tok_tag": tok_tag, "total_requested": total,
        "train_tokens": n_train, "val_frac": val_frac, "seed": seed,
        "block": BLOCK, "sources": plan_rows,
        "note": "oversampled: sources tiled to target; val is unique tail",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{name}] train={n_train:,} tok from {len(srcs)} sources (tag {tok_tag})")
    for r in plan_rows:
        flag = "  (OVERSAMPLED)" if r["oversampled"] else ""
        print(f"   {r['key']:<24} w={r['renorm_weight']:.3f} -> "
              f"{r['train_tokens']/1e6:6.1f}M train  {r['val_tokens']/1e6:4.1f}M val  "
              f"({r['repeat']}x){flag}")
    print(f"   manifest: {out_dir / 'manifest.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-tag", default="8k", help="tokenizer cache tag to pack from")
    ap.add_argument("--name", default=None, help="mix name (default tiny_<budget>_<tag>)")
    ap.add_argument("--total", type=int, default=S.TARGET_TOKENS)
    ap.add_argument("--val-frac", type=float, default=S.VAL_FRAC)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default=None,
                    help="override source weights as 'key:w,key:w,...' (any source "
                         "not listed is set to 0). Lets you A/B a mixture without "
                         "editing sources.py; renormalized internally.")
    args = ap.parse_args()
    if args.weights:
        ov = {p.split(":")[0].strip(): float(p.split(":")[1]) for p in args.weights.split(",")}
        for s in S.SOURCES:
            s.weight = ov.get(s.key, 0.0)
        print("weight override ->", {s.key: s.weight for s in S.SOURCES if s.weight > 0})
    name = args.name or f"tiny_{args.total // 10**9}B_{args.tok_tag}"
    build(name, args.tok_tag, args.total, args.val_frac, args.seed)


if __name__ == "__main__":
    main()

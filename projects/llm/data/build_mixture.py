"""Pack a token mixture from per-source caches into train.bin / val.bin.

Reads the ``tok/<key>/ids.bin`` caches produced by tokenize_source.py, takes
each source's share of a total token budget (weights from the registry,
renormalized over whichever sources have a *complete* cache), block-shuffles the
pieces so no single source sits in one contiguous run, and writes:

    /mnt/ai/data/llm-mix/mix/<name>/{train.bin, val.bin, manifest.json}

Because it renormalizes over available caches, dropping the (slow) Stack v2 code
slice for now "just works": build from web+math+tools today, then re-run once the
code caches finish to fold code back in at its 40% weight.

    uv run python projects/llm/data/build_mixture.py --name mix_1B  --total 1e9
    uv run python projects/llm/data/build_mixture.py --name mix_10B --total 10e9
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

import sources as S  # noqa: E402

DTYPE = np.uint16
BLOCK = 1_000_000  # token granularity for cross-source shuffling


def _tok_dir(sft: bool) -> Path:
    return Path("/mnt/ai/data/llm-mix") / ("tok_sft" if sft else "tok")


def _mix_dir(sft: bool) -> Path:
    return Path("/mnt/ai/data/llm-mix") / ("mix_sft" if sft else "mix")


def available_sources(sft: bool) -> list[tuple[str, float, int]]:
    """Return (key, weight, n_tokens) for every source with a complete cache."""
    out = []
    tok = _tok_dir(sft)
    for src in S.SOURCES:
        meta_p = tok / src.key / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("n_tokens", 0) > 0:
            out.append((src.key, src.weight, meta["n_tokens"]))
    return out


def build(name: str, total: int, val_frac: float, seed: int, sft: bool = False):
    TOK = _tok_dir(sft)
    MIX = _mix_dir(sft)
    avail = available_sources(sft)
    if not avail:
        raise SystemExit(
            "no tokenized sources found under "
            f"{TOK} — run tokenize_source.py first"
        )
    wsum = sum(w for _, w, _ in avail)
    rng = np.random.default_rng(seed)

    out_dir = MIX / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_f = open(out_dir / "train.bin", "wb")
    val_f = open(out_dir / "val.bin", "wb")
    train_mf = open(out_dir / "train_mask.bin", "wb") if sft else None
    val_mf = open(out_dir / "val_mask.bin", "wb") if sft else None

    def _mask(key):
        return np.memmap(TOK / key / "mask.bin", dtype=np.uint8, mode="r")

    plan_rows = []
    train_blocks = []  # (key, start, length) collected then shuffled
    for key, w, navail in avail:
        want = int(round(w / wsum * total))
        take = min(want, navail)
        n_val = int(take * val_frac)
        n_train = take - n_val
        data = np.memmap(TOK / key / "ids.bin", dtype=DTYPE, mode="r")
        # val = a contiguous tail slice (held out); train = the rest, blocked
        np.asarray(data[n_train : n_train + n_val]).tofile(val_f)
        if sft:
            np.asarray(_mask(key)[n_train : n_train + n_val]).tofile(val_mf)
        for start in range(0, n_train, BLOCK):
            length = min(BLOCK, n_train - start)
            train_blocks.append((key, start, length))
        plan_rows.append(
            {
                "key": key,
                "weight": w,
                "renorm_weight": round(w / wsum, 4),
                "want_tokens": want,
                "taken_tokens": take,
                "train_tokens": n_train,
                "val_tokens": n_val,
                "avail_tokens": navail,
                "repeat": round(want / navail, 2) if navail else None,
                "capped": want > navail,
            }
        )

    # shuffle blocks across sources, then stream them out in that order
    order = rng.permutation(len(train_blocks))
    caches = {key: np.memmap(TOK / key / "ids.bin", dtype=DTYPE, mode="r")
              for key, _, _ in avail}
    mcaches = {key: _mask(key) for key, _, _ in avail} if sft else {}
    n_train_tokens = 0
    for i in order:
        key, start, length = train_blocks[i]
        np.asarray(caches[key][start : start + length]).tofile(train_f)
        if sft:
            np.asarray(mcaches[key][start : start + length]).tofile(train_mf)
        n_train_tokens += length
    train_f.close()
    val_f.close()
    if sft:
        train_mf.close()
        val_mf.close()

    manifest = {
        "name": name,
        "sft": sft,
        "total_requested": total,
        "train_tokens": n_train_tokens,
        "val_frac": val_frac,
        "seed": seed,
        "block": BLOCK,
        "sources": plan_rows,
        "note": "renormalized over sources with a complete cache",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{name}] train={n_train_tokens:,} tokens from {len(avail)} sources")
    for r in plan_rows:
        flag = "  (CAPPED)" if r["capped"] else ""
        print(f"   {r['key']:<16} {r['renorm_weight']:.3f} -> "
              f"{r['train_tokens'] / 1e6:.1f}M train  ({r['repeat']}x){flag}")
    print(f"   manifest: {out_dir / 'manifest.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="mix name, e.g. mix_1B")
    p.add_argument("--total", type=float, required=True, help="total tokens, e.g. 1e9")
    p.add_argument("--val-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft", action="store_true", help="pack SFT caches (ids + supervise mask)")
    args = p.parse_args()
    build(args.name, int(args.total), args.val_frac, args.seed, sft=args.sft)


if __name__ == "__main__":
    main()

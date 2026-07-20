"""Search the packed training mixture for substrings — see what the model was
actually trained on.

The mix (build_mixture.py) is a uint16 token memmap at
``/mnt/ai/data/tiny-llm/mix/<mix>/{train,val}.bin``. Searching it directly means
DECODING back to text (BPE is context-dependent, so a token-subsequence search
misses most real occurrences). Decoding the whole 2B-token mix per query is
slow (~5 min, tokenizer-decode-bound), so this tool decodes ONCE into a text
cache — one document per line — and then searches it with ripgrep, which is
sub-second.

    <mix>/<split>.decoded.txt        # one document (eos-delimited) per line

The first search builds the cache (a few minutes, parallel decode); later
searches just grep it. Pass --rebuild to force a fresh decode.

"What the model trained on": a 1B-token run consumes ~1.0B of the 1.99B-token
train.bin (shuffle over 512-windows ≈ half an epoch). train.bin is block-
shuffled across sources, so any prefix is representative. --max-tokens caps the
search to the first N tokens' worth of text (default: whole split).

Usage:
    uv run python search_data.py "once upon a time"          # snippets
    uv run python search_data.py "photosynthesis" -i --context 120
    uv run python search_data.py "Elara" --count-only         # frequency
    uv run python search_data.py "\\bAPI\\b" --regex
    uv run python search_data.py --rebuild "x"                # force re-decode
    uv run python search_data.py --docs "Elara" -m 5          # whole documents
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from chimera.tokenizers.bpe import BPETokenizer

DTYPE = np.uint16
SPECIALS = ("<|endoftext|>", "<|startoftext|>")  # decoded doc delimiters -> newline
SUB_CHUNK = 4_000_000  # tokens decoded per call inside a worker

_TOK = None  # per-worker tokenizer (set in _worker_init)


def _mix_dir(data_dir: str, mix: str) -> Path:
    return Path(data_dir) / "tiny-llm" / "mix" / mix


def _resolve_tokenizer(data_dir: str, mix: str, override: str | None) -> str:
    if override:
        return override
    manifest = _mix_dir(data_dir, mix) / "manifest.json"
    tag = json.loads(manifest.read_text())["tok_tag"] if manifest.exists() else "8k"
    return str(Path(data_dir) / "tiny-llm" / "tokenizer" / tag)


def _worker_init(tok_path):
    global _TOK
    _TOK = BPETokenizer.from_pretrained(tok_path)


def _decode_range(job):
    """Decode tokens [start, end) of split_path to a text shard (one doc/line)."""
    split_path, start, end, shard_path = job
    arr = np.memmap(split_path, dtype=DTYPE, mode="r")
    with open(shard_path, "w", encoding="utf-8") as f:
        for s in range(start, end, SUB_CHUNK):
            ids = np.asarray(arr[s : min(s + SUB_CHUNK, end)]).tolist()
            text = _TOK._tok.decode(ids, skip_special_tokens=False)
            for sp in SPECIALS:
                text = text.replace(sp, "\n")
            f.write(text)
    return shard_path


def build_cache(split_path: Path, cache_path: Path, tok_path: str, workers: int):
    total = len(np.memmap(split_path, dtype=DTYPE, mode="r"))
    print(
        f"building text cache: decoding {total:,} tokens with {workers} workers "
        f"-> {cache_path}  (one-time)",
        file=sys.stderr,
    )
    per = -(-total // workers)  # ceil
    bounds = [(i * per, min((i + 1) * per, total)) for i in range(workers)]
    tmp = Path(tempfile.mkdtemp(prefix="decode_", dir=cache_path.parent))
    jobs = [
        (str(split_path), a, b, str(tmp / f"shard_{i:03d}.txt"))
        for i, (a, b) in enumerate(bounds)
        if a < b
    ]
    t0 = time.time()
    with Pool(workers, initializer=_worker_init, initargs=(tok_path,)) as pool:
        shards = pool.map(_decode_range, jobs)
    with open(cache_path, "wb") as out:
        for sh in shards:
            with open(sh, "rb") as f:
                shutil.copyfileobj(f, out, length=16 << 20)
    shutil.rmtree(tmp, ignore_errors=True)
    nbytes = cache_path.stat().st_size
    # sidecar: token/byte totals for the --max-tokens -> byte-prefix mapping
    cache_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "tokens": total,
                "bytes": nbytes,
                "split_mtime": split_path.stat().st_mtime,
            }
        )
    )
    print(
        f"cache built: {nbytes / 1e9:.2f} GB in {time.time() - t0:.0f}s "
        f"({total / 1e6 / (time.time() - t0):.0f}M tok/s aggregate)",
        file=sys.stderr,
    )


def ensure_cache(
    split_path: Path, cache_path: Path, tok_path: str, workers: int, rebuild: bool
):
    meta_p = cache_path.with_suffix(".meta.json")
    fresh = (
        cache_path.exists()
        and meta_p.exists()
        and json.loads(meta_p.read_text())["split_mtime"] == split_path.stat().st_mtime
    )
    if rebuild or not fresh:
        build_cache(split_path, cache_path, tok_path, workers)
    return json.loads(meta_p.read_text())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("query", nargs="+", help="substring(s) to search for")
    ap.add_argument("--data-dir", default="/mnt/ai/data")
    ap.add_argument("--mix", default="tiny_2B_8k", help="mixture under tiny-llm/mix/")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer dir (default: mix manifest)"
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        help="cap search to the first N tokens' worth of text (-1 = whole split)",
    )
    ap.add_argument(
        "-m",
        "--max-matches",
        type=int,
        default=20,
        help="max snippets per query (-1 = all); ignored with --count-only",
    )
    ap.add_argument(
        "--context", type=int, default=90, help="chars of context each side"
    )
    ap.add_argument("-i", "--ignore-case", action="store_true")
    ap.add_argument("--regex", action="store_true", help="treat queries as regex")
    ap.add_argument(
        "--count-only", action="store_true", help="report match counts only"
    )
    ap.add_argument(
        "--docs",
        action="store_true",
        help="print the whole matching document line, not a windowed snippet",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() // 2),
        help="parallel decode workers for the one-time cache build",
    )
    ap.add_argument(
        "--rebuild", action="store_true", help="force rebuild the text cache"
    )
    args = ap.parse_args()

    split_path = _mix_dir(args.data_dir, args.mix) / f"{args.split}.bin"
    if not split_path.exists():
        sys.exit(f"no such split: {split_path}")
    tok_path = _resolve_tokenizer(args.data_dir, args.mix, args.tokenizer)
    cache_path = _mix_dir(args.data_dir, args.mix) / f"{args.split}.decoded.txt"
    meta = ensure_cache(split_path, cache_path, tok_path, args.workers, args.rebuild)

    # --max-tokens -> byte prefix (proportional; the mix is block-shuffled)
    hi = meta["bytes"]
    if 0 <= args.max_tokens < meta["tokens"]:
        hi = int(meta["bytes"] * args.max_tokens / meta["tokens"])
    scanned = (
        f"{meta['tokens']:,}" if hi == meta["bytes"] else f"~first {args.max_tokens:,}"
    )
    denom_tok = meta["tokens"] if hi == meta["bytes"] else max(args.max_tokens, 1)
    print(
        f"mix={args.mix} split={args.split} cache={cache_path.name} "
        f"({meta['bytes'] / 1e9:.2f}GB, {meta['tokens']:,} tok)  scanning {scanned} tokens"
    )
    print(f"queries: {args.query}\n" + "-" * 90)

    fd = os.open(cache_path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)

    def matches(q):
        """Yield match-start byte offsets for query q within [0, hi)."""
        if args.regex or args.ignore_case:
            pat = re.compile(
                q.encode() if args.regex else re.escape(q.encode()),
                re.IGNORECASE if args.ignore_case else 0,
            )
            for m in pat.finditer(mm):
                if m.start() >= hi:
                    return
                yield m.start(), m.end() - m.start()
        else:
            needle = q.encode()
            pos = 0
            while (i := mm.find(needle, pos, hi)) != -1:
                yield i, len(needle)
                pos = i + 1

    def snippet(i, qlen):
        if args.docs:  # widen to the surrounding document line (bounded)
            lo = mm.rfind(b"\n", max(0, i - 4000), i)
            hi_ = mm.find(b"\n", i, i + 4000)
            s = lo + 1 if lo != -1 else max(0, i - 4000)
            e = hi_ if hi_ != -1 else min(len(mm), i + 4000)
        else:
            s, e = max(0, i - args.context), min(len(mm), i + qlen + args.context)
        raw = mm[s:e].decode("utf-8", "replace").replace("\n", " ⏎ ")
        return ("…" if s > 0 else "") + raw + ("…" if e < len(mm) else "")

    t0 = time.time()
    for q in args.query:
        count = shown = 0
        for i, qlen in matches(q):
            count += 1
            if not args.count_only and (
                args.max_matches < 0 or shown < args.max_matches
            ):
                print(f"[{q!r} #{shown + 1}] {snippet(i, qlen)}")
                shown += 1
            if (
                not args.count_only
                and args.max_matches >= 0
                and shown >= args.max_matches
            ):
                break  # snippet mode: stop this query once we have enough
        if args.count_only:
            per_m = count / denom_tok * 1e6
            print(f"{q!r}: {count:,} occurrence(s)  ≈ {per_m:.2f} per 1M tokens")
        print("-" * 90)
    mm.close()
    os.close(fd)
    print(f"(searched in {time.time() - t0:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Train byte-level BPE tokenizers on the tiny-LM mixture (4k / 8k / 16k vocab).

Samples a *weighted* character corpus from the staged raw parquet (per the
pretrain mixture weights in ``sources.py``), caches it once as JSONL, then trains
a suite of vocab sizes off that identical corpus so vocab size is the only
variable. Config choices (locked with the user):

    * byte-level BPE, no UNK (chimera.tokenizers.BPETokenizer, hf backend)
    * split_digits=True     -> runs of digits split to single chars (arithmetic)
    * minimal chat specials  -> EOS, BOS, PAD, <|im_start|>, <|im_end|>
      (chat SFT reuses this tokenizer; think/tool markers omitted — no tools here)
    * weighted by the PRETRAIN mix only (chat source excluded from the corpus;
      its ChatML markers are still reserved as specials, and chat English ⊂ the
      general English already sampled)

Writes per vocab:
    /mnt/ai/data/tiny-llm/tokenizer/<tag>/tokenizer.json   # BPETokenizer.from_pretrained
    /mnt/ai/data/tiny-llm/tokenizer/<tag>/meta.json
and a top-level summary.json with chars/token per source per vocab (held-out).

Usage:
    uv run python train_tokenizer.py                 # 1GB sample, vocabs 4k/8k/16k
    uv run python train_tokenizer.py --chars 500_000_000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

import pyarrow.parquet as pq  # noqa: E402

import sources as S  # noqa: E402
from chimera.data.chat_template import BOS, EOS, IM_END, IM_START, PAD  # noqa: E402
from chimera.tokenizers import BPETokenizer  # noqa: E402

OUT_ROOT = Path("/mnt/ai/data/tiny-llm/tokenizer")
SPECIALS = [EOS, BOS, PAD, IM_START, IM_END]  # minimal chat set (ids 0..4)
VOCAB_SIZES = [4096, 8192, 16384]
DEFAULT_CHARS = (
    1_000_000_000  # ~1GB weighted sample — BPE merges saturate well before this
)


def _pretrain_sources():
    return [s for s in S.SOURCES if s.weight > 0]


def _shards(src: S.Source) -> list[Path]:
    root = Path(S.RAW_ROOT) / src.key
    return sorted(root.rglob("*.parquet"))


def _iter_text(shards: list[Path], text_col: str):
    """Stream text values across a source's shards (train reads from the front)."""
    for sh in shards:
        pf = pq.ParquetFile(sh)
        if text_col not in pf.schema_arrow.names:
            continue
        for batch in pf.iter_batches(batch_size=1000, columns=[text_col]):
            for v in batch.column(0).to_pylist():
                if v:
                    yield v


def _vocab_tag(v: int) -> str:
    return f"{v // 1024}k" if v % 1024 == 0 else str(v)


# --------------------------------------------------------------------------- #
# Sample the weighted corpus once, cache as JSONL (one doc per line).
# --------------------------------------------------------------------------- #
def sample_corpus(total_chars: int, path: Path) -> dict:
    srcs = _pretrain_sources()
    wsum = sum(s.weight for s in srcs)
    realized, n_docs = {}, 0
    t0 = time.time()
    with open(path, "w") as f:
        for src in srcs:
            cap = int(total_chars * src.weight / wsum)
            got = 0
            for text in _iter_text(_shards(src), src.text_column):
                f.write(json.dumps(text) + "\n")
                got += len(text)
                n_docs += 1
                if got >= cap:
                    break
            realized[src.key] = got
            print(
                f"[{src.key}] {got / 1e6:.0f}M chars "
                f"(target {cap / 1e6:.0f}M, weight {src.weight / wsum:.3f})  "
                f"{(time.time() - t0) / 60:.1f}min",
                flush=True,
            )
    meta = {
        "total_chars": sum(realized.values()),
        "n_docs": n_docs,
        "realized_chars": realized,
    }
    print(f"corpus: {meta['total_chars'] / 1e6:.0f}M chars / {n_docs:,} docs -> {path}")
    return meta


def _read_corpus(path: Path):
    with open(path) as f:
        for line in f:
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Held-out compression eval: chars/token per source per vocab (tail slice).
# --------------------------------------------------------------------------- #
def _heldout_text(src: S.Source, n_chars: int = 2_000_000) -> str:
    """Read ~n_chars from the LAST shard (training reads from the front)."""
    shards = _shards(src)
    if not shards:
        return ""
    buf, got = [], 0
    for text in _iter_text(shards[-1:], src.text_column):
        buf.append(text)
        got += len(text)
        if got >= n_chars:
            break
    return "".join(buf)[:n_chars]


def eval_compression(tok_paths: dict[int, Path]) -> dict:
    toks = {v: BPETokenizer.from_pretrained(str(p)) for v, p in tok_paths.items()}
    heldout = {s.key: _heldout_text(s) for s in _pretrain_sources()}
    rows = {}
    print("\ncompression — chars/token (higher = better):")
    hdr = "  ".join(f"{_vocab_tag(v):>7}" for v in VOCAB_SIZES)
    print(f"  {'source':<24} {hdr}")
    for key, text in heldout.items():
        if not text:
            continue
        cr = {}
        for v in VOCAB_SIZES:
            n = len(toks[v]._tok.encode(text, add_special_tokens=False).ids)
            cr[v] = round(len(text) / max(n, 1), 3)
        rows[key] = cr
        print(f"  {key:<24} " + "  ".join(f"{cr[v]:>7.3f}" for v in VOCAB_SIZES))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=DEFAULT_CHARS)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_ROOT / "corpus.jsonl"
    corpus_meta = sample_corpus(args.chars, corpus_path)

    tok_paths = {}
    for v in VOCAB_SIZES:
        out_dir = OUT_ROOT / _vocab_tag(v)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        tok = BPETokenizer(backend="hf")
        tok.train(
            _read_corpus(corpus_path),
            vocab_size=v,
            special_tokens=SPECIALS,
            min_frequency=args.min_frequency,
            split_digits=True,
        )
        tok_path = out_dir / "tokenizer.json"
        tok.save(tok_path)
        secs = round(time.time() - t0, 1)
        meta = {
            "vocab_size": tok.vocab_size,
            "requested_vocab_size": v,
            "special_tokens": SPECIALS,
            "special_ids": {t: tok._tok.token_to_id(t) for t in SPECIALS},
            "split_digits": True,
            "min_frequency": args.min_frequency,
            "seconds": secs,
            "corpus": corpus_meta,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[{_vocab_tag(v)}] vocab={tok.vocab_size} in {secs}s -> {tok_path}")
        tok_paths[v] = tok_path

    compression = eval_compression(tok_paths)
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(
            {
                "vocab_sizes": VOCAB_SIZES,
                "specials": SPECIALS,
                "split_digits": True,
                "corpus": corpus_meta,
                "compression_chars_per_token": compression,
            },
            indent=2,
        )
    )
    print(f"\nsummary -> {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()

"""Tokenize the staged tiny-llm sources into uint16 caches (project tokenizer).

Reads the LOCAL staged parquet (download.py output) under
``/mnt/ai/data/tiny-llm/raw/<key>/``, tokenizes each row's text column with the
project's own BPE tokenizer (4k/8k/16k), appends ``<|endoftext|>`` between
documents, and writes a flat uint16 stream to::

    /mnt/ai/data/tiny-llm/tok/<tok_tag>/<key>/{ids.bin, meta.json}

Caches are namespaced by tokenizer tag (``8k`` etc.) so multiple vocab sizes
coexist. Each source is tokenized up to a cap sized to its share of the 2B budget
(with headroom for the val split); TinyStories has < its target unique tokens, so
its whole staged slice is tokenized and build_mixture.py oversamples it.

Pretrain only (unmasked next-token). Chat (smol-smoltalk) is skipped — SFT is a
later, separately-masked stage.

Usage:
    uv run python tokenize_source.py --tokenizer /mnt/ai/data/tiny-llm/tokenizer/8k
    uv run python tokenize_source.py --tokenizer .../8k --keys fineweb-edu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

import pyarrow.parquet as pq  # noqa: E402

import sources as S  # noqa: E402
from chimera.tokenizers import BPETokenizer  # noqa: E402

DTYPE = np.uint16
EOS_TOKEN = "<|endoftext|>"
TOK_ROOT = Path("/mnt/ai/data/tiny-llm/tok")


def _shards(key: str) -> list[Path]:
    return sorted((Path(S.RAW_ROOT) / key).rglob("*.parquet"))


class Writer:
    """Buffered uint16 writer that stops at a token cap."""

    def __init__(self, path: Path, cap: int):
        self.f = open(path, "wb")
        self.cap = cap
        self.n = 0
        self._buf: list[int] = []

    def add_ids(self, ids: list[int]):
        self._buf.extend(ids)
        if len(self._buf) >= 1_000_000:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        take = self._buf
        if self.n + len(take) > self.cap:
            take = take[: self.cap - self.n]
        np.asarray(take, dtype=DTYPE).tofile(self.f)
        self.n += len(take)
        self._buf = []

    def full(self) -> bool:
        return self.n + len(self._buf) >= self.cap

    def close(self):
        self.flush()
        self.f.close()


def tokenize_source(
    key: str, cap: int, tokenizer: str, tok_tag: str, batch: int = 1024
) -> dict:
    src = S.get(key)
    out_dir = TOK_ROOT / tok_tag / key
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path, meta_path = out_dir / "ids.bin", out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("complete") and meta.get("n_tokens", 0) >= cap:
            print(f"[{key}] cached ({meta['n_tokens']:,} tok >= cap {cap:,}) — skip")
            return meta

    tok = BPETokenizer.from_pretrained(tokenizer)
    eos = tok._tok.token_to_id(EOS_TOKEN)
    enc_batch = getattr(tok._tok, "encode_batch_fast", tok._tok.encode_batch)

    w = Writer(ids_path, cap)
    col = src.text_column
    texts: list[str] = []
    n_docs = 0
    t0 = time.time()

    def encode(texts):
        for e in enc_batch(texts, add_special_tokens=False):
            w.add_ids(e.ids)
            w.add_ids([eos])

    done = False
    for sh in _shards(key):
        pf = pq.ParquetFile(sh)
        if col not in pf.schema_arrow.names:
            print(f"[{key}] WARNING: text column {col!r} not in {sh.name}; skip")
            continue
        for rb in pf.iter_batches(batch_size=batch, columns=[col]):
            for v in rb.column(0).to_pylist():
                if v:
                    texts.append(v if isinstance(v, str) else str(v))
            if len(texts) >= batch:
                encode(texts)
                n_docs += len(texts)
                texts = []
                if w.full():
                    done = True
                    break
                if n_docs % (batch * 100) == 0:
                    _progress(key, w.n, cap, n_docs, t0)
        if done:
            break
    if texts and not w.full():
        encode(texts)
        n_docs += len(texts)
    w.close()

    meta = {
        "key": key,
        "n_tokens": w.n,
        "n_docs": n_docs,
        "dtype": "uint16",
        "tokenizer": tokenizer,
        "tok_tag": tok_tag,
        "cap": cap,
        "complete": True,
        "seconds": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[{key}] done: {w.n:,} tok, {n_docs:,} docs, {meta['seconds']}s "
        f"({w.n / max(meta['seconds'], 1e-9) / 1e6:.2f}M tok/s)"
    )
    return meta


def _progress(key, n, cap, n_docs, t0):
    dt = time.time() - t0
    rate = n / max(dt, 1e-9)
    eta = (cap - n) / max(rate, 1e-9)
    print(
        f"[{key}] {n / 1e6:.0f}M/{cap / 1e6:.0f}M tok  {n_docs:,} docs  "
        f"{rate / 1e6:.2f}M tok/s  eta {eta / 60:.1f}min",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/mnt/ai/data/tiny-llm/tokenizer/8k")
    ap.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="sources to tokenize (default: all pretrain sources)",
    )
    ap.add_argument(
        "--headroom",
        type=float,
        default=1.15,
        help="cap = weight*budget*headroom (over-tokenize a bit so the "
        "val split + block rounding never starve the target)",
    )
    args = ap.parse_args()

    tok_tag = Path(args.tokenizer).name
    srcs = (
        [S.get(k) for k in args.keys]
        if args.keys
        else [s for s in S.SOURCES if s.weight > 0]
    )
    print(
        f"tokenizer={args.tokenizer} (tag {tok_tag})  budget={S.TARGET_TOKENS / 1e9:.1f}B"
    )
    for src in srcs:
        cap = math.ceil(src.weight * S.TARGET_TOKENS * args.headroom)
        tokenize_source(src.key, cap, args.tokenizer, tok_tag)
    print(f"\ncaches -> {TOK_ROOT / tok_tag}")


if __name__ == "__main__":
    main()

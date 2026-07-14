"""Tokenize one mixture source into a uint16 memmap, capped at a token budget.

Renders each row to text per the source's ``kind`` (raw text / openmath / ChatML
chat / Stack-v2-from-S3), tokenizes with the LFM2.5 BPE, appends ``<|endoftext|>``
between documents, and writes a flat ``uint16`` token stream to
``/mnt/ai/data/llm-mix/tok/<key>/ids.bin`` (+ ``meta.json``). Everything is
tokenized *unmasked* (fully-supervised next-token) — masking is an SFT-stage
concern, out of scope for the pretraining mixture cache.

Per-source cache is sized to the *largest* mixture budget (default 10B), so both
the 1B and 10B packed mixes (see build_mixture.py) draw from the same caches.

    uv run python projects/llm/data/tokenize_source.py --source fineweb-edu
    uv run python projects/llm/data/tokenize_source.py --source stackv2-python
    uv run python projects/llm/data/tokenize_source.py --all --budget 10e9
    uv run python projects/llm/data/tokenize_source.py --source finemath --max-tokens 5_000_000  # smoke
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

import sources as S  # noqa: E402
from chimera.data import chat_template as ct  # noqa: E402
from chimera.tokenizers import BPETokenizer  # noqa: E402

CACHE = Path("/mnt/ai/data/llm-mix/tok")
CACHE_SFT = Path("/mnt/ai/data/llm-mix/tok_sft")
DTYPE = np.uint16
EOS_TOKEN = "<|endoftext|>"
DEFAULT_BUDGET = 10_000_000_000


# --------------------------------------------------------------------------- #
# Row -> text rendering per source kind
# --------------------------------------------------------------------------- #
def render(src, row) -> str:
    k = src.kind
    if k == "text":
        return row.get("text") or row.get("content") or ""
    if k == "openmath":
        prob = row.get("problem") or row.get("question") or ""
        sol = row.get("generated_solution") or row.get("solution") or ""
        return f"Problem:\n{prob}\n\nSolution:\n{sol}"
    if k == "chat":
        conv = S._coerce_conv(row.get("conversations") or row.get("messages"))
        return ct.render(conv, tools=row.get("tools"))
    raise ValueError(f"unexpected kind {k!r} for render()")


# --------------------------------------------------------------------------- #
# Shard listing / row iteration (download-as-needed, stop early at the cap)
# --------------------------------------------------------------------------- #
def _shard_paths(src) -> list[str]:
    from huggingface_hub import HfApi

    fp = src.file_prefix
    if fp.endswith((".parquet", ".json", ".jsonl")):
        return [fp]
    files = [
        f
        for f in HfApi().list_repo_files(src.hf_repo, repo_type="dataset")
        if f.startswith(fp) and f.endswith(".parquet")
    ]
    return sorted(files)


def _iter_rows(src, columns, start_shard=0):
    """Yield (shard_idx, row_dict) across shards, downloading each as reached."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    paths = _shard_paths(src)
    for si in range(start_shard, len(paths)):
        local = hf_hub_download(src.hf_repo, paths[si], repo_type="dataset")
        if local.endswith((".json", ".jsonl")):
            with open(local) as f:
                text = f.read()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = [json.loads(x) for x in text.splitlines() if x.strip()]
            for row in data:
                yield si, row
        else:
            pf = pq.ParquetFile(local)
            avail = set(pf.schema_arrow.names)
            cols = [c for c in columns if c in avail] if columns else None
            for batch in pf.iter_batches(batch_size=1024, columns=cols):
                for row in batch.to_pylist():
                    yield si, row


COLUMNS = {
    "text": ["text", "content"],
    "openmath": ["problem", "generated_solution"],
    "chat": ["conversations", "messages", "tools"],
    "stackv2": ["blob_id", "src_encoding"],
}


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
class Writer:
    def __init__(self, path: Path, cap: int):
        self.path = path
        self.cap = cap
        self.f = open(path, "wb")
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


# --------------------------------------------------------------------------- #
# Tokenize one source
# --------------------------------------------------------------------------- #
def tokenize_source(key: str, cap: int, s3_workers: int = 128, batch: int = 512,
                    tokenizer: str = "LiquidAI/LFM2.5-230M"):
    src = S.get(key)
    out_dir = CACHE / key
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / "ids.bin"
    meta_path = out_dir / "meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("complete") and meta.get("n_tokens", 0) >= cap:
            print(f"[{key}] already complete ({meta['n_tokens']:,} tokens) — skip")
            return meta

    tok = BPETokenizer.from_pretrained(tokenizer)
    eos = tok._tok.token_to_id(EOS_TOKEN)
    enc_batch = getattr(tok._tok, "encode_batch_fast", tok._tok.encode_batch)

    w = Writer(ids_path, cap)
    cols = COLUMNS[src.kind]
    t0 = time.time()
    n_docs = 0

    def encode_texts(texts):
        for e in enc_batch(texts, add_special_tokens=False):
            w.add_ids(e.ids)
            w.add_ids([eos])

    if src.kind == "stackv2":
        pool = ThreadPoolExecutor(max_workers=s3_workers)
        pending_rows = []

        def flush_rows():
            nonlocal n_docs
            blobs = [(r.get("blob_id"), r.get("src_encoding") or "utf-8") for r in pending_rows]
            texts = list(pool.map(lambda b: _safe_content(*b), blobs))
            texts = [t for t in texts if t]
            encode_texts(texts)
            n_docs += len(texts)
            pending_rows.clear()

        for _, row in _iter_rows(src, cols):
            pending_rows.append(row)
            if len(pending_rows) >= batch:
                flush_rows()
                if w.full():
                    break
                if n_docs % (batch * 20) == 0:
                    _progress(key, w.n, cap, n_docs, t0)
        if pending_rows and not w.full():
            flush_rows()
        pool.shutdown()
    else:
        texts = []
        for _, row in _iter_rows(src, cols):
            t = render(src, row)
            if t:
                texts.append(t)
            if len(texts) >= batch:
                encode_texts(texts)
                n_docs += len(texts)
                texts = []
                if w.full():
                    break
                if n_docs % (batch * 50) == 0:
                    _progress(key, w.n, cap, n_docs, t0)
        if texts and not w.full():
            encode_texts(texts)
            n_docs += len(texts)

    w.close()
    meta = {
        "key": key,
        "n_tokens": w.n,
        "n_docs": n_docs,
        "dtype": "uint16",
        "cap": cap,
        "complete": w.n >= cap or True,  # reaching end-of-data also counts as complete
        "seconds": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{key}] done: {w.n:,} tokens, {n_docs:,} docs, {meta['seconds']}s "
          f"({w.n / max(meta['seconds'], 1e-9) / 1e6:.2f}M tok/s)")
    return meta


# --------------------------------------------------------------------------- #
# SFT: masked ChatML tokenization (supervise assistant / tool-call turns only)
# --------------------------------------------------------------------------- #
def _render_row_masked(src, row, enc, eos_id):
    """Return (ids, mask) for one row via the canonical chat template; mask=1 on
    supervised (assistant) tokens. openmath is cast to a user/assistant turn so
    its chain-of-thought solution is the supervised target.

    Masking policy lives in chimera.data.chat_template: assistant content, its
    <think> block, its tool calls, and the closing <|im_end|> are supervised;
    headers, system/tools, user turns, and tool responses are masked.
    """
    if src.kind == "openmath":
        prob = row.get("problem") or row.get("question") or ""
        sol = row.get("generated_solution") or row.get("solution") or ""
        msgs = [{"role": "user", "content": prob},
                {"role": "assistant", "content": sol}]
        return ct.render_masked(msgs, encode=enc, eos_id=eos_id)
    conv = S._coerce_conv(row.get("conversations") or row.get("messages"))
    return ct.render_masked(conv, encode=enc, tools=row.get("tools"), eos_id=eos_id)


def tokenize_source_sft(key: str, cap: int, batch: int = 256,
                        tokenizer: str = "LiquidAI/LFM2.5-230M"):
    src = S.get(key)
    if src.kind not in ("chat", "openmath"):
        print(f"[{key}] kind={src.kind!r} is not an SFT source — skip")
        return None
    out_dir = CACHE_SFT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path, mask_path, meta_path = out_dir / "ids.bin", out_dir / "mask.bin", out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("complete") and meta.get("n_tokens", 0) >= cap:
            print(f"[{key}] already complete ({meta['n_tokens']:,} tokens) — skip")
            return meta

    tok = BPETokenizer.from_pretrained(tokenizer)
    eos = tok._tok.token_to_id(EOS_TOKEN)
    enc = lambda t: tok._tok.encode(t, add_special_tokens=False).ids
    fi, fm = open(ids_path, "wb"), open(mask_path, "wb")
    n = n_sup = n_docs = 0
    t0 = time.time()
    for _, row in _iter_rows(src, None):  # all columns (conversations/tools/...)
        rids, rmask = _render_row_masked(src, row, enc, eos)
        if not rids:
            continue
        if n + len(rids) > cap:
            rids, rmask = rids[: cap - n], rmask[: cap - n]
        np.asarray(rids, dtype=DTYPE).tofile(fi)
        np.asarray(rmask, dtype=np.uint8).tofile(fm)
        n += len(rids)
        n_sup += int(sum(rmask))
        n_docs += 1
        if n >= cap:
            break
        if n_docs % 5000 == 0:
            _progress(key, n, cap, n_docs, t0)
    fi.close()
    fm.close()
    meta = {
        "key": key, "n_tokens": n, "n_docs": n_docs, "dtype": "uint16",
        "has_mask": True, "supervised_frac": round(n_sup / max(n, 1), 4),
        "cap": cap, "complete": True, "seconds": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{key}] SFT done: {n:,} tokens ({meta['supervised_frac']:.1%} supervised), "
          f"{n_docs:,} docs, {meta['seconds']}s")
    return meta


def _safe_content(blob_id, enc):
    try:
        return S._swh_content(blob_id, enc)
    except Exception:
        return ""


def _progress(key, n, cap, n_docs, t0):
    dt = time.time() - t0
    rate = n / max(dt, 1e-9)
    eta = (cap - n) / max(rate, 1e-9)
    print(f"[{key}] {n / 1e6:.1f}M/{cap / 1e6:.0f}M tok  {n_docs:,} docs  "
          f"{rate / 1e6:.2f}M tok/s  eta {eta / 60:.1f}min", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="source key (see browse.py --list)")
    p.add_argument("--all", action="store_true", help="tokenize every non-deferred source")
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="total mixture budget (caps = weight*budget)")
    p.add_argument("--max-tokens", type=float, default=None, help="hard per-source cap override (smoke tests)")
    p.add_argument("--s3-workers", type=int, default=128, help="parallel SWH S3 fetchers (stackv2)")
    p.add_argument("--tokenizer", default="LiquidAI/LFM2.5-230M",
                   help="tokenizer: HF hub id or local path to a custom tokenizer.json/dir "
                        "(see train_tokenizer.py). Caches are tokenizer-specific.")
    p.add_argument("--sft", action="store_true",
                   help="masked ChatML tokenization for SFT (chat/openmath sources -> ids+mask)")
    args = p.parse_args()

    def cap_for(src):
        if args.max_tokens is not None:
            return int(args.max_tokens)
        return int(math.ceil(src.weight * args.budget))

    if args.all:
        targets = [s for s in S.SOURCES if not s.deferred]
        if args.sft:  # only conversational sources make sense for SFT
            targets = [s for s in targets if s.kind in ("chat", "openmath")]
    elif args.source:
        targets = [S.get(args.source)]
    else:
        p.error("pass --source KEY or --all")

    for src in targets:
        if args.sft:
            tokenize_source_sft(src.key, cap_for(src), tokenizer=args.tokenizer)
        else:
            tokenize_source(src.key, cap_for(src), s3_workers=args.s3_workers,
                            tokenizer=args.tokenizer)


if __name__ == "__main__":
    main()

"""Train a custom byte-level BPE tokenizer on the LLM mixture.

Samples a *weighted* text corpus straight from the mixture sources (same rows,
same per-``kind`` rendering as ``tokenize_source.py`` — including Stack v2 content
pulled from Software Heritage S3), trains a byte-level BPE with the ChatML special
tokens baked in, and writes:

    /mnt/ai/data/llm-mix/tokenizer/<name>/tokenizer.json   # load with BPETokenizer
    /mnt/ai/data/llm-mix/tokenizer/<name>/meta.json        # vocab, specials, corpus mix

Each source contributes a share of the character budget proportional to its
mixture weight (renormalized over the sources actually included), so the merges
reflect the blend the model will train on — code identifiers, math notation, and
ChatML/tool-call markup all get their fair say, instead of inheriting a tokenizer
tuned for generic web text.

The result drops straight into the existing pipeline: point ``tokenize_source``
and the trainers at it with ``tokenizer=<dir>`` (or the ``tokenizer.json``),
since ``BPETokenizer.from_pretrained`` now accepts a local path.

Library module — driven from the "Train a custom BPE tokenizer" section of
``main.ipynb`` (sample a weighted corpus once, train a suite of vocab sizes from
it). Key helpers: ``Corpus``, ``_materialize_corpus``/``_read_corpus``,
``_train_one``, ``_eval_compression``.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

import sources as S  # noqa: E402
from chimera.data.chat_template import SPECIAL_TOKENS as DEFAULT_SPECIALS  # noqa: E402
from tokenize_source import COLUMNS, _iter_rows, _safe_content, render  # noqa: E402

from chimera.tokenizers import BPETokenizer  # noqa: E402

OUT_ROOT = Path("/mnt/ai/data/llm-mix/tokenizer")

# Special tokens the chat template relies on (structural + semantic markers).
# Reserved in the tokenizer at stable low ids (see chat_template.SPECIAL_TOKENS
# and BPETokenizer._train_hf).

# uint16 memmaps store token ids downstream, so the vocab must fit in 16 bits.
UINT16_MAX = 65_535


# --------------------------------------------------------------------------- #
# Weighted corpus iterator (streams text; never materializes the whole sample)
# --------------------------------------------------------------------------- #
def _iter_source_texts(src, char_cap: int, s3_workers: int, batch: int = 512):
    """Yield rendered documents from one source until ``char_cap`` chars emitted."""
    cols = COLUMNS[src.kind]
    n = 0
    if src.kind == "stackv2":
        pool = ThreadPoolExecutor(max_workers=s3_workers)
        buf = []
        try:
            for _, row in _iter_rows(src, cols):
                buf.append(row)
                if len(buf) >= batch:
                    blobs = [(r.get("blob_id"), r.get("src_encoding") or "utf-8") for r in buf]
                    for t in pool.map(lambda b: _safe_content(*b), blobs):
                        if t:
                            yield t
                            n += len(t)
                    buf.clear()
                    if n >= char_cap:
                        return
        finally:
            pool.shutdown()
    else:
        for _, row in _iter_rows(src, cols):
            t = render(src, row)
            if t:
                yield t
                n += len(t)
                if n >= char_cap:
                    return


class Corpus:
    """Chains per-source samples to a char budget, tracking realized counts."""

    def __init__(self, targets, total_chars: int, s3_workers: int):
        self.targets = targets
        self.total_chars = total_chars
        self.s3_workers = s3_workers
        self.wsum = sum(s.weight for s in targets)
        self.realized: dict[str, int] = {}

    def __iter__(self):
        t0 = time.time()
        for src in self.targets:
            cap = int(self.total_chars * src.weight / self.wsum)
            print(f"[{src.key}] sampling up to {cap / 1e6:.0f}M chars "
                  f"(weight {src.weight / self.wsum:.3f})", flush=True)
            got = 0
            next_mark = 50_000_000
            for text in _iter_source_texts(src, cap, self.s3_workers):
                got += len(text)
                if got >= next_mark:
                    print(f"[{src.key}]   {got / 1e6:.0f}M chars "
                          f"({(time.time() - t0) / 60:.1f}min elapsed)", flush=True)
                    next_mark += 50_000_000
                yield text
            self.realized[src.key] = got
            print(f"[{src.key}] done: {got / 1e6:.0f}M chars", flush=True)


# --------------------------------------------------------------------------- #
# Corpus cache: sample the (S3-expensive) corpus once, reuse for every vocab size
# --------------------------------------------------------------------------- #
# One JSON-encoded document per line — preserves embedded newlines exactly, so
# training a suite of vocab sizes re-reads the identical corpus without re-hitting
# S3/HF. Trains from an iterator over this file, so RAM stays flat regardless of
# corpus size.
def _materialize_corpus(corpus: "Corpus", path: Path) -> dict:
    n_docs = n_chars = 0
    with open(path, "w") as f:
        for text in corpus:
            f.write(json.dumps(text) + "\n")
            n_docs += 1
            n_chars += len(text)
    return {"n_docs": n_docs, "n_chars": n_chars, "realized_chars": dict(corpus.realized)}


def _read_corpus(path: Path):
    with open(path) as f:
        for line in f:
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Optional: compression comparison vs the incumbent tokenizer
# --------------------------------------------------------------------------- #
def _eval_compression(tok_path: Path, targets, baseline_id: str, sample_chars: int):
    """Report chars/token (higher = better compression) on fresh held-out text."""
    new = BPETokenizer.from_pretrained(str(tok_path))
    base = BPETokenizer.from_pretrained(baseline_id)
    print(f"\ncompression (chars/token, higher is better) vs {baseline_id}:")
    print(f"  {'source':<16} {'new':>8} {'base':>8} {'delta':>8}")
    per_src_cap = max(sample_chars // max(len(targets), 1), 200_000)
    for src in targets:
        text = "".join(_iter_source_texts(src, per_src_cap, s3_workers=32))
        if not text:
            continue
        n_new = len(new._tok.encode(text, add_special_tokens=False).ids)
        n_base = len(base._tok.encode(text, add_special_tokens=False).ids)
        cr_new = len(text) / max(n_new, 1)
        cr_base = len(text) / max(n_base, 1)
        print(f"  {src.key:<16} {cr_new:>8.3f} {cr_base:>8.3f} "
              f"{(cr_new - cr_base):>+8.3f}")


def _vocab_tag(v: int) -> str:
    """4096 -> '4k', 32768 -> '32k' (falls back to the raw int if not a clean kib)."""
    return f"{v // 1024}k" if v % 1024 == 0 else str(v)


def _train_one(vocab: int, corpus_src, min_frequency: int, out_dir: Path,
               provenance: dict) -> dict:
    """Train + save one vocab size from an already-sampled corpus. corpus_src is a
    zero-arg callable returning a fresh iterator over the corpus documents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"
    t0 = time.time()
    tok = BPETokenizer(backend="hf")
    tok.train(corpus_src(), vocab_size=vocab,
              special_tokens=DEFAULT_SPECIALS, min_frequency=min_frequency)
    tok.save(tok_path)
    secs = round(time.time() - t0, 1)
    meta = {
        "vocab_size": tok.vocab_size,
        "requested_vocab_size": vocab,
        "special_tokens": DEFAULT_SPECIALS,
        "special_ids": {t: tok._tok.token_to_id(t) for t in DEFAULT_SPECIALS},
        "min_frequency": min_frequency,
        "seconds": secs,
        **provenance,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{out_dir.name}] trained vocab={tok.vocab_size} in {secs}s -> {tok_path}")
    return meta


# --------------------------------------------------------------------------- #


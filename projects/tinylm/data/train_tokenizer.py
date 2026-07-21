"""Train the frozen tinylm tokenizer suite (8k / 12k / 16k) on a fixed corpus.

Follows the project README tokenizer plan: sample ONE fixed, future-facing
corpus with a seed and explicit per-source character budgets — this represents
the model's lifetime inputs, not any single pretraining run — cache it once,
then train every vocab-size candidate from those identical bytes so vocab size
is the only variable.

Config (locked with the user):
    * byte-level BPE, no UNK, no normalization (chimera.tokenizers, hf backend)
    * split_digits=False        -> unsplit digits compress dates/numbers better;
      commonsense/grounding matter more than arithmetic here (README contract)
    * full canonical specials    -> chat/reasoning/tool markers reserved at fixed
      low ids from the first run (chimera.data.chat_template.SPECIAL_TOKENS)
    * 500M-char corpus, per-source shares below (README future-facing mix, minus
      QuAC which has no loadable HF form — its share folds into CoQA + SQuAD)

Sources are read through their chimera.data DataModules (same document rendering
the pretraining stream uses) via streaming, so nothing is fully downloaded just
to sample a slice. Chat sources are rendered with the canonical ChatML template.

Writes, git-tracked, into the repo:
    projects/tinylm/data/tokenizers/<tag>/tokenizer.json   # from_pretrained-loadable
    projects/tinylm/data/tokenizers/<tag>/meta.json        # hash, ids, corpus meta
    projects/tinylm/data/tokenizers/report.md              # comparison report
    projects/tinylm/data/tokenizers/summary.json
The 500M-char corpus itself is cached OFF-repo (too big to track):
    /mnt/ai/data/tinylm/tokenizer/corpus.jsonl

Usage:
    uv run python train_tokenizer.py                 # 500M chars, vocabs 8k/12k/16k
    uv run python train_tokenizer.py --chars 250_000_000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

from datasets import load_dataset  # noqa: E402

from chimera.data import (  # noqa: E402
    CoQADataModule,
    CosmopediaV2DataModule,
    FineWebEduTextDataModule,
    SQuADTextDataModule,
    StackExchangeDataModule,
    TinyStoriesV2DataModule,
    WikipediaDataModule,
)
from chimera.data.chat_template import SPECIAL_TOKENS, render  # noqa: E402
from chimera.tokenizers import BPETokenizer  # noqa: E402

REPO_OUT = Path(__file__).resolve().parent / "tokenizers"  # git-tracked
CORPUS_PATH = Path("/mnt/ai/data/tinylm/tokenizer/corpus.jsonl")  # off-repo cache
VOCAB_SIZES = [8192, 12288, 16384]
DEFAULT_CHARS = 500_000_000
SEED = 0

# Per-source character shares of the fixed corpus (sum to 100). README
# future-facing mix; QuAC dropped (no loadable HF form) and its share folded
# into the CoQA/SQuAD grounded-QA group.
SHARES = {
    "fineweb-edu": 35.0,
    "cosmopedia-v2": 20.0,
    "tinystories-v2": 15.0,
    "stackexchange": 10.0,
    "wikipedia": 5.0,
    "coqa": 3.75,          # grounded QA (7.5% split across coqa + squad)
    "squad": 3.75,
    "oasst1": 2.5,         # ChatML conversations (7.5% split across the three)
    "no_robots": 2.5,
    "ultrachat": 2.5,
}

# Probe strings for the round-trip + tokenization report (README: Unicode,
# Markdown, JSON, URLs, contractions, dates, ChatML).
PROBES = {
    "unicode": "café — naïve façade, 日本語, emoji 😀🎉, math ∑∫≈",
    "markdown": "# Title\n\n- **bold** and _italic_\n\n```py\nx = 1\n```\n",
    "json": '{"name": "get_weather", "arguments": {"city": "Paris", "n": 3}}',
    "url": "See https://example.com/path?q=1&r=2#frag for details.",
    "contractions": "I can't; you're right, it's don't-and-won't o'clock.",
    "dates": "On 2026-07-20 at 14:23:50, 1,234,567 items shipped.",
    "chatml": render(
        [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello."}]
    ),
}


def _vocab_tag(v: int) -> str:
    return f"{v // 1024}k" if v % 1024 == 0 else str(v)


# --------------------------------------------------------------------------- #
# Streaming document iterators per source (same rendering as the pretrain stream)
# --------------------------------------------------------------------------- #
def _stream_ds(dm):
    """Open a source's HF dataset in streaming mode, honoring its config/shards."""
    kwargs = {"split": dm.TRAIN_SPLIT, "streaming": True}
    if dm.DATA_FILES is not None:
        kwargs["data_files"] = dm.DATA_FILES
    if dm.CONFIG_NAME is not None:
        kwargs["name"] = dm.CONFIG_NAME
    return load_dataset(dm.HF_REPO, **kwargs)


def _plain_docs(dm):
    """Documents rendered exactly as the DataModule would render them."""
    yield from dm.iter_texts(_stream_ds(dm))


def _chatml_docs(repo: str, split: str, msg_col: str):
    """Render each conversation with the canonical ChatML template."""
    ds = load_dataset(repo, split=split, streaming=True)
    for ex in ds:
        msgs = ex[msg_col]
        if msgs:
            yield render(msgs)


def _oasst1_docs():
    """Reconstruct English root-to-leaf conversation paths from OASST1 trees.

    OASST1 is a flat message table; group by tree, follow the highest-ranked
    child at each turn from each root, keep reviewed English messages, and
    render the resulting user/assistant path as ChatML.
    """
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    ds = ds.filter(lambda r: r["lang"] == "en" and r["review_result"])
    by_id = {r["message_id"]: r for r in ds}
    children: dict[str, list] = {}
    roots = []
    for r in by_id.values():
        pid = r["parent_id"]
        if pid is None or pid not in by_id:
            roots.append(r)
        else:
            children.setdefault(pid, []).append(r)

    def _rank(r):  # rank None (unranked) sorts last
        return r["rank"] if r["rank"] is not None else 1e9

    for root in roots:
        msgs, node = [], root
        while node is not None:
            role = "user" if node["role"] == "prompter" else "assistant"
            msgs.append({"role": role, "content": node["text"]})
            kids = sorted(children.get(node["message_id"], []), key=_rank)
            node = kids[0] if kids else None
        if len(msgs) >= 2:
            yield render(msgs)


def _source_docs(name: str):
    if name == "fineweb-edu":
        return _plain_docs(FineWebEduTextDataModule(data_dir="/mnt/ai/data"))
    if name == "cosmopedia-v2":
        return _plain_docs(CosmopediaV2DataModule(data_dir="/mnt/ai/data"))
    if name == "tinystories-v2":
        return _plain_docs(TinyStoriesV2DataModule(data_dir="/mnt/ai/data"))
    if name == "stackexchange":
        return _plain_docs(StackExchangeDataModule(data_dir="/mnt/ai/data"))
    if name == "wikipedia":
        return _plain_docs(WikipediaDataModule(data_dir="/mnt/ai/data"))
    if name == "coqa":
        return _plain_docs(CoQADataModule(data_dir="/mnt/ai/data"))
    if name == "squad":
        return _plain_docs(SQuADTextDataModule(data_dir="/mnt/ai/data"))
    if name == "oasst1":
        return _oasst1_docs()
    if name == "no_robots":
        return _chatml_docs("HuggingFaceH4/no_robots", "train", "messages")
    if name == "ultrachat":
        return _chatml_docs("HuggingFaceH4/ultrachat_200k", "train_sft", "messages")
    raise ValueError(f"unknown source {name!r}")


# --------------------------------------------------------------------------- #
# Sample the fixed corpus once (front-of-stream = deterministic), cache as JSONL.
# A trailing held-out slice per source is kept separately for compression eval.
# --------------------------------------------------------------------------- #
def sample_corpus(total_chars: int, path: Path, heldout_chars: int = 1_000_000) -> dict:
    wsum = sum(SHARES.values())
    realized, n_docs = {}, 0
    heldout: dict[str, str] = {}
    t0 = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name, share in SHARES.items():
            cap = int(total_chars * share / wsum)
            got, hbuf, hgot = 0, [], 0
            for text in _source_docs(name):
                if not text:
                    continue
                if got < cap:
                    f.write(json.dumps(text) + "\n")
                    got += len(text)
                    n_docs += 1
                elif hgot < heldout_chars:  # trailing docs -> held-out eval slice
                    hbuf.append(text)
                    hgot += len(text)
                else:
                    break
            realized[name] = got
            heldout[name] = "\n".join(hbuf)[:heldout_chars]
            print(
                f"[{name}] {got / 1e6:.1f}M chars (target {cap / 1e6:.1f}M, "
                f"share {share / wsum:.3f})  held-out {hgot / 1e6:.2f}M  "
                f"{(time.time() - t0) / 60:.1f}min",
                flush=True,
            )
    total = sum(realized.values())
    corpus_hash = _file_hash(path)
    meta = {
        "seed": SEED,
        "total_chars": total,
        "requested_chars": total_chars,
        "n_docs": n_docs,
        "shares_pct": SHARES,
        "realized_chars": realized,
        "corpus_hash": corpus_hash,
    }
    print(f"corpus: {total / 1e6:.0f}M chars / {n_docs:,} docs -> {path}")
    return meta, heldout


def _file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_corpus(path: Path):
    with open(path) as f:
        for line in f:
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Per-candidate evaluation on the held-out slices.
# --------------------------------------------------------------------------- #
def evaluate(tok: BPETokenizer, heldout: dict[str, str]) -> dict:
    raw = tok._tok
    used = set()
    per_source = {}
    for name, text in heldout.items():
        if not text:
            continue
        ids = raw.encode(text, add_special_tokens=False).ids
        used.update(ids)
        n_tok = max(len(ids), 1)
        per_source[name] = {
            "chars_per_token": round(len(text) / n_tok, 3),
            "bytes_per_token": round(len(text.encode("utf-8")) / n_tok, 3),
        }

    # doc-length distribution + context-fit fractions on the held-out docs
    doc_lens = []
    for name, text in heldout.items():
        for doc in text.split("\n"):
            if doc:
                doc_lens.append(len(raw.encode(doc, add_special_tokens=False).ids))
    doc_lens.sort()

    def _p(q):
        return doc_lens[min(len(doc_lens) - 1, int(q * len(doc_lens)))] if doc_lens else 0

    def _frac_within(n):
        return round(sum(l <= n for l in doc_lens) / max(len(doc_lens), 1), 4)

    all_text = "\n".join(t for t in heldout.values() if t)
    total_ids = raw.encode(all_text, add_special_tokens=False).ids
    # round-trip + special-token atomicity checks. Decode with specials kept:
    # the byte-level stream is lossless, and reserved markers are atomic tokens
    # (the default skip_special_tokens=True would drop them from the ChatML probe).
    round_trips = {
        k: (raw.decode(raw.encode(v).ids, skip_special_tokens=False) == v)
        for k, v in PROBES.items()
    }
    special_atomic = {
        t: (len(raw.encode(t, add_special_tokens=False).ids) == 1)
        for t in SPECIAL_TOKENS
    }
    return {
        "aggregate_chars_per_token": round(len(all_text) / max(len(total_ids), 1), 3),
        "aggregate_bytes_per_token": round(
            len(all_text.encode("utf-8")) / max(len(total_ids), 1), 3
        ),
        "per_source": per_source,
        "tokens_per_doc_mean": round(sum(doc_lens) / max(len(doc_lens), 1), 1),
        "tokens_per_doc_p95": _p(0.95),
        "frac_within_512": _frac_within(512),
        "frac_within_2048": _frac_within(2048),
        "frac_within_8192": _frac_within(8192),
        "vocab_utilization": round(len(used) / tok.vocab_size, 4),
        "round_trips_ok": round_trips,
        "all_round_trips_ok": all(round_trips.values()),
        "special_tokens_atomic": special_atomic,
        "all_specials_atomic": all(special_atomic.values()),
    }


def _write_report(path: Path, results: dict):
    tags = list(results.keys())
    lines = ["# tinylm tokenizer suite\n"]
    lines.append("Fixed 500M-char future-facing corpus; vocab size is the only variable.\n")
    lines.append("## Aggregate\n")
    lines.append("| metric | " + " | ".join(tags) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in tags) + " |")
    rows = [
        ("chars/token (agg)", "aggregate_chars_per_token"),
        ("bytes/token (agg)", "aggregate_bytes_per_token"),
        ("tokens/doc mean", "tokens_per_doc_mean"),
        ("tokens/doc p95", "tokens_per_doc_p95"),
        ("frac ≤512 tok", "frac_within_512"),
        ("frac ≤2048 tok", "frac_within_2048"),
        ("vocab utilization", "vocab_utilization"),
        ("round trips ok", "all_round_trips_ok"),
        ("specials atomic", "all_specials_atomic"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | " + " | ".join(str(results[t][key]) for t in tags) + " |")
    lines.append("\n## chars/token per source\n")
    srcs = list(next(iter(results.values()))["per_source"].keys())
    lines.append("| source | " + " | ".join(tags) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in tags) + " |")
    for s in srcs:
        lines.append(
            f"| {s} | "
            + " | ".join(str(results[t]["per_source"][s]["chars_per_token"]) for t in tags)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=DEFAULT_CHARS)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    REPO_OUT.mkdir(parents=True, exist_ok=True)
    corpus_meta, heldout = sample_corpus(args.chars, CORPUS_PATH)

    results = {}
    for v in VOCAB_SIZES:
        tag = _vocab_tag(v)
        out_dir = REPO_OUT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        tok = BPETokenizer(backend="hf")
        tok.train(
            _read_corpus(CORPUS_PATH),
            vocab_size=v,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=args.min_frequency,
            split_digits=False,
        )
        tok_path = out_dir / "tokenizer.json"
        tok.save(tok_path)
        secs = round(time.time() - t0, 1)
        ev = evaluate(tok, heldout)
        results[tag] = ev
        meta = {
            "vocab_size": tok.vocab_size,
            "requested_vocab_size": v,
            "tokenizer_hash": _file_hash(tok_path),
            "special_tokens": SPECIAL_TOKENS,
            "special_ids": {t: tok._tok.token_to_id(t) for t in SPECIAL_TOKENS},
            "split_digits": False,
            "min_frequency": args.min_frequency,
            "seconds": secs,
            "corpus": corpus_meta,
            "evaluation": ev,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"[{tag}] vocab={tok.vocab_size} in {secs}s  "
            f"chars/tok={ev['aggregate_chars_per_token']}  "
            f"util={ev['vocab_utilization']}  -> {tok_path}"
        )

    (REPO_OUT / "summary.json").write_text(
        json.dumps(
            {
                "vocab_sizes": VOCAB_SIZES,
                "specials": SPECIAL_TOKENS,
                "split_digits": False,
                "corpus": corpus_meta,
                "results": results,
            },
            indent=2,
        )
    )
    _write_report(REPO_OUT / "report.md", results)
    print(f"\nsummary -> {REPO_OUT / 'summary.json'}\nreport  -> {REPO_OUT / 'report.md'}")


if __name__ == "__main__":
    main()

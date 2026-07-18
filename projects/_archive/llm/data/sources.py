"""Registry of the planned pretraining mixture, with streaming sample access.

This project trains a small LM on a blend of code, web text, and math/reasoning.
Each entry here names the concrete Hugging Face dataset that backs a slice of the
mixture, its target weight (fraction of the *whole* blend), and a streaming
``sample()`` that yields a handful of real rows so the data can be eyeballed
before committing to a tokenization/packing pipeline.

Planned mixture (weights = fraction of the whole blend; see TARGET_TOKENS/plan())
---------------
    code            40%   The Stack v2 (dedup): Python 28 / Shell 6 / JSON 6
    web             25%   FineWeb-Edu (sample-10BT)
    math / cot      20%   FineMath 4+ (web math) 10 + OpenMathReasoning CoT 10
    tool-call       15%   Toucan-1.5M (anchor, ~6B tok) + smaller FC/agentic sets
                          (xLAM, Dria, ToolACE, APIGen-MT, Hermes) for diversity

Supply note: every category is drawn <<1 epoch at our scale. Toucan-1.5M (~6B
tokens, Apache-2.0) removed the old tool-call scarcity; see the training-plan
section of ``main.ipynb`` for per-source target tokens and repeat factors.

Nothing here downloads a full corpus. Sharded parquet slices download only the
smallest shard and read the first rows locally; the small tool-call JSON files
download once (cached). The Stack v2 stores *pointers*, not code — each row
carries a ``blob_id`` whose bytes live in the public Software Heritage S3 bucket,
so those loaders fetch content on demand (see :func:`_swh_content`).

Gating / credentials
--------------------
- The Stack v2 is a gated dataset: accept the terms at
  https://huggingface.co/datasets/bigcode/the-stack-v2-dedup and be logged in
  (``huggingface-cli login`` or ``HF_TOKEN``) or its rows won't stream.
- Content bytes come from ``s3://softwareheritage/content/`` via anonymous
  (unsigned) access. If that ever returns AccessDenied, S3 credentials or
  requester-pays may be required in your environment.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from chimera.data.chat_template import decode_unicode_escapes as _decode_unicode_escapes

# Datasets/models/caches live on the big volume (see project memory).
os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")


# --------------------------------------------------------------------------- #
# Peeking one shard (download the smallest shard, read the first rows locally)
# --------------------------------------------------------------------------- #
# These datasets are Xet-backed. In this environment neither the dataset-viewer
# /rows API (unreachable) nor ranged HTTP reads (Xet CDN rejects plain byte-range
# GETs) work, so the one dependable content path is huggingface_hub's native Xet
# downloader — which fetches whole files. We therefore grab the *smallest* shard
# of a slice (e.g. a partial final FineWeb-Edu shard is ~0.5GB vs ~2GB), cache it
# under HF_HOME, and read only the first rows + needed columns from it locally.
# One shard is a chunk of the corpus, not the whole dataset, and it's cached, so
# repeated browsing is free.


@dataclass
class Shard:
    path: str
    size: int  # bytes
    cached: bool  # already on disk before this call


def _smallest_shard(repo: str, prefix: str) -> Shard:
    from huggingface_hub import HfApi
    from huggingface_hub import try_to_load_from_cache

    api = HfApi()
    files = [
        f
        for f in api.list_repo_files(repo, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(f"no parquet under {prefix!r} in {repo}")
    infos = api.get_paths_info(repo, files, repo_type="dataset", expand=True)

    def _size(p) -> int:
        return getattr(p, "size", None) or getattr(getattr(p, "lfs", None), "size", 0)

    best = min(infos, key=_size)
    cached = (
        try_to_load_from_cache(repo, best.path, repo_type="dataset") is not None
    )
    return Shard(path=best.path, size=_size(best), cached=cached)


def _peek_parquet(
    repo: str, prefix: str, n: int, columns: Optional[list[str]] = None
) -> tuple[list[dict], Shard]:
    """Download the smallest shard (cached) and return its first ``n`` rows."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shard = _smallest_shard(repo, prefix)
    local = hf_hub_download(repo, shard.path, repo_type="dataset")
    pf = pq.ParquetFile(local)
    avail = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in avail] if columns else None
    batch = next(pf.iter_batches(batch_size=n, columns=cols))
    return batch.to_pylist()[:n], shard


def _peek_json(repo: str, filename: str, n: int) -> tuple[list[dict], Shard]:
    """Download a whole .json/.jsonl file (cached) and return its first ``n`` rows.

    Used for the small tool-call datasets that ship a single JSON blob rather than
    sharded parquet; there is no partial-read path for these, but the files are
    small and cached after the first fetch.
    """
    import json

    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo, filename, repo_type="dataset") is not None
    info = HfApi().get_paths_info(repo, [filename], repo_type="dataset", expand=True)[0]
    size = getattr(info, "size", None) or getattr(getattr(info, "lfs", None), "size", 0)
    local = hf_hub_download(repo, filename, repo_type="dataset")
    with open(local) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data[:n], Shard(path=filename, size=size, cached=cached)


def _peek_slice(
    src: "Source", n: int, columns: Optional[list[str]] = None
) -> tuple[list[dict], Shard]:
    """Dispatch to the parquet or JSON peek based on the slice's file layout."""
    if src.file_prefix.endswith((".json", ".jsonl")):
        return _peek_json(src.hf_repo, src.file_prefix, n)
    return _peek_parquet(src.hf_repo, src.file_prefix, n, columns=columns)


# --------------------------------------------------------------------------- #
# Software Heritage content fetch (for The Stack v2)
# --------------------------------------------------------------------------- #
_S3 = None


def _s3_client():
    """Lazily build an anonymous S3 client for the Software Heritage bucket."""
    global _S3
    if _S3 is None:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        # max_pool_connections defaults to 10, which serializes a big thread pool
        # onto 10 sockets — the bottleneck for bulk Stack v2 content fetches.
        _S3 = boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=512,
                retries={"max_attempts": 3, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=30,
            ),
        )
    return _S3


def _swh_content(blob_id: str, src_encoding: str = "utf-8") -> str:
    """Download and decode one file's bytes from Software Heritage S3.

    Stack v2 rows are gzip-compressed blobs keyed by ``blob_id``; ``src_encoding``
    is the file's original text encoding (a column on every row).
    """
    obj = _s3_client().get_object(
        Bucket="softwareheritage", Key=f"content/{blob_id}"
    )
    raw = gzip.decompress(obj["Body"].read())
    return raw.decode(src_encoding, errors="replace")


# --------------------------------------------------------------------------- #
# Sample record + source definition
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One previewable example: the training ``text`` plus a bit of provenance."""

    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Source:
    key: str  # cli handle, e.g. "stackv2-python"
    title: str  # human label
    category: str  # code | web | math | tools | chat
    weight: float  # PRETRAIN fraction of the whole mixture (categories sum to ~1.0)
    hf_repo: str  # backing dataset (or "" when deferred)
    config: Optional[str]  # HF config / subset name
    split: str  # HF split
    file_prefix: str  # repo path prefix selecting this slice's parquet shards
    avail_tokens: float  # est. UNIQUE tokens available (LFM2.5 tok; measured/published)
    notes: str  # gating, licensing, schema quirks
    _sampler: Callable[["Source", int], Iterator[Sample]]
    # kind drives how the tokenizer renders a row: "text" (raw next-token),
    # "openmath" (problem+solution), "chat" (ChatML from a message list),
    # "stackv2" (fetch content from SWH S3). See tokenize_source.py.
    kind: str = "text"
    deferred: bool = False
    # SFT-stage weight, independent of the pretrain ``weight`` (the SFT mix has a
    # different composition — chat-led, not code/web-led). When set, the SFT
    # builder uses this instead of ``weight``; when None it falls back to
    # ``weight``. A source with ``weight=0.0`` + ``sft_weight>0`` is SFT-only
    # (excluded from the pretrain mix, included in the SFT mix).
    sft_weight: Optional[float] = None
    last_shard: Optional["Shard"] = None  # shard read by the most recent sample()

    def sample(self, n: int = 5) -> Iterator[Sample]:
        """Yield up to ``n`` real rows, streamed (no full download)."""
        if self.deferred:
            raise RuntimeError(
                f"source {self.key!r} is deferred (not built yet): {self.notes}"
            )
        yield from self._sampler(self, n)


# --------------------------------------------------------------------------- #
# Per-source samplers
# --------------------------------------------------------------------------- #
def _sample_stackv2(src: Source, n: int) -> Iterator[Sample]:
    # Stack v2 rows are lightweight pointers; read just the columns we need, then
    # pull each file's bytes from Software Heritage S3.
    cols = ["blob_id", "src_encoding", "path", "language", "length_bytes"]
    rows, shard = _peek_parquet(src.hf_repo, src.file_prefix, n, columns=cols)
    src.last_shard = shard
    for row in rows:
        try:
            text = _swh_content(row["blob_id"], row.get("src_encoding") or "utf-8")
        except Exception as e:  # surface S3/gating failures without crashing browse
            text = f"<failed to fetch content for blob {row.get('blob_id')}: {e}>"
        yield Sample(
            text=text,
            meta={
                "path": row.get("path"),
                "language": row.get("language"),
                "length_bytes": row.get("length_bytes"),
            },
        )


def _text_field(row: dict) -> str:
    """Pick the main text column, tolerating schema drift across dataset versions."""
    for k in ("text", "content", "markdown"):
        if isinstance(row.get(k), str):
            return row[k]
    # last resort: the longest string field on the row
    strings = {k: v for k, v in row.items() if isinstance(v, str)}
    if not strings:
        return f"<no text field; keys={list(row)}>"
    return max(strings.values(), key=len)


def _sample_plain_text(src: Source, n: int) -> Iterator[Sample]:
    rows, shard = _peek_parquet(
        src.hf_repo, src.file_prefix, n, columns=["text", "content", "url", "score"]
    )
    src.last_shard = shard
    for row in rows:
        yield Sample(
            text=_text_field(row),
            meta={k: row[k] for k in ("url", "score", "language") if k in row},
        )


def _sample_openmath(src: Source, n: int) -> Iterator[Sample]:
    """Render OpenMathReasoning as problem + chain-of-thought solution."""
    cols = ["problem", "generated_solution", "problem_type", "expected_answer"]
    rows, shard = _peek_parquet(src.hf_repo, src.file_prefix, n, columns=cols)
    src.last_shard = shard
    for row in rows:
        problem = row.get("problem") or row.get("question") or ""
        solution = row.get("generated_solution") or row.get("solution") or ""
        text = f"Problem:\n{problem}\n\nSolution:\n{solution}".strip()
        yield Sample(
            text=text,
            meta={
                "problem_type": row.get("problem_type"),
                "expected_answer": row.get("expected_answer"),
            },
        )


def _render_conversation(conv: list) -> str:
    """Flatten a ShareGPT/ChatML-style message list into readable text."""
    lines = []
    for m in conv:
        if not isinstance(m, dict):
            lines.append(_decode_unicode_escapes(str(m)))
            continue
        role = m.get("role") or m.get("from") or "?"
        content = _decode_unicode_escapes(m.get("content") or m.get("value") or "")
        lines.append(f"<{role}>\n{content}")
    return "\n\n".join(lines)


def _coerce_conv(v) -> list:
    """Return a message list from a value that may be a list or a JSON/py-repr string."""
    import ast
    import json

    if isinstance(v, list):
        return v
    if isinstance(v, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                out = parse(v)
                if isinstance(out, list):
                    return out
            except Exception:
                pass
        return [{"role": "raw", "content": v}]
    return []


def _sample_toolcall(src: Source, n: int) -> Iterator[Sample]:
    """Render tool-call datasets whose rows carry a message list.

    Handles the common column names (`conversations` / `messages`), lists stored
    as JSON/py-repr strings, and an optional `tools` schema block.
    """
    cols = ["conversations", "messages", "tools", "system", "type", "domain", "subset_name"]
    rows, shard = _peek_slice(src, n, columns=cols)
    src.last_shard = shard
    for row in rows:
        conv = _coerce_conv(row.get("conversations") or row.get("messages"))
        text = _render_conversation(conv)
        if row.get("tools"):
            text = f"<tools>\n{_decode_unicode_escapes(str(row['tools']))}\n\n{text}"
        yield Sample(
            text=text,
            meta={
                "type": row.get("type"),
                "domain": row.get("domain") or row.get("subset_name"),
            },
        )


def _sample_deferred(src: Source, n: int) -> Iterator[Sample]:
    raise RuntimeError(f"{src.key} is deferred")
    yield  # pragma: no cover


# --------------------------------------------------------------------------- #
# The mixture
# --------------------------------------------------------------------------- #
# Default total training budget the composition is planned against. Weights below
# are fractions of this. avail_tokens is UNIQUE supply (LFM2.5 tokenizer): measured
# on cached shards for web/math/tools; published order-of-magnitude for Stack v2
# (its content is gated + S3-only, so it can't be scanned locally). plan() turns
# these into per-source target tokens + repetition factors (see --plan).
TARGET_TOKENS = 10_000_000_000  # 10B: ~100x a 100M-param muP model (healthy overtrain)

_B = 1_000_000_000
_M = 1_000_000

SOURCES: list[Source] = [
    # ---- code : 40% (The Stack v2 dedup, gated; content from SWH S3) --------
    # Code supply is effectively unbounded at our scale (<<1 epoch drawn).
    Source(
        key="stackv2-python",
        title="The Stack v2 — Python",
        category="code",
        weight=0.28,
        hf_repo="bigcode/the-stack-v2-dedup",
        config="Python",
        split="train",
        file_prefix="data/Python/",
        avail_tokens=60 * _B,  # published order-of-magnitude
        notes="Gated; rows are blob pointers, content fetched from SWH S3.",
        kind="stackv2",
        _sampler=_sample_stackv2,
    ),
    Source(
        key="stackv2-bash",
        title="The Stack v2 — Shell/Bash",
        category="code",
        weight=0.06,
        hf_repo="bigcode/the-stack-v2-dedup",
        config="Shell",
        split="train",
        file_prefix="data/Shell/",
        avail_tokens=4 * _B,
        notes="Gated; SWH S3 content fetch. 'Shell' config covers bash/sh/zsh.",
        kind="stackv2",
        _sampler=_sample_stackv2,
    ),
    Source(
        key="stackv2-json",
        title="The Stack v2 — JSON (tool-call/structured)",
        category="code",
        weight=0.06,
        hf_repo="bigcode/the-stack-v2-dedup",
        config="JSON",
        split="train",
        file_prefix="data/JSON/",
        avail_tokens=40 * _B,
        notes="Gated; SWH S3 content fetch. Proxy for tool-call/structured data.",
        kind="stackv2",
        _sampler=_sample_stackv2,
    ),
    # ---- web : 25% ---------------------------------------------------------
    Source(
        key="fineweb-edu",
        title="FineWeb-Edu (sample-10BT)",
        category="web",
        weight=0.25,
        hf_repo="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        file_prefix="sample/10BT/",
        avail_tokens=10 * _B,  # the sample IS 10B; swap to 100BT/350BT/1.3T for more
        notes="Open web-text; 'text' column. Same corpus as the base pretrain.",
        _sampler=_sample_plain_text,
    ),
    # ---- math / cot : 20% --------------------------------------------------
    Source(
        key="finemath",
        title="FineMath 4+ (web math)",
        category="math",
        weight=0.10,
        hf_repo="HuggingFaceTB/finemath",
        config="finemath-4plus",
        split="train",
        file_prefix="finemath-4plus/",
        avail_tokens=9.5 * _B,  # measured
        notes="CommonCrawl math, step-by-step explanations; 'text' column.",
        _sampler=_sample_plain_text,
    ),
    Source(
        key="openmath-cot",
        title="OpenMathReasoning — CoT",
        category="math",
        weight=0.10,
        hf_repo="nvidia/OpenMathReasoning",
        config="default",
        split="cot",
        file_prefix="data/cot-",
        avail_tokens=24.9 * _B,  # measured; note: problem→CoT, SFT-like
        notes="Long chain-of-thought solutions to AoPS problems (SFT-like).",
        kind="openmath",
        sft_weight=0.20,  # math capability in the chat-led SFT mix
        _sampler=_sample_openmath,
    ),
    # ---- tool-call / agentic : 15% ----------------------------------------
    # Toucan-1.5M (~6B tokens, Apache-2.0) anchors the slice, so the tools category
    # is no longer supply-constrained; the smaller sets add format diversity
    # (pythonic calls, deep human-in-the-loop trajectories, single-turn FC).
    Source(
        key="tool-toucan",
        title="Toucan-1.5M (synthetic tool-agent trajectories)",
        category="tools",
        weight=0.10,
        hf_repo="Agent-Ark/Toucan-1.5M",
        config=None,
        split="train",
        file_prefix="OSS/",  # smallest-shard peek picks within this generator subset
        avail_tokens=6.0 * _B,  # measured: ~4016 tok/traj x 1.5M; Apache-2.0
        notes="Largest open tool-agent set (1.5M multi-turn trajectories, real MCP tools); Apache-2.0.",
        kind="chat",
        sft_weight=0.15,  # tool-agent capability in the chat-led SFT mix
        _sampler=_sample_toolcall,
    ),
    Source(
        key="tool-xlam",
        title="xLAM function-calling 60k (Hermes/ChatML reformat)",
        category="tools",
        weight=0.01,
        hf_repo="minpeter/xlam-function-calling-60k-hermes",
        config=None,
        split="train",
        file_prefix="result.parquet",
        avail_tokens=51 * _M,  # measured; single-turn; UNGATED mirror of Salesforce xLAM
        notes="Single-turn FC; ungated ChatML mirror of the gated Salesforce/xlam-60k.",
        kind="chat",
        sft_weight=0.03,  # format-diversity tool sets in the chat-led SFT mix
        _sampler=_sample_toolcall,
    ),
    Source(
        key="tool-pythonic",
        title="Dria pythonic function-calling",
        category="tools",
        weight=0.01,
        hf_repo="driaforall/pythonic-function-calling",
        config=None,
        split="train",
        file_prefix="data/train-00000-of-00001.parquet",
        avail_tokens=49 * _M,  # measured; multi-turn, pythonic call style
        notes="Multi-turn pythonic (code-style) tool calls; complements JSON-style FC.",
        kind="chat",
        sft_weight=0.03,  # format-diversity tool sets in the chat-led SFT mix
        _sampler=_sample_toolcall,
    ),
    Source(
        key="tool-toolace",
        title="ToolACE",
        category="tools",
        weight=0.01,
        hf_repo="Team-ACE/ToolACE",
        config=None,
        split="train",
        file_prefix="data.json",
        avail_tokens=10.4 * _M,  # measured; multi-turn, 26k+ API pool, Apache-2.0
        notes="Multi-turn, high-diversity APIs; Apache-2.0.",
        kind="chat",
        sft_weight=0.03,  # format-diversity tool sets in the chat-led SFT mix
        _sampler=_sample_toolcall,
    ),
    Source(
        key="tool-apigen-mt",
        title="APIGen-MT-5k (deep agentic trajectories)",
        category="tools",
        weight=0.01,
        hf_repo="Salesforce/APIGen-MT-5k",
        config=None,
        split="train",
        file_prefix="apigen-mt_5k.json",
        avail_tokens=30 * _M,  # measured; avg 18.5 msgs/conv, function_call→observation loops
        notes="Deepest multi-turn agentic trajectories, BUT license is CC-BY-NC (non-commercial).",
        kind="chat",
        sft_weight=0.03,  # format-diversity tool sets in the chat-led SFT mix
        _sampler=_sample_toolcall,
    ),
    Source(
        key="tool-hermes",
        title="Hermes function-calling v1 (multi-turn)",
        category="tools",
        weight=0.01,
        hf_repo="NousResearch/hermes-function-calling-v1",
        config=None,
        split="train",
        file_prefix="func-calling.json",
        avail_tokens=10.4 * _M,  # measured (all files); Apache-2.0
        notes="ShareGPT multi-turn FC + JSON-mode; Apache-2.0. (repo also has more files)",
        kind="chat",
        sft_weight=0.03,  # format-diversity tool sets in the chat-led SFT mix
        _sampler=_sample_toolcall,
    ),
    # ---- general chat : SFT-only (weight=0.0 keeps it out of the pretrain mix) --
    # UltraChat is the general open-domain instruction/chat anchor for SFT. It is
    # the ONLY SFT source with no pretrain-mix overlap (openmath-cot/toucan/tools
    # all appear in the pretrain data), so it carries the cleanest chat signal.
    Source(
        key="ultrachat",
        title="UltraChat 200k (general multi-turn chat)",
        category="chat",
        weight=0.0,  # SFT-only; excluded from the pretrain mixture
        hf_repo="HuggingFaceH4/ultrachat_200k",
        config=None,
        split="train_sft",
        file_prefix="data/train_sft-",
        avail_tokens=0.256 * _B,  # measured: 255.7M tok (207.9k convs), 77.5% supervised
        notes="General open-domain chat (no tools); anchors the chat-led SFT mix. MIT.",
        kind="chat",
        sft_weight=0.50,  # chat-dominant: general chat leads the SFT mix
        _sampler=_sample_toolcall,
    ),
    # ---- narrative / prose : LAMBADA-mix-only (weight=0.0) -----------------
    # Long-form narrative prose for a LAMBADA-optimized pretraining blend.
    # LAMBADA rewards discourse-level, long-range coreference — exactly what
    # book-length fiction supplies. Kept out of the default pretrain mix
    # (weight=0.0, like ultrachat); the LAMBADA-mix builder assigns explicit
    # weights at build time.
    Source(
        key="bookcorpus",
        title="BookCorpusOpen (novels)",
        category="prose",
        weight=0.0,  # LAMBADA-mix-only; excluded from the default pretrain mix
        hf_repo="lucadiliello/bookcorpusopen",
        config="default",
        split="train",
        file_prefix="data/train-",
        avail_tokens=1.6 * _B,  # est: 6.64GB text / ~4 chars/tok (17.9k books)
        notes="Open BookCorpus replacement (full-length novels); 'text' column. Unknown license.",
        _sampler=_sample_plain_text,
    ),
    Source(
        key="gutenberg",
        title="Project Gutenberg English eBooks",
        category="prose",
        weight=0.0,  # LAMBADA-mix-only; excluded from the default pretrain mix
        hf_repo="sedthh/gutenberg_english",
        config="default",
        split="train",
        file_prefix="data/train-",
        avail_tokens=4.5 * _B,  # est: 18.1GB text / ~4 chars/tok (48.3k books)
        notes="Gutenberg English books; TEXT column is UPPERCASE (also SOURCE/METADATA) — tokenize_source.render()/COLUMNS['text'] read lowercase 'text'/'content', so add a 'TEXT' fallback before tokenizing. MIT.",
        _sampler=_sample_plain_text,
    ),
    Source(
        key="pg19",
        title="PG-19 (long-form pre-1919 books)",
        category="prose",
        weight=0.0,  # LAMBADA-mix-only; excluded from the default pretrain mix
        hf_repo="emozilla/pg19",  # parquet mirror; canonical deepmind/pg19 ships no HF parquet
        config="default",
        split="train",
        file_prefix="data/train-",
        avail_tokens=2.9 * _B,  # est: 11.45GB text / ~4 chars/tok (28.6k books)
        notes="PG-19 long-range LM benchmark; 'text' column. Uses emozilla/pg19 parquet mirror — canonical deepmind/pg19 is a loader script + GCS blobs with NO HF parquet, incompatible with the parquet-prefix loader. Apache-2.0.",
        _sampler=_sample_plain_text,
    ),
    Source(
        key="fineweb",
        title="FineWeb (sample-10BT, plain web)",
        category="web",
        weight=0.0,  # LAMBADA-mix-only; excluded from the default pretrain mix
        hf_repo="HuggingFaceFW/fineweb",
        config="sample-10BT",
        split="train",
        file_prefix="sample/10BT/",
        avail_tokens=10 * _B,  # the sample IS 10B tokens (GPT-2 tok); larger dumps available
        notes="Plain FineWeb (NOT fineweb-edu) — broader, less-filtered web prose; 'text' column. ODC-By.",
        _sampler=_sample_plain_text,
    ),
    # ---- synthetic simple-English stories : opt-in (weight=0.0) ------------
    # TinyStoriesV2-GPT4, repackaged as our own HF dataset. Very short (~197
    # tok/story), simple-vocabulary synthetic narratives — a clean tiny-scale
    # pretraining corpus / curriculum warm-up. Kept out of the default mix
    # (weight=0.0, like the prose sources); assign a build-time weight to opt in.
    Source(
        key="tinystories",
        title="TinyStoriesV2-GPT4 (synthetic simple-English stories)",
        category="prose",
        weight=0.0,  # opt-in; excluded from the default pretrain mix
        hf_repo="karanravindra/tinystories-v2",
        config="default",
        split="train",
        file_prefix="data/train-",
        avail_tokens=0.542 * _B,  # measured: 542.3M tok (LFM2.5); 2.72M stories
        notes="Our repackaging of roneneldan/TinyStories GPT-4 V2; 'text' column, id+text schema. Public.",
        _sampler=_sample_plain_text,
    ),
]

SOURCES_BY_KEY: dict[str, Source] = {s.key: s for s in SOURCES}


def get(key: str) -> Source:
    if key not in SOURCES_BY_KEY:
        raise KeyError(
            f"unknown source {key!r}; options: {', '.join(SOURCES_BY_KEY)}"
        )
    return SOURCES_BY_KEY[key]


def category_weights() -> dict[str, float]:
    out: dict[str, float] = {}
    for s in SOURCES:
        out[s.category] = out.get(s.category, 0.0) + s.weight
    return out


def plan(target_tokens: int = TARGET_TOKENS) -> list[dict]:
    """Turn weights + available supply into a training plan.

    For each source: target tokens (weight × budget) and the repetition factor
    needed = target / unique-available. repeat > 1 means the slice must be seen
    more than once (epochs); >~3 in pretraining is a red flag for overfitting.
    """
    rows = []
    for s in SOURCES:
        tgt = s.weight * target_tokens
        repeat = (tgt / s.avail_tokens) if s.avail_tokens else float("inf")
        rows.append(
            {
                "key": s.key,
                "category": s.category,
                "weight": s.weight,
                "target_tokens": tgt,
                "avail_tokens": s.avail_tokens,
                "repeat": repeat,
            }
        )
    return rows

"""Registry of the tiny-LM pretraining + SFT mixture.

Target: a **5–20M param** general-language + chat model. At this scale the
binding constraint is *distribution width*, not token volume — so the blend is
dominated by narrow, clean synthetic prose (TinyStoriesV2) with knowledge /
expository / web registers layered in at small weights. See ``DATASETS.md`` for
the full research writeup behind these choices.

Weights are the fraction of the **pretrain** token budget (they sum to ~1.0).
Chat lives in a separate short SFT phase (``sft_weight``), not blended into
pretrain — smol-smoltalk is the SmolTalk subset trimmed for <1B models.

Budget
------
``TARGET_TOKENS`` = 2B — the recommended sweet spot for 5–20M params (≈1 clean
epoch of TinyStories + <1 epoch of everything else; no forced repetition). The
range considered was 1–10BT; going higher just means re-staging larger slices of
the effectively-unlimited sources (finephrase / fineweb-edu) and letting
TinyStories' *share* fall so its epoch count stays ≤~3.

Staging (NOT tokenizing)
------------------------
This project brings its own tokenizer (trained later), so nothing here tokenizes.
``download.py`` stages raw parquet shards into
``/mnt/ai/data/tiny-llm/raw/<key>/`` sized to cover each source's target unique
tokens with headroom. ``avail_tokens`` below is an *estimate* (bytes/≈4) until a
tokenizer exists and we can measure exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# Datasets/models/caches live on the big volume (see project memory).
os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")

RAW_ROOT = "/mnt/ai/data/tiny-llm/raw"

TARGET_TOKENS = 2_000_000_000  # 2B — recommended for 5–20M params
VAL_FRAC = 0.005

_B = 1_000_000_000
_M = 1_000_000


@dataclass
class Source:
    key: str                       # cli handle / raw-dir name, e.g. "tinystories"
    title: str                     # human label
    category: str                  # prose | textbook | synthetic-web | web | chat
    weight: float                  # PRETRAIN fraction of the blend (sum ~1.0)
    hf_repo: str                   # backing HF dataset
    file_prefix: str               # repo path prefix selecting this slice's parquet
    text_column: str               # column holding the training text
    avail_tokens: float            # est. UNIQUE tokens available (bytes/≈4)
    role: str                      # one-line role in the mixture
    notes: str = ""                # gating / licensing / schema quirks
    # How much to physically stage. n_shards=None -> stage the whole slice; else
    # the first N parquet shards under file_prefix (sorted). Sized to cover the
    # source's target unique tokens (weight x TARGET_TOKENS) with headroom.
    n_shards: Optional[int] = None
    # SFT-stage weight, independent of pretrain ``weight``. weight=0.0 +
    # sft_weight>0 => SFT-only (out of pretrain, in the chat SFT mix).
    sft_weight: Optional[float] = None
    license: str = ""


SOURCES: list[Source] = [
    # ---- fluency backbone : 50% -------------------------------------------
    Source(
        key="tinystories",
        title="TinyStoriesV2-GPT4 (synthetic simple-English stories)",
        category="prose",
        weight=0.50,
        hf_repo="karanravindra/tinystories-v2",  # our repackaging of roneneldan V2-GPT4
        file_prefix="data/train-",
        text_column="text",
        avail_tokens=0.54 * _B,     # measured previously (~542M tok, 2.72M stories)
        role="fluency backbone — narrow, clean, coherent English",
        notes="id+text schema. Public. cdla-sharing-1.0 upstream.",
        n_shards=None,              # small — stage the whole thing
        license="cdla-sharing-1.0",
    ),
    # ---- knowledge / expository : 22% -------------------------------------
    Source(
        key="tiny-strange-textbooks",
        title="Tiny Strange Textbooks (synthetic textbooks)",
        category="textbook",
        weight=0.22,
        hf_repo="nampdn-ai/tiny-strange-textbooks",
        file_prefix="data_part_",
        text_column="text",
        avail_tokens=4.0 * _B,      # 16GB raw / ~4 chars/tok
        role="knowledge / expository style (downweighted — higher complexity)",
        notes="Gated upstream (accept terms); accessible with HF_TOKEN. apache-2.0.",
        n_shards=3,                 # ~0.85B unique staged vs 0.44B target
        license="apache-2.0",
    ),
    # ---- explanatory + QA register (chat-adjacent) : 18% ------------------
    Source(
        key="finephrase-tutorial",
        title="finephrase — tutorial (synthetic web rephrasings)",
        category="synthetic-web",
        weight=0.09,
        hf_repo="HuggingFaceFW/finephrase",
        file_prefix="tutorial/",
        text_column="text",
        avail_tokens=160.0 * _B,    # whole config; we stage a bounded slice
        role="explanatory register — bridge toward chat",
        notes="Config is >1TB — MUST slice. ODC-BY. text column verified at stage time.",
        n_shards=12,                # ~0.28B staged vs 0.18B target
        license="odc-by",
    ),
    Source(
        key="finephrase-faq",
        title="finephrase — faq (synthetic web rephrasings, Q&A)",
        category="synthetic-web",
        weight=0.09,
        hf_repo="HuggingFaceFW/finephrase",
        file_prefix="faq/",
        text_column="text",
        avail_tokens=160.0 * _B,
        role="QA register — closest to conversational turn-taking",
        notes="Config is >1TB — MUST slice. ODC-BY.",
        n_shards=12,
        license="odc-by",
    ),
    # ---- natural web grounding : 10% --------------------------------------
    Source(
        key="fineweb-edu",
        title="FineWeb-Edu (sample-10BT)",
        category="web",
        weight=0.10,
        hf_repo="HuggingFaceFW/fineweb-edu",
        file_prefix="sample/10BT/",
        text_column="text",
        avail_tokens=10.0 * _B,     # the sample IS 10B; stage 1 shard for our budget
        role="natural-web grounding (keep small — broad web hurts coherence here)",
        notes="Educational web; 'text' column. ODC-BY. NOT small-model-specific.",
        n_shards=1,                 # ~0.7B staged vs 0.20B target
        license="odc-by",
    ),
    # ---- chat SFT : SFT-only (weight=0.0 keeps it out of pretrain) --------
    Source(
        key="smol-smoltalk",
        title="smol-smoltalk (SmolTalk trimmed for <1B models)",
        category="chat",
        weight=0.0,                 # SFT-only
        hf_repo="HuggingFaceTB/smol-smoltalk",
        file_prefix="data/train-",
        text_column="messages",     # conversation list -> rendered at SFT time
        avail_tokens=0.20 * _B,
        role="chat SFT — trimmed SmolTalk (no advanced math / long FC)",
        notes="484,570 rows / 971MB. apache-2.0. Separate SFT phase, not pretrain.",
        n_shards=None,              # small — stage whole
        sft_weight=1.0,
        license="apache-2.0",
    ),
]

SOURCES_BY_KEY: dict[str, Source] = {s.key: s for s in SOURCES}


def get(key: str) -> Source:
    if key not in SOURCES_BY_KEY:
        raise KeyError(f"unknown source {key!r}; options: {', '.join(SOURCES_BY_KEY)}")
    return SOURCES_BY_KEY[key]


def category_weights() -> dict[str, float]:
    out: dict[str, float] = {}
    for s in SOURCES:
        out[s.category] = out.get(s.category, 0.0) + s.weight
    return out


def plan(target_tokens: int = TARGET_TOKENS) -> list[dict]:
    """weight x budget -> per-source target tokens + repetition factor."""
    rows = []
    for s in SOURCES:
        if s.weight <= 0:
            continue
        tgt = s.weight * target_tokens
        repeat = (tgt / s.avail_tokens) if s.avail_tokens else float("inf")
        rows.append({
            "key": s.key, "category": s.category, "weight": s.weight,
            "target_tokens": tgt, "avail_tokens": s.avail_tokens, "repeat": repeat,
        })
    return rows


if __name__ == "__main__":
    print(f"pretrain budget: {TARGET_TOKENS/1e9:.1f}B tokens")
    print("category weights:", {k: round(v, 3) for k, v in category_weights().items()})
    print(f"{'key':<24}{'wt':>6}{'target':>10}{'avail':>10}{'repeat':>8}")
    for r in plan():
        print(f"{r['key']:<24}{r['weight']:>6.2f}{r['target_tokens']/1e6:>9.0f}M"
              f"{r['avail_tokens']/1e6:>9.0f}M{r['repeat']:>8.2f}x")

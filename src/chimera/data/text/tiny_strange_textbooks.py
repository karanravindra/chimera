"""
tiny-strange-textbooks BPE DataModule.

tiny-strange-textbooks (``nampdn-ai/tiny-strange-textbooks``, Apache-2.0) is the
scaled-up sibling of tiny-textbooks from the same author: 2.7M deduped synthetic
textbooks (~16GB raw) in the "Textbooks Are All You Need" / phi-1.5 register —
the same expository register the tinylm data-mix ablation showed drives the
knowledge/reasoning benchmarks, but ~6x the token count of tiny-textbooks and a
different synthesis pipeline for diversity. Gated:auto on the Hub (access
auto-granted; downloading needs an authenticated HF token), like tiny-textbooks.

The document text is the single ``text`` column. The repo ships only a ``train``
split (root-level ``data_part_N.parquet`` shards, ~200k docs each), so
``DATA_FILES`` bounds the download to the first four shards (~0.8B tokens, past a
tiny model's cap) and ``VAL_FROM_TRAIN`` carves validation off the head. All
machinery lives in :class:`chimera.data.hf_text.HFTextDataModule`.

Standalone it trains its own tokenizer on the text; mixed into a
:class:`chimera.data.ConcatTextDataModule` it adopts the mixture owner's
tokenizer instead.

Usage:
    dm = TinyStrangeTextbooksDataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule


class TinyStrangeTextbooksDataModule(HFTextDataModule):
    HF_REPO = "nampdn-ai/tiny-strange-textbooks"
    DIR_NAME = "tiny-strange-textbooks"
    TEXT_COLUMN = "text"
    UNIT = "doc"
    # First four of 14 root shards (~0.8B tokens) — covers a tiny model's cap
    # without pulling the full ~4.4GB / ~16GB-raw corpus.
    DATA_FILES = [
        "data_part_0.parquet",
        "data_part_1.parquet",
        "data_part_2.parquet",
        "data_part_3.parquet",
    ]
    VAL_FROM_TRAIN = 0.01

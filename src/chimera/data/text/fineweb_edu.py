"""
FineWeb-Edu BPE DataModule (mixture-ready).

FineWeb-Edu (``HuggingFaceFW/fineweb-edu``) is a large, high-quality English web
corpus for language-model pretraining — real-world knowledge and register that
the synthetic story/textbook corpora lack. This is the
:class:`chimera.data.hf_text.HFTextDataModule` variant, so it shares the mixture
owner's tokenizer and drops straight into a
:class:`chimera.data.ConcatTextDataModule` blend.

The ``sample-10BT`` config is ~28GB across ten ~2.1GB parquet shards (~1B tokens
each). A single shard already exceeds a tiny model's token budget, so
``DATA_FILES`` bounds the download to one shard rather than the whole config.
FineWeb-Edu ships only a ``train`` split, so ``VAL_FROM_TRAIN`` carves the
validation set off its head.

Usage:
    dm = FineWebEduTextDataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule


class FineWebEduTextDataModule(HFTextDataModule):
    HF_REPO = "HuggingFaceFW/fineweb-edu"
    DIR_NAME = "fineweb-edu"
    TEXT_COLUMN = "text"
    UNIT = "doc"
    # One ~2.1GB shard of the 10BT sample (~1B tokens) — plenty for a tiny
    # model's cap, without pulling the full ~28GB config.
    DATA_FILES = "sample/10BT/000_00000.parquet"
    VAL_FROM_TRAIN = 0.01

"""
Cosmopedia v2 BPE DataModule.

Cosmopedia v2 (the ``cosmopedia-v2`` config of ``HuggingFaceTB/smollm-corpus``,
ODC-BY) is the enhanced rebuild of Cosmopedia: ~39M synthetic textbooks, blog
posts, and stories generated for small-model pretraining. It is the same
"textbooks are all you need" register as tiny-textbooks but far larger and more
diverse — the axis the data-mix ablation showed drives the knowledge/reasoning
benchmarks (sciq, arc, piqa). All machinery lives in
:class:`chimera.data.hf_text.HFTextDataModule`.

The config is ~122GB across 104 ~1.2GB parquet shards (~270M tokens each), so
``DATA_FILES`` bounds the download to the first couple of shards (~0.5B tokens,
enough to cover a tiny model's cap). The corpus ships only a ``train`` split, so
``VAL_FROM_TRAIN`` carves the validation set off its head. The document text is
the ``text`` column (``prompt`` is the generation seed).

Standalone it trains its own tokenizer on the text; mixed into a
:class:`chimera.data.ConcatTextDataModule` it adopts the mixture owner's
tokenizer instead.

Usage:
    dm = CosmopediaV2DataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule

HF_REPO = "HuggingFaceTB/smollm-corpus"


class CosmopediaV2DataModule(HFTextDataModule):
    HF_REPO = HF_REPO
    DIR_NAME = "cosmopedia-v2"
    TEXT_COLUMN = "text"
    UNIT = "doc"
    # First two ~1.2GB shards (~0.5B tokens) of 104 — covers a tiny model's cap
    # without pulling the full ~122GB config.
    DATA_FILES = [
        "cosmopedia-v2/train-00000-of-00104.parquet",
        "cosmopedia-v2/train-00001-of-00104.parquet",
    ]
    VAL_FROM_TRAIN = 0.01

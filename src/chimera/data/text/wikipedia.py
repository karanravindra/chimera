"""
Wikipedia BPE DataModule.

Wikipedia (``wikimedia/wikipedia``, the community parquet rebuild, CC BY-SA) is
clean long-form expository/reference prose — the register the tinylm roadmap
wants for continued pretraining and for the tokenizer's "lifetime inputs"
corpus. All machinery lives in :class:`chimera.data.hf_text.HFTextDataModule`;
this class points it at the ``20231101.en`` snapshot.

The English snapshot is ~41 ~230MB parquet shards; ``DATA_FILES`` bounds the
download to the first shard (far beyond a tiny model's slice). The corpus ships
only a ``train`` split, so ``VAL_FROM_TRAIN`` carves validation off its head.

Usage:
    dm = WikipediaDataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule


class WikipediaDataModule(HFTextDataModule):
    HF_REPO = "wikimedia/wikipedia"
    DIR_NAME = "wikipedia"
    TEXT_COLUMN = "text"
    UNIT = "article"
    # first shard alone (~230MB) covers a tiny model's cap; skip the other 40
    DATA_FILES = ["20231101.en/train-00000-of-00041.parquet"]
    VAL_FROM_TRAIN = 0.01

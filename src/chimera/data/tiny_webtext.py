"""
tiny-webtext BPE DataModule.

tiny-webtext (``nampdn-ai/tiny-webtext``, MIT) is 4.5M records (~6GB raw) derived
from Falcon RefinedWeb and augmented with "critical thinking" reasoning — the
real-web-knowledge axis the synthetic textbook corpora lack, but with an
analytical register rather than raw crawl text. Gated:auto on the Hub.

Note the schema: this is NOT a single-column text corpus. Each row is a
``human`` prompt and a ``bot`` reasoned response (plus repeat-ratio / POS
metadata). For base pretraining the two turns are concatenated into one document
via :attr:`~chimera.data.hf_text.HFTextDataModule.TEXT_COLUMNS`, so the model
sees the prompt→reasoned-answer flow as running text. (It is closer to an
instruction/QA corpus than raw web text — worth weighing when deciding its share
of a base-pretraining mix.)

Files live under ``train/en/`` as many small shards; the whole English set is
only ~1.3GB, so ``DATA_FILES`` globs all of them (the token cap stops
tokenization early) and ``VAL_FROM_TRAIN`` carves validation off the head. All
machinery lives in :class:`chimera.data.hf_text.HFTextDataModule`.

Usage:
    dm = TinyWebTextDataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule


class TinyWebTextDataModule(HFTextDataModule):
    HF_REPO = "nampdn-ai/tiny-webtext"
    DIR_NAME = "tiny-webtext"
    # No single text column: join the prompt and reasoned response per row.
    TEXT_COLUMNS = ["human", "bot"]
    TEXT_JOIN = "\n\n"
    UNIT = "doc"
    # All English shards (~1.3GB total — smaller than one FineWeb shard); the
    # token cap bounds how much is actually tokenized.
    DATA_FILES = "train/en/*.parquet"
    VAL_FROM_TRAIN = 0.01

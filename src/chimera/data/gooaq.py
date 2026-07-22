"""
GooAQ BPE DataModule.

GooAQ (``sentence-transformers/gooaq``, the cleaned 3M-pair export of
allenai/gooaq, apache-2.0) is naturalistic short-form QA mined from Google
autocomplete questions and answer-box answers. Its role in a pretrain mix is
response FORMAT: direct question -> direct answer, unlike the continuous prose
of the textbook/web sources. Each pair is rendered as one document::

    Question: {question}
    Answer: {answer}

All machinery lives in :class:`chimera.data.hf_text.HFTextDataModule`.
``DATA_FILES`` bounds the download to the first parquet shard (still far beyond
a tiny model's slice); the corpus ships only a ``train`` split, so
``VAL_FROM_TRAIN`` carves validation off its head.

Usage:
    dm = GooAQDataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule


class GooAQDataModule(HFTextDataModule):
    HF_REPO = "sentence-transformers/gooaq"
    DIR_NAME = "gooaq"
    UNIT = "pair"
    # first shard alone covers a tiny model's cap; skip the rest of the config
    DATA_FILES = ["pair/train-00000-of-00002.parquet"]
    VAL_FROM_TRAIN = 0.01

    def _row_text(self, row) -> str:
        return f"Question: {row['question']}\nAnswer: {row['answer']}"

    def iter_texts(self, ds, batch_size: int = 1024):
        for batch in ds.iter(batch_size=batch_size):
            for q, a in zip(batch["question"], batch["answer"]):
                yield f"Question: {q}\nAnswer: {a}"

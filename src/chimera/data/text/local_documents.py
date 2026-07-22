"""
LocalDocumentsDataModule — fold a directory of local text/markdown files into
a text mixture as an always-on source.

Each file is one document; the train stream is the documents repeated
``repeat`` times (a handful of tiny files would otherwise be invisible next to
hundreds of millions of web tokens), the val stream is one copy (val loss on
it measures how well the model absorbed the documents). Cache keys include a
content hash of the files plus ``repeat``, so editing a document or changing
the repeat forces a consistent rebuild instead of reusing a stale ids cache.

The source is excluded from a ConcatTextDataModule mixture-trained tokenizer
(sample and cache key) — a few KB of docs shouldn't perturb the vocab, and
excluding them keeps the existing blended tokenizer + every other source's ids
caches valid when the docs are added or edited.
"""

import hashlib
from pathlib import Path

from .hf_text import HFTextDataModule


class LocalDocumentsDataModule(HFTextDataModule):
    TEXT_COLUMN = "text"
    UNIT = "doc"

    # keep the blended-mixture tokenizer (path key + training sample) unchanged
    exclude_from_mixture_tokenizer = True

    def __init__(self, doc_dir: str, repeat: int = 1, glob: str = "*.md", **kwargs):
        kwargs.setdefault("max_train_tokens", None)
        kwargs.setdefault("max_val_tokens", None)
        self.doc_dir = Path(doc_dir)
        self.glob = glob
        self.repeat = repeat
        self.files = sorted(self.doc_dir.glob(glob))
        assert self.files, f"no files matching {glob!r} in {self.doc_dir}"
        h = hashlib.blake2b(digest_size=6)
        for f in self.files:
            h.update(f.name.encode())
            h.update(f.read_bytes())
        # content hash + repeat in the cache dir name => edits rebuild caches
        self.DIR_NAME = f"documents-{h.hexdigest()}-r{repeat}"
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "documents"

    def _load_dataset(self, split: str):
        from datasets import Dataset

        texts = [f.read_text() for f in self.files]
        if split == self.TRAIN_SPLIT:
            texts = texts * self.repeat
        return Dataset.from_dict({"text": texts})

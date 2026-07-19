"""
ConcatTextDataModule — mix any HFTextDataModule sources into one token stream.

Composes submodules instead of subclassing per combination: pass a list of
:class:`chimera.data.hf_text.HFTextDataModule` instances and this module
concatenates their flat token streams into a single training stream. The
DataLoader's chunk-level shuffle then interleaves the sources within every
epoch, so no separate sampling/weighting machinery is needed — a source's
weight is simply how many tokens it contributes (cap per source via its own
``max_train_tokens``).

Tokenizer: the FIRST submodule owns it. It trains/loads its tokenizer exactly
as it would standalone; every other submodule adopts it via
``set_shared_tokenizer`` and keys its ids caches on the owner's fingerprint,
so one vocab spans the whole mixture and no cache can pair with the wrong
tokenizer. Put the source whose text should define the vocab first.

Document convention (EOS/BOS wrapping) comes from each submodule's own
``add_eos`` / ``add_bos`` — they must agree across sources (asserted) so the
doc mask and per-document position reset see one consistent stream. Every
source stream ends on a document boundary, so the seams are clean.

Validation: ``val_dataloader()`` serves the concatenation of all sources' val
streams (the overall mixture val); per-source loaders are available via
``val_dataloaders_by_source()`` and per-source datasets via ``val_datasets``.

Usage:
    dm = ConcatTextDataModule(
        [
            TinyStoriesV2DataModule(data_dir=..., vocab_size=16_384, add_bos=True),
            TinyTextbooksDataModule(data_dir=..., add_bos=True),
        ],
        batch_size=128,
    )
    dm.prepare_data(); dm.setup("fit")
    dm.source_train_tokens  # {"tinystories-v2": ..., "tiny-textbooks": ...}
"""

import hashlib
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

import lightning as pl

from chimera.tokenizers import BPETokenizer

from ._text import TokenDataset
from .chat_template import SPECIAL_TOKENS
from .hf_text import HFTextDataModule


class ConcatTextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datamodules: Sequence[HFTextDataModule],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        train_tokenizer_on_mixture: bool = False,
        tokenizer_sample_chars: int = 200_000_000,
    ):
        super().__init__()
        assert len(datamodules) >= 1, "need at least one submodule"
        self.datamodules = list(datamodules)
        # When set, train ONE tokenizer on a sample drawn round-robin across all
        # sources (see setup) instead of letting the first source own it — so
        # the vocab compresses the whole mixture, not just the owner's register.
        self.train_tokenizer_on_mixture = train_tokenizer_on_mixture
        self.tokenizer_sample_chars = tokenizer_sample_chars

        owner = self.datamodules[0]
        self.seq_len = owner.seq_len
        # loader knobs default to the tokenizer owner's settings
        self.batch_size = batch_size if batch_size is not None else owner.batch_size
        self.num_workers = num_workers if num_workers is not None else owner.num_workers
        self.pin_memory = pin_memory if pin_memory is not None else owner.pin_memory

        for dm in self.datamodules[1:]:
            assert dm.seq_len == owner.seq_len, (
                f"seq_len mismatch: {dm.name}={dm.seq_len} vs {owner.name}={owner.seq_len}"
            )
            assert (dm.add_eos, dm.add_bos) == (owner.add_eos, owner.add_bos), (
                f"document convention mismatch on {dm.name}: all sources must "
                "use the same add_eos/add_bos so the stream has one convention"
            )

        # unique per-source keys, in submodule order (suffix duplicates)
        names: list[str] = []
        for dm in self.datamodules:
            n, k = dm.name, 2
            while n in names:
                n, k = f"{dm.name}-{k}", k + 1
            names.append(n)
        self.source_names = names

        # proxied from the tokenizer owner in setup()
        self.tokenizer = None
        self.vocab_size: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.bos_id: Optional[int] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.val_datasets: dict[str, Dataset] = {}
        self.source_train_tokens: dict[str, int] = {}

    def prepare_data(self):
        for dm in self.datamodules:
            dm.prepare_data()

    def _blended_tokenizer_path(self):
        """Cache path for the mixture-trained tokenizer.

        Keyed on the source set, backend, vocab, and sample size, so any change
        to what the tokenizer saw trains/loads a different file rather than
        silently reusing a mismatched vocab (same discipline as the per-source
        tokenizer paths in :class:`HFTextDataModule`).
        """
        owner = self.datamodules[0]
        names = "+".join(sorted(self.source_names))
        key = hashlib.blake2b(names.encode(), digest_size=6).hexdigest()
        tag = f"{owner.tokenizer_backend}_v{owner.vocab_size}_c{self.tokenizer_sample_chars}"
        return owner.data_dir / "mixture_tokenizers" / f"tok_{tag}_{key}.json"

    def _train_or_load_blended_tokenizer(self):
        """Train (or load) one tokenizer on a round-robin sample of all sources.

        Each source contributes an equal char budget so no single register
        dominates the vocab. Returns ``(tokenizer, fingerprint)`` ready to hand
        every submodule via ``set_shared_tokenizer``.
        """
        owner = self.datamodules[0]
        path = self._blended_tokenizer_path()
        if not path.exists():
            per_source = max(1, self.tokenizer_sample_chars // len(self.datamodules))
            docs: list[str] = []
            for dm in self.datamodules:
                ds = dm._load_dataset(dm.TRAIN_SPLIT)
                total = 0
                for text in dm.iter_texts(ds):
                    docs.append(text)
                    total += len(text)
                    if total >= per_source:
                        break
            tok = BPETokenizer(backend=owner.tokenizer_backend)
            # Train on the list of documents as a MULTI-element iterator — HF
            # tokenizers chunks + parallelizes it, keeping trainer memory bounded.
            # NOT one "\n".join(...) mega-string: a single giant element makes the
            # Rust trainer hold the whole pre-tokenized corpus at once (OOM + hours).
            # The canonical chat/tool special tokens stay reserved at fixed low ids
            # so this base vocab carries unchanged into chat/tool-call SFT later.
            tok.train(
                docs,
                vocab_size=owner.vocab_size,
                special_tokens=SPECIAL_TOKENS,
            )
            del docs
            path.parent.mkdir(parents=True, exist_ok=True)
            tok.save(path)
        else:
            tok = BPETokenizer.load(path, backend=owner.tokenizer_backend)
        fingerprint = hashlib.blake2b(path.read_bytes(), digest_size=6).hexdigest()
        return tok, fingerprint

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        owner = self.datamodules[0]
        if self.train_tokenizer_on_mixture and owner.tokenizer_backend != "pretrained":
            # One vocab trained on the whole mixture: build it first, then hand
            # every source (owner included) the same tokenizer + fingerprint.
            tokenizer, fingerprint = self._train_or_load_blended_tokenizer()
            for dm in self.datamodules:
                dm.set_shared_tokenizer(tokenizer, fingerprint)
                dm.setup(stage)
        else:
            # Owner first: it resolves the tokenizer everything else encodes with.
            owner.setup(stage)
            fingerprint = owner._tokenizer_fingerprint()
            for dm in self.datamodules[1:]:
                dm.set_shared_tokenizer(owner.tokenizer, fingerprint)
                dm.setup(stage)

        self.tokenizer = owner.tokenizer
        self.vocab_size = owner.vocab_size
        self.eos_id = owner.eos_id
        self.bos_id = owner.bos_id

        # Concatenate the train streams; every source stream ends on a document
        # boundary (EOS), so each seam is a clean doc break and the chunk-level
        # shuffle interleaves sources within every epoch.
        train_streams = [dm.train_dataset.data for dm in self.datamodules]
        self.source_train_tokens = {
            name: len(s) for name, s in zip(self.source_names, train_streams)
        }
        self.train_dataset = TokenDataset(torch.cat(train_streams), self.seq_len)

        self.val_datasets = {
            name: dm.val_dataset
            for name, dm in zip(self.source_names, self.datamodules)
        }
        self.val_dataset = TokenDataset(
            torch.cat([dm.val_dataset.data for dm in self.datamodules]), self.seq_len
        )

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def _val_dl(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return self._val_dl(self.val_dataset)

    def val_dataloaders_by_source(self) -> dict[str, DataLoader]:
        """One val loader per source, for per-domain val metrics."""
        return {name: self._val_dl(ds) for name, ds in self.val_datasets.items()}


if __name__ == "__main__":
    import os

    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    from .tiny_textbooks import TinyTextbooksDataModule
    from .tinystories_v2 import TinyStoriesV2DataModule

    dm = ConcatTextDataModule(
        [
            TinyStoriesV2DataModule(data_dir="/mnt/ai/data", add_bos=True),
            TinyTextbooksDataModule(data_dir="/mnt/ai/data", add_bos=True),
        ],
        batch_size=8,
    )
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}  sources={dm.source_train_tokens}")
    print(f"train batch: x={x.shape}, y={y.shape}")

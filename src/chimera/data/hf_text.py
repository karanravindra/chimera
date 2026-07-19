"""
Generic HF text-corpus BPE DataModule base.

Machinery shared by the flat-token-stream text DataModules (TinyStoriesV2,
tiny-textbooks, ...): download a Hugging Face dataset, train (or load) a
byte-level BPE tokenizer on its text, tokenize each split into one flat token
stream with per-document EOS/BOS markers, cache the streams keyed on the
tokenizer's content hash, and serve non-overlapping next-token ``(input,
target)`` chunks.

Subclasses configure the dataset via class attributes::

    class TinyStoriesV2DataModule(HFTextDataModule):
        HF_REPO = "noanabeshima/TinyStoriesV2"
        DIR_NAME = "tinystories-v2"   # cache dir under data_dir + tqdm label
        TEXT_COLUMN = "text"          # column holding the document text
        VAL_SPLIT = "validation"      # HF split used for validation
        UNIT = "story"                # tqdm unit

Tokenizer sharing (for :class:`chimera.data.ConcatTextDataModule`): a module
normally trains/loads its own tokenizer in ``setup()``, but
``set_shared_tokenizer(tok, fingerprint)`` lets a mixture hand every source the
same tokenizer — the module then tokenizes its text with it and keys its ids
caches on the shared fingerprint, so one vocab spans the whole mixture and
caches can never pair with the wrong tokenizer.

The ``hf`` tokenizer backend is trained with the canonical chat/tool special
tokens (:data:`chimera.data.chat_template.SPECIAL_TOKENS`) reserved at fixed
low ids, so a base tokenizer carries straight into chat / tool-call SFT
without a vocab change. Documents are concatenated into a single stream with
an end-of-document token (``eos_token``) appended after each one; ``add_bos``
additionally prepends a start token so every document begins with an explicit
anchor at position 0. Both require a tokenizer that defines the token — the
``hf`` and ``pretrained`` backends do; the from-scratch byte-level BPE has
none, so they are no-ops there.
"""

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

import lightning as pl

from chimera.tokenizers import BPETokenizer

from ._text import (
    TokenDataset,
    load_cached_ids,
    save_cached_ids,
    tokenize_with_progress,
)
from .chat_template import BOS, EOS, SPECIAL_TOKENS


class HFTextDataModule(pl.LightningDataModule):
    HF_REPO: str
    DIR_NAME: str
    TEXT_COLUMN: str = "text"
    # Corpora with no single text column (e.g. human/bot QA pairs) instead set
    # TEXT_COLUMNS to join those fields into one document per row, TEXT_JOIN
    # between them. None => use the single TEXT_COLUMN.
    TEXT_COLUMNS: Optional[Sequence[str]] = None
    TEXT_JOIN: str = "\n\n"
    TRAIN_SPLIT: str = "train"
    VAL_SPLIT: str = "validation"
    UNIT: str = "doc"

    # HF dataset config passed as ``name=`` (e.g. FineWeb-Edu's "sample-10BT").
    CONFIG_NAME: Optional[str] = None
    # Repo-relative parquet file(s) to build from, passed as ``data_files=``.
    # Bounds the download to a few shards instead of pulling a whole config —
    # essential for the multi-billion-token web corpora, where a couple of
    # shards already exceed a small model's token budget.
    DATA_FILES = None
    # Fraction of the train split carved off its head for validation, for
    # corpora with no native validation split. None => use VAL_SPLIT verbatim.
    VAL_FROM_TRAIN: Optional[float] = None

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        seq_len: int = 512,
        vocab_size: int = 8192,
        tokenizer_backend: str = "hf",
        pretrained_id: str = "LiquidAI/LFM2.5-230M",
        add_eos: bool = True,
        eos_token: str = EOS,
        add_bos: bool = False,
        bos_token: str = BOS,
        max_train_tokens: Optional[int] = 10_000_000,
        max_val_tokens: Optional[int] = 1_000_000,
        tokenizer_train_chars: int = 5_000_000,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tokenizer_backend = tokenizer_backend
        self.pretrained_id = pretrained_id
        self.add_eos = add_eos
        self.eos_token = eos_token
        self.add_bos = add_bos
        self.bos_token = bos_token
        self.max_train_tokens = max_train_tokens
        self.max_val_tokens = max_val_tokens
        self.tokenizer_train_chars = tokenizer_train_chars
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer: Optional[BPETokenizer] = None
        # set via set_shared_tokenizer(); overrides the own-file fingerprint
        self._shared_fingerprint: Optional[str] = None
        # ids of the document separator / start tokens, resolved in setup()
        self.eos_id: Optional[int] = None
        self.bos_id: Optional[int] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def name(self) -> str:
        """Short per-source key (mixture labels, per-source metrics)."""
        return self.DIR_NAME

    @property
    def _dir(self) -> Path:
        return self.data_dir / self.DIR_NAME

    @property
    def _tokenizer_path(self) -> Path:
        """Tokenizer cache path, keyed on everything that defines the tokenizer.

        A trained tokenizer depends on its backend, target vocab, and how much
        text it saw; a pretrained one on its Hub id. Encoding those in the
        filename means changing any of them trains/loads a *different* file
        rather than silently reusing a mismatched tokenizer.
        """
        hp = self.hparams
        if hp.tokenizer_backend == "pretrained":
            tag = "pretrained_" + hp.pretrained_id.replace("/", "_")
        else:
            tag = f"{hp.tokenizer_backend}_v{hp.vocab_size}_c{hp.tokenizer_train_chars}"
        return self._dir / f"tokenizer_{tag}.json"

    def set_shared_tokenizer(self, tokenizer: BPETokenizer, fingerprint: str):
        """Adopt an externally owned tokenizer (see ConcatTextDataModule).

        Must be called before ``setup()``. The module then skips training its
        own tokenizer and keys its ids caches on ``fingerprint`` (the owner's
        content hash) so streams stay bound to the tokenizer that made them.
        """
        assert self.train_dataset is None, "set_shared_tokenizer must precede setup()"
        self.tokenizer = tokenizer
        self._shared_fingerprint = fingerprint

    def _tokenizer_fingerprint(self) -> str:
        """Short content hash of the tokenizer in use.

        The ids caches are keyed on this so a token stream can never be paired
        with a different tokenizer than the one that produced it: any change to
        the tokenizer yields a new fingerprint -> a new ids path -> a forced,
        consistent rebuild. (This is exactly the desync that silently corrupted
        validation — the val ids were tokenized with a tokenizer that was later
        retrained, and nothing forced them to rebuild.)
        """
        if self._shared_fingerprint is not None:
            return self._shared_fingerprint
        return hashlib.blake2b(
            self._tokenizer_path.read_bytes(), digest_size=6
        ).hexdigest()

    def _ids_path(
        self, split: str, max_tokens: Optional[int], fingerprint: str
    ) -> Path:
        """Path of the cached flat token stream for ``split`` and this config.

        Bound to ``fingerprint`` (the tokenizer's content hash) so ids and
        tokenizer can never drift apart.
        """
        eos = "eos" if self.hparams.add_eos else "noeos"
        # only tagged when on, so pre-existing (no-bos) caches keep matching
        bos = "_bos" if self.hparams.add_bos else ""
        cap = "all" if max_tokens is None else str(max_tokens)
        return self._dir / f"ids_{split}_tok-{fingerprint}_{eos}{bos}_{cap}.pt"

    def _row_text(self, row) -> str:
        """The document text for one row (single column, or joined columns)."""
        if self.TEXT_COLUMNS is None:
            return row[self.TEXT_COLUMN]
        return self.TEXT_JOIN.join(str(row[c]) for c in self.TEXT_COLUMNS)

    def iter_texts(self, ds, batch_size: int = 1024):
        """Yield document strings from a HF dataset, reading columns in batches.

        Batched column reads are far faster than per-row access on the Arrow
        dataset; handles both the single-column and joined-columns cases.
        """
        for batch in ds.iter(batch_size=batch_size):
            if self.TEXT_COLUMNS is None:
                yield from batch[self.TEXT_COLUMN]
            else:
                cols = [batch[c] for c in self.TEXT_COLUMNS]
                for parts in zip(*cols):
                    yield self.TEXT_JOIN.join(str(p) for p in parts)

    def _hf_split(self, split: str) -> str:
        """Translate a logical split name to the HF split spec to load.

        Logical names (:attr:`TRAIN_SPLIT` / :attr:`VAL_SPLIT`) are what the ids
        caches and progress labels are keyed on. When :attr:`VAL_FROM_TRAIN` is
        set the corpus has no native validation split, so validation is carved
        off the head of the train split and train takes the remainder — while
        the cache/label names stay clean.
        """
        if self.VAL_FROM_TRAIN is None:
            return split
        pct = max(1, round(self.VAL_FROM_TRAIN * 100))
        if split == self.VAL_SPLIT:
            return f"{self.TRAIN_SPLIT}[:{pct}%]"
        if split == self.TRAIN_SPLIT:
            return f"{self.TRAIN_SPLIT}[{pct}%:]"
        return split

    def _load_dataset(self, split: str):
        from datasets import load_dataset

        kwargs = {"cache_dir": str(self.data_dir / "hf_cache")}
        if self.DATA_FILES is not None:
            kwargs["data_files"] = self.DATA_FILES
        if self.CONFIG_NAME is not None:
            kwargs["name"] = self.CONFIG_NAME
        return load_dataset(
            self.HF_REPO,
            split=self._hf_split(split),
            **kwargs,
        )

    def prepare_data(self):
        # download + cache only, called once on a single process
        self._dir.mkdir(parents=True, exist_ok=True)
        # The ids caches are keyed on the tokenizer's content hash, so we can
        # only match them once the tokenizer exists (own file or shared). If
        # both streams are already cached for it, skip the download entirely;
        # otherwise fetch the raw data (needed to train the tokenizer and/or
        # tokenize the splits).
        if self._shared_fingerprint is not None or self._tokenizer_path.exists():
            fp = self._tokenizer_fingerprint()
            if (
                self._ids_path(self.TRAIN_SPLIT, self.max_train_tokens, fp).exists()
                and self._ids_path(self.VAL_SPLIT, self.max_val_tokens, fp).exists()
            ):
                return
        self._load_dataset(self.TRAIN_SPLIT)
        self._load_dataset(self.VAL_SPLIT)

    def _load_or_train_tokenizer(self) -> BPETokenizer:
        if self._tokenizer_path.exists():
            return BPETokenizer.load(
                self._tokenizer_path, backend=self.tokenizer_backend
            )

        if self.tokenizer_backend == "pretrained":
            # fixed tokenizer from the Hub (e.g. LiquidAI/LFM2.5-230M); no training
            tok = BPETokenizer.from_pretrained(self.pretrained_id)
            tok.save(self._tokenizer_path)
            return tok

        # concatenate train rows until we have enough characters to train on
        ds = self._load_dataset(self.TRAIN_SPLIT)
        parts: list[str] = []
        total = 0
        for row in ds:
            text = self._row_text(row)
            parts.append(text)
            total += len(text)
            if total >= self.tokenizer_train_chars:
                break

        tok = BPETokenizer(backend=self.tokenizer_backend)
        # Reserve the canonical chat/tool special tokens at fixed low ids so this
        # base tokenizer can be reused unchanged for chat/tool-call SFT later
        # (only the ``hf`` backend honors these; ``scratch`` ignores them).
        tok.train(
            "\n".join(parts),
            vocab_size=self.vocab_size,
            special_tokens=SPECIAL_TOKENS,
        )
        tok.save(self._tokenizer_path)
        return tok

    def _resolve_special_id(self, enabled: bool, token: str) -> Optional[int]:
        """Id of a named special token, or None if disabled/unavailable.

        Only the fast (``hf`` / ``pretrained``) backends carry named special
        tokens; a from-scratch byte-level BPE has none, so ``add_eos`` /
        ``add_bos`` are no-ops there.
        """
        if not enabled:
            return None
        tok = self.tokenizer._tok if self.tokenizer is not None else None
        if tok is None:
            return None
        return tok.token_to_id(token)

    def _build_split(
        self, split: str, max_tokens: Optional[int], fingerprint: str
    ) -> torch.Tensor:
        """Return the cached (or freshly tokenized) token stream for ``split``."""
        ids_path = self._ids_path(split, max_tokens, fingerprint)
        data = load_cached_ids(ids_path)
        if data is not None:
            return data

        ds = self._load_dataset(split)

        # tokenize_with_progress batches these for encode_batch and stops early
        # once max_tokens is reached.
        ids = tokenize_with_progress(
            self.tokenizer,
            self.iter_texts(ds),
            desc=f"Tokenizing {self.DIR_NAME} [{split}]",
            total=len(ds),
            unit=self.UNIT,
            eos_id=self.eos_id,
            bos_id=self.bos_id,
            max_tokens=max_tokens,
        )
        # int16 (vocab << 32767) not int64: 4x less RAM for the stream, which
        # matters once several capped sources concatenate into one pool. Datasets
        # cast per-item via TokenDataset.__getitem__().long().
        data = torch.tensor(ids, dtype=torch.int16)
        save_cached_ids(ids_path, data)
        return data

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        if self.tokenizer is None:
            self.tokenizer = self._load_or_train_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_id = self._resolve_special_id(self.add_eos, self.eos_token)
        self.bos_id = self._resolve_special_id(self.add_bos, self.bos_token)
        if self.add_eos and self.eos_id is None:
            print(
                f"add_eos=True but tokenizer has no {self.eos_token!r} token "
                f"(backend={self.tokenizer_backend!r}); documents will be "
                "concatenated without a separator."
            )
        if self.add_bos and self.bos_id is None:
            print(
                f"add_bos=True but tokenizer has no {self.bos_token!r} token "
                f"(backend={self.tokenizer_backend!r}); documents will start "
                "without a start marker."
            )

        # Bind both streams to this exact tokenizer via its content hash.
        fingerprint = self._tokenizer_fingerprint()
        train_data = self._build_split(
            self.TRAIN_SPLIT, self.max_train_tokens, fingerprint
        )
        val_data = self._build_split(self.VAL_SPLIT, self.max_val_tokens, fingerprint)
        self.train_dataset = TokenDataset(train_data, self.seq_len)
        self.val_dataset = TokenDataset(val_data, self.seq_len)

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

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

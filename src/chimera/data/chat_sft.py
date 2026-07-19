"""
Chat-SFT DataModules: HF chat/QA datasets -> packed, loss-masked token streams.

The SFT counterpart of :class:`chimera.data.hf_text.HFTextDataModule`. Each row
becomes a ChatML conversation (:func:`chimera.data.chat_template.render_masked`),
encoded with a FIXED tokenizer loaded from disk (the pretrain vocab — SFT never
retrains it; the chat special tokens are already reserved at fixed low ids).
Conversations are packed into one flat ids stream plus a parallel labels stream
(token id on supervised/assistant positions, -100 elsewhere), separated by EOS
so FlexAttention document masking sees one conversation per document. Served as
non-overlapping ``(input, shifted-labels)`` chunks via
:class:`chimera.data._text.MaskedTokenDataset`.

Streams are cached keyed on the tokenizer's content hash + caps (same
discipline as the pretrain ids caches). Subclasses configure the dataset via
class attributes (HF_REPO / DIR_NAME / splits, as in HFTextDataModule) and a
``row_to_messages`` hook::

    class GooAQChatDataModule(ChatSFTDataModule):
        HF_REPO = "sentence-transformers/gooaq"
        ...
        def row_to_messages(self, row):
            return [{"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]}]

Mixing: instantiate several modules with the same tokenizer_path and
concatenate their (ids, labels) streams — every stream ends on an EOS boundary.
"""

import hashlib
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

import lightning as pl

from tqdm import tqdm

from chimera.tokenizers import BPETokenizer

from ._text import MaskedTokenDataset, load_cached_ids, save_cached_ids
from .chat_template import BOS, EOS, render_masked


class ChatSFTDataModule(pl.LightningDataModule):
    HF_REPO: str
    DIR_NAME: str
    TRAIN_SPLIT: str = "train"
    VAL_SPLIT: str = "validation"
    UNIT: str = "conv"
    CONFIG_NAME: Optional[str] = None
    DATA_FILES = None
    # carve validation off the head of train when there is no native val split
    VAL_FROM_TRAIN: Optional[float] = None

    def __init__(
        self,
        tokenizer_path: str,
        data_dir: str = "./data",
        batch_size: int = 64,
        seq_len: int = 512,
        max_train_tokens: Optional[int] = 10_000_000,
        max_val_tokens: Optional[int] = 500_000,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer_path = Path(tokenizer_path)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_train_tokens = max_train_tokens
        self.max_val_tokens = max_val_tokens
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer: Optional[BPETokenizer] = None
        self.vocab_size: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.bos_id: Optional[int] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    # -- subclass hook -------------------------------------------------------
    def row_to_messages(self, row) -> list[dict]:
        raise NotImplementedError

    # -- shared machinery ----------------------------------------------------
    @property
    def name(self) -> str:
        return self.DIR_NAME

    @property
    def _dir(self) -> Path:
        return self.data_dir / self.DIR_NAME

    def _fingerprint(self) -> str:
        return hashlib.blake2b(
            self.tokenizer_path.read_bytes(), digest_size=6
        ).hexdigest()

    def _stream_path(self, split: str, max_tokens: Optional[int]) -> Path:
        cap = "all" if max_tokens is None else str(max_tokens)
        return self._dir / f"sft_{split}_tok-{self._fingerprint()}_{cap}.pt"

    def _hf_split(self, split: str) -> str:
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
            kwargs["verification_mode"] = "no_checks"
        if self.CONFIG_NAME is not None:
            kwargs["name"] = self.CONFIG_NAME
        return load_dataset(self.HF_REPO, split=self._hf_split(split), **kwargs)

    def prepare_data(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        if (
            self._stream_path(self.TRAIN_SPLIT, self.max_train_tokens).exists()
            and self._stream_path(self.VAL_SPLIT, self.max_val_tokens).exists()
        ):
            return
        self._load_dataset(self.TRAIN_SPLIT)
        self._load_dataset(self.VAL_SPLIT)

    def _build_split(self, split: str, max_tokens: Optional[int]):
        path = self._stream_path(split, max_tokens)
        cached = load_cached_ids(path)
        if cached is not None:
            return cached[0], cached[1]

        enc = self.tokenizer._tok
        encode = lambda text: enc.encode(text, add_special_tokens=False).ids  # noqa: E731

        ds = self._load_dataset(split)
        ids: list[int] = []
        labels: list[int] = []
        pbar = tqdm(ds, desc=f"Rendering {self.DIR_NAME} [{split}]", unit=self.UNIT)
        for row in pbar:
            conv_ids, conv_mask = render_masked(
                self.row_to_messages(row), encode, eos_id=self.eos_id
            )
            conv_ids = [self.bos_id] + conv_ids
            conv_mask = [0] + conv_mask
            ids.extend(conv_ids)
            labels.extend(i if m else -100 for i, m in zip(conv_ids, conv_mask))
            pbar.set_postfix(tokens=len(ids))
            if max_tokens is not None and len(ids) >= max_tokens:
                break

        # int32 not int16: labels carry -100 alongside ids up to the vocab size
        data = torch.stack(
            [torch.tensor(ids, dtype=torch.int32), torch.tensor(labels, dtype=torch.int32)]
        )
        save_cached_ids(path, data)
        return data[0], data[1]

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return
        self.tokenizer = BPETokenizer.load(self.tokenizer_path, backend="hf")
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_id = self.tokenizer._tok.token_to_id(EOS)
        self.bos_id = self.tokenizer._tok.token_to_id(BOS)
        assert self.eos_id is not None and self.bos_id is not None, (
            "SFT tokenizer must reserve the BOS/EOS special tokens"
        )

        tr_ids, tr_labels = self._build_split(self.TRAIN_SPLIT, self.max_train_tokens)
        va_ids, va_labels = self._build_split(self.VAL_SPLIT, self.max_val_tokens)
        self.train_dataset = MaskedTokenDataset(tr_ids, tr_labels, self.seq_len)
        self.val_dataset = MaskedTokenDataset(va_ids, va_labels, self.seq_len)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def _dl(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._dl(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_dataset, shuffle=False)


class GooAQChatDataModule(ChatSFTDataModule):
    """Closed-book simple QA: GooAQ pairs as single-turn chat."""

    HF_REPO = "sentence-transformers/gooaq"
    DIR_NAME = "gooaq-chat"
    UNIT = "pair"
    DATA_FILES = ["pair/train-00000-of-00002.parquet"]
    VAL_FROM_TRAIN = 0.01

    def row_to_messages(self, row) -> list[dict]:
        return [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]


class SQuADChatDataModule(ChatSFTDataModule):
    """Grounded QA: passage + question in the user turn, span answer back."""

    HF_REPO = "rajpurkar/squad"
    DIR_NAME = "squad-chat"
    UNIT = "qa"

    def row_to_messages(self, row) -> list[dict]:
        return [
            {"role": "user", "content": f"{row['context']}\n\n{row['question']}"},
            {"role": "assistant", "content": row["answers"]["text"][0]},
        ]


class EverydayConversationsDataModule(ChatSFTDataModule):
    """Multi-turn small-talk (smoltalk/everyday-conversations): chat style."""

    HF_REPO = "HuggingFaceTB/smoltalk"
    DIR_NAME = "everyday-conversations"
    CONFIG_NAME = "everyday-conversations"
    VAL_SPLIT = "test"

    def row_to_messages(self, row) -> list[dict]:
        return row["messages"]

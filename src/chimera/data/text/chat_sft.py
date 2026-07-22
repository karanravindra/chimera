"""
Chat-SFT DataModules: HF chat/QA datasets -> packed, loss-masked token streams.

The SFT counterpart of :class:`chimera.data.text.hf_text.HFTextDataModule`. Each row
becomes a ChatML conversation (:func:`chimera.data.text.chat_template.render_masked`),
encoded with a FIXED tokenizer loaded from disk (the pretrain vocab — SFT never
retrains it; the chat special tokens are already reserved at fixed low ids).
Conversations are packed into one flat ids stream plus a parallel labels stream
(token id on supervised/assistant positions, -100 elsewhere), separated by EOS
so FlexAttention document masking sees one conversation per document. Served as
non-overlapping ``(input, shifted-labels)`` chunks via
:class:`chimera.data.text.datasets.MaskedTokenDataset`.

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

from tqdm import tqdm

from chimera.tokenizers import BPETokenizer

from ._hf_base import _HFCorpusBase
from .datasets import MaskedTokenDataset
from .chat_template import BOS, EOS, render_masked


class ChatSFTDataModule(_HFCorpusBase):
    UNIT: str = "conv"

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
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            seq_len=seq_len,
            max_train_tokens=max_train_tokens,
            max_val_tokens=max_val_tokens,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.save_hyperparameters()
        self.tokenizer_path = Path(tokenizer_path)

    # -- subclass hook -------------------------------------------------------
    def row_to_messages(self, row) -> list[dict]:
        raise NotImplementedError

    # -- tokenizer -----------------------------------------------------------
    def _tokenizer_fingerprint(self) -> str:
        return hashlib.blake2b(
            self.tokenizer_path.read_bytes(), digest_size=6
        ).hexdigest()

    def _load_tokenizer(self):
        self.tokenizer = BPETokenizer.load(self.tokenizer_path, backend="hf")
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_id = self.tokenizer._tok.token_to_id(EOS)
        self.bos_id = self.tokenizer._tok.token_to_id(BOS)
        if self.eos_id is None or self.bos_id is None:
            raise ValueError("SFT tokenizer must reserve the BOS/EOS special tokens")

    def _prepare_tokenizer(self):
        self._load_tokenizer()

    def _setup_tokenizer(self):
        if self.tokenizer is None:
            self._load_tokenizer()

    # -- rendering fingerprint -----------------------------------------------
    def _renderer_methods(self):
        return (type(self).row_to_messages, render_masked)

    # -- cache / tokenize / datasets -----------------------------------------
    def _cache_path(self, split: str, max_tokens: Optional[int]) -> Path:
        cap = "all" if max_tokens is None else str(max_tokens)
        return self._dir / (
            f"sft_v2_{split}_src-{self._dataset_fingerprint()}_"
            f"tok-{self._tokenizer_fingerprint()}_{cap}.pt"
        )

    def _tokenize_split(self, ds, split: str, max_tokens: Optional[int]):
        enc = self.tokenizer._tok
        encode = lambda text: enc.encode(text, add_special_tokens=False).ids  # noqa: E731

        ids: list[int] = []
        labels: list[int] = []
        pbar = tqdm(ds, desc=f"Rendering {self.DIR_NAME} [{split}]", unit=self.UNIT)
        for row in pbar:
            conv_ids, conv_mask = render_masked(
                self.row_to_messages(row), encode, eos_id=self.eos_id
            )
            conv_ids = [self.bos_id] + conv_ids
            conv_mask = [0] + conv_mask
            conv_labels = [i if m else -100 for i, m in zip(conv_ids, conv_mask)]
            if max_tokens is not None and len(ids) + len(conv_ids) > max_tokens:
                remaining = max_tokens - len(ids)
                if not ids and remaining > 0:
                    conv_ids = conv_ids[:remaining]
                    conv_labels = conv_labels[:remaining]
                    conv_ids[-1] = self.eos_id
                    conv_labels[-1] = -100
                    ids.extend(conv_ids)
                    labels.extend(conv_labels)
                break
            ids.extend(conv_ids)
            labels.extend(conv_labels)
            pbar.set_postfix(tokens=len(ids))
            if max_tokens is not None and len(ids) >= max_tokens:
                break

        # int32 not int16: labels carry -100 alongside ids up to the vocab size
        return torch.stack(
            [
                torch.tensor(ids, dtype=torch.int32),
                torch.tensor(labels, dtype=torch.int32),
            ]
        )

    def _make_datasets(self, train_payload, val_payload):
        return (
            MaskedTokenDataset(train_payload[0], train_payload[1], self.seq_len),
            MaskedTokenDataset(val_payload[0], val_payload[1], self.seq_len),
        )

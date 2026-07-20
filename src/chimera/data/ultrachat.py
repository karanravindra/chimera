"""
UltraChat SFT DataModule for PyTorch Lightning.

UltraChat 200k (``HuggingFaceH4/ultrachat_200k``) is a filtered multi-turn
chat dataset used for supervised finetuning. This module downloads the
``train_sft``/``test_sft`` splits, renders each conversation in ChatML with the
same pretrained tokenizer used for FineWeb-Edu pretraining (default
``LiquidAI/LFM2.5-230M``, whose vocab already carries ``<|im_start|>`` /
``<|im_end|>``), and packs the conversations into one flat token stream — the
same packing style as pretraining, served as non-overlapping ``(input, target)``
chunks.

Each turn is rendered as ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` and
conversations are separated with ``<|endoftext|>``. Alongside the id stream a
parallel *label* stream is built for SFT loss masking: assistant content tokens
(plus their closing ``<|im_end|>``, so the model learns to stop) keep their id,
everything else — system/user turns, role headers, separators — is ``-100``
(the CrossEntropy/CCE ``ignore_index``). Both streams are cached to disk per
exact configuration so reruns skip download and tokenization.

Because segments are tokenized independently and joined by explicit special-token
ids, the mask boundaries are exact by construction (no re-alignment needed).

Usage:
    dm = UltraChatDataModule(data_dir="./data", batch_size=8, seq_len=2048)
    trainer.fit(model, datamodule=dm)
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import lightning as pl

from chimera.tokenizers import BPETokenizer

from ._text import MaskedTokenDataset

HF_REPO = "HuggingFaceH4/ultrachat_200k"
IGNORE_INDEX = -100


class UltraChatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 8,
        seq_len: int = 2048,
        pretrained_id: str = "LiquidAI/LFM2.5-230M",
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        im_start_token: str = "<|im_start|>",
        im_end_token: str = "<|im_end|>",
        max_train_tokens: Optional[int] = 10_000_000,
        max_val_tokens: Optional[int] = 1_000_000,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pretrained_id = pretrained_id
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.max_train_tokens = max_train_tokens
        self.max_val_tokens = max_val_tokens
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer: Optional[BPETokenizer] = None
        self.vocab_size: Optional[int] = None
        self.bos_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.im_start_id: Optional[int] = None
        self.im_end_id: Optional[int] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    @property
    def _dir(self) -> Path:
        return self.data_dir / "ultrachat"

    @property
    def _tokenizer_path(self) -> Path:
        return self._dir / f"tokenizer_{self.pretrained_id.replace('/', '_')}.json"

    def _cache_path(self, split: str) -> Path:
        """Path of the cached (ids, labels) streams for this exact configuration."""
        hp = self.hparams
        tok_tag = hp.pretrained_id.replace("/", "_")
        cap = hp.max_train_tokens if split == "train_sft" else hp.max_val_tokens
        cap_tag = "all" if cap is None else str(cap)
        return self._dir / f"stream_{split}_{tok_tag}_{cap_tag}.pt"

    def _load_dataset(self, split: str):
        from datasets import load_dataset

        return load_dataset(
            HF_REPO,
            split=split,
            cache_dir=str(self.data_dir / "hf_cache"),
        )

    def prepare_data(self):
        # download + cache only, called once on a single process
        self._dir.mkdir(parents=True, exist_ok=True)
        if all(self._cache_path(s).exists() for s in ("train_sft", "test_sft")):
            return
        for split in ("train_sft", "test_sft"):
            if not self._cache_path(split).exists():
                self._load_dataset(split)

    def _load_tokenizer(self) -> BPETokenizer:
        if self._tokenizer_path.exists():
            return BPETokenizer.load(self._tokenizer_path, backend="pretrained")
        tok = BPETokenizer.from_pretrained(self.pretrained_id)
        tok.save(self._tokenizer_path)
        return tok

    def _encode(self, text: str) -> list[int]:
        """Encode plain text WITHOUT the tokenizer's special-token template.

        The LFM2.5 tokenizer's post-processor prepends ``<|startoftext|>`` to
        every ``encode()`` call; chat segments are joined from explicitly-placed
        special ids, so segment encodes must stay template-free.
        """
        return self.tokenizer._tok.encode(text, add_special_tokens=False).ids

    def _encode_batch(self, texts: list[str]) -> list[list[int]]:
        tok = self.tokenizer._tok
        encode_batch = getattr(tok, "encode_batch_fast", tok.encode_batch)
        return [e.ids for e in encode_batch(texts, add_special_tokens=False)]

    def _special_id(self, token: str) -> int:
        tid = self.tokenizer._tok.token_to_id(token)
        if tid is None:
            raise ValueError(
                f"tokenizer {self.pretrained_id!r} has no {token!r} token; a "
                "ChatML-capable pretrained tokenizer is required for SFT"
            )
        return tid

    def _build_streams(self, split: str, max_tokens: Optional[int]):
        """Tokenize a split into parallel flat (ids, labels) streams.

        Message contents are tokenized in batches (parallel Rust encoder) and
        joined with explicitly-placed special-token ids, so the supervised
        region of each assistant turn is exact by construction.
        """
        ds = self._load_dataset(split)
        newline = self._encode("\n")
        # per-role header ids for "<|im_start|>{role}\n" (roles in UltraChat are
        # user/assistant; system appears in other chat sets — handled uniformly)
        headers = {
            role: [self.im_start_id] + self._encode(f"{role}\n")
            for role in ("system", "user", "assistant")
        }

        ids: list[int] = []
        labels: list[int] = []
        bar = tqdm(total=len(ds), desc=f"Tokenizing ultrachat {split}", unit="conv")

        def append(piece_ids: list[int], supervised: bool) -> None:
            ids.extend(piece_ids)
            labels.extend(piece_ids if supervised else [IGNORE_INDEX] * len(piece_ids))

        for batch in ds.iter(batch_size=256):
            conversations = batch["messages"]
            # batch-encode every message content in this chunk at once
            contents = [m["content"] for conv in conversations for m in conv]
            encoded = iter(self._encode_batch(contents))
            for conv in conversations:
                # one BOS per conversation, mirroring pretraining's per-document
                # <|startoftext|> ... <|endoftext|> framing
                append([self.bos_id], False)
                for m in conv:
                    role = m["role"]
                    content_ids = next(encoded)
                    supervised = role == "assistant"
                    append(headers.get(role, headers["user"]), False)
                    append(content_ids, supervised)
                    # the closing <|im_end|> is supervised on assistant turns so
                    # the model learns to terminate its responses
                    append([self.im_end_id], supervised)
                    append(newline, False)
                append([self.eos_id], False)
            bar.update(len(conversations))
            bar.set_postfix(tokens=len(ids))
            if max_tokens is not None and len(ids) >= max_tokens:
                break
        bar.close()

        if max_tokens is not None and len(ids) > max_tokens:
            del ids[max_tokens:]
            del labels[max_tokens:]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long
        )

    def _load_or_build_streams(self, split: str, max_tokens: Optional[int]):
        path = self._cache_path(split)
        if path.exists():
            data = torch.load(path)
            return data["ids"], data["labels"]
        ids, labels = self._build_streams(split, max_tokens)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": ids, "labels": labels}, path)
        return ids, labels

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        self.tokenizer = self._load_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.bos_id = self._special_id(self.bos_token)
        self.eos_id = self._special_id(self.eos_token)
        self.im_start_id = self._special_id(self.im_start_token)
        self.im_end_id = self._special_id(self.im_end_token)

        train_ids, train_labels = self._load_or_build_streams(
            "train_sft", self.max_train_tokens
        )
        val_ids, val_labels = self._load_or_build_streams(
            "test_sft", self.max_val_tokens
        )
        self.train_dataset = MaskedTokenDataset(train_ids, train_labels, self.seq_len)
        self.val_dataset = MaskedTokenDataset(val_ids, val_labels, self.seq_len)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def render_prompt(self, messages: list[dict]) -> list[int]:
        """Encode a chat prompt in ChatML, ending with an open assistant header.

        ``messages`` is a list of ``{"role": ..., "content": ...}``; the returned
        ids end right after ``<|im_start|>assistant\\n`` so generation continues
        with the assistant's reply (stop at :attr:`im_end_id`).
        """
        newline = self._encode("\n")
        ids: list[int] = [self.bos_id]
        for m in messages:
            ids += [self.im_start_id] + self._encode(f"{m['role']}\n")
            ids += self._encode(m["content"])
            ids += [self.im_end_id] + newline
        ids += [self.im_start_id] + self._encode("assistant\n")
        return ids

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


if __name__ == "__main__":
    import os

    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    dm = UltraChatDataModule(
        data_dir="/mnt/ai/data", max_train_tokens=500_000, max_val_tokens=100_000
    )
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}")
    print(f"train batch: x={x.shape}, y={y.shape}")
    n_sup = (y != IGNORE_INDEX).float().mean().item()
    print(f"supervised fraction of targets: {n_sup:.2%}")

    # visualize masking on the first sample: supervised target spans in [green]
    ids = x[0].tolist()
    lab = y[0].tolist()
    raw = dm.tokenizer._tok
    pieces = []
    for i, tid in enumerate(ids):
        tok_text = raw.decode([tid], skip_special_tokens=False)
        # y[i] is the target for predicting position i+1
        supervised = i > 0 and lab[i - 1] != IGNORE_INDEX
        pieces.append(f"\033[92m{tok_text}\033[0m" if supervised else tok_text)
    print("".join(pieces[:400]))

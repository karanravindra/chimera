"""DataModule for a pre-packed pretraining token mixture.

Serves a mixture built offline by ``projects/llm/data/build_mixture.py``: a flat
``uint16`` ``train.bin`` / ``val.bin`` under
``<data_dir>/llm-mix/mix/<mix_name>/``. Training then looks exactly like the
FineWeb-Edu pretrain (same ``(x, y)`` next-token chunks, same LFM2.5 tokenizer),
but the token stream is memory-mapped so multi-billion-token mixes don't need to
fit in RAM.

The mixture composition (which sources, what weights, how many tokens) lives in
the mix's ``manifest.json``; this module just reads the packed streams.

Usage:
    dm = MixtureDataModule(data_dir="/mnt/ai/data", mix_name="mix_1B",
                           batch_size=32, seq_len=2048)
    dm.setup("fit"); trainer.fit(model, datamodule=dm)
"""

import json
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, Dataset

import lightning as pl

from chimera.tokenizers import BPETokenizer

from . import chat_template
from ._text import MemmapMaskedTokenDataset, MemmapTokenDataset


class MixtureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/mnt/ai/data",
        mix_name: str = "mix_1B",
        batch_size: int = 32,
        seq_len: int = 2048,
        pretrained_id: str = "LiquidAI/LFM2.5-230M",
        sft: bool = False,
        num_workers: int = 7,
        pin_memory: bool = True,
        root_subdir: str = "llm-mix",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.mix_name = mix_name
        # Container dir under data_dir holding {mix,mix_sft}/<name>/. Default is the
        # llm project's "llm-mix"; other projects (e.g. tiny-llm) pass their own so
        # their packed mixes live alongside their raw data.
        self.root_subdir = root_subdir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pretrained_id = pretrained_id
        self.sft = sft
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer: Optional[BPETokenizer] = None
        self.vocab_size: Optional[int] = None
        self.bos_token = "<|startoftext|>"
        self.eos_token = "<|endoftext|>"
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"
        self.bos_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        self.im_start_id: Optional[int] = None
        self.im_end_id: Optional[int] = None
        self.manifest: Optional[dict] = None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        # Per-source val slices (pretrain only): val.bin is written in manifest
        # source order, so we window it per source to log per-dataset val curves
        # (grokking). Populated in setup(); empty -> single combined val loader.
        self.val_source_windows: list[tuple[str, int, int]] = []
        self.val_datasets: dict[str, Dataset] = {}
        self.val_source_names: list[str] = ["val"]

    @property
    def _dir(self) -> Path:
        sub = "mix_sft" if self.sft else "mix"
        return self.data_dir / self.root_subdir / sub / self.mix_name

    def prepare_data(self):
        if not (self._dir / "train.bin").exists():
            raise FileNotFoundError(
                f"mixture {self.mix_name!r} not found at {self._dir}; build it "
                "first with projects/llm/data/build_mixture.py"
                + (" --sft" if self.sft else "")
            )

    def _make_dataset(self, split: str) -> Dataset:
        ids = self._dir / f"{split}.bin"
        mask = self._dir / f"{split}_mask.bin"
        if self.sft and mask.exists():
            return MemmapMaskedTokenDataset(ids, mask, self.seq_len)
        return MemmapTokenDataset(ids, self.seq_len)

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        self.tokenizer = BPETokenizer.from_pretrained(self.pretrained_id)
        self.vocab_size = self.tokenizer.vocab_size
        tt = self.tokenizer._tok.token_to_id
        self.bos_id = tt(self.bos_token)
        self.eos_id = tt(self.eos_token)
        self.im_start_id = tt(self.im_start_token)
        self.im_end_id = tt(self.im_end_token)

        manifest_p = self._dir / "manifest.json"
        if manifest_p.exists():
            self.manifest = json.loads(manifest_p.read_text())

        self.train_dataset = self._make_dataset("train")
        self.val_dataset = self._make_dataset("val")

        # Carve per-source val slices from the (source-ordered) val.bin so we can
        # log per-dataset val loss/bpt/bpb. Pretrain only (SFT val is masked and
        # kept as one combined loader). Needs >=2 sources with val tokens.
        if self.manifest and not self.sft:
            val_path = self._dir / "val.bin"
            off = 0
            for r in self.manifest.get("sources", []):
                n = int(r.get("val_tokens", 0))
                if n > 0:
                    self.val_source_windows.append((r["key"], off, n))
                off += n
            if len(self.val_source_windows) >= 2:
                self.val_datasets = {
                    key: MemmapTokenDataset(val_path, self.seq_len, start=start, length=length)
                    for key, start, length in self.val_source_windows
                }
                self.val_source_names = [k for k, _, _ in self.val_source_windows]

    def _encode(self, text: str) -> list:
        return self.tokenizer._tok.encode(text, add_special_tokens=False).ids

    def render_prompt(self, messages: list, tools=None, system=None) -> list:
        """Encode a chat prompt via the canonical template, ending with an open
        assistant header ready for generation (see chimera.data.chat_template)."""
        text = chat_template.render(
            messages, tools=tools, system=system, add_generation_prompt=True
        )
        return self._encode(text)

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

    def _val_dl(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        # >=2 sources -> one val loader PER source so metrics log per dataset
        # (dataloader_idx k -> val_source_names[k]); the module also aggregates a
        # combined val/* across them. Else the single combined loader (unchanged).
        if self.val_datasets:
            self.val_source_names = [k for k, _, _ in self.val_source_windows]
            return [self._val_dl(self.val_datasets[k]) for k in self.val_source_names]
        self.val_source_names = ["val"]
        return self._val_dl(self.val_dataset)


if __name__ == "__main__":
    import os

    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    dm = MixtureDataModule(mix_name="mix_smoke", batch_size=4, seq_len=512)
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}  eos_id={dm.eos_id}")
    print(f"train batch: x={x.shape}, y={y.shape}")

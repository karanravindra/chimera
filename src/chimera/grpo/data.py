"""Prompt data module for GRPO.

Unlike the repo's image data modules, GRPO batches are *prompts*, not tensors: each
training step tokenizes a handful of prompts, samples completions, and scores them, so the
data layer's only job is to hand the trainer rendered prompt strings plus their gold
answers. Tokenization is deferred to the step (it depends on generation-time padding), so
the collate is a passthrough and the batch is a plain ``list[dict]``.

At ``setup`` time we render every example's chat messages through the tokenizer's chat
template once (``add_generation_prompt=True``) so the per-step cost is just tokenization.
The module is task-driven: point it at a :class:`tasks.Task` and it loads that task's splits
and prompt format.
"""

from __future__ import annotations

from collections.abc import Callable

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from chimera.grpo.tasks import Task


class _RenderedDataset(Dataset):
    """Lazily render an HF dataset row into a prompt record on ``__getitem__``.

    Rendering (a chat-template application) is cheap, so we defer it instead of
    materializing the whole rendered corpus into a ``list[dict]`` at ``setup``. The eager
    list was resident per DataLoader worker (duplicated under ``persistent_workers`` +
    forked workers); a thin Dataset keeps only the underlying HF dataset in memory.
    """

    def __init__(self, ds, render: Callable[[dict], dict]):
        self._ds = ds
        self._render = render

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> dict:
        return self._render(self._ds[index])


class PromptDataModule(LightningDataModule):
    """Serves ``{"prompt": str, "gold": str, "question": str}`` batches for a :class:`Task`.

    Args:
        task: the task whose splits/prompt format to use.
        tokenizer: HF tokenizer used to render the chat template into prompt strings.
        data_dir: dataset cache root (passed to the task's loaders).
        batch_size: number of **prompts** per training step (rollouts = batch_size * G).
        num_workers: DataLoader workers (items are small strings; 0-2 is plenty).
        val_size: held-out prompts for periodic pass@1.
        train_size: optional cap on training prompts (for fast smoke runs); ``None`` = all.
    """

    def __init__(
        self,
        task: Task,
        tokenizer,
        *,
        data_dir: str = "/mnt/ai/data",
        batch_size: int = 4,
        num_workers: int = 2,
        val_size: int = 200,
        train_size: int | None = None,
    ):
        super().__init__()
        self.task = task
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.train_size = train_size
        self.drop_last = False  # run_training flips this on for CUDA-graph compile modes
        self._train: Dataset | None = None
        self._val: Dataset | None = None

    def _render(self, example: dict) -> dict:
        """Turn one raw dataset row into a ready-to-tokenize prompt record."""
        messages = self.task.build_prompt(example)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt,
            "gold": self.task.gold_of(example),
            "question": example.get(self.task.question_key, ""),
        }

    def setup(self, stage: str | None = None) -> None:
        if self._train is not None:
            return
        train_ds, val_ds = self.task.load_splits(self.data_dir, self.val_size)
        if self.train_size is not None:
            train_ds = train_ds.select(range(min(self.train_size, len(train_ds))))
        # Wrap (don't materialize): each row is rendered lazily in __getitem__.
        self._train = _RenderedDataset(train_ds, self._render)
        self._val = _RenderedDataset(val_ds, self._render)

    @staticmethod
    def _collate(batch: list[dict]) -> list[dict]:
        # Passthrough: the step tokenizes; keep prompts/golds as Python objects.
        return batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            persistent_workers=self.num_workers > 0,
        )

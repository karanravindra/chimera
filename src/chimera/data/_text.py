"""Shared helpers for character/token-level text DataModules."""

from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class TokenDataset(Dataset):
    """Wraps an encoded 1D tensor as non-overlapping ``(input, target)`` chunks.

    For a chunk starting at position ``i`` the target is the input shifted by one
    token, i.e. the model predicts the next token at every step. Works with any
    1D tensor of token ids (character ids, BPE ids, ...).
    """

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        i = idx * self.seq_len
        x = self.data[i : i + self.seq_len].long()
        y = self.data[i + 1 : i + 1 + self.seq_len].long()
        return x, y


def iter_text_chunks(text: str, chunk_chars: int = 1_000_000) -> Iterable[str]:
    """Yield ``text`` in fixed-size character chunks.

    Lets a single continuous document be tokenized with a live progress bar. The
    byte-level BPE does not merge across chunk boundaries, but at this chunk size
    the effect on the resulting token stream is negligible.
    """
    for i in range(0, len(text), chunk_chars):
        yield text[i : i + chunk_chars]


def tokenize_with_progress(
    tokenizer,
    texts: Iterable[str],
    *,
    desc: str = "Tokenizing",
    total: Optional[int] = None,
    unit: str = "doc",
    eos_id: Optional[int] = None,
    max_tokens: Optional[int] = None,
    batch_size: int = 1024,
) -> list[int]:
    """Encode an iterable of texts into one flat list of ids, showing a tqdm bar.

    Texts are encoded in batches via ``tokenizer.encode_batch`` (parallel on the
    fast backends), which is much faster than encoding one at a time. Optionally
    appends ``eos_id`` after each text (a document separator) and stops once
    ``max_tokens`` ids have been collected — so with a cap we never tokenize more
    than the last batch beyond it, and the surplus is trimmed off at the end.
    """
    ids: list[int] = []
    bar = tqdm(total=total, desc=desc, unit=unit)

    def flush(batch: list[str]) -> None:
        for enc in tokenizer.encode_batch(batch):
            ids.extend(enc)
            if eos_id is not None:
                ids.append(eos_id)
        bar.update(len(batch))
        bar.set_postfix(tokens=len(ids))

    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
            if max_tokens is not None and len(ids) >= max_tokens:
                break
    else:
        # only reached if the loop wasn't cut short by the cap
        if batch:
            flush(batch)
    bar.close()

    if max_tokens is not None and len(ids) > max_tokens:
        del ids[max_tokens:]
    return ids


def load_cached_ids(path: Union[str, Path]) -> Optional[torch.Tensor]:
    """Return the cached 1D token tensor at ``path``, or ``None`` if absent."""
    path = Path(path)
    if path.exists():
        return torch.load(path)
    return None


def save_cached_ids(path: Union[str, Path], data: torch.Tensor) -> None:
    """Persist a 1D token tensor so future ``setup()`` calls skip tokenization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)

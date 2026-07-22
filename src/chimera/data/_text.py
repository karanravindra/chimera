"""Shared helpers for character/token-level text DataModules."""

import os
from pathlib import Path
import tempfile
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


class WindowSampledDataset(Dataset):
    """Document-aware random-window dataset over a flat integer token stream.

    :class:`TokenDataset` slices fixed non-overlapping windows that cross document
    boundaries freely — fine for short-context training, but useless for teaching
    long-range dependencies: FlexAttention document masking resets attention and
    RoPE positions at every EOS, so a window packed from unrelated short documents
    trains nothing at long range. This dataset instead serves a ``seq_len + 1``
    window sampled at a RANDOM contiguous offset *inside a single long document*,
    so every position in the window is mutually visible.

    Document spans are recovered from the inline ``eos_id`` markers already in the
    stream (``[bos?] doc tokens eos`` per document — the existing on-disk format,
    no new layout). Only documents with at least ``min_doc_len`` tokens qualify;
    shorter ones belong to the short/broad pool and are never zero-padded here.

    Because a window is a slice of one document with no interior EOS,
    :func:`chimera.models.attention.build_block_mask_and_pos` derives
    ``pos_ids = 0..seq_len`` (window-relative) for it automatically. A mid-document
    window carries no ``bos_id`` (the draw starts past a real doc's BOS) and no
    leading/trailing EOS; the trailing EOS is present only when the window reaches
    the true document end.

    ``(x, y)`` are the ``seq_len``-length input and its one-token shift — the same
    contract as :class:`TokenDataset`. Call :meth:`set_epoch` each epoch (and wire
    :func:`window_worker_init_fn` as the DataLoader ``worker_init_fn``) so offsets
    resample without correlating across workers.
    """

    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int,
        eos_id: int,
        min_doc_len: Optional[int] = None,
        max_windows_per_doc: int = 4,
        seed: int = 0,
    ):
        self.data = data
        self.seq_len = seq_len
        self.eos_id = eos_id
        # A window needs seq_len + 1 tokens (input + shifted target); a qualifying
        # document must hold at least that. Callers can raise the floor to demand
        # "genuinely long" docs (e.g. only score the tail of the length band).
        self.min_doc_len = max(seq_len + 1, min_doc_len or 0)
        self.max_windows_per_doc = max_windows_per_doc
        self.seed = seed
        self._epoch = 0
        # Advanced on every __getitem__ so repeated draws of the same document
        # (unavoidable under a with-replacement sampler) get FRESH offsets, not a
        # deterministic-per-idx one. Reseeded by set_epoch / window_worker_init_fn.
        self._gen = torch.Generator().manual_seed(seed)

        # Recover document spans from inline EOS. Doc k occupies (prev_eos+1,
        # eos_pos[k]] — inclusive of its trailing EOS; doc 0 starts at index 0.
        eos_pos = (data == eos_id).nonzero(as_tuple=True)[0].tolist()
        starts, ends = [], []
        prev = 0
        for e in eos_pos:
            starts.append(prev)
            ends.append(e)  # index of this doc's EOS token (inclusive end)
            prev = e + 1

        # Per-doc content start: skip a leading BOS so mid-doc windows never
        # synthesize one (the window beginning exactly at doc start keeps its BOS).
        self._doc_start = []
        self._doc_end = []  # inclusive index of the doc's EOS
        self._doc_nwin = []
        for s, e in zip(starts, ends):
            length = e - s + 1  # tokens including trailing EOS
            if length < self.min_doc_len:
                continue
            self._doc_start.append(s)
            self._doc_end.append(e)
            # how many disjoint windows the doc could yield, capped
            n = min(self.max_windows_per_doc, length // (seq_len + 1))
            self._doc_nwin.append(max(1, n))

        # Cumulative map: flat idx -> (doc index, window ordinal within doc).
        self._cum = []
        total = 0
        for n in self._doc_nwin:
            total += n
            self._cum.append(total)
        self._len = total

    def set_epoch(self, epoch: int) -> None:
        """Reseed so offsets are redrawn this epoch (call once per epoch)."""
        self._epoch = epoch
        self._gen = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)

    def __len__(self) -> int:
        return self._len

    def _locate(self, idx: int) -> int:
        # binary search the cumulative window counts -> owning doc index
        import bisect

        return bisect.bisect_right(self._cum, idx)

    def __getitem__(self, idx: int):
        doc = self._locate(idx)
        s, e = self._doc_start[doc], self._doc_end[doc]
        # Valid window starts run [s, e - seq_len] so the window stays inside the
        # doc (last token index <= e, the doc's EOS). Start == s keeps the real
        # BOS (aligned first window); any later start begins past it, so mid-doc
        # windows carry no BOS. No interior EOS exists (EOS only at index e), so a
        # window contains one only when it reaches the true doc end.
        lo = s
        hi = e - self.seq_len
        if hi <= lo:
            start = s
        else:
            start = int(torch.randint(lo, hi + 1, (1,), generator=self._gen).item())
        window = self.data[start : start + self.seq_len + 1]
        x = window[:-1].long()
        y = window[1:].long()
        return x, y


def window_worker_init_fn(worker_id: int) -> None:
    """Decorrelate :class:`WindowSampledDataset` offsets across DataLoader workers.

    Each worker gets its own copy of the dataset; folding the worker id into the
    per-item seed keeps the same-index draws from lining up across workers.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return

    def descendants(dataset):
        children = getattr(dataset, "datasets", None)
        if children is None:
            yield dataset
            return
        for child in children:
            yield from descendants(child)

    for ds in descendants(info.dataset):
        if isinstance(ds, WindowSampledDataset):
            ds._gen = torch.Generator().manual_seed(
                (ds.seed * 1_000_003 + ds._epoch) * 1_000_003 + worker_id + 1
            )


class MaskedTokenDataset(Dataset):
    """Like :class:`TokenDataset` but with a parallel per-token label stream.

    ``ids`` and ``labels`` are equal-length 1D tensors; ``labels`` carries the
    supervision target for each position — the token id where the loss applies,
    or ``-100`` (the CrossEntropy/CCE ``ignore_index``) where it doesn't (e.g.
    prompt/user tokens during SFT). A chunk's input is ``ids[i : i+L]`` and its
    target is ``labels`` shifted by one, so masking decisions made once at
    tokenization time flow straight through the standard ``(x, y)`` batch shape.
    """

    def __init__(self, ids: torch.Tensor, labels: torch.Tensor, seq_len: int):
        assert len(ids) == len(labels), "ids and labels must be parallel streams"
        self.ids = ids
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.ids) - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        i = idx * self.seq_len
        x = self.ids[i : i + self.seq_len].long()
        y = self.labels[i + 1 : i + 1 + self.seq_len].long()
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
    bos_id: Optional[int] = None,
    max_tokens: Optional[int] = None,
    batch_size: int = 1024,
    chunk_tokens: int = 20_000_000,
) -> torch.Tensor:
    """Encode texts into a compact flat token tensor while preserving documents.

    Texts are encoded in batches via ``tokenizer.encode_batch`` (parallel on the
    fast backends), which is much faster than encoding one at a time. Optionally
    prepends ``bos_id`` before and/or appends ``eos_id`` after each text (document
    start marker / separator) and stops once ``max_tokens`` ids have been
    collected — so with a cap we never tokenize more than the last batch beyond
    it, and the surplus is trimmed off at the end.

    Memory: the running ids are flushed to a compact tensor chunk every
    ``chunk_tokens`` and the Python buffer cleared, so peak RAM stays ~chunk-sized.
    ``int16`` is used only when every tokenizer id fits; larger vocabularies use
    ``int32``. When a total cap lands inside a document, the document is truncated
    before its EOS rather than dropping that boundary marker.
    """
    dtype = (
        torch.int16
        if tokenizer.vocab_size - 1 <= torch.iinfo(torch.int16).max
        else torch.int32
    )
    bar = tqdm(total=total, desc=desc, unit=unit)
    chunks: list[torch.Tensor] = []
    buf: list[int] = []
    n_total = 0

    def spill() -> None:
        # move the Python buffer into a compact int16 chunk and free it
        if buf:
            chunks.append(torch.tensor(buf, dtype=dtype))
            buf.clear()

    def flush(batch: list[str]) -> bool:
        nonlocal n_total
        consumed = 0
        for enc in tokenizer.encode_batch(batch, add_special_tokens=False):
            doc = ([bos_id] if bos_id is not None else []) + list(enc)
            if eos_id is not None:
                doc.append(eos_id)

            if max_tokens is not None and n_total + len(doc) > max_tokens:
                remaining = max_tokens - n_total
                # If nothing has been emitted, retain a capped first document so
                # a single very long row cannot produce an empty dataset.
                if n_total == 0 and remaining > 0:
                    prefix = [bos_id] if bos_id is not None else []
                    suffix = [eos_id] if eos_id is not None else []
                    room = max(0, remaining - len(prefix) - len(suffix))
                    doc = prefix + list(enc[:room]) + suffix
                    doc = doc[:remaining]
                    if eos_id is not None and doc:
                        doc[-1] = eos_id
                    buf.extend(doc)
                    n_total += len(doc)
                    consumed += 1
                bar.update(consumed)
                bar.set_postfix(tokens=n_total)
                return True

            buf.extend(doc)
            n_total += len(doc)
            consumed += 1
            if len(buf) >= chunk_tokens:
                spill()

        bar.update(consumed)
        bar.set_postfix(tokens=n_total)
        return max_tokens is not None and n_total >= max_tokens

    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            capped = flush(batch)
            batch = []
            if capped:
                break
    else:
        # only reached if the loop wasn't cut short by the cap
        if batch:
            flush(batch)
    spill()
    bar.close()

    return torch.cat(chunks) if chunks else torch.empty(0, dtype=dtype)


def load_cached_ids(path: Union[str, Path]):
    """Load a versioned cache payload; legacy unversioned caches are ignored."""
    path = Path(path)
    if path.exists():
        payload = torch.load(path, weights_only=False)
        if isinstance(payload, dict) and payload.get("version") == 2:
            return payload["data"]
    return None


def save_cached_ids(
    path: Union[str, Path], data, metadata: Optional[dict] = None
) -> None:
    """Atomically persist a versioned token cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save({"version": 2, "metadata": metadata or {}, "data": data}, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)

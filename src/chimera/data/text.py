"""Lightning datamodules for streaming text pretraining corpora.

All corpora share one streaming / tokenizing / packing / caching implementation
(`StreamingTextDataModule`). Concrete corpora (FineWeb-Edu, DCLM, FineMath, and a
code corpus) are thin subclasses that only set dataset-specific defaults
(`sources`, `text_field`, ...). They all honor the same split + token-cap config.

Splitting train/val supports two modes that compose with per-split token caps:
  * a distinct HF ``val_split`` (e.g. WikiText's ``validation``), or
  * carving a validation set out of a single ``train_split`` either positionally
    (the legacy first-N-then-next-M behavior) or by deterministic, seeded
    document-level routing (``val_split_strategy="random"``).

Token caps are applied while streaming, so only as many documents as needed are
pulled from the Hub (token caps are honored without downloading the full set).
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import tiktoken
import torch
from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from chimera.utils.seed import seed_worker


# A source is a (dataset_name, dataset_config) pair. Multiple sources are
# chained so a single corpus can span several HF configs (e.g. languages).
Source = tuple[str, str | None]


def _serialize_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _normalize_sources(
    dataset_name: str, dataset_config: str | list[str] | None
) -> list[Source]:
    """Expand ``dataset_config`` (a str, list, or None) into concrete sources."""
    if isinstance(dataset_config, (list, tuple)):
        return [(dataset_name, cfg) for cfg in dataset_config]
    return [(dataset_name, dataset_config)]


def _iter_texts(sources: list[Source], split: str, text_field: str):
    """Yield non-empty text strings from the chained streaming sources."""
    for name, config in sources:
        dataset = load_dataset(name, config, split=split, streaming=True)
        for row in dataset:
            text = row.get(text_field, "")
            if text:
                yield text


def _encode_document(tokenizer: tiktoken.Encoding, text: str) -> list[int]:
    """Tokenize one document and append an end-of-text separator.

    Uses ``encode_ordinary`` (not ``encode``) so a document containing the literal
    string ``<|endoftext|>`` is encoded as ordinary text instead of raising — the
    default ``disallowed_special="all"`` would crash the stream on such web text.
    The ``eot_token`` separator (nanoGPT style) marks the document boundary so packed
    blocks don't blend unrelated documents under causal attention. Empty documents
    yield ``[]`` (no lone separator), preserving the empty-doc skip in callers.
    """
    ids = tokenizer.encode_ordinary(text)
    if not ids:
        return ids
    ids.append(tokenizer.eot_token)
    return ids


def _take_tokens(
    sources: list[Source],
    split: str,
    text_field: str,
    tokenizer: tiktoken.Encoding,
    max_tokens: int,
    *,
    desc: str,
) -> list[int]:
    """Stream + tokenize until ``max_tokens`` tokens are collected (truncating the
    final document so the cap is exact). Mirrors the original FineWeb-Edu loader."""
    token_ids: list[int] = []
    if max_tokens <= 0:
        return token_ids

    progress = tqdm(
        total=max_tokens,
        desc=desc,
        unit="tok",
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True,
    )
    for text in _iter_texts(sources, split, text_field):
        encoded = _encode_document(tokenizer, text)
        if not encoded:
            continue
        remaining = max_tokens - len(token_ids)
        if remaining <= 0:
            break
        token_ids.extend(encoded[:remaining])
        progress.update(len(token_ids) - progress.n)
        if len(token_ids) >= max_tokens:
            break
    progress.close()

    if len(token_ids) < max_tokens:
        print(
            f"Warning: requested {max_tokens:,} tokens for {desc} but only "
            f"loaded {len(token_ids):,}."
        )
    return token_ids


def _route_tokens(
    sources: list[Source],
    split: str,
    text_field: str,
    tokenizer: tiktoken.Encoding,
    max_train_tokens: int,
    max_val_tokens: int,
    *,
    strategy: str,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Carve train/val token streams out of a single ``split``.

    ``positional`` (legacy): take the first ``max_train_tokens`` tokens as train
    and the next ``max_val_tokens`` as val, so the split is a deterministic
    function of position. ``random``: route whole documents to train or val with
    a seeded RNG (no block straddles the split boundary), filling both caps.
    """
    if strategy == "positional":
        combined = _take_tokens(
            sources,
            split,
            text_field,
            tokenizer,
            max_train_tokens + max_val_tokens,
            desc="Loading train+val",
        )
        return combined[:max_train_tokens], combined[max_train_tokens:]

    if strategy != "random":
        raise ValueError(f"Unknown val_split_strategy: {strategy!r}")

    rng = torch.Generator().manual_seed(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    progress = tqdm(
        total=max_train_tokens + max_val_tokens,
        desc="Routing train/val",
        unit="tok",
        unit_scale=True,
        unit_divisor=1000,
        dynamic_ncols=True,
    )
    for text in _iter_texts(sources, split, text_field):
        if len(train_ids) >= max_train_tokens and len(val_ids) >= max_val_tokens:
            break
        encoded = _encode_document(tokenizer, text)
        if not encoded:
            continue
        to_val = torch.rand(1, generator=rng).item() < (
            max_val_tokens / max(max_train_tokens + max_val_tokens, 1)
        )
        target, cap = (
            (val_ids, max_val_tokens) if to_val else (train_ids, max_train_tokens)
        )
        remaining = cap - len(target)
        if remaining <= 0:
            continue
        target.extend(encoded[:remaining])
        progress.update((len(train_ids) + len(val_ids)) - progress.n)
    progress.close()
    return train_ids, val_ids


class PackedTokenDataset(Dataset):
    """Packs a flat token stream into non-overlapping ``block_size`` blocks."""

    def __init__(self, token_ids: list[int] | torch.Tensor, block_size: int):
        # uint16 (2 bytes) holds the full 50257-token vocab exactly and is 4x
        # smaller than int64 in RAM and on the host->device copy. The Lightning
        # module widens it back to int64 on-device for the embedding lookup.
        self.tokens = torch.as_tensor(token_ids, dtype=torch.uint16).contiguous()
        self.block_size = block_size
        self.num_blocks = self.tokens.numel() // block_size
        self.tokens = self.tokens[: self.num_blocks * block_size]

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.block_size
        return self.tokens[start : start + self.block_size]


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": True,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, **loader_kwargs)


class StreamingTextDataModule(LightningDataModule):
    """Shared streaming-text datamodule for all pretraining corpora.

    Subclasses set ``dataset_name`` / ``dataset_config`` / ``text_field`` defaults;
    everything else (splitting, token caps, packing, caching, the tokenizer, and
    the per-token byte lookup used for bits-per-byte) is shared.
    """

    # Overridden by subclasses.
    dataset_name: str = ""
    dataset_config: str | list[str] | None = None
    text_field: str = "text"

    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        dataset_config: str | list[str] | None = None,
        text_field: str | None = None,
        # Legacy total-cap knobs (preserved): cap total tokens then split by fraction.
        token_limit: int = 600_000_000,
        train_fraction: float = 0.9,
        # Per-split overrides; when set they take precedence over the legacy knobs.
        max_train_tokens: int | None = None,
        max_val_tokens: int | None = None,
        # Split selection.
        train_split: str = "train",
        val_split: str | None = None,
        val_split_strategy: str = "positional",
        # Packing / loading.
        tokenizer_name: str = "gpt2",
        cache_dir: str | Path | None = Path("data") / "llm_cache",
        block_size: int = 128,
        val_block_size: int | None = None,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name or self.dataset_name
        self.dataset_config = (
            dataset_config if dataset_config is not None else self.dataset_config
        )
        self.text_field = text_field or self.text_field
        self.sources = _normalize_sources(self.dataset_name, self.dataset_config)

        self.token_limit = token_limit
        self.train_fraction = train_fraction
        # Derive per-split caps from the legacy knobs when not given explicitly,
        # preserving the original "first train_fraction is train" behavior.
        if max_val_tokens is None:
            max_val_tokens = round(token_limit * (1.0 - train_fraction))
        if max_train_tokens is None:
            max_train_tokens = token_limit - max_val_tokens
        self.max_train_tokens = max_train_tokens
        self.max_val_tokens = max_val_tokens

        self.train_split = train_split
        self.val_split = val_split
        self.val_split_strategy = val_split_strategy

        self.tokenizer_name = tokenizer_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.block_size = block_size
        # Sweeps train at a swept block size but must validate at one FIXED
        # length so val/loss stays comparable across trials. None = block_size.
        self.val_block_size = val_block_size or block_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        # Token caches/packing store ids as uint16 (see PackedTokenDataset and
        # _write_cache) -- fine for gpt2's 50257 vocab but silently wraps any id
        # >= 65536. Fail loudly rather than corrupt the token stream.
        if self.tokenizer.n_vocab > 65535:
            raise ValueError(
                f"{tokenizer_name} vocab ({self.tokenizer.n_vocab}) exceeds the "
                f"uint16 token cache limit (65535); widen the token dtype or use "
                f"a smaller tokenizer."
            )
        self._train_dataset: PackedTokenDataset | None = None
        self._val_dataset: PackedTokenDataset | None = None
        self._token_ids: torch.Tensor | None = None

    # -- public surface -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def token_ids(self) -> torch.Tensor:
        if self._token_ids is None:
            raise RuntimeError("DataModule has not been set up yet.")
        return self._token_ids

    def token_byte_lengths(self) -> torch.Tensor:
        """UTF-8 byte length of each token id, used for bits-per-byte metrics.

        Because tiktoken BPE is a byte-level bijection, summing these per-token
        lengths over a packed block exactly reconstructs the number of UTF-8
        bytes in the original text. Special tokens (e.g. ``<|endoftext|>``) carry
        no original-text bytes and are forced to 0, so the ``eot_token``
        separators inserted between documents during packing do not inflate the
        byte count. (``decode_single_token_bytes`` would otherwise return the
        literal ``b"<|endoftext|>"`` — 13 bytes — for the gpt2 encoding.)
        """
        special_ids = {
            self.tokenizer.encode_single_token(s)
            for s in self.tokenizer.special_tokens_set
        }
        lengths = torch.zeros(self.vocab_size, dtype=torch.long)
        for token_id in range(self.vocab_size):
            if token_id in special_ids:
                continue
            try:
                lengths[token_id] = len(
                    self.tokenizer.decode_single_token_bytes(token_id)
                )
            except (KeyError, ValueError):
                lengths[token_id] = 0
        return lengths

    def config_dict(self) -> dict[str, object]:
        """Return a JSON-safe summary of the effective runtime configuration."""
        total_tokens = max(self.max_train_tokens + self.max_val_tokens, 1)
        return {
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "text_field": self.text_field,
            "sources": [
                {"dataset_name": name, "dataset_config": config}
                for name, config in self.sources
            ],
            "token_limit": self.token_limit,
            "train_fraction": self.train_fraction,
            "effective_train_fraction": self.max_train_tokens / total_tokens,
            "effective_val_fraction": self.max_val_tokens / total_tokens,
            "max_train_tokens": self.max_train_tokens,
            "max_val_tokens": self.max_val_tokens,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "val_split_strategy": self.val_split_strategy,
            "tokenizer_name": self.tokenizer_name,
            "cache_dir": _serialize_path(self.cache_dir),
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "seed": self.seed,
        }

    def prepare_data(self) -> None:
        # Materialize the token cache here: Lightning runs prepare_data on ONE
        # process per node (vs setup on every rank), so under DDP this is what
        # stops 4 ranks from concurrently streaming + writing the same cache
        # file. setup() then hits the warm cache on all ranks.
        self._load_token_splits()

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None and self._val_dataset is not None:
            return

        train_ids, val_ids = self._load_token_splits()
        if len(train_ids) < self.block_size:
            raise ValueError(
                "Need at least block_size tokens to build packed train blocks."
            )

        self._token_ids = torch.cat(
            [
                torch.as_tensor(train_ids, dtype=torch.long),
                torch.as_tensor(val_ids, dtype=torch.long),
            ]
        )
        self._train_dataset = PackedTokenDataset(train_ids, self.block_size)
        self._val_dataset = PackedTokenDataset(val_ids, self.val_block_size)

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self.setup("fit")
        assert self._train_dataset is not None
        generator = torch.Generator().manual_seed(self.seed)
        return make_loader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            self.setup("validate")
        assert self._val_dataset is not None
        return make_loader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    # -- token loading ------------------------------------------------------

    def _load_token_splits(self) -> tuple[list[int], list[int]]:
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        if self.val_split is not None:
            # Distinct HF splits: cap each split independently.
            train_ids = _take_tokens(
                self.sources,
                self.train_split,
                self.text_field,
                self.tokenizer,
                self.max_train_tokens,
                desc=f"Loading {self.dataset_name} train",
            )
            val_ids = _take_tokens(
                self.sources,
                self.val_split,
                self.text_field,
                self.tokenizer,
                self.max_val_tokens,
                desc=f"Loading {self.dataset_name} val",
            )
        else:
            train_ids, val_ids = _route_tokens(
                self.sources,
                self.train_split,
                self.text_field,
                self.tokenizer,
                self.max_train_tokens,
                self.max_val_tokens,
                strategy=self.val_split_strategy,
                seed=self.seed,
            )

        if cache_path is not None:
            self._write_cache(cache_path, train_ids, val_ids)
        return train_ids, val_ids

    # -- caching ------------------------------------------------------------

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        cache_key = {
            # Bump when the token *content* changes so stale caches are not reused.
            # v2: documents joined with <|endoftext|> separators + encode_ordinary.
            "format_version": 2,
            "sources": self.sources,
            "text_field": self.text_field,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "val_split_strategy": self.val_split_strategy,
            "max_train_tokens": self.max_train_tokens,
            "max_val_tokens": self.max_val_tokens,
            "tokenizer_name": self.tokenizer_name,
            "seed": self.seed,
        }
        digest = sha256(
            json.dumps(cache_key, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return self.cache_dir / f"tokens_{digest}.pt"

    def _read_cache(self, cache_path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - corrupt cache should not be fatal
            print(f"Warning: failed to load token cache {cache_path}: {exc}")
            return None
        # Tensors straight through: .tolist() on a 100M+-token cache took ~a
        # minute and ballooned RAM (python ints are ~28 bytes vs 2 for uint16);
        # PackedTokenDataset accepts tensors as-is.
        return cached["train"], cached["val"]

    def _write_cache(
        self, cache_path: Path, train_ids: list[int], val_ids: list[int]
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # uint16 quarters the on-disk cache vs int64 (vocab < 65536, enforced in
        # the datamodule). _read_cache returns the tensors as-is; PackedTokenDataset
        # accepts uint16 tensors directly.
        torch.save(
            {
                "train": torch.tensor(train_ids, dtype=torch.uint16),
                "val": torch.tensor(val_ids, dtype=torch.uint16),
            },
            cache_path,
        )


class MixedStreamingDataModule(LightningDataModule):
    """Concatenate several ``StreamingTextDataModule`` corpora into one mix.

    Each component keeps its own token caps, split strategy, and cache. Tokens
    are packed into blocks per corpus (no block straddles a corpus boundary)
    and the packed datasets are concatenated; the train loader's shuffle then
    interleaves the corpora at block granularity.
    """

    def __init__(
        self,
        components: list[StreamingTextDataModule],
        *,
        block_size: int = 128,
        val_block_size: int | None = None,
        batch_size: int = 64,
        val_batch_size: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("MixedStreamingDataModule needs at least one component.")
        tokenizer_names = {dm.tokenizer_name for dm in components}
        if len(tokenizer_names) > 1:
            raise ValueError(f"Components disagree on tokenizer: {tokenizer_names}")
        self.components = list(components)
        self.block_size = block_size
        # Fixed-length validation for sweeps (see StreamingTextDataModule).
        self.val_block_size = val_block_size or block_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._train_dataset: torch.utils.data.ConcatDataset | None = None
        self._val_dataset: torch.utils.data.ConcatDataset | None = None

    @property
    def tokenizer_name(self) -> str:
        return self.components[0].tokenizer_name

    @property
    def tokenizer(self) -> tiktoken.Encoding:
        return self.components[0].tokenizer

    @property
    def vocab_size(self) -> int:
        return self.components[0].vocab_size

    def token_byte_lengths(self) -> torch.Tensor:
        return self.components[0].token_byte_lengths()

    def config_dict(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "tokenizer_name": self.tokenizer_name,
            "datasets": {dm.dataset_name: dm.config_dict() for dm in self.components},
        }

    def prepare_data(self) -> None:
        # Once per node (see StreamingTextDataModule.prepare_data): build every
        # component's cache before the per-rank setup() calls race on them.
        for dm in self.components:
            dm._load_token_splits()

    def setup(self, stage: str | None = None) -> None:
        if self._train_dataset is not None and self._val_dataset is not None:
            return
        train_sets: list[PackedTokenDataset] = []
        val_sets: list[PackedTokenDataset] = []
        for dm in self.components:
            train_ids, val_ids = dm._load_token_splits()
            print(
                f"{dm.dataset_name}: {len(train_ids):,} train tokens, "
                f"{len(val_ids):,} val tokens"
            )
            train_sets.append(PackedTokenDataset(train_ids, self.block_size))
            val_sets.append(PackedTokenDataset(val_ids, self.val_block_size))
        self._train_dataset = torch.utils.data.ConcatDataset(train_sets)
        self._val_dataset = torch.utils.data.ConcatDataset(val_sets)

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self.setup("fit")
        assert self._train_dataset is not None
        generator = torch.Generator().manual_seed(self.seed)
        return make_loader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            self.setup("validate")
        assert self._val_dataset is not None
        return make_loader(
            self._val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class FineWebEduDataModule(StreamingTextDataModule):
    """FineWeb-Edu (English educational web text)."""

    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_config = "sample-10BT"
    text_field = "text"


class DCLMDataModule(StreamingTextDataModule):
    """DCLM-baseline (DataComp-LM filtered Common Crawl)."""

    dataset_name = "mlfoundations/dclm-baseline-1.0-parquet"
    dataset_config = "default"
    text_field = "text"


class FineMathDataModule(StreamingTextDataModule):
    """FineMath (math-heavy web text)."""

    dataset_name = "HuggingFaceTB/finemath"
    dataset_config = "finemath-3plus"
    text_field = "text"


class CodeDataModule(StreamingTextDataModule):
    """Code pretraining corpus (stand-in for Stack-Edu).

    Stack-Edu (``HuggingFaceTB/stack-edu``) only stores ``blob_id`` metadata and
    requires fetching content from Software Heritage S3, and the inline-text
    ``codeparrot/github-code-clean`` is a script-based dataset that modern
    ``datasets`` refuses to load. We default to the parquet-native
    ``codeparrot/codeparrot-clean`` (Python, ``content`` field); point
    ``dataset_name`` / ``dataset_config`` / ``text_field`` at any inline-text
    code/markdown corpus to override.
    """

    dataset_name = "codeparrot/codeparrot-clean"
    dataset_config = None
    text_field = "content"

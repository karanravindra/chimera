"""
Generic HF text-corpus BPE DataModule (pretrain).

The flat-token-stream loader for the pretrain corpora (TinyStoriesV2,
tiny-textbooks, ...): train (or load) a byte-level BPE tokenizer on the dataset's
text, tokenize each split into one flat token stream with per-document EOS/BOS
markers, cache the streams keyed on the tokenizer's content hash, and serve
non-overlapping next-token ``(input, target)`` chunks. The dataset loading,
split carving, fingerprinting, cache round-trip, and dataloader scaffolding live
in :class:`chimera.data._hf_base._HFCorpusBase`; this class supplies the
tokenizer and the flat-stream tokenization.

Subclasses configure the dataset via class attributes::

    class TinyStoriesV2DataModule(HFTextDataModule):
        HF_REPO = "noanabeshima/TinyStoriesV2"
        DIR_NAME = "tinystories-v2"   # cache dir under data_dir + tqdm label
        TEXT_COLUMN = "text"          # column holding the document text
        VAL_SPLIT = "validation"      # HF split used for validation
        UNIT = "story"                # tqdm unit

Rendering: one document per row is produced by :meth:`_row_text` (a single
``TEXT_COLUMN`` or joined ``TEXT_COLUMNS``). :meth:`iter_texts` streams
``_row_text`` over the dataset in Arrow batches; subclasses that need cross-row
grouping (e.g. SQuAD passages) override ``iter_texts`` directly.

Tokenizer sharing (for :class:`chimera.data.ConcatTextDataModule`): a module
normally trains/loads its own tokenizer, but ``set_shared_tokenizer(tok,
fingerprint)`` lets a mixture hand every source the same tokenizer — the module
then tokenizes with it and keys its ids caches on the shared fingerprint, so one
vocab spans the whole mixture and caches can never pair with the wrong tokenizer.

The ``hf`` tokenizer backend is trained with the canonical chat/tool special
tokens (:data:`chimera.data.chat_template.SPECIAL_TOKENS`) reserved at fixed low
ids, so a base tokenizer carries straight into chat / tool-call SFT without a
vocab change. Documents are concatenated with an ``eos_token`` after each one;
``add_bos`` additionally prepends a start token. Both require a tokenizer that
defines the token — the ``hf`` and ``pretrained`` backends do; the from-scratch
byte-level BPE has none, so they are no-ops there.
"""

import hashlib
from pathlib import Path
from typing import Optional, Sequence

from ._hf_base import _HFCorpusBase
from ._text import TokenDataset, tokenize_with_progress
from .chat_template import BOS, EOS, SPECIAL_TOKENS

from chimera.tokenizers import BPETokenizer


class HFTextDataModule(_HFCorpusBase):
    TEXT_COLUMN: str = "text"
    # Corpora with no single text column (e.g. human/bot QA pairs) instead set
    # TEXT_COLUMNS to join those fields into one document per row, TEXT_JOIN
    # between them. None => use the single TEXT_COLUMN.
    TEXT_COLUMNS: Optional[Sequence[str]] = None
    TEXT_JOIN: str = "\n\n"

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

        self.vocab_size = vocab_size
        self.tokenizer_backend = tokenizer_backend
        self.pretrained_id = pretrained_id
        self.add_eos = add_eos
        self.eos_token = eos_token
        self.add_bos = add_bos
        self.bos_token = bos_token
        self.tokenizer_train_chars = tokenizer_train_chars

        # set via set_shared_tokenizer(); overrides the own-file fingerprint
        self._shared_fingerprint: Optional[str] = None

    # -- tokenizer -----------------------------------------------------------
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
            tag = (
                f"{hp.tokenizer_backend}_v{hp.vocab_size}_"
                f"c{hp.tokenizer_train_chars}_src-{self._dataset_fingerprint()}"
            )
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
        consistent rebuild.
        """
        if self._shared_fingerprint is not None:
            return self._shared_fingerprint
        return hashlib.blake2b(
            self._tokenizer_path.read_bytes(), digest_size=6
        ).hexdigest()

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

    def _prepare_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = self._load_or_train_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_id = self._resolve_special_id(self.add_eos, self.eos_token)
        self.bos_id = self._resolve_special_id(self.add_bos, self.bos_token)

    def _setup_tokenizer(self):
        if self.tokenizer is None:
            if not self._tokenizer_path.exists():
                raise RuntimeError("prepare_data() must run before setup()")
            self.tokenizer = BPETokenizer.load(
                self._tokenizer_path, backend=self.tokenizer_backend
            )
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

    # -- rendering -----------------------------------------------------------
    def _row_text(self, row) -> str:
        """The document text for one row (single column, or joined columns)."""
        if self.TEXT_COLUMNS is None:
            return row[self.TEXT_COLUMN]
        return self.TEXT_JOIN.join(str(row[c]) for c in self.TEXT_COLUMNS)

    def iter_texts(self, ds, batch_size: int = 1024):
        """Yield document strings from a HF dataset, reading columns in batches.

        Batched column reads are far faster than per-row access on the Arrow
        dataset; each row is rebuilt into a small dict and rendered by
        :meth:`_row_text`, so a subclass customizes rendering by overriding only
        ``_row_text``. Subclasses needing cross-row grouping override this.
        """
        for batch in ds.iter(batch_size=batch_size):
            cols = list(batch.keys())
            n = len(batch[cols[0]])
            for i in range(n):
                yield self._row_text({c: batch[c][i] for c in cols})

    def _renderer_methods(self):
        return (type(self)._row_text, type(self).iter_texts)

    def _fingerprint_extra(self) -> dict:
        return {
            "text_column": self.TEXT_COLUMN,
            "text_columns": self.TEXT_COLUMNS,
            "text_join": self.TEXT_JOIN,
            "eos_token": self.eos_token if self.add_eos else None,
            "bos_token": self.bos_token if self.add_bos else None,
        }

    # -- cache / tokenize / datasets -----------------------------------------
    def _cache_path(self, split: str, max_tokens: Optional[int]) -> Path:
        """Path of the cached flat token stream for ``split`` and this config.

        Bound to the tokenizer's content hash so ids and tokenizer can never
        drift apart.
        """
        fingerprint = self._tokenizer_fingerprint()
        eos = "eos" if self.hparams.add_eos else "noeos"
        # only tagged when on, so pre-existing (no-bos) caches keep matching
        bos = "_bos" if self.hparams.add_bos else ""
        cap = "all" if max_tokens is None else str(max_tokens)
        source = self._dataset_fingerprint()
        return self._dir / (
            f"ids_v2_{split}_src-{source}_tok-{fingerprint}_{eos}{bos}_{cap}.pt"
        )

    def _tokenize_split(self, ds, split: str, max_tokens: Optional[int]):
        # tokenize_with_progress batches these for encode_batch and stops early
        # once max_tokens is reached; returns a flat int16/int32 tensor built via
        # bounded chunks so large sources do not need Python-list RAM.
        return tokenize_with_progress(
            self.tokenizer,
            self.iter_texts(ds),
            desc=f"Tokenizing {self.DIR_NAME} [{split}]",
            total=len(ds),
            unit=self.UNIT,
            eos_id=self.eos_id,
            bos_id=self.bos_id,
            max_tokens=max_tokens,
        )

    def _make_datasets(self, train_payload, val_payload):
        return (
            TokenDataset(train_payload, self.seq_len),
            TokenDataset(val_payload, self.seq_len),
        )

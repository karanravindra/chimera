"""
Byte-level BPE tokenizer with two interchangeable backends.

- ``backend="scratch"``: a minimal, dependency-free byte-level BPE implementation
  (minbpe-style) — trains merges by repeatedly combining the most frequent adjacent
  pair of tokens starting from the 256 raw bytes.
- ``backend="hf"``: a wrapper around the Hugging Face ``tokenizers`` library
  (fast, Rust-backed). ``tokenizers`` is imported lazily, so the scratch backend
  works without it installed.
- ``backend="pretrained"``: loads a fixed, already-trained tokenizer from the
  Hugging Face Hub (e.g. ``LiquidAI/LFM2.5-230M``) via
  :meth:`BPETokenizer.from_pretrained`. Shares the ``tokenizers``-backed
  encode/decode/save paths with ``"hf"``; ``train`` is a no-op.

All backends round-trip losslessly (``decode(encode(s)) == s``) and share the same
``train`` / ``encode`` / ``decode`` / ``save`` / ``load`` interface.

Usage:
    tok = BPETokenizer(backend="scratch")
    tok.train("hello world", vocab_size=300)
    ids = tok.encode("hello")
    text = tok.decode(ids)
    tok.save("tokenizer.json")
    tok = BPETokenizer.load("tokenizer.json")

    # a fixed pretrained tokenizer from the Hub
    tok = BPETokenizer.from_pretrained("LiquidAI/LFM2.5-230M")
"""

import json
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

Backend = Literal["scratch", "hf", "pretrained"]

# backends backed by a Hugging Face ``tokenizers.Tokenizer`` (self._tok)
_FAST_BACKENDS = ("hf", "pretrained")


def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, backend: Backend = "scratch"):
        if backend not in ("scratch", "hf", "pretrained"):
            raise ValueError(
                f"unknown backend {backend!r}, expected 'scratch', 'hf', or 'pretrained'"
            )
        self.backend: Backend = backend

        # scratch state
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

        # hf / pretrained state (a tokenizers.Tokenizer)
        self._tok = None

    @classmethod
    def from_pretrained(cls, identifier: str, revision: str = "main"):
        """Load a fixed, already-trained tokenizer.

        ``identifier`` may be a Hugging Face Hub id (e.g. ``LiquidAI/LFM2.5-230M``)
        *or* a local path — either a ``tokenizer.json`` file or a directory
        containing one (as written by our own :meth:`save` / the tokenizer
        trainer). Local paths are tried first so a custom tokenizer drops into any
        caller that already threads a ``pretrained_id`` string through.
        """
        from tokenizers import Tokenizer

        self = cls(backend="pretrained")
        p = Path(identifier)
        if p.suffix == ".json" and p.is_file():
            self._tok = Tokenizer.from_file(str(p))
        elif p.is_dir() and (p / "tokenizer.json").is_file():
            self._tok = Tokenizer.from_file(str(p / "tokenizer.json"))
        else:
            self._tok = Tokenizer.from_pretrained(identifier, revision=revision)
        return self

    # -- training ---------------------------------------------------------

    def train(
        self,
        text: Union[str, Iterable[str]],
        vocab_size: int,
        special_tokens: Optional[list[str]] = None,
        min_frequency: int = 2,
        split_digits: bool = False,
    ):
        """Train a byte-level BPE.

        ``text`` may be a single string or an iterable of strings (the ``hf``
        backend streams the iterable, so it never has to materialize the whole
        corpus). ``special_tokens`` are reserved atomic tokens (e.g. ChatML
        markers) added to the vocabulary at fixed low ids; they are only honored
        by the ``hf`` backend. ``min_frequency`` is the minimum pair count for a
        merge (``hf`` only). ``split_digits`` pre-tokenizes runs of digits into
        single characters (Llama/GPT-4 style) so no merge ever spans two digits —
        better arithmetic generalization; ``hf`` backend only.
        """
        if self.backend == "pretrained":
            return self  # a pretrained tokenizer is fixed; nothing to train
        if self.backend == "scratch":
            self._train_scratch(text, vocab_size)
        else:
            self._train_hf(text, vocab_size, special_tokens, min_frequency,
                           split_digits)
        return self

    def _train_scratch(self, text: Union[str, Iterable[str]], vocab_size: int):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 for byte-level BPE")
        if not isinstance(text, str):
            text = "".join(text)

        ids = list(text.encode("utf-8"))
        vocab = {i: bytes([i]) for i in range(256)}
        merges: dict[tuple[int, int], int] = {}

        for idx in range(256, vocab_size):
            stats = _get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            if stats[pair] < 2:
                break
            ids = _merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def _train_hf(
        self,
        text: Union[str, Iterable[str]],
        vocab_size: int,
        special_tokens: Optional[list[str]] = None,
        min_frequency: int = 2,
        split_digits: bool = False,
    ):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        specials = list(special_tokens or [])
        tok = Tokenizer(models.BPE(unk_token=None))
        byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
        if split_digits:
            # Split runs of digits into single chars *before* byte-level, so a
            # merge never spans two digits (helps arithmetic). ByteLevel decoder
            # still reconstructs bytes exactly — digits remain byte-encoded.
            tok.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Digits(individual_digits=True), byte_level]
            )
        else:
            tok.pre_tokenizer = byte_level
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            # listed first -> reserved at ids 0..len(specials)-1, before the byte
            # alphabet and merges, so their ids are stable across retrains.
            special_tokens=specials,
        )
        iterator = [text] if isinstance(text, str) else text
        tok.train_from_iterator(iterator, trainer=trainer)
        self._tok = tok

    # -- encode / decode --------------------------------------------------

    def encode(self, text: str) -> list[int]:
        if self.backend in _FAST_BACKENDS:
            return self._tok.encode(text).ids

        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode many texts at once.

        For the fast backends this dispatches to the Rust tokenizer's parallel
        batch encoder (multi-threaded, far faster than looping ``encode``). The
        scratch backend has no parallel path, so it falls back to a Python loop.
        """
        if self.backend in _FAST_BACKENDS:
            # encode_batch_fast (tokenizers >= 0.20) skips offset bookkeeping we
            # don't need; fall back to encode_batch on older versions.
            encode_batch = getattr(self._tok, "encode_batch_fast", None)
            if encode_batch is None:
                encode_batch = self._tok.encode_batch
            return [enc.ids for enc in encode_batch(texts)]
        return [self.encode(t) for t in texts]

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids]
        if self.backend in _FAST_BACKENDS:
            return self._tok.decode(ids)
        tokens = b"".join(self.vocab[i] for i in ids)
        return tokens.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        if self.backend in _FAST_BACKENDS:
            return self._tok.get_vocab_size() if self._tok is not None else 0
        return len(self.vocab)

    # -- persistence ------------------------------------------------------

    def save(self, path: Union[str, Path]):
        path = Path(path)
        if self.backend in _FAST_BACKENDS:
            self._tok.save(str(path))
            return
        payload = {
            "backend": "scratch",
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Union[str, Path], backend: Optional[Backend] = None):
        path = Path(path)

        if backend in _FAST_BACKENDS:
            self = cls(backend=backend)
            from tokenizers import Tokenizer

            self._tok = Tokenizer.from_file(str(path))
            return self

        # scratch: infer from the JSON payload written by save()
        payload = json.loads(path.read_text())
        if payload.get("backend") != "scratch":
            raise ValueError(
                f"{path} is not a scratch tokenizer; pass backend='hf' to load it"
            )
        self = cls(backend="scratch")
        self.merges = {(a, b): idx for a, b, idx in payload["merges"]}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        return self

"""
tiny-textbooks BPE DataModule.

``nampdn-ai/tiny-textbooks`` (420k synthetic textbook documents written by
Nous-Hermes-Llama2-13b from web seed text, Apache 2.0) is a "textbooks are all
you need"-style corpus aimed at small language models — the world-knowledge /
expository register that pure children's-story corpora lack. Full ``train``
split is ~400M tokens at a 16k byte-BPE vocab. The repo is gated:auto on the
Hub: access is auto-granted but downloading needs an authenticated HF token.

The ``textbook`` column holds the synthesized textbook (the ``text`` column is
the raw web seed it was written from) — training uses the textbook only. The
HF ``test`` split serves as validation. All machinery lives in
:class:`chimera.data.hf_text.HFTextDataModule`.

Standalone it trains its own tokenizer on textbook text; mixed into a
:class:`chimera.data.ConcatTextDataModule` it adopts the mixture owner's
tokenizer instead.

Usage:
    dm = TinyTextbooksDataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule

HF_REPO = "nampdn-ai/tiny-textbooks"


class TinyTextbooksDataModule(HFTextDataModule):
    HF_REPO = HF_REPO
    DIR_NAME = "tiny-textbooks"
    TEXT_COLUMN = "textbook"
    VAL_SPLIT = "test"
    UNIT = "doc"

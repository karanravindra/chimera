"""
TinyStoriesV2 BPE DataModule for PyTorch Lightning.

TinyStoriesV2 (``noanabeshima/TinyStoriesV2``) is the GPT-4-only rebuild of
roneneldan/TinyStories — ~2.7M short synthetic children's stories using a small
vocabulary, a common toy corpus for tiny language models. All machinery
(tokenizer training/caching, per-document EOS/BOS markers, flat-stream ids
caches, dataloaders) lives in :class:`chimera.data.text.hf_text.HFTextDataModule`;
this class just points it at the dataset.

TinyStoriesV2 ships native ``train`` and
``validation`` splits, so each is tokenized into its own stream rather than
carving a validation fraction off the train tokens.

Usage:
    dm = TinyStoriesV2DataModule(data_dir="./data", batch_size=64, seq_len=512)
    trainer.fit(model, datamodule=dm)
"""

from .hf_text import HFTextDataModule


class TinyStoriesV2DataModule(HFTextDataModule):
    HF_REPO = "noanabeshima/TinyStoriesV2"
    DIR_NAME = "tinystories-v2"
    TEXT_COLUMN = "text"
    VAL_SPLIT = "validation"
    UNIT = "story"

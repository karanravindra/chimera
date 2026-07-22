"""
Stack Exchange BPE DataModule.

Stack Exchange (``donfu/oa-stackexchange``, the OpenAssistant Q/A export, CC
BY-SA) is natural question/explanation structure across many communities — the
role the tinylm roadmap wants for continued pretraining (real questions with
authored, often long-form answers), and the README tokenizer corpus's Stack
Exchange share. It is the practical substitute for the Dolma Stack Exchange
slice: Dolma's loader is a dataset script, which the current ``datasets`` no
longer supports.

Each row is one accepted-answer Q/A pair (``INSTRUCTION`` / ``RESPONSE``);
rendered as one document::

    {question}

    {answer}

All machinery lives in :class:`chimera.data.text.hf_text.HFTextDataModule`. The
corpus ships only a ``train`` split, so ``VAL_FROM_TRAIN`` carves validation
off its head.

Usage:
    dm = StackExchangeDataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule


class StackExchangeDataModule(HFTextDataModule):
    HF_REPO = "donfu/oa-stackexchange"
    DIR_NAME = "stackexchange"
    UNIT = "qa"
    VAL_FROM_TRAIN = 0.01

    def _row_text(self, row) -> str:
        return f"{row['INSTRUCTION']}\n\n{row['RESPONSE']}"

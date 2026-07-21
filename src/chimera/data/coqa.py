"""
CoQA-as-text BPE DataModule.

CoQA (``stanfordnlp/coqa``, CC BY-SA / MIT / Apache depending on source
passage) is grounded conversational QA: a passage followed by a chain of
free-form questions whose answers are grounded in the passage, with follow-ups
that refer back to earlier turns. Its role in a pretrain mix is the GROUNDED,
multi-turn answer format — the same "answer from the given text" register as
SQuAD but conversational — and it is part of the README tokenizer corpus's
grounded-QA share.

Each row is one passage + its whole question/answer chain, rendered as one
document::

    {story}

    Question: {question 1}
    Answer: {answer 1}

    Question: {question 2}
    Answer: {answer 2}

All machinery lives in :class:`chimera.data.hf_text.HFTextDataModule`; the
corpus ships native ``train`` and ``validation`` splits.

Usage:
    dm = CoQADataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule


def _render(story: str, questions, answers) -> str:
    qas = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]
    return "\n\n".join([story, *qas])


class CoQADataModule(HFTextDataModule):
    HF_REPO = "stanfordnlp/coqa"
    DIR_NAME = "coqa"
    VAL_SPLIT = "validation"
    UNIT = "passage"

    def _row_text(self, row) -> str:
        return _render(row["story"], row["questions"], row["answers"]["input_text"])

    def iter_texts(self, ds, batch_size: int = 1024):
        for batch in ds.iter(batch_size=batch_size):
            for story, questions, answers in zip(
                batch["story"], batch["questions"], batch["answers"]
            ):
                yield _render(story, questions, answers["input_text"])


if __name__ == "__main__":
    import os

    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    dm = CoQADataModule(data_dir="/mnt/ai/data", max_train_tokens=5_000_000)
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}")
    print(f"train batch: x={x.shape}, y={y.shape}")

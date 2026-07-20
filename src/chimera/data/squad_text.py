"""
SQuAD-as-text BPE DataModule.

SQuAD v1.1 (``rajpurkar/squad``, cc-by-sa-4.0) is extractive reading
comprehension: ~88k crowd-written questions whose answers are literal spans of
Wikipedia paragraphs. Its role in a pretrain mix is the GROUNDED response
format — answer from the given text — complementing GooAQ's closed-book
Question:/Answer: pairs. Rows sharing a paragraph are consecutive in the
dataset, so they are grouped into one document per passage::

    {context}

    Question: {question 1}
    Answer: {answer 1}

    Question: {question 2}
    Answer: {answer 2}

All machinery lives in :class:`chimera.data.hf_text.HFTextDataModule`; the
corpus is small (~20M chars of unique passages) and ships a native
``validation`` split.

Usage:
    dm = SQuADTextDataModule(data_dir="/mnt/ai/data", add_bos=True)
    dm.prepare_data(); dm.setup("fit")
"""

from .hf_text import HFTextDataModule


class SQuADTextDataModule(HFTextDataModule):
    HF_REPO = "rajpurkar/squad"
    DIR_NAME = "squad"
    UNIT = "passage"

    def _row_text(self, row) -> str:
        # single-row rendering (tokenizer-training sample path)
        answer = row["answers"]["text"][0]
        return f"{row['context']}\n\nQuestion: {row['question']}\nAnswer: {answer}"

    def iter_texts(self, ds, batch_size: int = 1024):
        """One document per passage: context + all its Q/A pairs.

        Rows for the same paragraph are consecutive in SQuAD, so grouping
        consecutive equal contexts recovers the per-passage structure without
        a global shuffle-and-sort.
        """
        context, qas = None, []
        for batch in ds.iter(batch_size=batch_size):
            for ctx, q, ans in zip(
                batch["context"], batch["question"], batch["answers"]
            ):
                if ctx != context:
                    if qas:
                        yield "\n\n".join([context] + qas)
                    context, qas = ctx, []
                qas.append(f"Question: {q}\nAnswer: {ans['text'][0]}")
        if qas:
            yield "\n\n".join([context] + qas)


if __name__ == "__main__":
    import os

    os.environ.setdefault("HF_HOME", "/mnt/ai/data/hf")
    dm = SQuADTextDataModule(data_dir="/mnt/ai/data", max_train_tokens=5_000_000)
    dm.prepare_data()
    dm.setup("fit")
    x, y = next(iter(dm.train_dataloader()))
    print(f"vocab_size={dm.vocab_size}")
    print(f"train batch: x={x.shape}, y={y.shape}")

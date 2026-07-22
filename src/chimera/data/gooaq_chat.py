"""GooAQ single-turn chat SFT module (closed-book simple QA)."""

from .chat_sft import ChatSFTDataModule


class GooAQChatDataModule(ChatSFTDataModule):
    """Closed-book simple QA: GooAQ pairs as single-turn chat."""

    HF_REPO = "sentence-transformers/gooaq"
    DIR_NAME = "gooaq-chat"
    UNIT = "pair"
    DATA_FILES = ["pair/train-00000-of-00002.parquet"]
    VAL_FROM_TRAIN = 0.01

    def row_to_messages(self, row) -> list[dict]:
        return [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]

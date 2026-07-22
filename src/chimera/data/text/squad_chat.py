"""SQuAD grounded-QA chat SFT module."""

from .chat_sft import ChatSFTDataModule


class SQuADChatDataModule(ChatSFTDataModule):
    """Grounded QA: passage + question in the user turn, span answer back."""

    HF_REPO = "rajpurkar/squad"
    DIR_NAME = "squad-chat"
    UNIT = "qa"

    def row_to_messages(self, row) -> list[dict]:
        return [
            {"role": "user", "content": f"{row['context']}\n\n{row['question']}"},
            {"role": "assistant", "content": row["answers"]["text"][0]},
        ]

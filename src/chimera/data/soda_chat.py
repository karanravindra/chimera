"""SODA social-commonsense dialog chat SFT module."""

from .chat_sft import ChatSFTDataModule


class SODAChatDataModule(ChatSFTDataModule):
    """Social-commonsense dialog (SODA). Two speakers alternate; rendered as a
    user/assistant chat (turn 0 = user). Synthetic — sample lightly (a low cap)
    to avoid imprinting its repetitive conversational style."""

    HF_REPO = "allenai/soda"
    DIR_NAME = "soda-chat"
    UNIT = "dialog"
    VAL_FROM_TRAIN = 0.01

    def row_to_messages(self, row) -> list[dict]:
        # alternate roles by turn (SODA dialogs strictly alternate two speakers)
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": t}
            for i, t in enumerate(row["dialogue"])
            if t and t.strip()
        ]

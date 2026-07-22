"""Everyday-conversations (smoltalk) multi-turn small-talk chat SFT module."""

from .chat_sft import ChatSFTDataModule


class EverydayConversationsDataModule(ChatSFTDataModule):
    """Multi-turn small-talk (smoltalk/everyday-conversations): chat style."""

    HF_REPO = "HuggingFaceTB/smoltalk"
    DIR_NAME = "everyday-conversations"
    CONFIG_NAME = "everyday-conversations"
    VAL_SPLIT = "test"

    def row_to_messages(self, row) -> list[dict]:
        return row["messages"]

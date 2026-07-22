"""No Robots human-authored instruction-breadth chat SFT module."""

from .chat_sft import ChatSFTDataModule


class NoRobotsChatDataModule(ChatSFTDataModule):
    """Human-authored instruction breadth (No Robots): summarize, rewrite,
    extract, classify, brainstorm, open-ended. The ``messages`` column is already
    ChatML-shaped. NOTE: CC BY-NC 4.0 — review before any non-research use."""

    HF_REPO = "HuggingFaceH4/no_robots"
    DIR_NAME = "no-robots"
    UNIT = "conv"
    VAL_SPLIT = "test"

    def row_to_messages(self, row) -> list[dict]:
        return row["messages"]

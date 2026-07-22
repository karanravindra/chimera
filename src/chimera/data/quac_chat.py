"""QuAC grounded information-seeking dialog chat SFT module."""

from .chat_sft import ChatSFTDataModule


class QuACChatDataModule(ChatSFTDataModule):
    """Grounded information-seeking dialog (QuAC) with explicit unanswerable
    questions — teaches the model NOT to guess beyond the passage. Parquet mirror
    (``yairfeldman/quac``; the official ``allenai/quac`` is a broken load script).
    Section passage in the first user turn; ``CANNOTANSWER`` spans are rewritten to
    a natural refusal so the model learns to decline rather than emit a sentinel."""

    HF_REPO = "yairfeldman/quac"
    DIR_NAME = "quac-chat"
    UNIT = "dialog"
    VAL_FROM_TRAIN = 0.01

    _CANNOT = "CANNOTANSWER"
    _REFUSAL = "I don't know based on the passage."

    def row_to_messages(self, row) -> list[dict]:
        context = row["context"]
        questions = row["questions"]
        answers = row["orig_answers"]["texts"]
        msgs: list[dict] = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            a = self._REFUSAL if a.strip() == self._CANNOT else a
            user = f"{context}\n\n{q}" if i == 0 else q
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": a})
        return msgs

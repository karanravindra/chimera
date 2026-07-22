"""CoQA grounded conversational-QA chat SFT module."""

from .chat_sft import ChatSFTDataModule


class CoQAChatDataModule(ChatSFTDataModule):
    """Grounded conversational QA (CoQA): a passage then a multi-turn Q/A dialog,
    all answers grounded in the passage. The story goes in the FIRST user turn
    (the proven SQuAD-chat pattern) so grounding is visible; follow-up questions
    are bare turns that reference earlier context. At seq-512 long dialogs get
    split across windows — the story stays visible for the turns within ~512 tok
    of it, which covers most CoQA conversations (short answers)."""

    HF_REPO = "stanfordnlp/coqa"
    DIR_NAME = "coqa-chat"
    UNIT = "passage"
    VAL_FROM_TRAIN = 0.01  # carve val off train (robust vs relying on a split)

    def row_to_messages(self, row) -> list[dict]:
        story = row["story"]
        questions = row["questions"]
        answers = row["answers"]["input_text"]
        msgs: list[dict] = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            user = f"{story}\n\n{q}" if i == 0 else q
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": a})
        return msgs

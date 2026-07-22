"""Source rows to canonical text examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .chat_template import iter_segments, normalize_messages
from .schema import TextExample, TextSegment


class TextAdapter(Protocol):
    key: str

    def iter_examples(self, rows) -> Iterable[TextExample]: ...


def _get(row, path: str):
    value = row
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else value[part]
    return value


def _chat_example(messages: Sequence[dict]) -> TextExample:
    normalized = normalize_messages(messages)
    segments = tuple(
        TextSegment(text=text, kind="assistant" if supervised else "context")
        for text, supervised in iter_segments(normalized)
    )
    return TextExample(segments)


@dataclass(frozen=True)
class PlainTextAdapter:
    columns: tuple[str, ...] = ("text",)
    join: str = "\n\n"
    key: str = "plain"

    def iter_examples(self, rows):
        for row in rows:
            text = self.join.join(str(_get(row, column)) for column in self.columns)
            if text:
                yield TextExample((TextSegment(text),))


@dataclass(frozen=True)
class PairTextAdapter:
    question: str
    answer: str
    question_prefix: str = "Question: "
    answer_prefix: str = "Answer: "
    key: str = "pair-text"

    def iter_examples(self, rows):
        for row in rows:
            text = (
                f"{self.question_prefix}{_get(row, self.question)}\n"
                f"{self.answer_prefix}{_get(row, self.answer)}"
            )
            yield TextExample((TextSegment(text),))


@dataclass(frozen=True)
class JoinedPairAdapter:
    left: str
    right: str
    join: str = "\n\n"
    key: str = "joined-pair"

    def iter_examples(self, rows):
        for row in rows:
            yield TextExample(
                (
                    TextSegment(
                        f"{_get(row, self.left)}{self.join}{_get(row, self.right)}"
                    ),
                )
            )


@dataclass(frozen=True)
class SQuADPassageAdapter:
    key: str = "squad-passage"

    def iter_examples(self, rows):
        context, qas = None, []
        for row in rows:
            if row["context"] != context:
                if qas:
                    yield TextExample((TextSegment("\n\n".join([context, *qas])),))
                context, qas = row["context"], []
            qas.append(
                f"Question: {row['question']}\nAnswer: {row['answers']['text'][0]}"
            )
        if qas:
            yield TextExample((TextSegment("\n\n".join([context, *qas])),))


@dataclass(frozen=True)
class CoQAPlainAdapter:
    key: str = "coqa-plain"

    def iter_examples(self, rows):
        for row in rows:
            qas = [
                f"Question: {question}\nAnswer: {answer}"
                for question, answer in zip(
                    row["questions"], row["answers"]["input_text"]
                )
            ]
            yield TextExample((TextSegment("\n\n".join([row["story"], *qas])),))


@dataclass(frozen=True)
class MessageListAdapter:
    column: str = "messages"
    key: str = "message-list"

    def iter_examples(self, rows):
        for row in rows:
            messages = row[self.column]
            if messages:
                yield _chat_example(messages)


@dataclass(frozen=True)
class SingleTurnChatAdapter:
    question: str
    answer: str
    context: str | None = None
    key: str = "single-turn-chat"

    def iter_examples(self, rows):
        for row in rows:
            question = str(_get(row, self.question))
            if self.context:
                question = f"{_get(row, self.context)}\n\n{question}"
            yield _chat_example(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": str(_get(row, self.answer))},
                ]
            )


@dataclass(frozen=True)
class ConversationalQAAdapter:
    context: str
    questions: str
    answers: str
    cannot_answer: str | None = None
    refusal: str = "I don't know based on the passage."
    key: str = "conversational-qa"

    def iter_examples(self, rows):
        for row in rows:
            messages = []
            for index, (question, answer) in enumerate(
                zip(_get(row, self.questions), _get(row, self.answers))
            ):
                if self.cannot_answer and answer.strip() == self.cannot_answer:
                    answer = self.refusal
                user = (
                    f"{_get(row, self.context)}\n\n{question}"
                    if index == 0
                    else question
                )
                messages.extend(
                    [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": answer},
                    ]
                )
            if messages:
                yield _chat_example(messages)


@dataclass(frozen=True)
class AlternatingDialogueAdapter:
    column: str = "dialogue"
    key: str = "alternating-dialogue"

    def iter_examples(self, rows):
        for row in rows:
            messages = [
                {"role": "user" if i % 2 == 0 else "assistant", "content": text}
                for i, text in enumerate(row[self.column])
                if text and text.strip()
            ]
            if messages:
                yield _chat_example(messages)


@dataclass(frozen=True)
class OASSTTreeAdapter:
    """Reconstruct one best-ranked English conversation from each OASST tree."""

    key: str = "oasst-tree"

    def iter_examples(self, rows):
        by_id = {
            row["message_id"]: row
            for row in rows
            if row["lang"] == "en" and row["review_result"]
        }
        children: dict[str, list] = {}
        roots = []
        for row in by_id.values():
            parent = row["parent_id"]
            if parent is None or parent not in by_id:
                roots.append(row)
            else:
                children.setdefault(parent, []).append(row)

        def rank(row):
            return row["rank"] if row["rank"] is not None else float("inf")

        for root in roots:
            messages, node = [], root
            while node is not None:
                messages.append(
                    {
                        "role": ("user" if node["role"] == "prompter" else "assistant"),
                        "content": node["text"],
                    }
                )
                descendants = sorted(children.get(node["message_id"], ()), key=rank)
                node = descendants[0] if descendants else None
            if len(messages) >= 2:
                yield _chat_example(messages)

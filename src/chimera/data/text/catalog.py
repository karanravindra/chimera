"""Typed catalog of raw sources and their reusable text views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Union

from chimera.data.manifest import CatalogLock
from chimera.data.source import DataFiles, HFSource
from chimera.data.splits import CarvedValidation, NativeSplits, SplitPolicy

from .adapters import (
    AlternatingDialogueAdapter,
    CoQAPlainAdapter,
    ConversationalQAAdapter,
    JoinedPairAdapter,
    MessageListAdapter,
    OASSTTreeAdapter,
    PairTextAdapter,
    PlainTextAdapter,
    SQuADPassageAdapter,
    SingleTurnChatAdapter,
    TextAdapter,
)


@dataclass(frozen=True)
class TextSourceSpec:
    key: str
    source: HFSource
    splits: SplitPolicy
    license: object
    provenance: str


@dataclass(frozen=True)
class TextViewSpec:
    key: str
    source: str
    adapter: TextAdapter
    objective: str = "next-token"
    unit: str = "doc"
    exclude_from_tokenizer: bool = False
    splits: SplitPolicy | None = None


@dataclass(frozen=True)
class LocalTextView:
    """A runtime-configured view over deterministic local text files."""

    key: str
    directory: Path
    glob: str = "*.md"
    repeat: int = 1
    adapter: TextAdapter = PlainTextAdapter(("text",))
    objective: str = "next-token"
    unit: str = "doc"
    exclude_from_tokenizer: bool = True

    def __post_init__(self):
        if self.repeat < 1:
            raise ValueError("local text repeat must be at least one")


TextView = Union[TextViewSpec, LocalTextView]


SOURCES: Mapping[str, TextSourceSpec] = {
    "tinystories-v2": TextSourceSpec(
        "tinystories-v2",
        HFSource("noanabeshima/TinyStoriesV2"),
        NativeSplits(),
        "cdla-sharing-1.0",
        "https://huggingface.co/datasets/roneneldan/TinyStories",
    ),
    "tiny-textbooks": TextSourceSpec(
        "tiny-textbooks",
        HFSource("nampdn-ai/tiny-textbooks", gated=True),
        NativeSplits(validation="test"),
        "apache-2.0",
        "https://huggingface.co/datasets/nampdn-ai/tiny-textbooks",
    ),
    "tiny-strange-textbooks": TextSourceSpec(
        "tiny-strange-textbooks",
        HFSource(
            "nampdn-ai/tiny-strange-textbooks",
            data_files=tuple(f"data_part_{i}.parquet" for i in range(4)),
            gated=True,
        ),
        CarvedValidation(),
        "apache-2.0",
        "https://huggingface.co/datasets/nampdn-ai/tiny-strange-textbooks",
    ),
    "tiny-webtext": TextSourceSpec(
        "tiny-webtext",
        HFSource("nampdn-ai/tiny-webtext", data_files="train/en/*.parquet", gated=True),
        CarvedValidation(),
        "mit",
        "https://huggingface.co/datasets/nampdn-ai/tiny-webtext",
    ),
    "fineweb-edu": TextSourceSpec(
        "fineweb-edu",
        HFSource(
            "HuggingFaceFW/fineweb-edu", data_files="sample/10BT/000_00000.parquet"
        ),
        CarvedValidation(),
        "odc-by",
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
    ),
    "cosmopedia-v2": TextSourceSpec(
        "cosmopedia-v2",
        HFSource(
            "HuggingFaceTB/smollm-corpus",
            data_files=tuple(
                f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(2)
            ),
        ),
        CarvedValidation(),
        "odc-by",
        "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus",
    ),
    "gooaq": TextSourceSpec(
        "gooaq",
        HFSource(
            "sentence-transformers/gooaq",
            data_files=("pair/train-00000-of-00002.parquet",),
        ),
        CarvedValidation(),
        "apache-2.0",
        "https://github.com/allenai/gooaq",
    ),
    "squad": TextSourceSpec(
        "squad",
        HFSource("rajpurkar/squad"),
        NativeSplits(),
        "cc-by-sa-4.0",
        "https://rajpurkar.github.io/SQuAD-explorer/",
    ),
    "coqa": TextSourceSpec(
        "coqa",
        HFSource("stanfordnlp/coqa"),
        NativeSplits(),
        ("other",),
        "https://stanfordnlp.github.io/coqa/",
    ),
    "stackexchange": TextSourceSpec(
        "stackexchange",
        HFSource("donfu/oa-stackexchange"),
        CarvedValidation(),
        "cc-by-sa-4.0",
        "https://huggingface.co/datasets/donfu/oa-stackexchange",
    ),
    "wikipedia": TextSourceSpec(
        "wikipedia",
        HFSource(
            "wikimedia/wikipedia",
            data_files=("20231101.en/train-00000-of-00041.parquet",),
        ),
        CarvedValidation(),
        ("cc-by-sa-3.0", "gfdl"),
        "https://dumps.wikimedia.org/",
    ),
    "smoltalk": TextSourceSpec(
        "smoltalk",
        HFSource("HuggingFaceTB/smoltalk", config="everyday-conversations"),
        NativeSplits(validation="test"),
        "unknown",
        "https://huggingface.co/datasets/HuggingFaceTB/smoltalk",
    ),
    "no-robots": TextSourceSpec(
        "no-robots",
        HFSource("HuggingFaceH4/no_robots"),
        NativeSplits(validation="test"),
        "cc-by-nc-4.0",
        "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
    ),
    "quac": TextSourceSpec(
        "quac",
        HFSource("yairfeldman/quac"),
        CarvedValidation(),
        "unknown",
        "https://quac.ai/",
    ),
    "soda": TextSourceSpec(
        "soda",
        HFSource("allenai/soda"),
        CarvedValidation(),
        ("cc-by-4.0",),
        "https://huggingface.co/datasets/allenai/soda",
    ),
    "oasst1": TextSourceSpec(
        "oasst1",
        HFSource("OpenAssistant/oasst1"),
        NativeSplits(),
        "apache-2.0",
        "https://huggingface.co/datasets/OpenAssistant/oasst1",
    ),
    "ultrachat-200k": TextSourceSpec(
        "ultrachat-200k",
        HFSource("HuggingFaceH4/ultrachat_200k"),
        NativeSplits(train="train_sft", validation="test_sft"),
        "mit",
        "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
    ),
}


VIEWS: Mapping[str, TextViewSpec] = {
    "tinystories-v2.pretrain": TextViewSpec(
        "tinystories-v2.pretrain",
        "tinystories-v2",
        PlainTextAdapter(("text",)),
        unit="story",
    ),
    "tiny-textbooks.pretrain": TextViewSpec(
        "tiny-textbooks.pretrain", "tiny-textbooks", PlainTextAdapter(("textbook",))
    ),
    "tiny-strange-textbooks.pretrain": TextViewSpec(
        "tiny-strange-textbooks.pretrain",
        "tiny-strange-textbooks",
        PlainTextAdapter(("text",)),
    ),
    "tiny-webtext.pretrain": TextViewSpec(
        "tiny-webtext.pretrain",
        "tiny-webtext",
        JoinedPairAdapter("human", "bot"),
    ),
    "fineweb-edu.pretrain": TextViewSpec(
        "fineweb-edu.pretrain", "fineweb-edu", PlainTextAdapter(("text",))
    ),
    "cosmopedia-v2.pretrain": TextViewSpec(
        "cosmopedia-v2.pretrain", "cosmopedia-v2", PlainTextAdapter(("text",))
    ),
    "gooaq.pretrain": TextViewSpec(
        "gooaq.pretrain", "gooaq", PairTextAdapter("question", "answer"), unit="pair"
    ),
    "squad.pretrain": TextViewSpec(
        "squad.pretrain", "squad", SQuADPassageAdapter(), unit="passage"
    ),
    "coqa.pretrain": TextViewSpec(
        "coqa.pretrain", "coqa", CoQAPlainAdapter(), unit="passage"
    ),
    "stackexchange.pretrain": TextViewSpec(
        "stackexchange.pretrain",
        "stackexchange",
        JoinedPairAdapter("INSTRUCTION", "RESPONSE"),
        unit="qa",
    ),
    "wikipedia.pretrain": TextViewSpec(
        "wikipedia.pretrain", "wikipedia", PlainTextAdapter(("text",)), unit="article"
    ),
    "gooaq.sft": TextViewSpec(
        "gooaq.sft",
        "gooaq",
        SingleTurnChatAdapter("question", "answer"),
        "assistant-only",
        "pair",
    ),
    "squad.sft": TextViewSpec(
        "squad.sft",
        "squad",
        SingleTurnChatAdapter("question", "answers.text.0", context="context"),
        "assistant-only",
        "qa",
    ),
    "coqa.sft": TextViewSpec(
        "coqa.sft",
        "coqa",
        ConversationalQAAdapter("story", "questions", "answers.input_text"),
        "assistant-only",
        "passage",
        splits=CarvedValidation(),
    ),
    "quac.sft": TextViewSpec(
        "quac.sft",
        "quac",
        ConversationalQAAdapter(
            "context", "questions", "orig_answers.texts", cannot_answer="CANNOTANSWER"
        ),
        "assistant-only",
        "dialog",
    ),
    "no-robots.sft": TextViewSpec(
        "no-robots.sft", "no-robots", MessageListAdapter(), "assistant-only", "conv"
    ),
    "smoltalk.everyday.sft": TextViewSpec(
        "smoltalk.everyday.sft",
        "smoltalk",
        MessageListAdapter(),
        "assistant-only",
        "conv",
    ),
    "soda.sft": TextViewSpec(
        "soda.sft", "soda", AlternatingDialogueAdapter(), "assistant-only", "dialog"
    ),
    "oasst1.sft": TextViewSpec(
        "oasst1.sft", "oasst1", OASSTTreeAdapter(), "assistant-only", "conv"
    ),
    "ultrachat-200k.sft": TextViewSpec(
        "ultrachat-200k.sft",
        "ultrachat-200k",
        MessageListAdapter(),
        "assistant-only",
        "conv",
    ),
}


LOCK_PATH = Path(__file__).with_name("catalog.lock.json")


def get_source(key: str) -> TextSourceSpec:
    try:
        return SOURCES[key]
    except KeyError as error:
        raise KeyError(f"unknown text source {key!r}") from error


def get_view(key: str | LocalTextView) -> TextView:
    if isinstance(key, LocalTextView):
        return key
    try:
        return VIEWS[key]
    except KeyError as error:
        raise KeyError(f"unknown text view {key!r}") from error


def load_rows(
    view: TextView,
    logical_split: str,
    *,
    data_dir: Path,
    data_files: DataFiles = None,
    streaming: bool = False,
):
    if isinstance(view, LocalTextView):
        if data_files is not None:
            raise ValueError("data_files cannot override a local text view")
        files = tuple(sorted(view.directory.glob(view.glob)))
        if not files:
            raise FileNotFoundError(
                f"no files matching {view.glob!r} in {view.directory}"
            )
        repeats = view.repeat if logical_split == "train" else 1
        return (
            {"text": path.read_text(encoding="utf-8"), "path": str(path)}
            for _ in range(repeats)
            for path in files
        )
    source_spec = get_source(view.source)
    lock = CatalogLock.load(LOCK_PATH).require(source_spec.key, source_spec.source.repo)
    source = (
        source_spec.source
        if data_files is None
        else source_spec.source.select(data_files=data_files)
    )
    return source.load(
        split=(view.splits or source_spec.splits).resolve(logical_split),
        revision=lock.revision,
        cache_dir=data_dir / "hf_cache",
        streaming=streaming,
    )

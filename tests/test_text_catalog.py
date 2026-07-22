from pathlib import Path

import pytest

from chimera.data.manifest import CatalogLock
from chimera.data.splits import CarvedValidation, NativeSplits
from chimera.data.text.catalog import (
    LOCK_PATH,
    SOURCES,
    VIEWS,
    LocalTextView,
    get_view,
    load_rows,
)
from chimera.data.text.compiler import artifact_directory, build_descriptor
from chimera.data.text.adapters import OASSTTreeAdapter


def test_every_catalog_source_is_locked_to_a_commit():
    locks = CatalogLock.load(LOCK_PATH)
    assert set(locks.entries) == set(SOURCES)
    for key, source in SOURCES.items():
        lock = locks.require(key, source.source.repo)
        assert len(lock.revision) == 40
        assert lock.gated == source.source.gated
        assert lock.license == source.license


def test_every_view_references_a_source_and_known_objective():
    assert VIEWS
    for key, view in VIEWS.items():
        assert get_view(key) is view
        assert view.source in SOURCES
        assert view.objective in {"next-token", "assistant-only"}


def test_split_policies_map_logical_splits():
    native = NativeSplits(validation="test")
    carved = CarvedValidation(fraction=0.01)
    assert native.resolve("validation") == "test"
    assert carved.resolve("validation") == "train[:1%]"
    assert carved.resolve("train") == "train[1%:]"


def test_local_view_is_content_addressed_and_repeated_only_for_train(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("first")
    view = LocalTextView("notes.pretrain", tmp_path, repeat=3)

    train = list(load_rows(view, "train", data_dir=Path("unused")))
    validation = list(load_rows(view, "validation", data_dir=Path("unused")))
    assert [row["text"] for row in train] == ["first"] * 3
    assert [row["text"] for row in validation] == ["first"]

    before = build_descriptor(
        view,
        "train",
        tokenizer_hash="tok",
        max_tokens=None,
        bos_id=1,
        eos_id=0,
        data_files=None,
        shard_tokens=100,
    )
    path.write_text("second")
    after = build_descriptor(
        view,
        "train",
        tokenizer_hash="tok",
        max_tokens=None,
        bos_id=1,
        eos_id=0,
        data_files=None,
        shard_tokens=100,
    )
    assert before != after


def test_oasst_adapter_selects_best_ranked_reviewed_english_path():
    rows = [
        {
            "message_id": "root",
            "parent_id": None,
            "lang": "en",
            "review_result": True,
            "rank": None,
            "role": "prompter",
            "text": "Question",
        },
        {
            "message_id": "best",
            "parent_id": "root",
            "lang": "en",
            "review_result": True,
            "rank": 0,
            "role": "assistant",
            "text": "Best answer",
        },
        {
            "message_id": "other",
            "parent_id": "root",
            "lang": "en",
            "review_result": True,
            "rank": 1,
            "role": "assistant",
            "text": "Other answer",
        },
    ]
    example = next(iter(OASSTTreeAdapter().iter_examples(rows)))
    assert "Best answer" in example.text
    assert "Other answer" not in example.text


def test_artifact_view_key_cannot_escape_data_directory(tmp_path):
    with pytest.raises(ValueError, match="unsafe text view key"):
        artifact_directory(tmp_path, {"view": "../../outside"})

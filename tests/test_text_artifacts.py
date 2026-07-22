import json

import pytest
import torch

from chimera.data.text.artifacts import (
    DocumentWindowArtifactDataset,
    PackedArtifactDataset,
    ShardedTokenStore,
    TextArtifactWriter,
)


def _write_artifact(tmp_path, *, labels=False):
    writer = TextArtifactWriter(
        tmp_path,
        build={"test": True},
        dtype=torch.int16,
        stores_labels=labels,
        shard_tokens=4,
    )
    writer.add_document([1, 2, 3], [-100, 2, 3] if labels else None)
    writer.add_document([4, 5, 6], [-100, 5, 6] if labels else None)
    return writer.finish()


def test_sharded_store_slices_across_document_aligned_shards(tmp_path):
    manifest = _write_artifact(tmp_path)
    assert manifest.tokens == 6
    assert manifest.documents == 2
    assert len(manifest.shards) == 2

    store = ShardedTokenStore(tmp_path, verify=True)
    assert store.slice(2, 5).tolist() == [3, 4, 5]
    assert store.document_spans == ((0, 3), (3, 6))

    dataset = PackedArtifactDataset(store, seq_len=2)
    assert len(dataset) == 2
    assert tuple(part.tolist() for part in dataset[1]) == ([3, 4], [4, 5])


def test_labeled_artifact_preserves_assistant_only_targets(tmp_path):
    _write_artifact(tmp_path, labels=True)
    store = ShardedTokenStore(tmp_path)
    x, y = PackedArtifactDataset(store, seq_len=2)[0]
    assert x.tolist() == [1, 2]
    assert y.tolist() == [2, 3]


def test_document_windows_never_cross_document_boundaries(tmp_path):
    writer = TextArtifactWriter(
        tmp_path,
        build={},
        dtype=torch.int16,
        stores_labels=False,
        shard_tokens=100,
    )
    writer.add_document(list(range(10, 20)))
    writer.add_document(list(range(30, 40)))
    writer.finish()
    dataset = DocumentWindowArtifactDataset(
        ShardedTokenStore(tmp_path), seq_len=4, max_windows_per_doc=2, seed=7
    )

    for index in range(len(dataset)):
        x, y = dataset[index]
        assert (x < 20).all() or (x >= 30).all()
        assert y[-1] == x[-1] + 1


def test_manifest_is_rejected_when_version_is_unknown(tmp_path):
    _write_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["version"] = 999
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsupported text artifact version"):
        ShardedTokenStore(tmp_path)

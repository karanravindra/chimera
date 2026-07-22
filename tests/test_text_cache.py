import torch

from chimera.data.text.datasets import tokenize_with_progress
from chimera.data.text.hf_text import HFTextDataModule


class _Tokenizer:
    def __init__(self, vocab_size, encoded):
        self.vocab_size = vocab_size
        self.encoded = encoded

    def encode_batch(self, texts, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [list(self.encoded[text]) for text in texts]


def test_large_vocab_uses_int32_without_overflow():
    tokenizer = _Tokenizer(64_402, {"doc": [40_000, 64_401]})
    data = tokenize_with_progress(tokenizer, ["doc"], eos_id=2, total=1)
    assert data.dtype == torch.int32
    assert data.tolist() == [40_000, 64_401, 2]


def test_total_cap_keeps_complete_document_boundary():
    tokenizer = _Tokenizer(100, {"one": [10, 11], "two": [20, 21, 22]})
    data = tokenize_with_progress(
        tokenizer, ["one", "two"], eos_id=2, max_tokens=6, total=2
    )
    assert data.tolist() == [10, 11, 2]


def test_first_long_document_is_truncated_before_eos():
    tokenizer = _Tokenizer(64_402, {"long": [40_000, 40_001, 40_002, 40_003]})
    data = tokenize_with_progress(
        tokenizer,
        ["long"],
        bos_id=1,
        eos_id=2,
        max_tokens=5,
        total=1,
    )
    assert data.dtype == torch.int32
    assert data.tolist() == [1, 40_000, 40_001, 40_002, 2]


class _Rows:
    def __init__(self, texts):
        self.texts = texts

    def __iter__(self):
        return iter([{"text": text} for text in self.texts])

    def __len__(self):
        return len(self.texts)

    def iter(self, batch_size):
        del batch_size
        yield {"text": self.texts}


class _LocalTextDataModule(HFTextDataModule):
    HF_REPO = "local/test"
    DIR_NAME = "local-test"


def test_prepare_builds_and_setup_only_reads_caches(tmp_path):
    kwargs = dict(
        data_dir=str(tmp_path),
        tokenizer_backend="scratch",
        vocab_size=256,
        tokenizer_train_chars=8,
        add_eos=False,
        max_train_tokens=20,
        max_val_tokens=20,
        num_workers=0,
    )
    builder = _LocalTextDataModule(**kwargs)
    builder._load_dataset = lambda split: _Rows([f"{split} text"])
    builder.prepare_data()

    cache_payloads = [
        torch.load(path, weights_only=False)
        for path in (tmp_path / "local-test").glob("ids_v2_*.pt")
    ]
    assert len(cache_payloads) == 2
    assert all(payload["version"] == 2 for payload in cache_payloads)

    reader = _LocalTextDataModule(**kwargs)

    def fail_if_loaded(split):
        raise AssertionError(f"setup attempted to load raw split {split}")

    reader._load_dataset = fail_if_loaded
    reader.setup("fit")
    assert len(reader.train_dataset.data) > 0
    assert len(reader.val_dataset.data) > 0

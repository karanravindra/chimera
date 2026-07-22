from chimera.data.text import (
    LocalTextView,
    MixtureSource,
    TextDataModule,
    TextMixtureSpec,
    TokenizerSpec,
)
from chimera.data.text.chat_template import SPECIAL_TOKENS
from chimera.tokenizers import BPETokenizer


def test_local_text_datamodule_builds_v3_artifacts_and_loaders(tmp_path):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "one.md").write_text("one two three four five six")
    (documents / "two.md").write_text("seven eight nine ten eleven twelve")

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = BPETokenizer(backend="hf")
    tokenizer.train(
        [path.read_text() for path in documents.iterdir()],
        vocab_size=300,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save(tokenizer_path)

    dm = TextDataModule(
        TextMixtureSpec(
            sources=(
                MixtureSource(
                    LocalTextView("notes.pretrain", documents),
                    max_train_tokens=None,
                    max_val_tokens=None,
                ),
            ),
            tokenizer=TokenizerSpec.pinned(tokenizer_path),
            add_bos=True,
            shard_tokens=8,
        ),
        data_dir=str(tmp_path / "data"),
        batch_size=1,
        seq_len=4,
        num_workers=0,
        pin_memory=False,
        verify_artifacts=True,
    )
    dm.prepare_data()
    dm.setup("fit")

    assert dm.vocab_size == tokenizer.vocab_size
    assert dm.source_train_tokens["notes.pretrain"] > 0
    assert list((tmp_path / "data/text/artifacts/v3").rglob("manifest.json"))
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == y.shape == (1, 4)

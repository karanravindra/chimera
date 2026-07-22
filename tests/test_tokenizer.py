import pytest

from chimera.tokenizers import BPETokenizer


def test_fast_backend_is_raw_and_lossless_by_default():
    pytest.importorskip("tokenizers")
    from tokenizers.processors import TemplateProcessing

    tokenizer = BPETokenizer(backend="hf").train(
        "hello world", vocab_size=300, special_tokens=["<bos>"]
    )
    bos_id = tokenizer._tok.token_to_id("<bos>")
    tokenizer._tok.post_processor = TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", bos_id)]
    )

    raw = tokenizer.encode("hello")
    processed = tokenizer.encode("hello", add_special_tokens=True)

    assert raw[0] != bos_id
    assert processed[0] == bos_id
    assert tokenizer.decode([bos_id, *raw]) == "<bos>hello"
    assert tokenizer.decode([bos_id, *raw], skip_special_tokens=True) == "hello"

"""Tests for chimera.models.attention doc-masking helpers.

The dense masks are the readable spec; doc_ids_and_pos feeds both the dense
document mask and the FlexAttention block mask, so pinning it against a
brute-force per-token walk covers the masking rule itself.
"""

import torch

from chimera.models.attention import (
    doc_ids_and_pos,
    make_causal_mask,
    make_document_mask,
)

EOS = 0


def brute_force_doc_ids_and_pos(row: list[int], eos_id: int):
    """Per-token doc id / position by walking the sequence: the EOS token is the
    last token of the doc it closes; positions restart at 0 after each EOS."""
    doc_ids, pos_ids = [], []
    doc, pos = 0, 0
    for tok in row:
        doc_ids.append(doc)
        pos_ids.append(pos)
        if tok == eos_id:
            doc += 1
            pos = 0
        else:
            pos += 1
    return doc_ids, pos_ids


def test_doc_ids_and_pos_matches_brute_force():
    x = torch.tensor(
        [
            [5, 3, EOS, 7, 7, 7, EOS, 2],  # two closed docs + one open
            [EOS, EOS, 4, 4, 4, 4, 4, 4],  # empty docs (back-to-back EOS)
            [9, 8, 7, 6, 5, 4, 3, 2],  # single doc, no EOS at all
        ]
    )
    doc_ids, pos_ids = doc_ids_and_pos(x, EOS)
    for b in range(x.shape[0]):
        want_doc, want_pos = brute_force_doc_ids_and_pos(x[b].tolist(), EOS)
        assert doc_ids[b].tolist() == want_doc
        assert pos_ids[b].tolist() == want_pos


def test_make_document_mask_blocks_cross_document_attention():
    x = torch.tensor([[5, 3, EOS, 7, 7]])
    mask = make_document_mask(x, eos_id=EOS)
    assert mask.shape == (1, 1, 5, 5)
    doc_of = [0, 0, 0, 1, 1]  # EOS closes doc 0
    for q in range(5):
        for kv in range(5):
            assert mask[0, 0, q, kv].item() == (doc_of[q] == doc_of[kv])


def test_make_document_mask_accepts_1d_input():
    x = torch.tensor([5, EOS, 7])
    assert make_document_mask(x, eos_id=EOS).shape == (1, 1, 3, 3)


def test_causal_and_document_mask_combined():
    x = torch.tensor([[5, 3, EOS, 7, 7]])
    combined = make_causal_mask(5) & make_document_mask(x, eos_id=EOS)
    # token 3 (first token of doc 1) attends only to itself
    assert combined[0, 0, 3].tolist() == [False, False, False, True, False]
    # token 1 attends to 0 and itself, not to the future
    assert combined[0, 0, 1].tolist() == [True, True, False, False, False]

"""Tests for the context-extension data layer.

WindowSampledDataset serves single-document random windows from the flat token
stream (the only data that trains long-range attention, since doc masking resets
at every EOS). These pin the invariants the training + banded-eval code relies
on: single-document windows, no false BOS, window-relative positions, per-epoch
resampling, and token-share == item-share mixing.
"""

import torch

from chimera.data._text import WindowSampledDataset, window_worker_init_fn
from chimera.models.attention import build_block_mask_and_pos

EOS, BOS = 0, 1


def _doc(n: int, start_tok: int) -> list[int]:
    """One document: BOS, n content tokens (>= 2, bounded to fit int16), EOS."""
    return [BOS] + [2 + ((start_tok + j) % 30000) for j in range(n)] + [EOS]


def _stream(lengths):
    toks = []
    for i, n in enumerate(lengths):
        toks += _doc(n, 100 + 7 * i)
    return torch.tensor(toks, dtype=torch.int16)


def test_boundary_recovery_filters_short_docs():
    # lengths: 3 (short), 50, 5 (short at ctx=8), 40
    data = _stream([3, 50, 5, 40])
    ds = WindowSampledDataset(data, seq_len=8, eos_id=EOS)
    # only docs with >= seq_len+1 (=9) content+EOS tokens qualify; the 50 and 40
    assert len(ds._doc_start) == 2
    assert len(ds) == sum(ds._doc_nwin)


def test_window_shape_and_shift():
    data = _stream([60])
    ds = WindowSampledDataset(data, seq_len=8, eos_id=EOS, seed=0)
    ds.set_epoch(0)
    for k in range(len(ds)):
        x, y = ds[k]
        assert x.shape == (8,) and y.shape == (8,)
        # y is x shifted by one within the same contiguous slice
        assert bool((y[:-1] == x[1:]).all())


def test_single_document_windows_only():
    data = _stream([80])
    ds = WindowSampledDataset(data, seq_len=8, eos_id=EOS, seed=1)
    ds.set_epoch(0)
    for k in range(len(ds) * 4):
        x, _ = ds[k % len(ds)]
        # no interior EOS in the input (EOS only ever the doc's final token, which
        # would land in y, not x); no BOS mid-window
        assert int((x[1:] == EOS).sum()) == 0
        assert int((x[1:] == BOS).sum()) == 0


def test_window_relative_positions():
    # a mid-doc window has no interior EOS -> build_block_mask_and_pos gives 0..N-1
    data = _stream([100])
    ds = WindowSampledDataset(data, seq_len=16, eos_id=EOS, seed=2)
    ds.set_epoch(0)
    x, _ = ds[0]
    _, pos_ids = build_block_mask_and_pos(x.unsqueeze(0), EOS)
    assert bool((pos_ids[0] == torch.arange(16)).all())


def test_per_epoch_resampling_changes_offsets():
    data = _stream([500])  # one long doc -> many possible offsets
    ds = WindowSampledDataset(data, seq_len=16, eos_id=EOS, seed=3)
    ds.set_epoch(0)
    first_e0 = ds[0][0].clone()
    ds.set_epoch(1)
    first_e1 = ds[0][0].clone()
    # different epoch -> (almost surely) a different offset for the same index
    assert not bool((first_e0 == first_e1).all())


def test_windows_per_doc_cap():
    data = _stream([1000])  # could yield many windows; capped
    ds = WindowSampledDataset(data, seq_len=8, eos_id=EOS, max_windows_per_doc=3)
    assert len(ds) == 3


def test_mixing_share_matches_weights():
    from torch.utils.data import ConcatDataset, WeightedRandomSampler

    long = _stream([200] * 10)
    # emulate the two pools: short = a plain index range, long = window ds
    long_ds = WindowSampledDataset(long, seq_len=8, eos_id=EOS)

    class _Range(torch.utils.data.Dataset):
        def __len__(self):
            return 400

        def __getitem__(self, i):
            return torch.zeros(8, dtype=torch.long), torch.zeros(8, dtype=torch.long)

    short_ds = _Range()
    n_s, n_l = len(short_ds), len(long_ds)
    short_share, long_share = 0.35, 0.65
    w = [short_share / n_s] * n_s + [long_share / n_l] * n_l
    sampler = WeightedRandomSampler(w, num_samples=20_000, replacement=True)
    _ = ConcatDataset([short_ds, short_ds])  # (sanity: ConcatDataset importable)
    picks = list(sampler)
    frac_long = sum(1 for i in picks if i >= n_s) / len(picks)
    assert abs(frac_long - long_share) < 0.03


def test_worker_init_fn_no_crash():
    data = _stream([100])
    ds = WindowSampledDataset(data, seq_len=8, eos_id=EOS)
    # no worker context -> no-op, must not raise
    window_worker_init_fn(0)
    assert len(ds) >= 1


def test_induction_batch_layout():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "projects/tinylm/pretrain"))
    from bpb_banded import _induction_batch

    d, n, prefix, vocab = 32, 20, 8, 256
    gen = torch.Generator().manual_seed(0)
    seqs, targets = _induction_batch(d, n, prefix, vocab, gen)
    # shape: prefix + A + B + gap(d) + A
    assert seqs.shape == (n, prefix + 2 + d + 1)
    # key repeated at prefix and at the final position; value right after first key
    assert bool((seqs[:, prefix] == seqs[:, -1]).all())  # both A's match
    assert bool((seqs[:, prefix + 1] == targets).all())  # B is the target
    assert bool((targets != seqs[:, prefix]).all())  # A != B
    # filler never collides with the key/value (disjoint id ranges)
    mid = vocab // 2
    filler = torch.cat([seqs[:, :prefix], seqs[:, prefix + 2 : -1]], dim=1)
    assert int((filler >= mid).sum()) == 0
    assert bool((targets >= mid).all())

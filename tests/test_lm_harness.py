from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")

from chimera.evals.lm_harness import ChimeraLM


class _NextTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.max_seen = 0
        self.training_seen = []

    def forward(self, ids):
        self.max_seen = max(self.max_seen, ids.shape[1])
        self.training_seen.append(self.training)
        logits = torch.full((*ids.shape, self.vocab_size), -20.0, device=ids.device)
        expected = (ids + 1) % self.vocab_size
        logits.scatter_(-1, expected.unsqueeze(-1), 20.0)
        return logits + self.anchor


def test_long_continuation_uses_rolling_windows_and_restores_mode(monkeypatch):
    model = _NextTokenModel(vocab_size=32)
    model.train()
    adapter = ChimeraLM(
        model=model,
        tokenizer=SimpleNamespace(_tok=SimpleNamespace()),
        eot_id=0,
        block_size=4,
    )
    continuation = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    encoded = [(("ctx", "cont"), [0], continuation)]
    monkeypatch.setattr(adapter, "_encode_pairs_cached", lambda pairs: encoded)

    result = adapter.loglikelihood(
        [SimpleNamespace(args=("ctx", "cont"))], disable_tqdm=True
    )

    assert result[0][1] is True
    assert model.max_seen <= 4
    assert model.training_seen and not any(model.training_seen)
    assert model.training is True

import torch

from chimera.optim import Muon


def _optimizer(matrix, bias):
    return Muon(
        [
            {"params": [matrix], "use_muon": True, "lr": 0.01},
            {"params": [bias], "use_muon": False, "lr": 0.001},
        ]
    )


def _step(optimizer, matrix, bias):
    matrix.grad = torch.randn_like(matrix)
    bias.grad = torch.randn_like(bias)
    optimizer.step()
    optimizer.zero_grad()


def test_state_dict_contains_muon_and_adamw_state():
    matrix = torch.nn.Parameter(torch.randn(4, 4))
    bias = torch.nn.Parameter(torch.randn(4))
    optimizer = _optimizer(matrix, bias)
    _step(optimizer, matrix, bias)

    state = optimizer.state_dict()["state"]
    assert len(state) == 2
    assert any("momentum_buffer" in value for value in state.values())
    assert any("exp_avg" in value for value in state.values())


def test_state_dict_round_trip_restores_all_moments():
    matrix = torch.nn.Parameter(torch.randn(4, 4))
    bias = torch.nn.Parameter(torch.randn(4))
    optimizer = _optimizer(matrix, bias)
    _step(optimizer, matrix, bias)
    saved = optimizer.state_dict()

    new_matrix = torch.nn.Parameter(matrix.detach().clone())
    new_bias = torch.nn.Parameter(bias.detach().clone())
    restored = _optimizer(new_matrix, new_bias)
    restored.load_state_dict(saved)

    assert "momentum_buffer" in restored.state[new_matrix]
    assert "exp_avg" in restored.state[new_bias]
    _step(restored, new_matrix, new_bias)

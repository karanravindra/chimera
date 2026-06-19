from collections.abc import Callable

from torch import nn


def _default_act() -> nn.Module:
    return nn.GELU(approximate="tanh")


class Mlp(nn.Module):
    """Two-layer feedforward block: ``Linear -> activation -> Linear``.

    ``out_features`` defaults to ``in_features``. The activation is swappable so
    the same block serves both the DiT MLP (GELU-tanh) and the time MLP (SiLU).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = _default_act,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

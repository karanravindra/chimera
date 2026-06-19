from typing import Callable, Literal

from torch import Tensor, nn

Activation = Literal[
    "silu",
    "relu",
    "gelu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "elu",
    "selu",
    "prelu",
    "rrelu",
    "celu",
]

ACTIVATIONS: dict[Activation, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "celu": nn.CELU,
}


class GLU(nn.Module):
    def __init__(self, dim: int, m: int, act: Activation = "silu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, m * 2, bias=False)
        self.fc2 = nn.Linear(m, dim, bias=False)
        self.act = ACTIVATIONS[act]()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(self.act(x1) * x2)

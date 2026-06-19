import torch.nn as nn


class StellarNet(nn.Module):
    """Simple MLP for stellar object classification (GALAXY / QSO / STAR)."""

    def __init__(
        self,
        in_features: int = 14,
        num_classes: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super(StellarNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

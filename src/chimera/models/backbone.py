import torch
from torch import nn

from chimera.nn import Conv2dNormAct


class Backbone(nn.Module):
    """CNN encoder: (B, 1, 28, 28) -> (B, embed_dim). `features` keeps the last
    conv map (before the global pool) so we can visualize where features focus.

    Shared by the DINO experiment (which trains it) and the DINO-aligned
    autoencoder (which loads the trained weights). Both must use the same
    ``embed_dim`` for ``load_state_dict`` to succeed.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()

        def conv_block(in_channels, out_channels):
            # Two 3x3 convs (a small residual stack) then halve the resolution.
            return nn.Sequential(
                Conv2dNormAct(in_channels, out_channels),
                Conv2dNormAct(out_channels, out_channels),
                nn.AvgPool2d(2),
            )

        self.norm = nn.BatchNorm2d(1)
        self.block1 = conv_block(1, 32)  # 28 -> 14
        self.block2 = conv_block(32, 64)  # 14 -> 7
        self.block3 = conv_block(64, 128)  # 7 -> 3
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pool -> (B, 128, 1, 1)
        self.fc1 = nn.Linear(128, embed_dim)

        self.features = None  # To store the last conv map for visualization

    def forward(self, x):
        x = self.norm(x)
        x = self.block1(x)
        x = self.block2(x)
        # Detached so this visualization side-effect can't retain the autograd
        # graph / extra activations. (Not safe under DataParallel; single-GPU only.)
        self.features = x.detach()  # Store the last conv map
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

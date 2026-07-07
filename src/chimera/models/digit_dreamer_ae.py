import torch.nn as nn

class Block(nn.Module):
    """A simple convolutional block with a conv, batch norm, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DigitDreamerAE(nn.Module):
    """A small convolutional autoencoder for MNIST with a *spatial* latent.

    The bottleneck is a ``1x1`` convolution rather than a flatten + linear, so
    ``encode`` returns a ``(B, latent_dim, 4, 4)`` feature map instead of a 1D
    vector. Keeping the latent spatial lets it be patchified into a token sequence
    for a latent diffusion / rectified-flow transformer (MM-DiT), while still
    working as a plain reconstruction autoencoder via ``forward``.

    ``latent_dim`` is the number of latent *channels* on the 4x4 grid.
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 4, base_channels: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.in_to_channels = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.encoder = nn.Sequential(
            Block(base_channels, base_channels),
            nn.MaxPool2d(2),
            Block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            Block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            Block(base_channels * 4, base_channels *4)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

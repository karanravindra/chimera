import torch.nn as nn


class CIFARAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        # 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
        )

        self.decoder_input = nn.Linear(latent_dim, 64 * 8 * 8)
        # 8x8 -> 16x16 -> 32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 64, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

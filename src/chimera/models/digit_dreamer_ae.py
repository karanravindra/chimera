import torch.nn as nn


class DigitDreamerAE(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 32):
        super().__init__()
        # 28x28 -> 14x14 -> 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )

        self.decoder_input = nn.Linear(latent_dim, 32 * 7 * 7)
        # 7x7 -> 14x14 -> 28x28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 32, 7, 7)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

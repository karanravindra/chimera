import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator (pix2pix / CycleGAN ``NLayerDiscriminator``).

    Classifies overlapping local patches as real/fake instead of scoring the
    whole image, so the gradient it sends back sharpens local texture — which is
    what an autoencoder's blurry reconstructions need. ``forward`` returns a
    ``(B, 1, H', W')`` map of per-patch logits (no final activation; pair it with
    a hinge or BCE-with-logits loss).
    """

    def __init__(
        self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3
    ):
        super().__init__()

        def block(cin, cout, stride):
            return [
                nn.Conv2d(
                    cin, cout, kernel_size=4, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # First layer has no norm.
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for i in range(1, n_layers):
            ch_prev, ch = ch, min(base_channels * 2**i, base_channels * 8)
            layers += block(ch_prev, ch, stride=2)
        # Penultimate stride-1 block widens the receptive field without downsizing.
        ch_prev, ch = ch, min(base_channels * 2**n_layers, base_channels * 8)
        layers += block(ch_prev, ch, stride=1)
        layers += [nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

from typing import Optional

import torch.nn as nn


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = 1,
    ):
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.skip = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=padding - (kernel_size // 2),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip(x)

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x + skip


class DigitNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels_per_block: tuple[int, ...] = (6, 16, 32, 32),
        layers_per_block: tuple[int, ...] = (1, 1, 1, 1),
    ):
        super().__init__()
        assert len(channels_per_block) == len(layers_per_block)

        layers: list[nn.Module] = []
        prev_channels = in_channels
        for i, (out_channels, num_layers) in enumerate(
            zip(channels_per_block, layers_per_block)
        ):
            # first block widens 28x28 -> 32x32 via padding=3
            padding = 3 if i == 0 else 1
            layers.append(
                Block(prev_channels, out_channels, kernel_size=3, padding=padding)
            )
            for _ in range(num_layers - 1):
                layers.append(Block(out_channels, out_channels, kernel_size=3))
            if i < len(channels_per_block) - 1:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels_per_block[-1], num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    @staticmethod
    def from_variant(
        variant: str, in_channels: int = 1, num_classes: int = 10
    ) -> "DigitNet":
        if variant == "tiny":
            return DigitNet(
                in_channels=in_channels,
                num_classes=num_classes,
                channels_per_block=(4, 8, 8, 8),
                layers_per_block=(1, 1, 1, 1),
            )
        elif variant == "small":
            return DigitNet(
                in_channels=in_channels,
                num_classes=num_classes,
                channels_per_block=(6, 12, 16, 16),
                layers_per_block=(1, 1, 1, 1),
            )
        elif variant == "medium":
            return DigitNet(
                in_channels=in_channels,
                num_classes=num_classes,
                channels_per_block=(8, 16, 24, 24),
                layers_per_block=(1, 1, 1, 1),
            )
        elif variant == "large":
            return DigitNet(
                in_channels=in_channels,
                num_classes=num_classes,
                channels_per_block=(12, 24, 32, 32),
                layers_per_block=(1, 2, 2, 1),
            )
        else:
            raise ValueError(f"Unknown model variant: {variant}")


if __name__ == "__main__":
    from torchinfo import summary

    for variant in ["tiny", "small", "medium", "large"]:
        model = DigitNet.from_variant(variant)
        print(f"Model variant: {variant}")
        summary(model, input_size=(1, 1, 28, 28))

import torch.nn as nn


class VGG(nn.Module):
    """A compact VGG-style CNN with BatchNorm, sized for 32x32 CIFAR images."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()

        def block(i, o):
            return [
                nn.Conv2d(i, o, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            ]

        self.features = nn.Sequential(
            *block(in_channels, 64),
            *block(64, 64),
            nn.MaxPool2d(2),  # 32 -> 16
            *block(64, 128),
            *block(128, 128),
            nn.MaxPool2d(2),  # 16 -> 8
            *block(128, 256),
            *block(256, 256),
            nn.MaxPool2d(2),  # 8 -> 4
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

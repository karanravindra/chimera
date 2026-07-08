import torch
from torch import nn


class CLEVRVQAModel(nn.Module):
    """Small CNN+GRU baseline for CLEVR answer classification."""

    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        image_dim: int = 256,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.image_encoder = nn.Sequential(
            self._conv_block(3, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 256),
            nn.MaxPool2d(2),
            self._conv_block(256, image_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.question_embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx
        )
        self.question_encoder = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        question_len: torch.Tensor,
    ):
        image_features = self.image_encoder(image)

        embedded = self.question_embedding(question)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            question_len.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.question_encoder(packed)
        question_features = hidden[-1]

        return self.classifier(torch.cat([image_features, question_features], dim=1))

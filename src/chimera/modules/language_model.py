import math

from lightning import LightningModule
from torch import nn


class LanguageModelModule(LightningModule):
    """Next-character language modeling with cross-entropy loss.

    Logs loss (nats) and bits-per-character (bpc = loss / ln 2), the standard
    text8 metric.
    """

    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        logits = self.model(x)  # (B, T, V)
        vocab_size = logits.size(-1)
        loss = self.criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        bpc = loss / math.log(2)
        return loss, bpc

    def training_step(self, batch, batch_idx):
        loss, bpc = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bpc", bpc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, bpc = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bpc", bpc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, bpc = self._step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bpc", bpc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

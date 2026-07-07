import math

from lightning import LightningModule
from torch import nn


class LanguageModelModule(LightningModule):
    """Next-token language modeling with cross-entropy loss.

    Logs loss (nats) and bits-per-token (bpt = loss / ln 2).
    """

    def __init__(self, model, optimizer, scheduler, use_cce=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Cut Cross Entropy (apple/ml-cross-entropy): fuse the lm_head projection
        # with cross-entropy so the (B, T, vocab) logits are never materialized —
        # a large memory saving with big vocabularies. Requires the model to
        # expose hidden states and lm_head_weight, and bf16/fp16 hidden states.
        self.use_cce = use_cce

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        if self.use_cce:
            from cut_cross_entropy import linear_cross_entropy

            hidden = self.model(x, return_hidden=True)  # (B, T, C)
            loss = linear_cross_entropy(
                hidden.reshape(-1, hidden.size(-1)),
                self.model.lm_head_weight,  # (V, C)
                y.reshape(-1),
            )
        else:
            logits = self.model(x)  # (B, T, V)
            vocab_size = logits.size(-1)
            loss = self.criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        bpt = loss / math.log(2)
        return loss, bpt

    def training_step(self, batch, batch_idx):
        loss, bpt = self._step(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/bpt", bpt, on_step=True, prog_bar=True)
        self.log("train/lr", self.scheduler.get_last_lr()[0], on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, bpt = self._step(batch)
        self.log("val/loss", loss, on_step=False, prog_bar=True)
        self.log("val/bpt", bpt, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, bpt = self._step(batch)
        self.log("test/loss", loss, on_step=False, prog_bar=True)
        self.log("test/bpt", bpt, on_step=False, prog_bar=True)
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

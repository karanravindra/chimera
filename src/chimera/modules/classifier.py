from lightning import LightningModule
from torch import nn


class ClassifierModule(LightningModule):
    def __init__(self, model, optimizer, scheduler, class_weights=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("val/loss", loss, on_step=False, prog_bar=True)
        self.log("val/acc", acc, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("test/loss", loss, on_step=False, prog_bar=True)
        self.log("test/acc", acc, on_step=False, prog_bar=True)
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

import matplotlib.pyplot as plt
import numpy as np
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torchmetrics.classification import MulticlassConfusionMatrix


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    return fig


class ClassifierModule(LightningModule):
    """Classification training module.

    ``log_confusion_matrix=True`` accumulates a confusion matrix over the whole
    val/test epoch (stateful torchmetrics, accumulate-then-compute — same
    pattern as rFID in ``AutoencoderModule``) and, when the trainer's logger is
    a ``WandbLogger``, logs it both as an image (``wandb.Image`` of a matplotlib
    figure) and as a raw-count table (``wandb.Table``) so it can be inspected or
    pivoted in the wandb UI.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        class_weights=None,
        num_classes: int = 10,
        class_names: list[str] | None = None,
        log_confusion_matrix: bool = False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.log_confusion_matrix = log_confusion_matrix
        if log_confusion_matrix:
            self.val_confusion = MulticlassConfusionMatrix(num_classes=num_classes)
            self.test_confusion = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return loss, acc, preds, y

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self._step(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, preds, y = self._step(batch)
        self.log("val/loss", loss, on_step=False, prog_bar=True)
        self.log("val/acc", acc, on_step=False, prog_bar=True)
        if self.log_confusion_matrix:
            self.val_confusion.update(preds, y)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc, preds, y = self._step(batch)
        self.log("test/loss", loss, on_step=False, prog_bar=True)
        self.log("test/acc", acc, on_step=False, prog_bar=True)
        if self.log_confusion_matrix:
            self.test_confusion.update(preds, y)
        return loss

    def _log_confusion(self, metric, stage: str):
        cm = metric.compute().cpu().numpy()
        metric.reset()

        wandb_logger = next(
            (lg for lg in self.loggers if isinstance(lg, WandbLogger)), None
        )
        if wandb_logger is None:
            return

        import wandb

        fig = _plot_confusion_matrix(
            cm, self.class_names, title=f"{stage.capitalize()} confusion matrix"
        )
        wandb_logger.log_image(key=f"{stage}/confusion_matrix", images=[fig])
        plt.close(fig)

        table = wandb.Table(
            columns=["actual", "predicted", "count"],
            data=[
                [self.class_names[i], self.class_names[j], int(cm[i, j])]
                for i in range(cm.shape[0])
                for j in range(cm.shape[1])
            ],
        )
        wandb_logger.experiment.log({f"{stage}/confusion_matrix_table": table})

    def on_validation_epoch_end(self):
        if self.log_confusion_matrix:
            self._log_confusion(self.val_confusion, "val")

    def on_test_epoch_end(self):
        if self.log_confusion_matrix:
            self._log_confusion(self.test_confusion, "test")

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

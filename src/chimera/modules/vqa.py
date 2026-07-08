import matplotlib.pyplot as plt
import numpy as np
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torchmetrics.classification import MulticlassConfusionMatrix


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str):
    fig, ax = plt.subplots(figsize=(8, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted answer")
    ax.set_ylabel("True answer")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    fig.tight_layout()
    return fig


class VQAModule(LightningModule):
    """Answer-classification training module for visual question answering."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        answer_names: list[str],
        log_confusion_matrix: bool = False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.answer_names = answer_names
        self.log_confusion_matrix = log_confusion_matrix
        self.criterion = nn.CrossEntropyLoss()

        if log_confusion_matrix:
            num_answers = len(answer_names)
            self.val_confusion = MulticlassConfusionMatrix(num_classes=num_answers)
            self.test_confusion = MulticlassConfusionMatrix(num_classes=num_answers)

    def forward(self, image, question, question_len):
        return self.model(image, question, question_len)

    def _step(self, batch):
        y = batch["answer"]
        logits = self.model(batch["image"], batch["question"], batch["question_len"])
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        return loss, acc, preds, y

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self._step(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        lr = (
            self.scheduler.get_last_lr()[0]
            if self.scheduler is not None
            else self.optimizer.param_groups[0]["lr"]
        )
        self.log("train/lr", lr, on_step=True, prog_bar=True)
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

    def predict_step(self, batch, batch_idx):
        logits = self.model(batch["image"], batch["question"], batch["question_len"])
        preds = logits.argmax(dim=1)
        return {
            "image_filename": batch["image_filename"],
            "question_text": batch["question_text"],
            "answer": [self.answer_names[idx] for idx in preds.cpu().tolist()],
        }

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
            cm, self.answer_names, title=f"{stage.capitalize()} confusion matrix"
        )
        wandb_logger.log_image(key=f"{stage}/confusion_matrix", images=[fig])
        plt.close(fig)

        table = wandb.Table(
            columns=["actual", "predicted", "count"],
            data=[
                [self.answer_names[i], self.answer_names[j], int(cm[i, j])]
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
        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

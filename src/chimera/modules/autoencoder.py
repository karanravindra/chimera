from lightning import LightningModule
from torch import nn
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


class AutoencoderModule(LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, _ = batch
        recon = self.model(x)
        loss = self.criterion(recon, x)

        psnr = peak_signal_noise_ratio(recon, x, data_range=1.0)
        self.log(f"{stage}/loss", loss, on_step=True, prog_bar=True)
        self.log(f"{stage}/psnr", psnr, prog_bar=True)

        if stage == "val":
            ssim = structural_similarity_index_measure(recon, x, data_range=1.0)
            self.log(f"{stage}/ssim", ssim, prog_bar=True)

        
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

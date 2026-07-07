from lightning import LightningModule
from torch import nn
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


class AutoencoderModule(LightningModule):
    """Reconstruction autoencoder training module.

    Base loss is L1. Two optional terms are off by default (so single-channel
    projects like MNIST are unaffected) and meant for RGB autoencoders:

    - ``lpips_weight > 0`` adds a VGG-LPIPS perceptual loss term, the main lever
      on reconstruction quality. Inputs are assumed to be in ``[0, 1]``.
    - ``compute_rfid`` accumulates a reconstruction-FID (real images vs. their
      reconstructions) over the whole val/test epoch and logs it once at epoch
      end. This is the correct stateful-metric use (accumulate then compute),
      unlike per-step losses which should stay functional.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        lpips_weight: float = 0.0,
        compute_rfid: bool = False,
        train_only: list[str] | None = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.L1Loss()

        # When set, only these named submodules of ``model`` are trained; the rest
        # are frozen (params + BatchNorm running stats). Used by the progressive
        # AFHQ curriculum, e.g. train_only=["to_latent", "from_latent"].
        self.train_only = train_only

        self.lpips_weight = lpips_weight
        if lpips_weight > 0:
            import lpips as lpips_lib

            self.lpips = lpips_lib.LPIPS(net="vgg")
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)

        self.compute_rfid = compute_rfid
        if compute_rfid:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid = FrechetInceptionDistance(feature=2048, normalize=True)

    @staticmethod
    def trainable_params(model, names: list[str]):
        """Return the parameters of the named submodules (for the optimizer)."""
        params = []
        for name in names:
            params += list(getattr(model, name).parameters())
        return params

    def _apply_freeze(self):
        """Freeze everything except ``self.train_only`` submodules.

        Sets the whole model to eval (so frozen BatchNorm running stats stay put)
        and re-enables train mode + gradients only on the named submodules. Called
        every train-epoch start because Lightning puts the module back in train
        mode there, which would otherwise un-freeze BatchNorm.
        """
        if not self.train_only:
            return
        self.model.eval()
        self.model.requires_grad_(False)
        for name in self.train_only:
            m = getattr(self.model, name)
            m.train()
            m.requires_grad_(True)

    def on_train_epoch_start(self):
        self._apply_freeze()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, _ = batch
        recon = self.model(x)

        on_step = stage == "train"

        l1 = self.criterion(recon, x)
        loss = l1
        psnr = peak_signal_noise_ratio(recon, x, data_range=1.0)
        self.log(
            f"{stage}/psnr", psnr, on_step=on_step, on_epoch=not on_step, prog_bar=True
        )

        if self.lpips_weight > 0:
            # lpips expects [-1, 1]; normalize=True lets it take [0, 1] inputs.
            lpips_val = self.lpips(recon, x, normalize=True).mean()
            loss = l1 + self.lpips_weight * lpips_val
            self.log(f"{stage}/l1", l1, on_step=on_step, on_epoch=not on_step)
            self.log(
                f"{stage}/lpips",
                lpips_val,
                on_step=on_step,
                on_epoch=not on_step,
                prog_bar=True,
            )

        self.log(
            f"{stage}/loss", loss, on_step=on_step, on_epoch=not on_step, prog_bar=True
        )

        if stage == "val":
            ssim = structural_similarity_index_measure(recon, x, data_range=1.0)
            self.log("val/ssim", ssim, prog_bar=True)

        if self.compute_rfid and stage in ("val", "test"):
            fid = self.fid if stage == "val" else self.fid
            fid.update(x.float().clamp(0, 1), real=True)
            fid.update(recon.float().clamp(0, 1), real=False)
            # Logging the metric object defers compute()/reset() to epoch end.
            self.log(f"{stage}/rfid", fid, on_step=False, on_epoch=True, prog_bar=True)

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

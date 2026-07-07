import torch
from lightning import LightningModule
from torch import nn


class RectifiedFlowModule(LightningModule):
    """Rectified-flow (flow-matching) training for a latent velocity network.

    The model predicts the constant velocity ``v = data - noise`` along the
    straight interpolant ``z_t = (1 - t) * noise + t * data`` (so ``t = 0`` is
    noise and ``t = 1`` is data). Training minimizes the MSE between the predicted
    and target velocity; sampling integrates the learned ODE with Euler steps and
    classifier-free guidance.

    Follows the ``(model, optimizer, scheduler)`` convention of the other modules
    (see ``AutoencoderModule``). Batches are ``(z, y)``: standardized latents and
    integer class labels.
    """

    def __init__(self, model, optimizer, scheduler, logit_normal: bool = True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logit_normal = logit_normal
        self.criterion = nn.L1Loss()

    def _sample_t(self, batch_size, device):
        if self.logit_normal:
            # SD3-style logit-normal: bias t toward the middle of [0, 1].
            return torch.sigmoid(torch.randn(batch_size, device=device))
        return torch.rand(batch_size, device=device)

    def _step(self, batch):
        z, y = batch
        b = z.shape[0]
        t = self._sample_t(b, z.device)
        noise = torch.randn_like(z)

        t_ = t.view(b, *([1] * (z.ndim - 1)))
        z_t = (1 - t_) * noise + t_ * z
        target = z - noise

        pred = self.model(z_t, t, y)
        return self.criterion(pred, target)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val/loss", loss, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test/loss", loss, on_step=False, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, y, n_steps: int = 50, cfg_scale: float = 1.0, latent_shape=None):
        """Generate standardized latents for labels ``y`` via Euler ODE integration.

        ``cfg_scale > 1`` applies classifier-free guidance by mixing the
        conditional and null-class (unconditional) velocities. The caller should
        un-standardize the returned latents and decode them with the autoencoder.
        Move the module to the target device before calling (Lightning leaves it on
        CPU after fit/test).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        y = y.to(device)
        b = y.shape[0]

        if latent_shape is None:
            m = self.model
            latent_shape = (m.latent_channels, m.latent_size, m.latent_size)

        z = torch.randn(b, *latent_shape, device=device)
        dt = 1.0 / n_steps
        null = torch.full_like(y, self.model.null_class)

        for i in range(n_steps):
            t = torch.full((b,), i * dt, device=device)
            if cfg_scale != 1.0:
                v_cond = self.model(z, t, y)
                v_uncond = self.model(z, t, null)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(z, t, y)
            z = z + v * dt
        return z

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

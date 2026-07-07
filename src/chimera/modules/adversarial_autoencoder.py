import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Differentiable augmentation (color + translation), applied to both real and
    fake images before the discriminator. Regularizes the PatchGAN on small data
    (AFHQ), following the DiffAugment recipe.
    """
    b = x.size(0)
    # brightness
    x = x + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    # saturation
    mean = x.mean(dim=1, keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=x.device) * 2) + mean
    # contrast
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=x.device) + 0.5) + mean
    # translation (up to 1/8 of each side, zero-padded)
    _, _, h, w = x.shape
    sh, sw = h // 8, w // 8
    tx = torch.randint(-sh, sh + 1, (1,)).item()
    ty = torch.randint(-sw, sw + 1, (1,)).item()
    x = torch.roll(x, shifts=(tx, ty), dims=(2, 3))
    return x


class AdversarialAutoencoderModule(LightningModule):
    """L1 + LPIPS + PatchGAN refinement of a reconstruction autoencoder.

    Manual-optimization GAN: a PatchGAN discriminator supplies an adversarial
    signal (hinge loss) on top of the L1 + VGG-LPIPS reconstruction loss, which
    sharpens texture the pixel/perceptual losses leave blurry. Meant for a final
    refinement phase — pass ``train_only=["out_from_channels"]`` to update only
    the output head while the rest of the AE stays frozen.

    Validation is GAN-free: it logs the reconstruction metrics (L1, LPIPS, PSNR,
    SSIM) and, if ``compute_rfid``, the reconstruction-FID over the whole epoch.
    Inputs are assumed to be in ``[0, 1]``.
    """

    def __init__(
        self,
        model,
        discriminator,
        opt_g,
        opt_d,
        sched_g=None,
        sched_d=None,
        lpips_weight: float = 1.0,
        gan_weight: float = 0.1,
        compute_rfid: bool = False,
        train_only: list[str] | None = None,
        use_diff_augment: bool = True,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.model = model
        self.discriminator = discriminator
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.sched_g = sched_g
        self.sched_d = sched_d

        self.l1 = nn.L1Loss()
        self.lpips_weight = lpips_weight
        self.gan_weight = gan_weight
        self.train_only = train_only
        self.use_diff_augment = use_diff_augment

        import lpips as lpips_lib

        self.lpips = lpips_lib.LPIPS(net="vgg")
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

        self.compute_rfid = compute_rfid
        if compute_rfid:
            from torchmetrics.image.fid import FrechetInceptionDistance

            self.val_fid = FrechetInceptionDistance(feature=2048, normalize=True)
            self.test_fid = FrechetInceptionDistance(feature=2048, normalize=True)

    @staticmethod
    def trainable_params(model, names: list[str]):
        params = []
        for name in names:
            params += list(getattr(model, name).parameters())
        return params

    def _apply_freeze(self):
        """Freeze the AE except ``self.train_only`` (params + BatchNorm stats).

        The discriminator is always fully trainable — only the generator (AE) is
        restricted. Re-applied each train-epoch start (Lightning resets to train
        mode there, which would un-freeze BatchNorm otherwise).
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

    def configure_optimizers(self):
        # Schedulers are stepped manually (manual optimization), so return only
        # the optimizers here.
        return [self.opt_g, self.opt_d]

    def forward(self, x):
        return self.model(x)

    def _aug(self, x):
        return diff_augment(x) if self.use_diff_augment else x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        opt_g, opt_d = self.optimizers()
        recon = self.model(x)

        # ---- discriminator (hinge) ----
        self.toggle_optimizer(opt_d)
        d_real = self.discriminator(self._aug(x))
        d_fake = self.discriminator(self._aug(recon.detach()))
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # ---- generator: L1 + LPIPS + adversarial ----
        self.toggle_optimizer(opt_g)
        l1 = self.l1(recon, x)
        lp = self.lpips(recon, x, normalize=True).mean()
        g_adv = -self.discriminator(self._aug(recon)).mean()
        g_loss = l1 + self.lpips_weight * lp + self.gan_weight * g_adv
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        for sched in (self.sched_g, self.sched_d):
            if sched is not None:
                sched.step()

        self.log_dict(
            {
                "train/l1": l1,
                "train/lpips": lp,
                "train/gan_g": g_adv,
                "train/gan_d": d_loss,
                "train/loss": g_loss,
            },
            on_step=True,
            prog_bar=True,
        )
        return g_loss

    def _eval_step(self, batch, stage):
        x, _ = batch
        recon = self.model(x)
        l1 = self.l1(recon, x)
        lp = self.lpips(recon, x, normalize=True).mean()
        loss = l1 + self.lpips_weight * lp
        psnr = peak_signal_noise_ratio(recon, x, data_range=1.0)
        ssim = structural_similarity_index_measure(recon, x, data_range=1.0)
        self.log_dict(
            {
                f"{stage}/l1": l1,
                f"{stage}/lpips": lp,
                f"{stage}/loss": loss,
                f"{stage}/psnr": psnr,
                f"{stage}/ssim": ssim,
            },
            prog_bar=True,
        )
        if self.compute_rfid:
            fid = self.val_fid if stage == "val" else self.test_fid
            fid.update(x.float().clamp(0, 1), real=True)
            fid.update(recon.float().clamp(0, 1), real=False)
            self.log(f"{stage}/rfid", fid, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

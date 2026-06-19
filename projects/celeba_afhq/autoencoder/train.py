"""Train a ConvAutoEncoder on CelebA-HQ or AFHQ with Weights & Biases logging.

One script for both StarGAN-v2 256x256 datasets (pick with ``--dataset``):

  * 8x spatial downsample (3 halving blocks) to an ``8 x (S/8) x (S/8)`` latent -- 8 latent
    channels -- via the deep-compression :class:`~chimera.models.ConvAutoEncoder`;
  * trained on MSE + an LPIPS perceptual loss (``--lpips-weight``); MSE, LPIPS, PSNR and
    SSIM are logged every phase (train/val/test);
  * reconstruction FID (rFID = FID between originals and their reconstructions, via
    torchmetrics) is logged on the val and test phases -- expensive, so eval-only;
  * saves a full Lightning checkpoint to ``outputs/<dataset>/<run_id>/last.ckpt`` every
    epoch and uploads it as a wandb model artifact; ``--resume`` continues the same run
    and rebuilds the architecture from the checkpoint's saved hyperparameters.

The LPIPS (VGG) and FID (Inception) networks are eval-only, held off the module's
state_dict (see ``LitAutoEncoder._ensure_metrics``) so they never bloat the checkpoint.

Examples
--------
    # fresh run on AFHQ
    uv run python projects/celeba_afhq/autoencoder/train.py --dataset afhq --epochs 100

    # resume run <id> on CelebA-HQ for more epochs (same wandb run, continues its ckpt)
    uv run python projects/celeba_afhq/autoencoder/train.py --dataset celeba_hq \
        --resume <run_id> --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import LightningModule, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from chimera.data import AFHQDataModule, CelebAHQDataModule
from chimera.models import ConvAutoEncoder
from chimera.utils.experiment import (
    add_common_args,
    find_ckpt,
    grid,
    init_wandb_logger,
    run_training,
)

DATAMODULES = {"celeba_hq": CelebAHQDataModule, "afhq": AFHQDataModule}
DOWNSAMPLE = 8  # 3 halving DCDownBlocks: S -> S/2 -> S/4 -> S/8
LATENT_CHANNELS = 8

OUTPUTS = Path(__file__).parent / "outputs"  # checkpoints live under OUTPUTS/<dataset>/<run_id>


class LitAutoEncoder(LightningModule):
    """Wraps ConvAutoEncoder: cast -> reconstruct -> MSE + LPIPS, with rFID/PSNR/SSIM."""

    def __init__(
        self,
        model_config: dict,
        image_size: int,
        lr: float = 1e-4,
        lpips_weight: float = 1.0,
        lpips_net: str = "vgg",
        fid_feature: int = 2048,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvAutoEncoder(**model_config)
        self.lr = lr
        self.lpips_weight = lpips_weight
        self.lpips_net = lpips_net
        self.fid_feature = fid_feature
        self.n_log_images = 8  # how many reconstructions to log each eval epoch
        self._sample = None  # (originals, recons) stashed from the first eval batch
        # Eval-only metric networks (LPIPS for the perceptual loss + logging, FID for rFID),
        # filled in lazily by _ensure_metrics. Kept in a plain dict on purpose -- see there.
        self._metrics: dict = {}

        # Guard the documented latent geometry: an SxS input must encode to a
        # latent_dim x (S/8) x (S/8) latent (8x downsample). Catches a block-count or
        # latent_dim change that would silently disagree with the docstring / config.
        with torch.no_grad():
            probe = self.model.encode(torch.zeros(1, model_config["input_dim"], image_size, image_size))
        expect = (model_config["latent_dim"], image_size // DOWNSAMPLE, image_size // DOWNSAMPLE)
        assert tuple(probe.shape[1:]) == expect, (
            f"expected a {expect} latent for a {image_size}x{image_size} input, got "
            f"{tuple(probe.shape[1:])}; MODEL_CONFIG block count / latent_dim disagree with {DOWNSAMPLE}x"
        )

    def _ensure_metrics(self) -> None:
        """Build the LPIPS + FID networks once, on the module's device.

        Held in a plain dict (not as ``nn.Module`` attributes) so these frozen, eval-only
        networks are invisible to ``state_dict`` -- the ~0.5GB VGG / Inception weights would
        otherwise bloat every per-epoch checkpoint and are trivially rebuilt -- and excluded
        from the optimizer, ``.train()`` and ``.to()``."""
        if self._metrics:
            return
        lpips = LearnedPerceptualImagePatchSimilarity(net_type=self.lpips_net, normalize=True)
        lpips.requires_grad_(False)  # input still gets gradients; net stays frozen
        fid = FrechetInceptionDistance(feature=self.fid_feature, normalize=True)
        self._metrics = {"lpips": lpips.to(self.device).eval(), "fid": fid.to(self.device)}

    def on_fit_start(self) -> None:
        self._ensure_metrics()  # LPIPS is needed for the very first training_step loss

    # -- losses / quality metrics ----------------------------------------------------

    def _reconstruct(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        images, _ = batch
        x = images.float()  # bf16 [0,1] from the collate -> float32 in [0,1]
        return x, self.model(x)  # recon is sigmoid'd into [0,1]

    def _log_quality(self, x: torch.Tensor, recon: torch.Tensor, stage: str) -> None:
        # PSNR/SSIM in fp32 (autocast off) so the logged numbers are precision-independent.
        with torch.autocast(self.device.type, enabled=False):
            xf, rf = x.float(), recon.detach().float().clamp(0, 1)
            psnr = peak_signal_noise_ratio(rf, xf, data_range=1.0)
            ssim = structural_similarity_index_measure(rf, xf, data_range=1.0)
        self.log(f"{stage}/psnr", psnr, prog_bar=True)
        self.log(f"{stage}/ssim", ssim, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, recon = self._reconstruct(batch)
        mse = F.mse_loss(recon, x)
        lpips = self._metrics["lpips"](recon, x)
        loss = mse + self.lpips_weight * lpips
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mse", mse)
        self.log("train/lpips", lpips, prog_bar=True)
        self._log_quality(x, recon, "train")
        return loss

    def _eval_step(self, batch, batch_idx: int, stage: str) -> None:
        x, recon = self._reconstruct(batch)
        mse = F.mse_loss(recon, x)
        lpips = self._metrics["lpips"](recon, x)
        self.log(f"{stage}/loss", mse + self.lpips_weight * lpips, prog_bar=True)
        self.log(f"{stage}/mse", mse)
        self.log(f"{stage}/lpips", lpips, prog_bar=True)
        self._log_quality(x, recon, stage)
        # rFID: accumulate originals as the "real" distribution, reconstructions as "fake".
        # Inception runs in fp32 (autocast off) so the feature stats aren't bf16-noisy.
        with torch.autocast(self.device.type, enabled=False):
            self._metrics["fid"].update(x.float().clamp(0, 1), real=True)
            self._metrics["fid"].update(recon.float().clamp(0, 1), real=False)
        if batch_idx == 0:  # stash a fixed set of reconstructions to log this epoch
            n = self.n_log_images
            self._sample = (x[:n].float().cpu().clamp(0, 1), recon[:n].float().cpu().clamp(0, 1))

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "test")

    def on_validation_epoch_start(self) -> None:
        self._ensure_metrics()
        self._metrics["fid"].reset()  # rFID is per-epoch over the whole eval set

    def on_test_epoch_start(self) -> None:
        self._ensure_metrics()
        self._metrics["fid"].reset()

    def on_validation_epoch_end(self) -> None:
        self._finalize_eval_epoch("val")

    def on_test_epoch_end(self) -> None:
        self._finalize_eval_epoch("test")

    def _finalize_eval_epoch(self, stage: str) -> None:
        self.log(f"{stage}/rfid", self._metrics["fid"].compute(), prog_bar=True)
        if self._sample is None or not isinstance(self.logger, WandbLogger):
            self._sample = None
            return
        # One image, three rows: originals / reconstructions / absolute difference.
        x, recon = self._sample
        panel = torch.cat([x, recon, (x - recon).abs()], dim=0)
        image = grid(panel, nrow=x.shape[0])  # nrow = samples -> each category on its own row
        self.logger.log_image(
            f"{stage}/reconstructions", [image], caption=["rows: original / reconstruction / |diff|"]
        )
        self._sample = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p, project=None, epochs=100)  # project defaults to <dataset>-autoencoder below
    p.set_defaults(batch_size=32, lr=1e-4)  # high-res images + a larger model than MNIST
    p.add_argument("--dataset", choices=list(DATAMODULES), required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=128, help="stem width; blocks are (1,2,4)x this")
    p.add_argument("--lpips-weight", type=float, default=1.0, help="weight of the LPIPS term in the loss")
    p.add_argument("--lpips-net", choices=["vgg", "alex", "squeeze"], default="vgg")
    p.add_argument("--fid-feature", type=int, default=2048, help="InceptionV3 feature dim for rFID")
    p.add_argument("--mmap", action="store_true", help="memory-map the dataset instead of loading into RAM")
    args = p.parse_args()
    args.project = args.project or f"{args.dataset}-autoencoder"

    seed_everything(args.seed, workers=True)  # seed python/numpy/torch + dataloader workers

    # 8x downsample = 3 halving blocks; widths grow (1, 2, 4)x base_channels, latent is
    # LATENT_CHANNELS x (image_size/8) x (image_size/8). The geometry is asserted at
    # construction (see LitAutoEncoder.__init__) so this can't silently drift.
    c = args.base_channels
    model_config = dict(
        input_dim=3,
        latent_dim=LATENT_CHANNELS,
        base_channels=c,
        dim_per_block=(c, 2 * c, 4 * c),
        layers_per_block=(2, 2, 2),
    )

    datamodule = DATAMODULES[args.dataset](
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        in_memory=not args.mmap,
    )

    outputs = OUTPUTS / args.dataset
    resume_ckpt = find_ckpt(args.resume, args.project, outputs) if args.resume else None
    if resume_ckpt:
        # Rebuild from the checkpoint's saved hyperparameters (model_config, image_size,
        # lpips/fid settings), not the current CLI flags, so a resume always reconstructs
        # the architecture the run was trained with.
        module = LitAutoEncoder.load_from_checkpoint(resume_ckpt, lr=args.lr)
    else:
        module = LitAutoEncoder(
            model_config,
            image_size=args.image_size,
            lr=args.lr,
            lpips_weight=args.lpips_weight,
            lpips_net=args.lpips_net,
            fid_feature=args.fid_feature,
        )

    # Read model/loss hyperparameters off the module so the logged config matches the live
    # model on both a fresh run (from args) and a resume (restored from the checkpoint).
    hp = module.hparams
    config = {
        "model": hp["model_config"],
        "dataset": args.dataset,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": module.lr,
            "seed": args.seed,
            "precision": "bf16-mixed",
            "grad_clip": args.grad_clip,
            "image_size": hp["image_size"],
            "downsample": DOWNSAMPLE,
            "latent_channels": LATENT_CHANNELS,
            "lpips_weight": module.lpips_weight,
            "lpips_net": module.lpips_net,
            "fid_feature": module.fid_feature,
        },
        "data": {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "num_workers": args.num_workers,
            "in_memory": not args.mmap,
        },
    }

    logger, run_id = init_wandb_logger(args.project, config, resume=args.resume)

    run_training(
        module=module,
        datamodule=datamodule,
        args=args,
        logger=logger,
        run_id=run_id,
        outputs=outputs,
        resume_ckpt=resume_ckpt,
        artifact_metadata=config,
        test=True,  # report rFID/PSNR/SSIM/LPIPS on the test split after training
    )


if __name__ == "__main__":
    main()

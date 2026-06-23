"""Train a ConvAutoEncoder on the union of CelebA-HQ and AFHQ with W&B logging.

A single autoencoder over the *merged* StarGAN-v2 256x256 datasets (CelebA-HQ faces +
AFHQ animal faces, concatenated into one train/eval set):

  * 16x spatial downsample (4 halving blocks) to an ``8 x (S/16) x (S/16)`` latent -- 8 latent
    channels -- via the deep-compression :class:`~chimera.models.ConvAutoEncoder`;
  * trained on MSE + an LPIPS perceptual loss (``--lpips-weight``) + a REPA representation-
    alignment loss (``--repa-weight``); MSE, LPIPS, REPA, PSNR and SSIM are logged every phase
    (train/val/test);
  * REPA (Representation Alignment, Yu et al. 2024) aligns the latent to the patch features of
    a frozen DINOv2 encoder (``--repa-model``, loaded from HuggingFace) via a small trainable
    projection head and a patch-wise cosine-similarity loss, structuring the latent space;
    ``--repa-weight 0`` disables it entirely (no DINOv2 load);
  * reconstruction FID (rFID = FID between originals and their reconstructions, via
    torchmetrics) is logged on the val and test phases -- expensive, so eval-only;
  * saves a full Lightning checkpoint to ``outputs/<run_id>/last.ckpt`` every epoch and
    uploads it as a wandb model artifact; ``--resume`` continues the same run and rebuilds
    the architecture from the checkpoint's saved hyperparameters.

The LPIPS (VGG), FID (Inception) and DINOv2 (REPA target) networks are eval-only, held off the
module's state_dict (see ``LitAutoEncoder._ensure_metrics``) so they never bloat the checkpoint;
the REPA projection head is small and trainable, so it IS checkpointed and optimized.

Examples
--------
    # fresh run on the merged CelebA-HQ + AFHQ set
    uv run python projects/celeba_afhq/autoencoder/train.py --epochs 100

    # resume run <id> for more epochs (same wandb run, continues from its checkpoint)
    uv run python projects/celeba_afhq/autoencoder/train.py --resume <run_id> --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import LightningModule, seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from chimera.data import AFHQDataModule, CelebAHQDataModule, ConcatImageDataModule
from chimera.models import DINOV2_HIDDEN_SIZE, ConvAutoEncoder, Dinov2Features
from chimera.utils.experiment import (
    add_common_args,
    find_ckpt,
    grid,
    init_wandb_logger,
    run_training,
)

# Sources merged (via ConcatImageDataModule) into one train/eval set; the AE is label-free,
# so the datasets' disjoint class ids don't matter.
SOURCE_DATAMODULES = {"celeba_hq": CelebAHQDataModule, "afhq": AFHQDataModule}
DATASET_NAME = "+".join(SOURCE_DATAMODULES)  # "celeba_hq+afhq"
DOWNSAMPLE = 8  # 4 halving DCDownBlocks: S -> S/2 -> S/4 -> S/8 -> S/16
LATENT_CHANNELS = 8

OUTPUTS = Path(__file__).parent / "outputs"  # checkpoints live under OUTPUTS/<run_id>


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
        repa_weight: float = 0.5,
        repa_model: str = "facebook/dinov2-small",
        repa_dino_size: int = 224,
        repa_proj_hidden: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvAutoEncoder(**model_config)
        self.lr = lr
        self.lpips_weight = lpips_weight
        self.lpips_net = lpips_net
        self.fid_feature = fid_feature
        self.repa_weight = repa_weight
        self.repa_model = repa_model
        self.repa_dino_size = repa_dino_size
        self.n_log_images = 8  # how many reconstructions to log each eval epoch
        self._sample = None  # (originals, recons) stashed from the first eval batch
        # Eval-only metric networks (LPIPS for the perceptual loss + logging, FID for rFID,
        # and -- when REPA is on -- the frozen DINOv2 target), filled in lazily by
        # _ensure_metrics. Kept in a plain dict on purpose -- see there.
        self._metrics: dict = {}

        # REPA (Representation Alignment): a trainable head projecting each latent token onto
        # the frozen DINOv2 patch-feature space, where a cosine-similarity loss aligns them.
        # Unlike the frozen metric nets this IS optimized + checkpointed, so it's a real
        # nn.Module attribute. Sized from the known DINOv2 width (no HF download in __init__).
        if repa_weight > 0:
            dino_dim = DINOV2_HIDDEN_SIZE[repa_model]
            self.repa_proj = nn.Sequential(
                nn.Linear(model_config["latent_dim"], repa_proj_hidden),
                nn.SiLU(),
                nn.Linear(repa_proj_hidden, repa_proj_hidden),
                nn.SiLU(),
                nn.Linear(repa_proj_hidden, dino_dim),
            )

        # Guard the documented latent geometry: an SxS input must encode to a
        # latent_dim x (S/16) x (S/16) latent (16x downsample). Catches a block-count or
        # latent_dim change that would silently disagree with the docstring / config.
        with torch.no_grad():
            probe = self.model.encode(
                torch.zeros(1, model_config["input_dim"], image_size, image_size)
            )
        expect = (
            model_config["latent_dim"],
            image_size // DOWNSAMPLE,
            image_size // DOWNSAMPLE,
        )
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
        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=self.lpips_net, normalize=True
        )
        lpips.requires_grad_(False)  # input still gets gradients; net stays frozen
        fid = FrechetInceptionDistance(feature=self.fid_feature, normalize=True)
        self._metrics = {
            "lpips": lpips.to(self.device).eval(),
            "fid": fid.to(self.device),
        }
        # Frozen DINOv2 REPA target -- same off-state_dict treatment as LPIPS/FID.
        if self.repa_weight > 0:
            dino = Dinov2Features(self.repa_model, image_size=self.repa_dino_size)
            self._metrics["dino"] = dino.to(self.device).eval()

    def on_fit_start(self) -> None:
        self._ensure_metrics()  # LPIPS is needed for the very first training_step loss

    # -- losses / quality metrics ----------------------------------------------------

    def _reconstruct(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        images, _ = batch
        x = images.float()  # bf16 [0,1] from the collate -> float32 in [0,1]
        if self.repa_weight > 0:  # also surface the latent for the REPA loss (single pass)
            recon, z = self.model(x, return_latent=True)
            return x, recon, z  # recon is sigmoid'd into [0,1]
        return x, self.model(x), None

    def _lpips(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Per-batch LPIPS (differentiable w.r.t. ``recon``) via the underlying LPIPS net.

        We deliberately do NOT call the ``LearnedPerceptualImagePatchSimilarity`` *metric*
        (``self._metrics["lpips"](recon, x)``): its ``forward`` appends every batch's score to
        an internal ``all_scores`` list that we never read (``compute()`` is never called) and
        never reset, so it grows for the whole run and makes per-step cost climb -- the cause
        of the per-epoch training slowdown. The metric only ever holds the frozen net for us;
        calling that net directly gives the same value statelessly and skips the metric's
        per-call input-range validation (a host-device sync) too."""
        lpips = self._metrics["lpips"]  # read normalize off the metric so the two can't drift
        return lpips.net(recon, x, normalize=lpips.normalize).squeeze().mean()

    def _repa(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """REPA loss: align the projected latent tokens to DINOv2 patch features of ``x``.

        ``z`` is the autoencoder latent ``(B, latent_dim, g, g)``; ``x`` is the clean ``[0,1]``
        input. The frozen DINOv2 features (computed under no_grad inside ``Dinov2Features``)
        are resized to the latent grid, then we maximize per-token cosine similarity between
        them and ``repa_proj(z)``. Returns ``1 - mean_cos`` (0 = perfectly aligned)."""
        dino = self._metrics["dino"]
        target = dino.as_grid(dino(x))  # (B, dino_dim, gh, gw), gradient-free
        g = z.shape[-1]
        target = F.interpolate(
            target.float(), size=g, mode="bilinear", align_corners=False
        )
        target = target.flatten(2).transpose(1, 2)  # (B, g*g, dino_dim)
        proj = self.repa_proj(z.flatten(2).transpose(1, 2))  # (B, g*g, dino_dim)
        proj = F.normalize(proj, dim=-1)
        target = F.normalize(target, dim=-1)
        return (1 - (proj * target).sum(dim=-1)).mean()

    def _log_quality(self, x: torch.Tensor, recon: torch.Tensor, stage: str) -> None:
        # PSNR/SSIM in fp32 (autocast off) so the logged numbers are precision-independent.
        with torch.autocast(self.device.type, enabled=False):
            xf, rf = x.float(), recon.detach().float().clamp(0, 1)
            psnr = peak_signal_noise_ratio(rf, xf, data_range=1.0)
            ssim = structural_similarity_index_measure(rf, xf, data_range=1.0)
        self.log(f"{stage}/psnr", psnr, prog_bar=True)
        self.log(f"{stage}/ssim", ssim, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, recon, z = self._reconstruct(batch)
        mse = F.mse_loss(recon, x)
        lpips = self._lpips(recon, x)
        loss = mse + self.lpips_weight * lpips
        if self.repa_weight > 0:
            repa = self._repa(z, x)
            loss = loss + self.repa_weight * repa
            self.log("train/repa", repa, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mse", mse)
        self.log("train/lpips", lpips, prog_bar=True)
        self._log_quality(x, recon, "train")
        return loss

    def _eval_step(self, batch, batch_idx: int, stage: str) -> None:
        x, recon, z = self._reconstruct(batch)
        mse = F.mse_loss(recon, x)
        lpips = self._lpips(recon, x)
        loss = mse + self.lpips_weight * lpips
        if self.repa_weight > 0:
            repa = self._repa(z, x)
            loss = loss + self.repa_weight * repa
            self.log(f"{stage}/repa", repa, prog_bar=True)
        self.log(f"{stage}/loss", loss, prog_bar=True)
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
            self._sample = (
                x[:n].float().cpu().clamp(0, 1),
                recon[:n].float().cpu().clamp(0, 1),
            )

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
        image = grid(
            panel, nrow=x.shape[0]
        )  # nrow = samples -> each category on its own row
        self.logger.log_image(
            f"{stage}/reconstructions",
            [image],
            caption=["rows: original / reconstruction / |diff|"],
        )
        self._sample = None

    def configure_optimizers(self):
        # self.parameters() = autoencoder + (when REPA is on) the projection head; the frozen
        # DINOv2/LPIPS/FID nets live in self._metrics (a plain dict), so they're excluded.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def build_model_config(base_channels: int) -> dict:
    """The ConvAutoEncoder config: 16x downsample = 4 halving blocks whose widths grow
    (1, 2, 4, 4)x base_channels, to a LATENT_CHANNELS x (S/16) x (S/16) latent. Shared by main()
    and benchmark.py so they construct the identical model."""
    c = base_channels
    return dict(
        input_dim=3,
        latent_dim=LATENT_CHANNELS,
        base_channels=c,
        dim_per_block=(c, 2 * c, 4 * c),
        layers_per_block=(1, 2, 3)
    )


def build_datamodule(
    *,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    in_memory: bool = True,
) -> ConcatImageDataModule:
    """Build the merged CelebA-HQ + AFHQ datamodule. Each source materializes + caches its own
    uint8 store at image_size; the merged module concatenates their splits and yields the
    bf16 [0,1] batches the model trains on. Shared by main() and benchmark.py."""
    sources = [
        cls(data_dir=data_dir, image_size=image_size, in_memory=in_memory)
        for cls in SOURCE_DATAMODULES.values()
    ]
    return ConcatImageDataModule(
        sources, batch_size=batch_size, num_workers=num_workers, in_memory=in_memory
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p, project="celeba-afhq-autoencoder", epochs=10)
    p.set_defaults(
        batch_size=64, lr=1e-3
    )  # high-res images + a larger model than MNIST
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="stem width; blocks are (1,2,4)x this",
    )
    p.add_argument(
        "--lpips-weight",
        type=float,
        default=0.1,
        help="weight of the LPIPS term in the loss",
    )
    p.add_argument("--lpips-net", choices=["vgg", "alex", "squeeze"], default="squeeze")
    p.add_argument(
        "--fid-feature", type=int, default=2048, help="InceptionV3 feature dim for rFID"
    )
    p.add_argument(
        "--repa-weight",
        type=float,
        default=0.5,
        help="weight of the REPA latent-alignment term; 0 disables REPA entirely",
    )
    p.add_argument(
        "--repa-model",
        choices=list(DINOV2_HIDDEN_SIZE),
        default="facebook/dinov2-small",
        help="frozen DINOv2 checkpoint providing the REPA alignment target",
    )
    p.add_argument(
        "--repa-dino-size",
        type=int,
        default=224,
        help="resolution the input is resized to for the DINOv2 target (multiple of 14)",
    )
    p.add_argument(
        "--mmap",
        action="store_true",
        help="memory-map the dataset instead of loading into RAM",
    )
    args = p.parse_args()

    seed_everything(
        args.seed, workers=True
    )  # seed python/numpy/torch + dataloader workers

    # 8x downsample, LATENT_CHANNELS x (image_size/8) x (image_size/8) latent. The geometry is
    # asserted at construction (see LitAutoEncoder.__init__) so this can't silently drift.
    model_config = build_model_config(args.base_channels)
    datamodule = build_datamodule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        in_memory=not args.mmap,
    )

    resume_ckpt = find_ckpt(args.resume, args.project, OUTPUTS) if args.resume else None
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
            repa_weight=args.repa_weight,
            repa_model=args.repa_model,
            repa_dino_size=args.repa_dino_size,
        )

    # Read model/loss hyperparameters off the module so the logged config matches the live
    # model on both a fresh run (from args) and a resume (restored from the checkpoint).
    hp = module.hparams
    config = {
        "model": hp["model_config"],
        "dataset": DATASET_NAME,
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
            "repa_weight": module.repa_weight,
            "repa_model": module.repa_model,
            "repa_dino_size": module.repa_dino_size,
        },
        "data": {
            "dataset": DATASET_NAME,
            "sources": list(SOURCE_DATAMODULES),
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
        outputs=OUTPUTS,
        resume_ckpt=resume_ckpt,
        artifact_metadata=config,
        test=True,  # report rFID/PSNR/SSIM/LPIPS on the test split after training
    )


if __name__ == "__main__":
    main()

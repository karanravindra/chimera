"""Train a TiTok-style ViT autoencoder on the union of CelebA-HQ and AFHQ with W&B logging.

A 1D tokenizer (Yu et al. 2024, "An Image is Worth 32 Tokens") over the *merged* StarGAN-v2
256x256 datasets (CelebA-HQ faces + AFHQ animal faces, concatenated into one train/eval set):

  * a ViT encoder fuses image patches with ``num_latent_tokens`` learnable query tokens and
    emits a **continuous** latent **token sequence** ``(B, K, latent_dim)`` -- unlike the conv
    autoencoder's spatial ``C x H/16 x W/16`` grid -- via :class:`~chimera.models.TiTokAutoEncoder`;
  * a ViT decoder reconstructs the image from those tokens (mask tokens + unpatchify);
  * the model is sized to a **ViT-Tiny** config (embed_dim 192, depth 12, 3 heads) by default;
  * trained on MSE + an LPIPS perceptual loss (``--lpips-weight``); MSE, LPIPS, PSNR and SSIM
    are logged every phase (train/val/test);
  * reconstruction FID (rFID = FID between originals and their reconstructions, via
    torchmetrics) is logged on the val and test phases -- expensive, so eval-only;
  * saves a full Lightning checkpoint to ``outputs/<run_id>/last.ckpt`` every epoch and uploads
    it as a wandb model artifact; ``--resume`` continues the same run and rebuilds the
    architecture from the checkpoint's saved hyperparameters.

The latent is intentionally **continuous** for now (a VQ/FSQ tokenizer slots in at the
``to_latent`` bottleneck later). REPA latent-alignment is off here: TiTok's latent is a token
sequence with no spatial grid, so the conv project's spatial DINOv2 loss does not apply.

The LPIPS (squeeze/vgg/alex) and FID (Inception) networks are eval-only and held off the
module's state_dict (see ``LitTiTok._ensure_metrics``) so they never bloat the checkpoint.

Examples
--------
    # fresh run on the merged CelebA-HQ + AFHQ set
    uv run python projects/text2image/titok/train.py --epochs 100

    # resume run <id> for more epochs (same wandb run, continues from its checkpoint)
    uv run python projects/text2image/titok/train.py --resume <run_id> --epochs 200
"""

from __future__ import annotations

import argparse
import math
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

from chimera.data import (
    AFHQDataModule,
    CelebAHQDataModule,
    ConcatImageDataModule,
    ReconstructionAugment,
)
from chimera.models import TiTokAutoEncoder
from chimera.optim import MuonWithAuxAdam, muon_adam_param_groups
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

# ViT-Tiny tokenizer bottleneck: K continuous tokens, each latent_dim wide. Shared by main()
# and benchmark.py so they construct the identical latent geometry.
NUM_LATENT_TOKENS = 16
LATENT_DIM = 16

OUTPUTS = Path(__file__).parent / "outputs"  # checkpoints live under OUTPUTS/<run_id>


class LitTiTok(LightningModule):
    """Wraps TiTokAutoEncoder: cast -> reconstruct -> MSE + LPIPS, with rFID/PSNR/SSIM."""

    def __init__(
        self,
        model_config: dict,
        image_size: int,
        lr: float = 1e-4,
        optimizer: str = "muon",
        muon_lr: float = 0.02,
        adam_lr: float = 8e-4,
        weight_decay: float = 0.05,
        min_lr_ratio: float = 0.05,
        lpips_weight: float = 1.0,
        lpips_net: str = "vgg",
        fid_feature: int = 2048,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TiTokAutoEncoder(**model_config)
        self.lr = lr
        self.optimizer = optimizer
        self.muon_lr = muon_lr
        self.adam_lr = adam_lr
        self.weight_decay = weight_decay
        self.min_lr_ratio = min_lr_ratio
        self.lpips_weight = lpips_weight
        self.lpips_net = lpips_net
        self.fid_feature = fid_feature
        self.n_log_images = 8  # how many reconstructions to log each eval epoch
        self._sample = None  # (originals, recons) stashed from the first eval batch
        # Eval-only metric networks (LPIPS for the perceptual loss + logging, FID for rFID),
        # filled in lazily by _ensure_metrics. Kept in a plain dict on purpose -- see there.
        self._metrics: dict = {}

        # Guard the documented latent geometry: an SxS input must encode to a continuous
        # (num_latent_tokens, latent_dim) token sequence. Catches a config change that would
        # silently disagree with the docstring / benchmark.
        with torch.no_grad():
            probe = self.model.encode(
                torch.zeros(1, model_config["input_dim"], image_size, image_size)
            )
        expect = (model_config["num_latent_tokens"], model_config["latent_dim"])
        assert tuple(probe.shape[1:]) == expect, (
            f"expected a {expect} latent for a {image_size}x{image_size} input, got "
            f"{tuple(probe.shape[1:])}; model_config num_latent_tokens / latent_dim disagree"
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

    def on_fit_start(self) -> None:
        self._ensure_metrics()  # LPIPS is needed for the very first training_step loss

    # -- losses / quality metrics ----------------------------------------------------

    def _reconstruct(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        images, _ = batch
        x = images.float()  # bf16 [0,1] from the collate -> float32 in [0,1]
        return x, self.model(x)  # recon is sigmoid'd into [0,1]

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
        lpips = self._lpips(recon, x)
        loss = mse + self.lpips_weight * lpips
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mse", mse)
        self.log("train/lpips", lpips, prog_bar=True)
        self._log_quality(x, recon, "train")
        return loss

    def _eval_step(self, batch, batch_idx: int, stage: str) -> None:
        x, recon = self._reconstruct(batch)
        mse = F.mse_loss(recon, x)
        lpips = self._lpips(recon, x)
        loss = mse + self.lpips_weight * lpips
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
        if self.optimizer == "muon":
            # Muon on the ViT's 2D hidden matmul weights, AdamW on the embeddings,
            # norms and biases. MuonWithAuxAdam presents a single torch optimizer so
            # Lightning's automatic optimization (grad clipping, LR logging) and the
            # cosine schedule below keep working -- the two LRs ride in separate param
            # groups and are each scaled by the scheduler. See chimera/optim/muon.py.
            groups = muon_adam_param_groups(
                self.model,
                muon_lr=self.muon_lr,
                adam_lr=self.adam_lr,
                weight_decay=self.weight_decay,
            )
            optimizer = MuonWithAuxAdam(groups)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        # Cosine decay to a floor of min_lr_ratio * peak (not 0): Southworth et al.
        # (arXiv:2605.24770) found a 0.05 minimum-LR ratio trains ViTs best. Driven via
        # LambdaLR so the single cosine factor scales each param group's own initial_lr
        # -- this floors the Muon and AdamW groups each at 5% of THEIR base LR, which a
        # scalar CosineAnnealingLR eta_min (one absolute floor for all groups) cannot do.
        t_max = self.trainer.max_epochs or 1
        floor = self.min_lr_ratio

        def cosine_with_floor(epoch: int) -> float:
            t = min(epoch, t_max) / t_max
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_floor)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def build_model_config(
    image_size: int,
    *,
    patch_size: int = 16,
    embed_dim: int = 192,
    depth: int = 12,
    num_heads: int = 3,
    num_latent_tokens: int = NUM_LATENT_TOKENS,
    latent_dim: int = LATENT_DIM,
    drop_path_rate: float = 0.0,
) -> dict:
    """The TiTokAutoEncoder config. Defaults are a canonical **ViT-Tiny** (embed_dim 192,
    depth 12, 3 heads, mlp_ratio 4) for both the encoder and decoder; the bottleneck is
    ``num_latent_tokens`` continuous tokens of width ``latent_dim``. ``drop_path_rate`` sets
    stochastic depth (ramped 0 -> rate across each stack) -- a ViT overfitting regularizer.
    Shared by main() and benchmark.py so they construct the identical model."""
    return dict(
        input_dim=3,
        image_size=image_size,
        patch_size=patch_size,
        num_latent_tokens=num_latent_tokens,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
    )


def build_datamodule(
    *,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    in_memory: bool = True,
    augment: bool = True,
    aug_min_scale: float = 0.4,
    aug_jitter: float = 0.3,
) -> ConcatImageDataModule:
    """Build the merged CelebA-HQ + AFHQ datamodule. Each source materializes + caches its own
    uint8 store at image_size; the merged module concatenates their splits and yields the
    bf16 [0,1] batches the model trains on. Shared by main() and benchmark.py.

    With ``augment=True`` a :class:`ReconstructionAugment` runs batched on-GPU as the
    ``gpu_transform`` -- per-sample random-resized-crop (area >= ``aug_min_scale``), horizontal
    flip, and brightness/contrast/saturation jitter (strength ``aug_jitter``) -- and
    ``augment_eval=False`` keeps it on training batches only, so val/test reconstruct clean images.
    This is the primary regularizer against the AE overfitting the ~45k-image set."""
    sources = [
        cls(data_dir=data_dir, image_size=image_size, in_memory=in_memory)
        for cls in SOURCE_DATAMODULES.values()
    ]
    transform = (
        ReconstructionAugment(min_scale=aug_min_scale, jitter=aug_jitter)
        if augment
        else None
    )
    return ConcatImageDataModule(
        sources,
        batch_size=batch_size,
        num_workers=num_workers,
        in_memory=in_memory,
        gpu_transform=transform,
        augment_eval=False,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p, project="text2image-titok", epochs=50)
    p.set_defaults(
        batch_size=64, lr=1e-3
    )  # high-res images + a ViT-Tiny tokenizer
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument(
        "--embed-dim", type=int, default=192, help="ViT hidden dim (192 = ViT-Tiny)"
    )
    p.add_argument(
        "--depth", type=int, default=6, help="transformer blocks per enc/dec stack"
    )
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument(
        "--num-latent-tokens",
        type=int,
        default=NUM_LATENT_TOKENS,
        help="K continuous latent tokens (the tokenizer's bottleneck)",
    )
    p.add_argument(
        "--latent-dim", type=int, default=LATENT_DIM, help="width of each latent token"
    )
    p.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        help="stochastic-depth rate (ramped 0->rate across each ViT stack); 0 disables. "
        "A ViT overfitting regularizer",
    )
    p.add_argument(
        "--optimizer",
        choices=["muon", "adamw"],
        default="muon",
        help="muon = Muon on 2D hidden weights + AdamW aux (default); adamw = plain AdamW",
    )
    p.add_argument(
        "--muon-lr", type=float, default=0.01, help="LR for the Muon (2D weight) group"
    )
    p.add_argument(
        "--adam-lr",
        type=float,
        default=3e-4,
        help="LR for the AdamW aux group (embeddings/norms/biases)",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="decoupled weight decay (0.05 = standard ViT range; helps overfitting)",
    )
    p.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train-only GPU augmentation (per-sample random-resized-crop + hflip); the main "
        "regularizer against overfitting -- eval stays on clean images. On by default; "
        "pass --no-augment to disable",
    )
    p.add_argument(
        "--aug-min-scale",
        type=float,
        default=0.4,
        help="min crop area fraction for --augment (lower = more aggressive)",
    )
    p.add_argument(
        "--aug-jitter",
        type=float,
        default=0.3,
        help="brightness/contrast/saturation jitter strength for --augment (0 disables)",
    )
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.05,
        help="cosine schedule floors at this fraction of peak LR (0 = decay to zero)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="stop if val/rfid hasn't improved for this many epochs (0 disables); the best "
        "checkpoint (not the last) is then kept and tested",
    )
    p.add_argument(
        "--lpips-weight",
        type=float,
        default=0.5,
        help="weight of the LPIPS term in the loss",
    )
    p.add_argument("--lpips-net", choices=["vgg", "alex", "squeeze"], default="alex")
    p.add_argument(
        "--fid-feature", type=int, default=2048, help="InceptionV3 feature dim for rFID"
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

    # ViT-Tiny tokenizer: encode an SxS image to a (num_latent_tokens, latent_dim) continuous
    # token sequence. The geometry is asserted at construction (see LitTiTok.__init__).
    model_config = build_model_config(
        args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_latent_tokens=args.num_latent_tokens,
        latent_dim=args.latent_dim,
        drop_path_rate=args.drop_path,
    )
    datamodule = build_datamodule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        in_memory=not args.mmap,
        augment=args.augment,
        aug_min_scale=args.aug_min_scale,
        aug_jitter=args.aug_jitter,
    )

    resume_ckpt = find_ckpt(args.resume, args.project, OUTPUTS) if args.resume else None
    if resume_ckpt:
        # Rebuild from the checkpoint's saved hyperparameters (model_config, image_size,
        # lpips/fid settings), not the current CLI flags, so a resume always reconstructs
        # the architecture the run was trained with.
        module = LitTiTok.load_from_checkpoint(resume_ckpt, lr=args.lr)
    else:
        module = LitTiTok(
            model_config,
            image_size=args.image_size,
            lr=args.lr,
            optimizer=args.optimizer,
            muon_lr=args.muon_lr,
            adam_lr=args.adam_lr,
            weight_decay=args.weight_decay,
            min_lr_ratio=args.min_lr_ratio,
            lpips_weight=args.lpips_weight,
            lpips_net=args.lpips_net,
            fid_feature=args.fid_feature,
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
            "optimizer": module.optimizer,
            "lr": module.lr,
            "muon_lr": module.muon_lr,
            "adam_lr": module.adam_lr,
            "weight_decay": module.weight_decay,
            "min_lr_ratio": module.min_lr_ratio,
            "drop_path": hp["model_config"]["drop_path_rate"],
            "early_stop_patience": args.early_stop_patience,
            "seed": args.seed,
            "precision": "bf16-mixed",
            "grad_clip": args.grad_clip,
            "image_size": hp["image_size"],
            "num_latent_tokens": hp["model_config"]["num_latent_tokens"],
            "latent_dim": hp["model_config"]["latent_dim"],
            "lpips_weight": module.lpips_weight,
            "lpips_net": module.lpips_net,
            "fid_feature": module.fid_feature,
        },
        "data": {
            "dataset": DATASET_NAME,
            "sources": list(SOURCE_DATAMODULES),
            "data_dir": args.data_dir,
            "num_workers": args.num_workers,
            "in_memory": not args.mmap,
            "augment": args.augment,
            "aug_min_scale": args.aug_min_scale if args.augment else None,
            "aug_jitter": args.aug_jitter if args.augment else None,
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
        # Keep + test the best-rFID epoch and stop once it plateaus (val == held-out test set
        # here), rather than training to the overfit endpoint.
        monitor="val/rfid",
        early_stop_patience=args.early_stop_patience,
    )


if __name__ == "__main__":
    main()

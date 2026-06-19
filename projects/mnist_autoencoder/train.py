"""Train a ConvAutoEncoder on MNIST with Weights & Biases logging.

Single, self-contained script:

  * resizes MNIST to 32x32 and applies light train-only augmentation (rotation +
    small translation) for regularization; the latent is 4x2x2 (C x H x W);
  * logs the full config and per-epoch metrics (loss, PSNR, SSIM) to wandb, plus
    validation reconstructions (images);
  * saves a full Lightning checkpoint to ``outputs/<run_id>/last.ckpt`` every epoch
    and uploads it as a wandb model artifact (so any run can be rebuilt later);
  * resumes a run exactly where it left off (optimizer + epoch + global step) while
    continuing the *same* wandb run.

Examples
--------
    # fresh run
    uv run python projects/mnist_autoencoder/train.py --epochs 20

    # resume run <id> for more epochs (same wandb run, continues from its checkpoint)
    uv run python projects/mnist_autoencoder/train.py --resume <run_id> --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.seed import isolate_rng
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchvision.transforms import v2
from torchvision.utils import make_grid

from chimera.data import MNISTDataModule
from chimera.models import ConvAutoEncoder

PROJECT_DEFAULT = "mnist-autoencoder"
IMAGE_SIZE = 32

# ConvAutoEncoder config: 4 downsample blocks take 32 -> 16 -> 8 -> 4 -> 2, so the
# latent is (latent_dim=4) x 2 x 2 (16 values). Logged to wandb and reused to
# rebuild the model. The latent shape is asserted at construction (see
# LitAutoEncoder.__init__) so this comment can't silently drift from the config.
MODEL_CONFIG = dict(
    input_dim=1,
    latent_dim=4,
    base_channels=4,
    dim_per_block=(8, 16, 16, 16),
    layers_per_block=(2, 2, 3, 3)
)

OUTPUTS = Path(__file__).parent / "outputs"


class LitAutoEncoder(LightningModule):
    """Wraps ConvAutoEncoder: resize -> (train) augment -> reconstruct -> MSE."""

    def __init__(self, model_config: dict, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConvAutoEncoder(**model_config)
        self.lr = lr
        # Light regularization, applied to training batches only (see _prepare).
        self.augment = v2.RandomAffine(degrees=10, translate=(0.1, 0.1))
        self.n_log_images = 8  # how many val reconstructions to log each epoch
        self._val_sample = None  # (originals, recons) stashed from the first val batch
        # Guard the documented latent geometry: a 32x32 input must encode to a
        # (latent_dim) x 2 x 2 latent. Catches a block-count change that would
        # silently make the comment / notebook ("4x2x2 = 16-dim") wrong.
        with torch.no_grad():
            probe = self.model.encode(torch.zeros(1, model_config["input_dim"], IMAGE_SIZE, IMAGE_SIZE))
        assert probe.shape[-2:] == (2, 2), (
            f"expected a 2x2 latent for a {IMAGE_SIZE}x{IMAGE_SIZE} input, got "
            f"{tuple(probe.shape[-2:])}; MODEL_CONFIG block count and the docstring disagree"
        )

    def _prepare(self, images: torch.Tensor, *, train: bool) -> torch.Tensor:
        """Cast to float [0, 1] and augment (train only). The DataModule already
        materializes images at IMAGE_SIZE, so no per-batch resize is needed."""
        x = images.float()
        if train:
            x = self.augment(x)
        return x

    def _step(self, batch, stage: str) -> torch.Tensor:
        images, _ = batch
        x = self._prepare(images, train=(stage == "train"))
        recon = self.model(x)
        loss = F.mse_loss(recon, x)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        # SSIM is an expensive windowed-conv metric; only the loss drives training,
        # so compute the reconstruction-quality metrics on eval batches only.
        if stage != "train":
            psnr = -10.0 * torch.log10(loss.detach().clamp_min(1e-12))
            ssim_val = ssim(recon.detach().float().clamp(0, 1), x.float(), data_range=1.0)
            self.log(f"{stage}/psnr", psnr, prog_bar=True)
            self.log(f"{stage}/ssim", ssim_val, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if batch_idx == 0:  # stash a fixed set of reconstructions to log this epoch
            with isolate_rng():  # don't let logging perturb the training RNG
                images, _ = batch
                x = self._prepare(images[: self.n_log_images], train=False)
                recon = self.model(x).clamp(0, 1)
                self._val_sample = (x.float().cpu(), recon.float().cpu())
        return loss

    def on_validation_epoch_end(self):
        # One image, three rows: originals / reconstructions / absolute difference.
        if self._val_sample is None or not isinstance(self.logger, WandbLogger):
            return
        x, recon = self._val_sample
        panel = torch.cat([x, recon, (x - recon).abs()], dim=0)
        grid = _grid(panel, nrow=x.shape[0])  # nrow = samples -> each category on its own row
        self.logger.log_image(
            "val/reconstructions", [grid], caption=["rows: original / reconstruction / |diff|"]
        )
        self._val_sample = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def _grid(images: torch.Tensor, nrow: int | None = None):
    """Tile a batch of [0,1] images into one grid, returned as an HxW (grayscale) numpy array."""
    grid = make_grid(images, nrow=nrow or images.shape[0], padding=2)
    grid = grid[0] if grid.shape[0] == 1 else grid.permute(1, 2, 0)
    return grid.numpy()


def _find_resume_ckpt(run_id: str, project: str) -> str:
    """Locate the checkpoint for a run: prefer the local copy, else the wandb artifact."""
    local = OUTPUTS / run_id / "last.ckpt"
    if local.exists():
        print(f"[resume] using local checkpoint {local}")
        return str(local)
    print(f"[resume] no local checkpoint; downloading model artifact for run {run_id}")
    api = wandb.Api()
    run = api.run(f"{api.default_entity}/{project}/{run_id}")
    for artifact in reversed(list(run.logged_artifacts())):
        if artifact.type != "model":
            continue
        directory = Path(artifact.download())
        ckpts = sorted(directory.glob("*.ckpt"))
        if ckpts:
            print(f"[resume] downloaded {ckpts[0]}")
            return str(ckpts[0])
    raise FileNotFoundError(f"No checkpoint found locally or as an artifact for run {run_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=7)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm (0 disables)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="/mnt/ai/data", help="where MNIST is downloaded/cached")
    p.add_argument("--project", default=PROJECT_DEFAULT)
    p.add_argument("--resume", metavar="RUN_ID", default=None, help="wandb run id to resume")
    args = p.parse_args()

    seed_everything(args.seed, workers=True)  # seed python/numpy/torch + dataloader workers

    config = {
        "model": MODEL_CONFIG,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "precision": "bf16-mixed",
            "grad_clip": args.grad_clip,
            "image_size": IMAGE_SIZE,
            "augment": "RandomAffine(degrees=10, translate=(0.1, 0.1))",
        },
        "data": {"dataset": "MNIST", "data_dir": args.data_dir, "num_workers": args.num_workers},
    }

    logger_kwargs = dict(project=args.project, config=config)
    if args.resume:
        logger_kwargs.update(id=args.resume, resume="must")
    logger = WandbLogger(**logger_kwargs)
    run_id = logger.experiment.id  # initializes the run; learn its id for the ckpt dir
    print(f"wandb run id: {run_id}{' (resumed)' if args.resume else ''}")

    ckpt_dir = OUTPUTS / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), save_last=True, save_top_k=0, every_n_epochs=1)

    datamodule = MNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=IMAGE_SIZE,  # materialize at 32x32 once; no per-batch resize
    )

    resume_ckpt = _find_resume_ckpt(args.resume, args.project) if args.resume else None
    if resume_ckpt:
        # Rebuild from the checkpoint's saved hyperparameters (model_config), not
        # the current module-level MODEL_CONFIG, so a resume always reconstructs
        # the architecture the run was trained with.
        module = LitAutoEncoder.load_from_checkpoint(resume_ckpt, lr=args.lr)
    else:
        module = LitAutoEncoder(MODEL_CONFIG, lr=args.lr)

    trainer = Trainer(
        max_epochs=args.epochs,
        precision="bf16-mixed",
        accelerator="auto",
        logger=logger,
        callbacks=[ckpt_cb],
        gradient_clip_val=args.grad_clip or None,
        gradient_clip_algorithm="norm",
        deterministic=True,
    )
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
    except KeyboardInterrupt:
        print("interrupted — uploading the latest checkpoint before exiting")

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        artifact = wandb.Artifact(name=run_id, type="model", metadata={"model": MODEL_CONFIG})
        artifact.add_file(str(last_ckpt))
        logger.experiment.log_artifact(artifact, aliases=["latest"])
        print(f"logged checkpoint artifact for run {run_id}")

    wandb.finish()


if __name__ == "__main__":
    main()

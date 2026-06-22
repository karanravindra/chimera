"""Train a class-conditioned diffusion transformer (rectified flow) on CelebA-HQ + AFHQ.

Two-stage latent generative model, the high-resolution sibling of
``projects/mnist/rectified_flow``:

  * a *frozen* pretrained ``ConvAutoEncoder`` (from the ``celeba_afhq/autoencoder`` project)
    maps a 3xSxS image to a small *spatial* latent ``(C, H, W)`` (8x downsample); latents are
    standardized per-channel before the flow and de-standardized after;
  * a :class:`~chimera.models.ClassVelocityDiT` learns the rectified-flow velocity
    ``v(z_t, t, y) ~= z1 - z0`` where ``z_t = (1 - t) z0 + t z1``, ``z0 ~ N(0, I)`` is noise
    and ``z1`` is a (normalized) image latent, conditioned on a class label ``y`` drawn from
    the **5-class union** of the two datasets (female, male, cat, dog, wild);
  * classifier-free guidance: the label is dropped to a learned null class during training and
    the velocity is extrapolated at sampling time (``--guidance-scale``);
  * each validation epoch we integrate the ODE from noise (Euler), decode with the AE, and log
    a class-conditioned sample grid to wandb (one row per class);
  * config + per-epoch losses logged to wandb, a full Lightning checkpoint saved every epoch
    and uploaded as a wandb model artifact, and ``--resume`` continues the same wandb run.

The DiT operates *directly* on the spatial latent map (patchified into tokens) -- the latent
is never flattened -- so it scales to the autoencoder's 8x(S/8)x(S/8) latent.

Examples
--------
    # fresh run against a trained autoencoder run (see celeba_afhq/autoencoder/outputs/<id>)
    uv run python projects/celeba_afhq/dit/train.py --ae-run <ae_run_id> --epochs 100

    # resume flow run <id> for more epochs (the AE must still be supplied)
    uv run python projects/celeba_afhq/dit/train.py --ae-run <ae_run_id> \
        --resume <dit_run_id> --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import LightningModule, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.seed import isolate_rng
from torchmetrics.image.fid import FrechetInceptionDistance

from chimera.data import AFHQDataModule, CelebAHQDataModule, ConcatImageDataModule
from chimera.models import ClassVelocityDiT, ConvAutoEncoder
from chimera.utils.experiment import (
    add_common_args,
    find_ckpt,
    grid,
    init_wandb_logger,
    run_training,
)

PROJECT_DEFAULT = "celeba-afhq-dit"
AE_PROJECT_DEFAULT = "celeba-afhq-autoencoder"

# Sources merged (via ConcatImageDataModule with unify_labels=True) into one train/eval set.
# The order here defines the unified class ids: CelebA-HQ {female=0, male=1}, then AFHQ
# {cat=2, dog=3, wild=4}. CLASS_NAMES must list them in that same id order.
SOURCE_DATAMODULES = {"celeba_hq": CelebAHQDataModule, "afhq": AFHQDataModule}
CLASS_NAMES = ["female", "male", "cat", "dog", "wild"]
NUM_CLASSES = len(CLASS_NAMES)  # 5 real classes; the null CFG class is index NUM_CLASSES.

OUTPUTS = Path(__file__).parent / "outputs"
AE_OUTPUTS = Path(__file__).parent.parent / "autoencoder" / "outputs"


class LitClassDiT(LightningModule):
    """Class-conditioned spatial-latent rectified flow over a frozen autoencoder."""

    def __init__(
        self,
        autoencoder: ConvAutoEncoder,
        ae_config: dict,
        latent_shape: tuple[int, int, int],
        dit_config: dict,
        lr: float = 1e-4,
        label_dropout: float = 0.1,
        guidance_scale: float = 3.0,
        sample_steps: int = 50,
        n_samples_per_class: int = 8,
        sample_seed: int = 0,
        fid_feature: int = 2048,
        gfid_every_n_epochs: int = 1,
    ):
        super().__init__()
        # The AE is a frozen, non-trainable dependency: don't checkpoint it as a hparam object,
        # but persist its *config* (its weights live in the state_dict, since it's a registered
        # submodule) so a run is fully rebuildable from the checkpoint alone.
        self.save_hyperparameters(ignore=["autoencoder"])

        self.ae = autoencoder.eval().requires_grad_(False)
        self.latent_shape = tuple(latent_shape)  # (C, H, W)
        c, h, w = self.latent_shape
        self.model = ClassVelocityDiT(
            latent_channels=c,
            latent_size=h,
            num_classes=NUM_CLASSES + 1,  # +1 for the null CFG class
            **dit_config,
        )

        self.lr = lr
        self.label_dropout = label_dropout
        self.guidance_scale = guidance_scale
        self.sample_steps = sample_steps
        self.n_samples_per_class = n_samples_per_class
        self.sample_seed = sample_seed
        self.null_class = NUM_CLASSES
        self.fid_feature = fid_feature
        self.gfid_every_n_epochs = gfid_every_n_epochs
        # Eval-only InceptionV3 for gFID (FID between real val images and generated samples),
        # built lazily and held in a plain dict -- not as an nn.Module attribute -- so the
        # ~85MB weights stay out of state_dict / checkpoints, the optimizer, and .to(); they
        # are frozen and trivially rebuilt (mirrors the autoencoder's rFID setup).
        self._metrics: dict = {}
        self._gfid_active = False  # set each validation epoch (frequency + enabled gate)

        # Per-channel latent standardization, filled in on_fit_start (and restored on resume).
        # Shape (1, C, 1, 1) so it broadcasts over the spatial latent map.
        self.register_buffer("latent_mean", torch.zeros(1, c, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, c, 1, 1))
        self.register_buffer("stats_ready", torch.zeros((), dtype=torch.bool))

    def train(self, mode: bool = True):
        """Keep the frozen autoencoder in eval mode regardless of the module's mode."""
        super().train(mode)
        self.ae.eval()
        return self

    # -- latent helpers --------------------------------------------------------------

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Images -> spatial latent (B, C, H, W), in float32 with autocast disabled.

        The DataModule materializes images at the AE's training resolution (same pipeline the
        AE trained on), so the batch is already the right size -- just cast to float."""
        with torch.autocast(self.device.type, enabled=False):
            return self.ae.encode(images.float())

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_std + self.latent_mean

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(Normalized) spatial latent -> image in [0, 1]."""
        with torch.autocast(self.device.type, enabled=False):
            return self.ae.decode(self.denormalize(z)).clamp(0, 1)

    def on_fit_start(self) -> None:
        """Estimate per-channel latent mean/std over a few train batches (once)."""
        if bool(self.stats_ready):
            return
        feats, seen = [], 0
        for images, _ in self.trainer.datamodule.train_dataloader():
            feats.append(self.encode(images.to(self.device)))
            seen += feats[-1].shape[0]
            if seen >= 4096:
                break
        z = torch.cat(feats, dim=0)  # (N, C, H, W)
        self.latent_mean.copy_(z.mean(dim=(0, 2, 3), keepdim=True))
        self.latent_std.copy_(z.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6))
        self.stats_ready.fill_(True)
        print(
            f"[latent stats] shape={self.latent_shape} from {z.shape[0]} samples | "
            f"mean|.|={self.latent_mean.abs().mean():.3f} std={self.latent_std.mean():.3f}"
        )

    def _ensure_metrics(self) -> None:
        """Build the InceptionV3 FID network once, on the module's device (see __init__ for
        why it's held off the state_dict)."""
        if self._metrics or self.fid_feature == 0:
            return
        fid = FrechetInceptionDistance(feature=self.fid_feature, normalize=True)
        self._metrics = {"fid": fid.to(self.device)}

    # -- training / validation -------------------------------------------------------

    def _flow_loss(self, batch, *, train: bool) -> torch.Tensor:
        images, y = batch
        z1 = self.normalize(self.encode(images))
        z0 = torch.randn_like(z1)
        t = torch.rand(z1.shape[0], device=z1.device)
        z_t = (1 - t)[:, None, None, None] * z0 + t[:, None, None, None] * z1
        target = z1 - z0

        if train and self.label_dropout > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.label_dropout
            y = torch.where(drop, torch.full_like(y, self.null_class), y)

        pred = self.model(z_t, t, y)
        return F.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        loss = self._flow_loss(batch, train=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._flow_loss(batch, train=False)
        self.log("val/loss", loss, prog_bar=True)
        self._update_gfid(batch)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Decide whether to score gFID this epoch (enabled + frequency) and, if so, build the
        FID net and reset it -- gFID is a per-epoch distribution over the whole val set."""
        every = max(1, self.gfid_every_n_epochs)
        self._gfid_active = self.fid_feature != 0 and (self.current_epoch + 1) % every == 0
        if self._gfid_active:
            self._ensure_metrics()
            self._metrics["fid"].reset()

    def _update_gfid(self, batch) -> None:
        """Accumulate the gFID distributions for one val batch: the real images as the "real"
        distribution and class-matched generated samples as the "fake" one. Samples are
        conditioned on the batch's own labels, so the generated set matches the real class mix.
        Inception runs in fp32 (autocast off) so the feature stats aren't bf16-noisy."""
        if not self._gfid_active:
            return
        images, y = batch
        gen = self.sample(y)  # module-default guidance_scale / sample_steps
        fid = self._metrics["fid"]
        with torch.autocast(self.device.type, enabled=False):
            fid.update(images.float().clamp(0, 1), real=True)
            fid.update(gen.float().clamp(0, 1), real=False)

    # -- sampling --------------------------------------------------------------------

    @torch.inference_mode()
    def sample(
        self,
        y: torch.Tensor,
        z0: torch.Tensor | None = None,
        *,
        guidance_scale: float | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Integrate the rectified-flow ODE from noise to latent, return decoded images.

        ``guidance_scale`` and ``steps`` default to the values the module was built with;
        callers (e.g. an analysis notebook) override them to sweep without rebuilding."""
        scale = self.guidance_scale if guidance_scale is None else guidance_scale
        steps = self.sample_steps if steps is None else steps
        if z0 is None:
            z0 = torch.randn(y.shape[0], *self.latent_shape, device=self.device)
        z = z0
        null = torch.full_like(y, self.null_class)
        dt = 1.0 / steps
        with torch.autocast(self.device.type, enabled=False):
            for step in range(steps):
                t = torch.full((y.shape[0],), step * dt, device=self.device)
                if scale == 1.0:
                    v = self.model(z, t, y)
                else:  # classifier-free guidance: one batched cond+uncond pass
                    zin, tin = torch.cat([z, z]), torch.cat([t, t])
                    v_cond, v_uncond = self.model(
                        zin, tin, torch.cat([y, null])
                    ).chunk(2)
                    v = v_uncond + scale * (v_cond - v_uncond)
                z = z + dt * v
        return self.decode(z)

    def on_validation_epoch_end(self) -> None:
        if self._gfid_active:
            self.log("val/gfid", self._metrics["fid"].compute(), prog_bar=True)
        if not isinstance(self.logger, WandbLogger):
            return
        n = self.n_samples_per_class
        # class-major order: row c (nrow=n) holds the n samples for class c.
        y = torch.arange(NUM_CLASSES, device=self.device).repeat_interleave(n)
        with isolate_rng():  # fixed noise -> stable grid across epochs
            torch.manual_seed(self.sample_seed)
            z0 = torch.randn(y.shape[0], *self.latent_shape, device=self.device)
        imgs = self.sample(y, z0).float().cpu()
        image = grid(imgs, nrow=n)
        self.logger.log_image(
            "samples/grid",
            [image],
            caption=[f"rows: {', '.join(CLASS_NAMES)} (cfg={self.guidance_scale})"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_autoencoder(ckpt_path: str) -> tuple[ConvAutoEncoder, dict, int]:
    """Rebuild a frozen ConvAutoEncoder from a LitAutoEncoder checkpoint (no script import).

    Returns the AE, its model config, and the ``image_size`` it was trained at (so the DiT
    encodes images at the same resolution the AE -- and its latent -- expect)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    config = hp["model_config"]
    image_size = hp["image_size"]
    ae = ConvAutoEncoder(**config)
    weights = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    ae.load_state_dict(weights)
    print(f"[ae] loaded ConvAutoEncoder {config} (image_size={image_size})")
    return ae.eval().requires_grad_(False), config, image_size


def _probe_latent_shape(
    ae: ConvAutoEncoder, datamodule: ConcatImageDataModule
) -> tuple[int, int, int]:
    """Encode one (already image_size) batch to learn the latent (C, H, W)."""
    datamodule.prepare_data()
    datamodule.setup("fit")
    images, _ = next(iter(datamodule.train_dataloader()))
    with torch.no_grad():
        z = ae.encode(images.float())
    return tuple(z.shape[1:])  # (C, H, W)


def build_datamodule(
    *, data_dir: str, image_size: int, batch_size: int, num_workers: int, in_memory: bool
) -> ConcatImageDataModule:
    """Merged CelebA-HQ + AFHQ datamodule with a unified 5-class label space (see
    SOURCE_DATAMODULES / CLASS_NAMES for the id order)."""
    sources = [
        cls(data_dir=data_dir, image_size=image_size, in_memory=in_memory)
        for cls in SOURCE_DATAMODULES.values()
    ]
    return ConcatImageDataModule(
        sources,
        batch_size=batch_size,
        num_workers=num_workers,
        in_memory=in_memory,
        unify_labels=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p, project=PROJECT_DEFAULT, epochs=10)
    p.set_defaults(batch_size=64, lr=1e-3)  # high-res latent; smaller batch than MNIST
    # autoencoder source (one of --ae-run / --ae-ckpt required)
    p.add_argument(
        "--ae-run", metavar="RUN_ID", default=None, help="autoencoder wandb run id"
    )
    p.add_argument(
        "--ae-project", default=AE_PROJECT_DEFAULT, help="autoencoder wandb project"
    )
    p.add_argument(
        "--ae-ckpt", default=None, help="direct path to an autoencoder checkpoint"
    )
    # rectified-flow / CFG
    p.add_argument(
        "--guidance-scale", type=float, default=3.0, help="CFG scale at sampling (1=off)"
    )
    p.add_argument(
        "--label-dropout", type=float, default=0.1, help="prob of dropping the label (CFG)"
    )
    p.add_argument(
        "--sample-steps", type=int, default=50, help="Euler ODE steps for sampling"
    )
    p.add_argument(
        "--mmap", action="store_true", help="memory-map the dataset instead of RAM"
    )
    # gFID (FID between real val images and generated samples)
    p.add_argument(
        "--fid-feature",
        type=int,
        default=2048,
        help="InceptionV3 feature dim for gFID (0 disables gFID logging)",
    )
    p.add_argument(
        "--gfid-every-n-epochs",
        type=int,
        default=5,
        help="score gFID every N validation epochs (generation is expensive; raise to throttle)",
    )
    # ClassVelocityDiT hyperparameters
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--time-dim", type=int, default=256)
    args = p.parse_args()

    if not args.ae_run and not args.ae_ckpt:
        p.error("supply --ae-run <id> or --ae-ckpt <path> to source the frozen autoencoder")

    seed_everything(args.seed, workers=True)

    ae_ckpt = args.ae_ckpt or find_ckpt(args.ae_run, args.ae_project, AE_OUTPUTS)
    autoencoder, ae_config, image_size = load_autoencoder(ae_ckpt)

    datamodule = build_datamodule(
        data_dir=args.data_dir,
        image_size=image_size,  # encode at the AE's training resolution
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        in_memory=not args.mmap,
    )

    resume_ckpt = find_ckpt(args.resume, args.project, OUTPUTS) if args.resume else None
    if resume_ckpt:
        # Rebuild the architecture from the checkpoint's saved hyperparameters
        # (latent_shape, dit_config) -- never the current CLI flags -- so a resume always
        # reconstructs the model the run was trained with, mirroring the autoencoder script.
        # The frozen AE is re-supplied here (it's excluded from the hparams); its weights are
        # then restored from the checkpoint along with the rest of the state.
        module = LitClassDiT.load_from_checkpoint(
            resume_ckpt, autoencoder=autoencoder, lr=args.lr
        )
        latent_shape, dit_config = module.latent_shape, module.hparams["dit_config"]
    else:
        latent_shape = _probe_latent_shape(autoencoder, datamodule)
        c, h, w = latent_shape
        if h != w:
            p.error(f"expected a square latent, got {latent_shape}")
        if h % args.patch_size != 0:
            p.error(f"--patch-size ({args.patch_size}) must divide the latent size ({h})")
        dit_config = dict(
            patch_size=args.patch_size,
            hidden_dim=args.hidden_dim,
            time_dim=args.time_dim,
            depth=args.depth,
            num_heads=args.num_heads,
        )
        module = LitClassDiT(
            autoencoder,
            ae_config=ae_config,
            latent_shape=latent_shape,
            dit_config=dit_config,
            lr=args.lr,
            label_dropout=args.label_dropout,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            fid_feature=args.fid_feature,
            gfid_every_n_epochs=args.gfid_every_n_epochs,
        )
    print(f"[latent] shape={latent_shape} patch_size={dit_config['patch_size']}")

    # Read the flow/sampling hyperparameters off the module so the logged config matches the
    # live model on both a fresh run (from args) and a resume (restored from the checkpoint).
    config = {
        "dit": {
            **dit_config,
            "latent_shape": list(latent_shape),
            "num_classes": NUM_CLASSES + 1,
        },
        "autoencoder": {
            "config": ae_config,
            "run": args.ae_run,
            "ckpt": ae_ckpt,
            "image_size": image_size,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": module.lr,
            "seed": args.seed,
            "precision": "bf16-mixed",
            "grad_clip": args.grad_clip,
            "guidance_scale": module.guidance_scale,
            "label_dropout": module.label_dropout,
            "sample_steps": module.sample_steps,
            "fid_feature": module.fid_feature,
            "gfid_every_n_epochs": module.gfid_every_n_epochs,
        },
        "data": {
            "dataset": "+".join(SOURCE_DATAMODULES),
            "sources": list(SOURCE_DATAMODULES),
            "classes": CLASS_NAMES,
            "image_size": image_size,
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
    )


if __name__ == "__main__":
    main()

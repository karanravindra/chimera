"""Train a class-conditioned rectified flow on MNIST in a pretrained AE latent space.

Two-stage generative model:

  * a *frozen* pretrained ``ConvAutoEncoder`` (from the ``mnist/autoencoder`` project)
    maps 32x32 MNIST images to a small spatial latent ``(C, H, W)``, flattened to a
    ``D = C*H*W`` vector; latents are standardized (per-dim mean/std) before the flow;
  * a ``VelocityDiT`` learns the rectified-flow velocity ``v(z_t, t, y) ~= z1 - z0`` where
    ``z_t = (1 - t) z0 + t z1``, ``z0 ~ N(0, I)`` is noise and ``z1`` is a (normalized)
    image latent, conditioned on the digit class ``y``;
  * classifier-free guidance: the label is dropped to a learned null class during training
    and the velocity is extrapolated at sampling time (``--guidance-scale``);
  * each validation epoch we integrate the ODE from noise (Euler), decode with the AE, and
    log a class-conditioned sample grid to wandb (one row per digit 0-9);
  * config + per-epoch losses logged to wandb, a full Lightning checkpoint saved every epoch
    and uploaded as a wandb model artifact, and ``--resume`` continues the same wandb run.

Examples
--------
    # fresh run against a trained autoencoder run (see mnist/autoencoder/outputs/<id>)
    uv run python projects/mnist/rectified_flow/train.py --ae-run <ae_run_id> --epochs 40

    # resume flow run <id> for more epochs (the AE must still be supplied)
    uv run python projects/mnist/rectified_flow/train.py --ae-run <ae_run_id> \
        --resume <flow_run_id> --epochs 80
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from lightning import LightningModule, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.seed import isolate_rng

from chimera.data import MNISTDataModule
from chimera.models import ConvAutoEncoder, VelocityDiT
from chimera.utils.experiment import (
    add_common_args,
    find_ckpt,
    grid,
    init_wandb_logger,
    run_training,
)

PROJECT_DEFAULT = "mnist-rectified-flow"
AE_PROJECT_DEFAULT = "mnist-autoencoder"
IMAGE_SIZE = 32
NUM_CLASSES = 10  # MNIST digits 0-9; the null CFG class is index NUM_CLASSES.

OUTPUTS = Path(__file__).parent / "outputs"
AE_OUTPUTS = Path(__file__).parent.parent / "autoencoder" / "outputs"


class LitRectifiedFlow(LightningModule):
    """Class-conditioned rectified flow over a frozen autoencoder's latent space."""

    def __init__(
        self,
        autoencoder: ConvAutoEncoder,
        ae_config: dict,
        latent_shape: tuple[int, int, int],
        flow_config: dict,
        lr: float = 1e-3,
        label_dropout: float = 0.1,
        guidance_scale: float = 3.0,
        sample_steps: int = 50,
        n_samples_per_class: int = 8,
        sample_seed: int = 0,
    ):
        super().__init__()
        # The AE is a frozen, non-trainable dependency: don't checkpoint it as a hparam
        # object, but persist its *config* (and its weights live in the state_dict, since
        # it's a registered submodule) so a run is fully rebuildable from the checkpoint
        # alone -- no wandb round-trip needed (see notebook.ipynb).
        self.save_hyperparameters(ignore=["autoencoder"])

        self.ae = autoencoder.eval().requires_grad_(False)
        self.latent_shape = tuple(latent_shape)  # (C, H, W)
        self.latent_dim = math.prod(self.latent_shape)
        self.model = VelocityDiT(
            latent_dim=self.latent_dim, num_classes=NUM_CLASSES + 1, **flow_config
        )

        self.lr = lr
        self.label_dropout = label_dropout
        self.guidance_scale = guidance_scale
        self.sample_steps = sample_steps
        self.n_samples_per_class = n_samples_per_class
        self.sample_seed = sample_seed
        self.null_class = NUM_CLASSES

        # Per-dim latent standardization, filled in on_fit_start (and restored on resume).
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_std", torch.ones(self.latent_dim))
        self.register_buffer("stats_ready", torch.zeros((), dtype=torch.bool))

    def train(self, mode: bool = True):
        """Keep the frozen autoencoder in eval mode regardless of the module's mode."""
        super().train(mode)
        self.ae.eval()
        return self

    # -- latent helpers --------------------------------------------------------------

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Images -> flat latent (B, D), in float32 with autocast disabled for stability.

        The DataModule materializes MNIST at IMAGE_SIZE (same pipeline the AE trained
        on), so the batch is already the right resolution -- just cast to float."""
        with torch.autocast(self.device.type, enabled=False):
            z = self.ae.encode(images.float())
        return z.flatten(1)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_std + self.latent_mean

    @torch.no_grad()
    def decode(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Flat (normalized) latent -> image in [0, 1]."""
        z = self.denormalize(z_flat).reshape(-1, *self.latent_shape)
        with torch.autocast(self.device.type, enabled=False):
            return self.ae.decode(z).clamp(0, 1)

    def on_fit_start(self) -> None:
        """Estimate per-dim latent mean/std over a few train batches (once)."""
        if bool(self.stats_ready):
            return
        feats, seen = [], 0
        for images, _ in self.trainer.datamodule.train_dataloader():
            feats.append(self.encode(images.to(self.device)))
            seen += feats[-1].shape[0]
            if seen >= 4096:
                break
        z = torch.cat(feats, dim=0)
        self.latent_mean.copy_(z.mean(0))
        self.latent_std.copy_(z.std(0).clamp_min(1e-6))
        self.stats_ready.fill_(True)
        print(
            f"[latent stats] D={self.latent_dim} from {z.shape[0]} samples | "
            f"mean|.|={self.latent_mean.abs().mean():.3f} std={self.latent_std.mean():.3f}"
        )

    # -- training / validation -------------------------------------------------------

    def _flow_loss(self, batch, *, train: bool) -> torch.Tensor:
        images, y = batch
        z1 = self.normalize(self.encode(images))
        z0 = torch.randn_like(z1)
        t = torch.rand(z1.shape[0], device=z1.device)
        z_t = (1 - t)[:, None] * z0 + t[:, None] * z1
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
        return loss

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
        callers (e.g. the analysis notebook) override them to sweep without rebuilding."""
        scale = self.guidance_scale if guidance_scale is None else guidance_scale
        steps = self.sample_steps if steps is None else steps
        if z0 is None:
            z0 = torch.randn(y.shape[0], self.latent_dim, device=self.device)
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
                    v_cond, v_uncond = self.model(zin, tin, torch.cat([y, null])).chunk(2)
                    v = v_uncond + scale * (v_cond - v_uncond)
                z = z + dt * v
        return self.decode(z)

    def on_validation_epoch_end(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        n = self.n_samples_per_class
        # class-major order: row c (nrow=n) holds the n samples for digit c.
        y = torch.arange(NUM_CLASSES, device=self.device).repeat_interleave(n)
        with isolate_rng():  # fixed noise -> stable grid across epochs
            torch.manual_seed(self.sample_seed)
            z0 = torch.randn(y.shape[0], self.latent_dim, device=self.device)
        imgs = self.sample(y, z0).float().cpu()
        image = grid(imgs, nrow=n)
        self.logger.log_image(
            "samples/grid", [image], caption=[f"rows = digits 0-9, cfg={self.guidance_scale}"]
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def load_autoencoder(ckpt_path: str) -> tuple[ConvAutoEncoder, dict]:
    """Rebuild a frozen ConvAutoEncoder from a LitAutoEncoder checkpoint (no script import)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["hyper_parameters"]["model_config"]
    ae = ConvAutoEncoder(**config)
    weights = {k[len("model.") :]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    ae.load_state_dict(weights)
    print(f"[ae] loaded ConvAutoEncoder {config}")
    return ae.eval().requires_grad_(False), config


def _probe_latent_shape(ae: ConvAutoEncoder, datamodule: MNISTDataModule) -> tuple[int, int, int]:
    """Encode one (already IMAGE_SIZE) batch to learn the latent (C, H, W)."""
    datamodule.prepare_data()
    datamodule.setup("fit")
    images, _ = next(iter(datamodule.train_dataloader()))
    with torch.no_grad():
        z = ae.encode(images.float())
    return tuple(z.shape[1:])  # (C, H, W)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p, project=PROJECT_DEFAULT, epochs=40)
    # autoencoder source (one of --ae-run / --ae-ckpt required)
    p.add_argument("--ae-run", metavar="RUN_ID", default=None, help="autoencoder wandb run id")
    p.add_argument("--ae-project", default=AE_PROJECT_DEFAULT, help="autoencoder wandb project")
    p.add_argument("--ae-ckpt", default=None, help="direct path to an autoencoder checkpoint")
    # rectified-flow / CFG
    p.add_argument("--guidance-scale", type=float, default=3.0, help="CFG scale at sampling (1=off)")
    p.add_argument("--label-dropout", type=float, default=0.1, help="prob of dropping the label (CFG)")
    p.add_argument("--sample-steps", type=int, default=50, help="Euler ODE steps for sampling")
    # VelocityDiT hyperparameters
    p.add_argument("--num-tokens", type=int, default=None, help="latent tokens (default: latent channels)")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--time-dim", type=int, default=128)
    args = p.parse_args()

    if not args.ae_run and not args.ae_ckpt:
        p.error("supply --ae-run <id> or --ae-ckpt <path> to source the frozen autoencoder")

    seed_everything(args.seed, workers=True)

    ae_ckpt = args.ae_ckpt or find_ckpt(args.ae_run, args.ae_project, AE_OUTPUTS)
    autoencoder, ae_config = load_autoencoder(ae_ckpt)

    datamodule = MNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=IMAGE_SIZE,  # materialize at 32x32 once, matching the AE's training pipeline
    )
    resume_ckpt = find_ckpt(args.resume, args.project, OUTPUTS) if args.resume else None
    if resume_ckpt:
        # Rebuild the architecture from the checkpoint's saved hyperparameters
        # (latent_shape, flow_config) -- never the current CLI flags -- so a resume always
        # reconstructs the model the run was trained with, mirroring the autoencoder script.
        # The frozen AE is re-supplied here (it's excluded from the hparams); its weights are
        # then restored from the checkpoint along with the rest of the state.
        module = LitRectifiedFlow.load_from_checkpoint(
            resume_ckpt, autoencoder=autoencoder, lr=args.lr
        )
        latent_shape, flow_config, latent_dim = (
            module.latent_shape,
            module.hparams["flow_config"],
            module.latent_dim,
        )
    else:
        latent_shape = _probe_latent_shape(autoencoder, datamodule)
        latent_dim = math.prod(latent_shape)
        num_tokens = args.num_tokens or latent_shape[0]
        if latent_dim % num_tokens != 0:
            p.error(f"--num-tokens ({num_tokens}) must divide the flat latent dim ({latent_dim})")
        flow_config = dict(
            hidden_dim=args.hidden_dim,
            time_dim=args.time_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            num_tokens=num_tokens,
        )
        module = LitRectifiedFlow(
            autoencoder,
            ae_config=ae_config,
            latent_shape=latent_shape,
            flow_config=flow_config,
            lr=args.lr,
            label_dropout=args.label_dropout,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
        )
    print(f"[latent] shape={latent_shape} flat_dim={latent_dim} num_tokens={flow_config['num_tokens']}")

    # Read the flow/sampling hyperparameters off the module so the logged config matches the
    # live model on both a fresh run (from args) and a resume (restored from the checkpoint).
    config = {
        "flow": {**flow_config, "latent_dim": latent_dim, "num_classes": NUM_CLASSES + 1},
        "autoencoder": {"config": ae_config, "run": args.ae_run, "ckpt": ae_ckpt},
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
        },
        "data": {"dataset": "MNIST", "data_dir": args.data_dir, "num_workers": args.num_workers},
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

"""Smoke tests for the chimera.train layer: config CLI parsing + a full run()."""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, TensorDataset

from chimera.train import RunResult, TrainConfig, run


@dataclass
class _Config(TrainConfig):
    run_dir: Path = Path("/mnt/ai/runs/test/task")
    wandb_project: str = "test-task"
    arch: str = "small"


def test_config_cli_parse():
    cfg = tyro.cli(
        _Config, args=["--lr", "3e-4", "--arch", "large", "--tags", "a", "b"]
    )
    assert cfg.lr == 3e-4
    assert cfg.arch == "large"
    assert cfg.tags == ("a", "b")
    assert cfg.run_dir == Path("/mnt/ai/runs/test/task")  # subclass default survives
    assert cfg.ema_decay is None


class _Module(LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 1)

    def _loss(self, batch):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x), y)

    def training_step(self, batch, _):
        loss = self._loss(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        self.log("val/loss", self._loss(batch))

    def test_step(self, batch, _):
        self.log("test/loss", self._loss(batch))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


class _Data(LightningDataModule):
    def _loader(self):
        x = torch.randn(16, 4)
        return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=8)

    train_dataloader = val_dataloader = test_dataloader = _loader


def test_run_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = _Config(
        run_dir=tmp_path,
        wandb_project="test-task",
        epochs=1,
        precision="32-true",
        wandb_offline=True,
        ema_decay=0.99,
    )
    result = run(
        cfg,
        _Module(),
        _Data(),
        trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False},
    )
    assert isinstance(result, RunResult)
    assert result.best_ckpt is not None and result.best_ckpt.exists()
    assert result.best_ckpt == tmp_path / "checkpoints" / "best.ckpt"
    assert "val/loss" in result.metrics and "test/loss" in result.metrics
    assert result.wandb_id is not None
    # EMA state made it into the shipped checkpoint (cfg.ema_decay wiring)
    assert "ema" in torch.load(result.best_ckpt, weights_only=False)

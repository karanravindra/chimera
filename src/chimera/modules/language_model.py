import math

import torch
from lightning import LightningModule
from torch import nn


class LanguageModelModule(LightningModule):
    """Next-token language modeling with cross-entropy loss.

    Logs, for every stage (train/val/test): loss (nats), bits-per-token
    (bpt = loss / ln 2, tokenizer-dependent), and -- when ``bytes_per_token`` is
    given -- bits-per-byte (bpb = bpt / bytes_per_token), the tokenizer-independent
    metric comparable across vocabularies (measure it with projects/llm/gpt/bpb.py
    and pass it from the training script).
    """

    def __init__(self, model, optimizer, scheduler, use_cce=False, bytes_per_token=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # bytes-per-token for the training corpus+tokenizer; enables bpb logging.
        # None -> only loss/bpt are logged (bpb needs the byte normalizer).
        self.bytes_per_token = bytes_per_token
        # Cut Cross Entropy (apple/ml-cross-entropy): fuse the lm_head projection
        # with cross-entropy so the (B, T, vocab) logits are never materialized —
        # a large memory saving with big vocabularies. Requires the model to
        # expose hidden states and lm_head_weight, and bf16/fp16 hidden states.
        self.use_cce = use_cce

        self.criterion = nn.CrossEntropyLoss()

        # Per-batch accumulators for the cross-dataloader aggregate val/test
        # metric (Lightning forbids logging one key across multiple dataloaders,
        # so we mean it ourselves at epoch end); per-source keys are logged
        # directly since each is unique to its dataloader.
        self._val_acc: list = []
        self._test_acc: list = []

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        if self.use_cce:
            from cut_cross_entropy import linear_cross_entropy

            hidden = self.model(x, return_hidden=True)  # (B, T, C)
            # CCE fuses hidden @ lm_head_weight and bypasses model.project(), so
            # apply the muP output multiplier here to match the non-CCE path.
            hidden = hidden * self.model.output_mult
            loss = linear_cross_entropy(
                hidden.reshape(-1, hidden.size(-1)),
                self.model.lm_head_weight,  # (V, C)
                y.reshape(-1),
            )
        else:
            logits = self.model(x)  # (B, T, V)
            vocab_size = logits.size(-1)
            loss = self.criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        bpt = loss / math.log(2)
        return loss, bpt

    def _log_stage(self, stage, loss, bpt, on_step):
        self.log(f"{stage}/loss", loss, on_step=on_step, on_epoch=not on_step, prog_bar=True)
        self.log(f"{stage}/bpt", bpt, on_step=on_step, on_epoch=not on_step, prog_bar=True)
        if self.bytes_per_token:
            self.log(f"{stage}/bpb", bpt / self.bytes_per_token,
                     on_step=on_step, on_epoch=not on_step, prog_bar=True)

    @staticmethod
    def _fully_masked(batch) -> bool:
        # SFT batches can be entirely non-supervised (all targets == ignore_index
        # -100), e.g. a window landing wholly in a prompt/tool-result span. The
        # mean cross-entropy over 0 valid tokens is 0/0 = NaN, which would poison
        # the epoch-averaged metric; skip such a batch (it carries no signal).
        _, y = batch
        return bool((y != -100).sum() == 0)

    def training_step(self, batch, batch_idx):
        if self._fully_masked(batch):
            return None
        loss, bpt = self._step(batch)
        self._log_stage("train", loss, bpt, on_step=True)
        lr = (
            self.scheduler.get_last_lr()[0]
            if self.scheduler is not None
            else self.optimizer.param_groups[0]["lr"]
        )
        self.log("train/lr", lr, on_step=True, prog_bar=True)
        return loss

    def on_before_zero_grad(self, optimizer):
        # DeepSeek aux-loss-free MoE: nudge each router's load-balancing bias once
        # per optimizer step (this hook fires after step(), before grads are
        # zeroed — correct under gradient accumulation), from the assignment
        # counts the gate forwards accumulated. No-op for dense models. Without
        # it the router never balances and experts collapse.
        raw = getattr(self.model, "_orig_mod", self.model)
        update = getattr(raw, "update_moe_bias", None)
        if update is not None:
            update()

    def _source_name(self, dataloader_idx):
        """Map a val/test dataloader index -> source key, when the datamodule
        serves one loader per source (per-dataset metrics). Returns None for the
        single-loader case (or if unavailable), i.e. log only the combined stage."""
        dm = getattr(self.trainer, "datamodule", None)
        names = getattr(dm, "val_source_names", None)
        # Fallback: train.py passes explicit dataloaders (no datamodule on the
        # trainer), so it also sets lm_module.val_source_names directly.
        if not names:
            names = getattr(self, "val_source_names", None)
        if not names or len(names) < 2 or dataloader_idx >= len(names):
            return None
        return names[dataloader_idx]

    def _log_eval(self, stage, loss, bpt, src, acc):
        # Aggregate across ALL dataloaders: accumulate per batch, mean at epoch
        # end (Lightning forbids logging one key from multiple dataloaders).
        acc.append((loss.detach(), bpt.detach()))
        # Per-source curves (grokking): val_<src>/* — unique key per dataloader,
        # so Lightning averages over that source's batches with no conflict.
        if src is not None:
            self.log(f"{stage}_{src}/loss", loss, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{stage}_{src}/bpt", bpt, on_epoch=True, add_dataloader_idx=False)
            if self.bytes_per_token:
                self.log(f"{stage}_{src}/bpb", bpt / self.bytes_per_token,
                         on_epoch=True, add_dataloader_idx=False)

    def _log_aggregate(self, stage, acc):
        if not acc:
            return
        loss = torch.stack([l for l, _ in acc]).mean()
        bpt = torch.stack([b for _, b in acc]).mean()
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/bpt", bpt, prog_bar=True)
        if self.bytes_per_token:
            self.log(f"{stage}/bpb", bpt / self.bytes_per_token, prog_bar=True)
        acc.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self._fully_masked(batch):
            return None
        loss, bpt = self._step(batch)
        self._log_eval("val", loss, bpt, self._source_name(dataloader_idx), self._val_acc)
        return loss

    def on_validation_epoch_end(self):
        self._log_aggregate("val", self._val_acc)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self._fully_masked(batch):
            return None
        loss, bpt = self._step(batch)
        self._log_eval("test", loss, bpt, self._source_name(dataloader_idx), self._test_acc)
        return loss

    def on_test_epoch_end(self):
        self._log_aggregate("test", self._test_acc)

    def configure_optimizers(self):
        # No scheduler -> constant LR (e.g. for clean muP LR sweeps).
        if self.scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

"""Shared LightningModule for the tinylm training stages (pretrain + SFT).

Factors the loss/optimizer/logging that was copy-pasted between the two raw
PyTorch loops into one place. The training math is unchanged from those loops:

    * Cut Cross Entropy (``linear_cross_entropy``) over the tied token-embedding /
      lm-head weight, fed the model's hidden states — on CUDA. FlexAttention
      packed-document block masks + per-document RoPE positions are rebuilt per
      batch inside the loss (``build_block_mask_and_pos``). A plain
      ``cross_entropy`` over logits is the CPU fallback (no flex / doc masking).
    * ``getattr(model, "_orig_mod", model)`` unwraps ``torch.compile`` for the
      tied weight (and for anything a caller does with ``.model`` post-fit).
    * Automatic optimization: the pre-built Muon(+AdamW) optimizer and a per-step
      LR scheduler are returned from ``configure_optimizers``. Gradient
      accumulation and clipping are the Trainer's job — a mean per-batch loss is
      returned and Lightning divides by ``accumulate_grad_batches`` before
      backward, reproducing the old manual microbatch division exactly.

Logging conventions (slash-separated so wandb groups them): ``train/loss`` on
step; ``val/loss`` (the ``ModelCheckpoint`` monitor) aggregated per epoch, plus
optional ``val/bpb`` (loss / ln2 / bytes-per-token) and per-source
``val/<src>/bpb`` when a mix-wide / per-source bytes-per-token is supplied.
"""

import math

import lightning.pytorch as pl
import torch.nn as nn

LN2 = math.log(2.0)


def _unwrap(model):
    """The eager module underneath a possible ``torch.compile`` wrapper."""
    return getattr(model, "_orig_mod", model)


class LanguageModelModule(pl.LightningModule):
    """Next-token LM training/eval shared by pretrain and SFT.

    Args:
        model: the GPT (optionally already ``torch.compile``-wrapped).
        optimizer: a pre-built optimizer over ``model``'s params (Muon+AdamW, or
            AdamW over LoRA params).
        scheduler: optional per-step LR scheduler (stepped every optimizer step).
        use_cce: use the fused CCE + FlexAttention path (CUDA). False -> the plain
            logits ``cross_entropy`` CPU fallback.
        logit_softcap: Gemma-style final-logit soft-cap passed to CCE.
        eos_id: document separator, for ``build_block_mask_and_pos``.
        bytes_per_token: mix-wide bytes/token; enables ``{train,val}/bpb`` logging.
        val_source_names: dataloader_idx -> source name, for per-source val bpb.
        source_bpt: source name -> bytes/token, for ``val/<src>/bpb``.
        doc_boundary_eos_id: when set, mask the loss target at every eos input
            position (predicting the next, unrelated document's first token).
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        *,
        use_cce: bool = True,
        logit_softcap: float | None = None,
        eos_id: int | None = None,
        bytes_per_token: float | None = None,
        val_source_names: list[str] | None = None,
        source_bpt: dict[str, float] | None = None,
        doc_boundary_eos_id: int | None = None,
        zero_grad_all_params: bool = False,
    ):
        super().__init__()
        self.model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.use_cce = bool(use_cce)
        self.logit_softcap = logit_softcap
        self.eos_id = eos_id
        self.bytes_per_token = bytes_per_token
        self.val_source_names = val_source_names
        self.source_bpt = source_bpt or {}
        self.doc_boundary_eos_id = doc_boundary_eos_id
        # LoRA: the frozen base weights keep requires_grad=True (the flex-attn
        # wrapper breaks on no-grad q/k/v) but are never stepped, so their grads
        # must be cleared model-wide — the optimizer only owns the LoRA params.
        self.zero_grad_all_params = bool(zero_grad_all_params)

        if self.use_cce:
            # Imported lazily: CCE + FlexAttention are CUDA-only; a CPU smoke run
            # (use_cce=False) must not need them installed/working.
            from cut_cross_entropy import linear_cross_entropy

            from chimera.models.attention import build_block_mask_and_pos

            self._cce = linear_cross_entropy
            self._build_mask = build_block_mask_and_pos

    # -- forward / loss ----------------------------------------------------
    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _target_mask(self, x, y):
        # Document masking: ignore the target at every eos input position (it would
        # predict the *next*, unrelated document's first token). SFT leaves this
        # None — its labels already carry -100 for non-assistant tokens.
        if self.doc_boundary_eos_id is not None:
            y = y.masked_fill(x == self.doc_boundary_eos_id, -100)  # CE/CCE ignore
        return y

    def _loss(self, x, y):
        y = self._target_mask(x, y)
        if self.use_cce:
            # FlexAttention block mask (causal + document) + per-document RoPE
            # position ids, rebuilt per batch. The smaller last val batch triggers
            # a one-time torch.compile recompile on the first val pass, then cached.
            block_mask, pos_ids = self._build_mask(x, self.eos_id)
            hidden = self.model(
                x, return_hidden=True, block_mask=block_mask, pos_ids=pos_ids
            )
            weight = _unwrap(self.model).token_emb.weight  # tied lm_head
            return self._cce(hidden, weight, y, softcap=self.logit_softcap)
        logits = self.model(x)  # CPU fallback: plain causal, no doc masking / flex
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )

    # -- train -------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._loss(x, y)
        bs = x.size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=bs)
        if self.bytes_per_token:
            self.log(
                "train/bpb",
                loss.detach() / LN2 / self.bytes_per_token,
                on_step=True,
                on_epoch=False,
                batch_size=bs,
            )
        return loss

    # -- eval (val / test share one body) ---------------------------------
    def _eval_step(self, batch, dataloader_idx, stage):
        x, y = batch
        loss = self._loss(x, y)
        bs = x.size(0)
        # Aggregate loss (batch-count/size mean over the epoch) — the checkpoint
        # monitor. add_dataloader_idx=False so multiple per-source val loaders all
        # reduce into the one `{stage}/loss` key.
        self.log(
            f"{stage}/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=bs,
            sync_dist=True,
        )
        if self.bytes_per_token:
            self.log(
                f"{stage}/bpb",
                loss.detach() / LN2 / self.bytes_per_token,
                on_epoch=True,
                add_dataloader_idx=False,
                batch_size=bs,
                sync_dist=True,
            )
        # Per-source headline: {stage}/<src>/bpb, each normalized by that source's
        # own bytes/token (so cross-source bpb is genuinely comparable).
        if self.val_source_names and self.source_bpt and dataloader_idx < len(
            self.val_source_names
        ):
            src = self.val_source_names[dataloader_idx]
            b = self.source_bpt.get(src)
            if b:
                self.log(
                    f"{stage}/{src}/bpb",
                    loss.detach() / LN2 / b,
                    on_epoch=True,
                    add_dataloader_idx=False,
                    batch_size=bs,
                    sync_dist=True,
                )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, dataloader_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, dataloader_idx, "test")

    # -- optim -------------------------------------------------------------
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        # Model-wide when requested (LoRA) so the un-stepped base weights don't
        # accumulate stale grads; otherwise the standard optimizer-only zero.
        if self.zero_grad_all_params:
            self.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        if self._scheduler is None:
            return self._optimizer
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": {
                "scheduler": self._scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

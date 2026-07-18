import math

import torch
import torch.nn.functional as F
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

    def __init__(self, model, optimizer, scheduler, use_cce=False,
                 bytes_per_token=None, mtp_weight=0.0,
                 nextlat_lambda_mse=0.0, nextlat_lambda_kl=0.0, nextlat_horizon=1):
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
        # Multi-Token Prediction auxiliary loss weight (lambda). The training loss
        # is L_main + mtp_weight * mean_k(L_mtp_k) when the model has MTP modules
        # (see chimera.models.gpt.MTPModule). 0 -> MTP off (also a no-op if the
        # model has no MTP modules). MTP affects TRAINING ONLY: val/test always
        # log the pure next-token loss, so val/loss & bpb stay comparable to a
        # non-MTP baseline. DeepSeek-V3 used lambda 0.3 early then 0.1.
        self.mtp_weight = mtp_weight
        # NextLat (arXiv:2511.05963) next-latent prediction auxiliary losses. When
        # the model has a dynamics_model, the training loss becomes
        # L_ntp + lambda_mse*L_smoothL1(sg[h_{t+i}], ĥ) + lambda_kl*L_KL. Both 0
        # (or no dynamics_model) -> NextLat off. Training-only: val/test log pure
        # next-token metrics (comparable to a non-NextLat baseline).
        self.nextlat_lambda_mse = nextlat_lambda_mse
        self.nextlat_lambda_kl = nextlat_lambda_kl
        self.nextlat_horizon = nextlat_horizon

        self.criterion = nn.CrossEntropyLoss()

        # Per-batch accumulators for the cross-dataloader aggregate val/test
        # metric (Lightning forbids logging one key across multiple dataloaders,
        # so we mean it ourselves at epoch end); per-source keys are logged
        # directly since each is unique to its dataloader.
        self._val_acc: list = []
        self._test_acc: list = []

    def forward(self, x):
        return self.model(x)

    def _target_mask(self, x, y):
        """Hook to mask the loss targets (-100 = ignore) before the CE. Identity
        here; subclasses override (e.g. document-boundary masking). Applied by
        both the plain and the MTP training path, so MTP auxiliary targets
        inherit the same masking."""
        return y

    def _ce_from_hidden(self, hidden, targets):
        """Cross-entropy of a (B, T, C) hidden state against (B, T) targets via
        the shared output head, honoring -100 ignore. Mirrors ``_step``'s CCE /
        dense split; used for the main head AND each MTP depth so they share one
        code path."""
        raw = getattr(self.model, "_orig_mod", self.model)
        if self.use_cce:
            from cut_cross_entropy import linear_cross_entropy

            hidden = hidden * raw.output_mult
            return linear_cross_entropy(
                hidden.reshape(-1, hidden.size(-1)),
                raw.lm_head_weight,  # (V, C)
                targets.reshape(-1),
            )
        # Dense: replicate GPT.project() (hidden @ head.T * output_mult).
        logits = (hidden @ raw.lm_head_weight.t()) * raw.output_mult
        return self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def _step(self, batch):
        x, y = batch
        y = self._target_mask(x, y)
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

    def _mtp_active(self) -> bool:
        raw = getattr(self.model, "_orig_mod", self.model)
        return self.mtp_weight > 0 and getattr(raw, "mtp_depth", 0) > 0

    def _training_loss(self, batch):
        """Single-forward MTP training loss. Returns (total, main_loss, bpt, aux):
        ``total`` = main + mtp_weight * mean_k(aux_k) is what we backprop;
        ``main_loss`` / ``bpt`` are the pure next-token metrics we log (kept
        comparable to a non-MTP baseline). Depth k predicts the token at offset
        k beyond the main next-token target, i.e. y shifted left by k with the
        exposed tail masked to -100."""
        x, y = batch
        y = self._target_mask(x, y)
        main_hidden, mtp_hiddens = self.model(x, return_mtp=True)
        main_loss = self._ce_from_hidden(main_hidden, y)
        B = y.size(0)
        aux = main_loss.new_zeros(())
        for k, hk in enumerate(mtp_hiddens, start=1):
            yk = torch.cat([y[:, k:], y.new_full((B, k), -100)], dim=1)
            aux = aux + self._ce_from_hidden(hk, yk)
        aux = aux / len(mtp_hiddens)
        total = main_loss + self.mtp_weight * aux
        bpt = main_loss / math.log(2)
        return total, main_loss, bpt, aux

    def _nextlat_active(self) -> bool:
        raw = getattr(self.model, "_orig_mod", self.model)
        return getattr(raw, "nextlat", False) and (
            self.nextlat_lambda_mse > 0 or self.nextlat_lambda_kl > 0
        )

    def _nextlat_aux(self, x, hidden, raw):
        """NextLat auxiliary losses (arXiv:2511.05963) — one recursive rollout of
        the latent-dynamics model p_ψ over ``nextlat_horizon`` steps, teacher-
        forced on the true tokens. Returns (mse, kl):

          * mse = mean_i SmoothL1( sg[h_{t+i}], ĥ_{t+i} )  — self-predictive; the
            stop-gradient on the target prevents representational collapse.
          * kl  = mean_i KL( sg[p(·|h_{t+i})] || p(·|ĥ_{t+i}) ), computed with the
            output head weights *detached* so the KL updates only p_ψ and the
            trunk (via ĥ), never the head (matches the reference impl).

        ĥ depends on the live ``hidden`` (no stop-grad on the input), so both
        losses backprop into the transformer trunk — that is what shapes it
        toward belief states. Positions whose target hidden belongs to an eos
        token are masked (they cross document boundaries)."""
        d = self.nextlat_horizon
        out_mult = raw.output_mult
        head_w = raw.lm_head_weight.detach()  # frozen head for the latent losses
        eos_id = raw.doc_mask_eos_id
        curr_eos = (
            (x == eos_id) if eos_id is not None
            else torch.zeros_like(x, dtype=torch.bool)
        )
        use_kl = self.nextlat_lambda_kl > 0
        pred = hidden                                   # ĥ rollout, starts at h_t
        target = hidden                                 # true future hidden (sg'd below)
        toks = raw.tok_emb(x) * raw.mup_input_mult      # next-token embeddings ("actions")
        # Teacher token-distribution from the true hidden (fully detached).
        teacher = (hidden.detach() @ head_w.t()) * out_mult if use_kl else None
        mse_tot = hidden.new_zeros(())
        kl_tot = hidden.new_zeros(())
        for i in range(d):
            pred = pred[:, :-1]
            toks = toks[:, 1:]
            target = target[:, 1:]
            if use_kl:
                teacher = teacher[:, 1:]
            pred = raw.dynamics_model(pred, toks)       # ĥ_{t+i} = p_ψ(ĥ_{t+i-1}, x_{t+i})
            mmask = (~curr_eos[:, i + 1:]).unsqueeze(-1).to(pred.dtype)
            mse_elem = F.smooth_l1_loss(pred, target.detach(), reduction="none") * mmask
            mse_tot = mse_tot + mse_elem.sum() / mmask.expand_as(mse_elem).sum().clamp_min(1.0)
            if use_kl:
                student = (pred @ head_w.t()) * out_mult
                kmask = (~curr_eos[:, i + 1:]).to(student.dtype)
                log_q = F.log_softmax(teacher.detach(), dim=-1)  # teacher (true)
                log_p = F.log_softmax(student, dim=-1)           # student (predicted)
                kl_pt = F.kl_div(log_p, log_q, log_target=True, reduction="none").sum(-1)
                kl_tot = kl_tot + (kl_pt * kmask).sum() / kmask.sum().clamp_min(1.0)
        return mse_tot / d, kl_tot / d

    def _nextlat_training_loss(self, batch):
        """Single-forward NextLat training loss. Returns (total, main_loss, bpt,
        mse, kl); ``total`` = L_ntp + lambda_mse*mse + lambda_kl*kl is
        backpropagated, while main_loss/bpt (pure next-token) are logged so the
        metrics stay comparable to a non-NextLat baseline."""
        x, y = batch
        y = self._target_mask(x, y)
        raw = getattr(self.model, "_orig_mod", self.model)
        hidden = self.model(x, return_hidden=True)  # (B, T, C), grad-connected
        main_loss = self._ce_from_hidden(hidden, y)
        mse, kl = self._nextlat_aux(x, hidden, raw)
        total = main_loss + self.nextlat_lambda_mse * mse + self.nextlat_lambda_kl * kl
        bpt = main_loss / math.log(2)
        return total, main_loss, bpt, mse, kl

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
        if self._nextlat_active():
            total, loss, bpt, mse, kl = self._nextlat_training_loss(batch)
            self.log("train/nextlat_mse", mse, on_step=True, prog_bar=False)
            self.log("train/nextlat_kl", kl, on_step=True, prog_bar=False)
        elif self._mtp_active():
            total, loss, bpt, aux = self._training_loss(batch)
            self.log("train/mtp_aux", aux, on_step=True, prog_bar=False)
        else:
            loss, bpt = self._step(batch)
            total = loss
        # Log the pure next-token loss/bpt (comparable across MTP on/off); the
        # returned `total` (main + weighted MTP aux) is what gets backpropagated.
        self._log_stage("train", loss, bpt, on_step=True)
        lr = (
            self.scheduler.get_last_lr()[0]
            if self.scheduler is not None
            else self.optimizer.param_groups[0]["lr"]
        )
        self.log("train/lr", lr, on_step=True, prog_bar=True)
        return total

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

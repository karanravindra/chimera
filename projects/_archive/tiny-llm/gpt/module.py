"""Tiny-LM Lightning module: per-source bpb metrics, minimal logging.

Subclasses the shared ``LanguageModelModule`` to change only what it logs — the
training/eval math (CE loss, CCE path, MoE-bias hook, masked-batch skipping) is
inherited unchanged. Conventions (locked with the user):

    * per-source eval metric is ``val/<src>/bpb`` ONLY — slash-separated so
      wandb nests every source under one "val" panel group. Each source is
      normalized by its OWN bytes/token (``source_bpt``), so cross-source bpb is
      genuinely comparable rather than the loss rescaled by one global constant.
    * ``bpt`` is dropped everywhere — it is exactly ``loss / ln2`` (a constant
      rescale of the loss), so logging it alongside loss/bpb is pure redundancy.
    * per-source loss is dropped — within a fixed source it is proportional to
      that source's bpb, so it adds nothing over ``val/<src>/bpb``.
    * aggregate ``{stage}/loss`` (nats — the objective + ModelCheckpoint monitor)
      and aggregate ``{stage}/bpb`` (normalized by the mix-wide ``bytes_per_token``)
      are kept.
"""

import torch

from chimera.modules import LanguageModelModule


class TinyLMModule(LanguageModelModule):
    def __init__(self, *args, source_bpt=None, doc_boundary_eos_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        # per-source bytes/token, {src: bpt}; set from bpb.measure() in train.py.
        # A source missing here (or 0) simply has its per-source bpb skipped.
        self.source_bpt = source_bpt or {}
        # Document masking: when the eos id is set, ignore the loss target at every
        # eos input position (it would predict the *next*, unrelated document's
        # first token — uninformed once attention can't cross the boundary). Pairs
        # with the model's doc_mask_eos_id; None -> no boundary masking.
        self.doc_boundary_eos_id = doc_boundary_eos_id

    def _target_mask(self, x, y):
        # Document masking: ignore the target at every eos input position (it
        # would predict the *next*, unrelated document's first token). Applied by
        # both the plain next-token path and the MTP auxiliary path via the base
        # class hook, so MTP targets inherit the same boundary masking.
        if self.doc_boundary_eos_id is not None:
            y = y.masked_fill(
                x == self.doc_boundary_eos_id, -100
            )  # CE/CCE ignore_index
        return y

    def _log_stage(self, stage, loss, bpt, on_step):
        # train on-step: loss + aggregate bpb (mix-wide normalizer). No bpt.
        self.log(
            f"{stage}/loss", loss, on_step=on_step, on_epoch=not on_step, prog_bar=True
        )
        if self.bytes_per_token:
            self.log(
                f"{stage}/bpb",
                bpt / self.bytes_per_token,
                on_step=on_step,
                on_epoch=not on_step,
                prog_bar=True,
            )

    def _log_eval(self, stage, loss, bpt, src, acc):
        # accumulate for the cross-source aggregate (mean at epoch end)
        acc.append((loss.detach(), bpt.detach()))
        # per-source headline: val/<src>/bpb, normalized by the source's own bytes/token
        if src is not None:
            b = self.source_bpt.get(src)
            if b:
                self.log(
                    f"{stage}/{src}/bpb",
                    bpt / b,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )

    def _log_aggregate(self, stage, acc):
        if not acc:
            return
        loss = torch.stack([l for l, _ in acc]).mean()
        bpt = torch.stack([b for _, b in acc]).mean()
        self.log(f"{stage}/loss", loss, prog_bar=True)  # checkpoint monitor
        if self.bytes_per_token:
            self.log(f"{stage}/bpb", bpt / self.bytes_per_token, prog_bar=True)
        acc.clear()

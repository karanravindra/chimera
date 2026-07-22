"""Exponential moving average of model weights as a Lightning callback.

Keeps a shadow copy of the trained parameters, updated every optimizer step as

    shadow = d * shadow + (1 - d) * param

with a *warmed-up* decay ``d`` so the average doesn't cling to the random init
while the weights are still moving fast early in training:

    d = min(decay, (1 + n) / (warmup_steps + n))          # n = updates so far

At ``n = 0`` the decay is ~``1/warmup_steps`` (shadow ≈ live weights); it climbs
toward the ``decay`` ceiling as ``n`` grows. Bigger ``warmup_steps`` -> slower
ramp. This mirrors timm's ``ModelEmaV2`` warmup and self-scales to the run length
(a 2k-step run settles around d≈0.95, a 30k-step run around d≈0.996).

The shadow is kept in **fp32** even under ``bf16-true`` training: a bf16 running
average loses the small ``(1 - d) * param`` increment to rounding and barely
moves, so the EMA must accumulate in fp32 and cast on read.

Evaluation uses the EMA weights: the callback swaps the shadow into the live
parameters at ``on_validation_start`` / ``on_test_start`` (in place via
``.data.copy_`` so torch.compile's cudagraph static addresses stay valid, per the
cudagraph-eval convention) and restores the raw weights afterwards so training
continues on the true optimizer trajectory. At ``on_fit_end`` the EMA weights are
loaded into the model *permanently*, so anything that runs on the in-memory model
after ``trainer.fit`` (a separate ``trainer.test``, sampling, downstream
benchmarks) uses the averaged model too. EMA state is also stashed in the
checkpoint under ``checkpoint["ema"]`` for resume/inspection.
"""

import lightning.pytorch as pl
import torch


class EMACallback(pl.Callback):
    """Maintain a warmed-up EMA of ``pl_module`` weights and evaluate with it.

    Args:
        decay: ceiling for the EMA decay (steady-state smoothing).
        warmup_steps: decay-ramp constant; the effective decay is
            ``min(decay, (1 + n) / (warmup_steps + n))`` at update ``n``.
        eval_with_ema: swap EMA weights in for validation/test (and load them
            permanently at fit end). Off -> the shadow is tracked but never used.
        update_every: update the shadow every k optimizer steps (1 = every step).
    """

    def __init__(
        self,
        decay: float = 0.999,
        warmup_steps: int = 100,
        eval_with_ema: bool = True,
        update_every: int = 1,
    ):
        super().__init__()
        self.decay = float(decay)
        self.warmup_steps = max(1, int(warmup_steps))
        self.eval_with_ema = bool(eval_with_ema)
        self.update_every = max(1, int(update_every))
        self.n_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._swapped = False
        self._last_step = -1
        self._pending_state: dict | None = None  # EMA restored from a checkpoint

    # -- setup -------------------------------------------------------------
    @staticmethod
    def _params(pl_module):
        """(name, param) for every trained parameter, in a stable order."""
        return [(n, p) for n, p in pl_module.named_parameters() if p.requires_grad]

    def on_fit_start(self, trainer, pl_module):
        # Build the fp32 shadow after the model is on-device (before sanity val).
        if not self.shadow:
            self.shadow = {
                n: p.detach().float().clone() for n, p in self._params(pl_module)
            }
        if self._pending_state is not None:  # resumed from checkpoint
            self.n_updates = int(self._pending_state.get("n_updates", 0))
            saved = self._pending_state.get("shadow", {})
            for n, t in self.shadow.items():
                if n in saved:
                    t.copy_(saved[n].to(t.device))
            self._pending_state = None

    def _decay(self) -> float:
        return min(
            self.decay, (1 + self.n_updates) / (self.warmup_steps + self.n_updates)
        )

    # -- update ------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        step = trainer.global_step
        if step == self._last_step:  # same optimizer step (grad accumulation) — skip
            return
        self._last_step = step
        if step % self.update_every != 0:
            return
        d = self._decay()
        with torch.no_grad():
            for n, p in self._params(pl_module):
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)
        self.n_updates += 1

    # -- eval swap ---------------------------------------------------------
    def _swap_in(self, pl_module):
        if not self.eval_with_ema or not self.shadow or self._swapped:
            return
        self._backup = {}
        with torch.no_grad():
            for n, p in self._params(pl_module):
                if n in self.shadow:
                    self._backup[n] = p.detach().clone()
                    p.data.copy_(self.shadow[n])  # fp32 -> param dtype, in place
        self._swapped = True

    def _swap_out(self, pl_module):
        if not self._swapped:
            return
        with torch.no_grad():
            for n, p in self._params(pl_module):
                if n in self._backup:
                    p.data.copy_(self._backup[n])
        self._backup = {}
        self._swapped = False

    def on_validation_start(self, trainer, pl_module):
        self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_in(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def on_fit_end(self, trainer, pl_module):
        # Ship the EMA model: load the averaged weights into the live model for
        # good, so post-fit test / generations / benchmarks use them.
        if not self.eval_with_ema or not self.shadow:
            return
        self._swap_out(pl_module)  # ensure not mid-swap
        with torch.no_grad():
            for n, p in self._params(pl_module):
                if n in self.shadow:
                    p.data.copy_(self.shadow[n])

    # -- checkpointing -----------------------------------------------------
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Ship the EMA model on disk: overwrite the saved parameters with the
        # shadow so best.ckpt IS the averaged model (the one we evaluate + ship).
        # Order-independent — Lightning runs ModelCheckpoint last, so relying on a
        # validation-time weight-swap being active at save is not robust; this
        # guarantees it regardless. Raw EMA state is also kept for resume.
        if self.eval_with_ema and self.shadow:
            sd = checkpoint.get("state_dict", {})
            for n, t in self.shadow.items():
                if n in sd:
                    sd[n] = t.detach().to(sd[n].dtype).cpu()
        checkpoint["ema"] = {
            "n_updates": self.n_updates,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "shadow": {n: t.detach().cpu() for n, t in self.shadow.items()},
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema" in checkpoint:
            self._pending_state = checkpoint["ema"]

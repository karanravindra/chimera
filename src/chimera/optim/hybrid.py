"""Run several optimizers / schedulers together as one.

Useful for pairing :class:`chimera.optim.muon.Muon` (2D hidden weights) with an
AdamW group for everything else: both are driven through a single
``optimizer.step()`` / ``scheduler.step()`` from the training loop.
"""


class HybridOptim:
    def __init__(self, opts):
        self.optimizers = opts

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # Accept a closure so this matches the torch.optim.Optimizer.step contract
        # (and MuonWithAuxAdam.step), e.g. for GradScaler / closure-based loops.
        loss = closure() if closure is not None else None
        for o in self.optimizers:
            o.step()
        return loss

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, sds):
        for o, sd in zip(self.optimizers, sds):
            o.load_state_dict(sd)


class HybridSched:
    def __init__(self, scheds):
        self.scheds = scheds

    def step(self):
        for s in self.scheds:
            s.step()

    def get_last_lr(self):
        return [lr for s in self.scheds for lr in s.get_last_lr()]

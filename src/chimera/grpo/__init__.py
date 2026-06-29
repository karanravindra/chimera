"""GRPO (Group Relative Policy Optimization) building blocks.

The reusable library behind ``projects/grpo`` (the runnable ``train.py`` / ``eval.py`` live
there; everything importable lives here). The pure algorithm core, the verifiable-reward
primitives, and the rollout-generation helper are exported eagerly (torch / stdlib only).
The task registry and the prompt DataModule are imported explicitly to avoid pulling
``datasets`` / ``lightning`` at package-import time -- mirroring :mod:`chimera.data`:

    from chimera.grpo.tasks import get_task
    from chimera.grpo.data import PromptDataModule
"""

from chimera.grpo.core import (
    compute_group_advantages,
    grpo_loss,
    mgpo_difficulty_weights,
    selective_log_softmax,
)
from chimera.grpo.generation import build_completion_mask, generate_completions
from chimera.grpo.rewards import (
    correctness_reward,
    extract_final_answer,
    format_reward,
)

__all__ = [
    "compute_group_advantages",
    "grpo_loss",
    "mgpo_difficulty_weights",
    "selective_log_softmax",
    "build_completion_mask",
    "generate_completions",
    "correctness_reward",
    "extract_final_answer",
    "format_reward",
]

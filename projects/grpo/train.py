"""GRPO fine-tuning of a small LLM on verifiable tasks (default: LiquidAI/LFM2.5-230M on GSM8K).

Group Relative Policy Optimization (DeepSeekMath, arXiv:2402.03300) implemented from scratch
as a ``LightningModule`` on the repo's training rails (``chimera.utils.experiment`` +
wandb), with HuggingFace ``transformers`` generation and a LoRA policy. Each training step:

  1. sample ``--num-generations`` (G) completions per prompt with ``model.generate``;
  2. score each completion with the task's verifiable reward functions (correctness + format);
  3. turn rewards into group-relative advantages (center per prompt, optionally std-scale);
  4. a single grad forward over prompt+completion for the completion token log-probs;
  5. token-level (DAPO) policy-gradient loss, optional KL to the LoRA-disabled base model;
  6. one optimizer step (mu = 1, so the PPO ratio is exactly 1 -- no theta_old cache).

The task is pluggable via :mod:`tasks` (``--task``); GSM8K ships today. The base model is
loaded in bf16 and adapted with LoRA, whose adapter weights peft keeps in fp32 for a clean
optimizer update -- so the frozen base stays cheap while training is precise. With ``beta>0``
the reference policy is the **adapter-disabled** base model (``model.disable_adapter()``), so
no separate reference model is held in memory.

Examples
--------
    # fresh GSM8K run
    uv run python projects/grpo/train.py --task gsm8k --epochs 2

    # resume run <id> (same wandb run, rebuilds model + adapter from the checkpoint)
    uv run python projects/grpo/train.py --resume <run_id> --epochs 4

    # fast smoke run (few prompts, small group, tiny val)
    uv run python projects/grpo/train.py --train-size 32 --val-size 16 \
        --batch-size 2 --num-generations 4 --max-completion-len 128 --epochs 1
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

# Make Lightning's deterministic=True safe: give cuBLAS a fixed workspace so its GEMMs have a
# deterministic algorithm available (otherwise some matmuls raise under deterministic mode).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from lightning import LightningModule, seed_everything
from lightning.pytorch.loggers import WandbLogger
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from chimera.grpo import (
    compute_group_advantages,
    generate_completions,
    grpo_loss,
    selective_log_softmax,
)
from chimera.grpo.data import PromptDataModule
from chimera.grpo.tasks import get_task
from chimera.optim import cosine_with_floor
from chimera.utils.experiment import (
    add_common_args,
    find_ckpt,
    init_wandb_logger,
    run_training,
)

OUTPUTS = Path(__file__).parent / "outputs"  # checkpoints live under OUTPUTS/<run_id>


class LitGRPO(LightningModule):
    """LoRA policy trained with from-scratch GRPO; periodic greedy pass@1 on a val subset."""

    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2.5-230M",
        task_name: str = "gsm8k",
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        num_generations: int = 8,
        max_prompt_len: int = 256,
        max_completion_len: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        beta: float = 0.0,
        scale_rewards: bool = True,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        min_lr_ratio: float = 0.1,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = get_task(task_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched left-padded generation

        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        # target_modules="all-linear" adapts every linear (attn + MLP + conv projections)
        # except the LM head -- robust across architectures, no per-model name lists. peft
        # keeps the adapter weights in fp32 by default (autocast_adapter_dtype), so AdamW
        # updates are precise even though the frozen base is bf16.
        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora)
        if grad_checkpoint:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()

        # plain scalars used by the step
        self.num_generations = num_generations
        self.max_prompt_len = max_prompt_len
        self.max_completion_len = max_completion_len
        self.temperature = temperature
        self.top_p = top_p
        self.beta = beta
        self.scale_rewards = scale_rewards
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_lr_ratio = min_lr_ratio
        self._val_correct: list[float] = []
        self._val_samples: list[tuple] = []

    # -- reward scoring --------------------------------------------------------------------

    def _score(self, texts: list[str], golds: list[str]):
        """Weighted-sum reward per completion + a per-function breakdown (all CPU tensors)."""
        total = torch.zeros(len(texts), dtype=torch.float32)
        per_func: dict[str, torch.Tensor] = {}
        for func, weight in zip(self.task.reward_funcs, self.task.reward_weights):
            vals = torch.tensor(
                [func(t, g) for t, g in zip(texts, golds)], dtype=torch.float32
            )
            per_func[func.__name__] = vals
            total += weight * vals
        return total, per_func

    def _completion_logprobs(self, roll: dict, *, reference: bool = False) -> torch.Tensor:
        """Per-token log-probs of the completion tokens under the current (or reference) policy.

        One forward over ``full_ids`` (prompt+completion); the logit at position ``t-1``
        predicts the token at ``t``, so completion log-probs come from logits
        ``[prompt_len-1 : -1]`` against targets ``[prompt_len:]``. The reference path disables
        the LoRA adapter and runs under ``no_grad`` -- a free reference policy for the KL term.
        """
        full_ids = roll["full_ids"]
        prompt_len = roll["prompt_len"]
        attention_mask = torch.cat([roll["prompt_mask"], roll["completion_mask"]], dim=1)
        adapter_ctx = self.model.disable_adapter() if reference else nullcontext()
        grad_ctx = torch.no_grad() if reference else nullcontext()
        with adapter_ctx, grad_ctx:
            logits = self.model(input_ids=full_ids, attention_mask=attention_mask).logits
        comp_logits = logits[:, prompt_len - 1 : -1, :]
        targets = full_ids[:, prompt_len:]
        return selective_log_softmax(comp_logits, targets)

    # -- training --------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        prompts = [b["prompt"] for b in batch]
        golds = [b["gold"] for b in batch]
        group = self.num_generations

        roll = generate_completions(
            self.model,
            self.tokenizer,
            prompts,
            num_return_sequences=group,
            max_prompt_len=self.max_prompt_len,
            max_new_tokens=self.max_completion_len,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            device=self.device,
        )

        golds_rep = [g for g in golds for _ in range(group)]  # row i -> prompt i // G
        rewards, per_func = self._score(roll["texts"], golds_rep)
        advantages = compute_group_advantages(
            rewards, group, scale_rewards=self.scale_rewards
        ).to(self.device)

        logprobs = self._completion_logprobs(roll)
        # beta > 0 runs a second full forward pass (the KL reference) over the batch,
        # roughly doubling per-step forward compute; beta == 0 skips it entirely.
        ref_logprobs = (
            self._completion_logprobs(roll, reference=True) if self.beta > 0 else None
        )
        loss, kl = grpo_loss(
            logprobs,
            advantages,
            roll["completion_mask"],
            ref_logprobs=ref_logprobs,
            beta=self.beta,
        )

        grouped = rewards.view(-1, group)
        bs = len(prompts)
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        self.log("train/reward", rewards.mean(), prog_bar=True, batch_size=bs)
        self.log("train/reward_std", grouped.std(dim=1).mean(), batch_size=bs)
        self.log(
            "train/frac_reward_zero_std",
            (grouped.std(dim=1) < 1e-6).float().mean(),
            prog_bar=True,
            batch_size=bs,
        )
        self.log(
            "completions/mean_length",
            roll["completion_mask"].sum(1).float().mean(),
            batch_size=bs,
        )
        for name, vals in per_func.items():
            self.log(f"train/reward/{name}", vals.mean(), batch_size=bs)
        if kl is not None:
            self.log("train/kl", kl, batch_size=bs)
        return loss

    # -- validation: greedy pass@1 ---------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_correct = []
        self._val_samples = []

    def validation_step(self, batch, batch_idx):
        prompts = [b["prompt"] for b in batch]
        golds = [b["gold"] for b in batch]
        questions = [b["question"] for b in batch]
        roll = generate_completions(
            self.model,
            self.tokenizer,
            prompts,
            num_return_sequences=1,
            max_prompt_len=self.max_prompt_len,
            max_new_tokens=self.max_completion_len,
            do_sample=False,
            device=self.device,
        )
        for text, gold, question in zip(roll["texts"], golds, questions):
            correct = self.task.correctness(text, gold)
            self._val_correct.append(correct)
            if batch_idx == 0 and len(self._val_samples) < 4:
                self._val_samples.append((question, text, gold, correct))

    def on_validation_epoch_end(self) -> None:
        accuracy = sum(self._val_correct) / max(len(self._val_correct), 1)
        self.log("val/accuracy", accuracy, prog_bar=True)
        if isinstance(self.logger, WandbLogger) and self._val_samples:
            import wandb

            table = wandb.Table(columns=["question", "completion", "gold", "correct"])
            for question, text, gold, correct in self._val_samples:
                table.add_data(question, text, str(gold), correct)
            self.logger.experiment.log({"val/samples": table})

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=self.lr, weight_decay=self.weight_decay
        )
        # Cosine decay to a floor of min_lr_ratio * peak (per-epoch), matching the repo's
        # ViT schedule convention (see projects/text2image/titok/train.py).
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            cosine_with_floor(self.trainer.max_epochs or 1, self.min_lr_ratio),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p, project="grpo-math", epochs=2)
    # GRPO defaults: a handful of prompts x G rollouts per step; tiny LR; eager (compile and
    # EMA fight generation / cost a full-model copy and are off by default for this run type).
    p.set_defaults(
        batch_size=4,
        lr=1e-5,
        num_workers=2,
        compile_mode="off",
        ema_decay=0.0,
    )
    p.add_argument("--task", default="gsm8k", help="registered task name (see tasks.py)")
    p.add_argument("--model-name", default="LiquidAI/LFM2.5-230M")
    p.add_argument(
        "--num-generations", type=int, default=8, help="G: completions sampled per prompt"
    )
    p.add_argument("--max-prompt-len", type=int, default=256)
    p.add_argument("--max-completion-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL-to-reference coefficient (0 disables the KL term and the reference forward)",
    )
    p.add_argument(
        "--scale-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="divide advantages by the group std (on by default; --no-scale-rewards disables)",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="enable gradient checkpointing (cuts activation memory; slows generation)",
    )
    p.add_argument("--val-size", type=int, default=200, help="held-out prompts for pass@1")
    p.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="cap on training prompts (for smoke runs); default uses all",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop if val/accuracy hasn't improved for this many epochs (0 disables)",
    )
    args = p.parse_args()

    seed_everything(args.seed, workers=True)

    resume_ckpt = find_ckpt(args.resume, args.project, OUTPUTS) if args.resume else None
    if resume_ckpt:
        # Rebuild model + adapter from the checkpoint's saved hyperparameters (model_name,
        # task, LoRA/gen config), not the current CLI flags, so resume reconstructs the run.
        module = LitGRPO.load_from_checkpoint(resume_ckpt, lr=args.lr)
    else:
        module = LitGRPO(
            model_name=args.model_name,
            task_name=args.task,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_generations=args.num_generations,
            max_prompt_len=args.max_prompt_len,
            max_completion_len=args.max_completion_len,
            temperature=args.temperature,
            top_p=args.top_p,
            beta=args.beta,
            scale_rewards=args.scale_rewards,
            lr=args.lr,
            weight_decay=args.weight_decay,
            min_lr_ratio=args.min_lr_ratio,
            grad_checkpoint=args.grad_checkpoint,
        )

    datamodule = PromptDataModule(
        module.task,
        module.tokenizer,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        train_size=args.train_size,
    )

    hp = module.hparams
    config = {
        "model": {"name": hp["model_name"], "task": hp["task_name"]},
        "lora": {
            "r": hp["lora_r"],
            "alpha": hp["lora_alpha"],
            "dropout": hp["lora_dropout"],
        },
        "grpo": {
            "num_generations": hp["num_generations"],
            "beta": hp["beta"],
            "scale_rewards": hp["scale_rewards"],
            "temperature": hp["temperature"],
            "top_p": hp["top_p"],
            "max_prompt_len": hp["max_prompt_len"],
            "max_completion_len": hp["max_completion_len"],
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": module.lr,
            "weight_decay": module.weight_decay,
            "min_lr_ratio": module.min_lr_ratio,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "precision": "bf16-mixed",
        },
        "data": {
            "task": args.task,
            "data_dir": args.data_dir,
            "val_size": args.val_size,
            "train_size": args.train_size,
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
        test=False,  # GSM8K test set is graded by the standalone eval.py
        monitor="val/accuracy",
        monitor_mode="max",
        early_stop_patience=args.early_stop_patience,
    )


if __name__ == "__main__":
    main()

"""Standalone pass@1 evaluation on a task's held-out test set (greedy decoding).

Use it to measure the model before and after GRPO: run once with no ``--resume`` for the
base-model baseline, then with ``--resume <run_id>`` to score the trained adapter on the
exact same test set. For GSM8K this is the official 1319-problem ``test`` split, which the
trainer never sees (it validates on a slice carved from ``train``).

Examples
--------
    # base-model baseline on the full GSM8K test set
    uv run python projects/grpo/eval.py --task gsm8k

    # a trained run's checkpoint
    uv run python projects/grpo/eval.py --task gsm8k --resume <run_id>

    # quick check on the first 100 problems
    uv run python projects/grpo/eval.py --task gsm8k --limit 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from chimera.grpo import generate_completions
from chimera.utils.experiment import find_ckpt
from train import OUTPUTS, LitGRPO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="gsm8k", help="registered task name (see tasks.py)")
    p.add_argument("--model-name", default="LiquidAI/LFM2.5-230M")
    p.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="evaluate this run's checkpoint instead of the base model",
    )
    p.add_argument(
        "--adapter",
        metavar="DIR",
        default=None,
        help="evaluate a saved LoRA adapter directory (e.g. from the scratchpad run harness)",
    )
    p.add_argument("--project", default="grpo-math")
    p.add_argument("--data-dir", default="/mnt/ai/data")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-prompt-len", type=int, default=256)
    p.add_argument("--max-completion-len", type=int, default=256)
    p.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N test problems"
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.resume:
        ckpt = find_ckpt(args.resume, args.project, OUTPUTS)
        module = LitGRPO.load_from_checkpoint(ckpt, map_location=device)
        label = f"run {args.resume}"
    elif args.adapter:
        # Build the base + a fresh adapter, then load the trained adapter weights over it.
        module = LitGRPO(model_name=args.model_name, task_name=args.task)
        module.model.load_adapter(args.adapter, adapter_name="trained")
        module.model.set_adapter("trained")
        label = f"adapter {args.adapter}"
    else:
        # A freshly-wrapped LoRA model has a zero-init adapter (identity), so this is the
        # untrained base-model baseline.
        module = LitGRPO(model_name=args.model_name, task_name=args.task)
        label = f"base model {args.model_name}"
    module.eval().to(device)
    tokenizer, task = module.tokenizer, module.task

    test_ds = task.load_test(args.data_dir)
    if args.limit is not None:
        test_ds = test_ds.select(range(min(args.limit, len(test_ds))))

    def render(ex: dict) -> dict:
        return {
            "prompt": tokenizer.apply_chat_template(
                task.build_prompt(ex), tokenize=False, add_generation_prompt=True
            ),
            "gold": task.gold_of(ex),
        }

    n_test = len(test_ds)
    correct = 0.0
    for start in tqdm(range(0, n_test, args.batch_size), desc=f"eval {args.task}"):
        # Render this batch's prompts lazily instead of materializing the whole test set.
        chunk = [render(test_ds[i]) for i in range(start, min(start + args.batch_size, n_test))]
        roll = generate_completions(
            module.model,
            tokenizer,
            [r["prompt"] for r in chunk],
            num_return_sequences=1,
            max_prompt_len=args.max_prompt_len,
            max_new_tokens=args.max_completion_len,
            do_sample=False,
            device=device,
        )
        for text, row in zip(roll["texts"], chunk):
            correct += task.correctness(text, row["gold"])

    accuracy = correct / max(n_test, 1)
    # correctness() may award partial credit, so print the count without a lossy int() cast
    # unless it is a whole number (the common binary-reward case).
    correct_str = str(int(correct)) if correct == int(correct) else f"{correct:.2f}"
    print(
        f"\n{task.name} pass@1 ({label}): {accuracy:.4f}  "
        f"({correct_str}/{n_test} on {'first ' + str(args.limit) if args.limit else 'full'} test set)"
    )


if __name__ == "__main__":
    main()

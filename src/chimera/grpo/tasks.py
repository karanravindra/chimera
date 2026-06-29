"""Task registry -- the extensibility seam for "start with math, add more later".

A :class:`Task` bundles everything GRPO needs to train and grade one verifiable objective:
how to load its train/val splits, how to turn an example into a chat prompt, how to read the
gold answer, and which reward functions to apply. The trainer and eval script are written
against this interface and select a task by name, so adding a new verifiable task (e.g.
competition MATH, a code-exec task, a logic puzzle) is a single new entry in :data:`TASKS`
plus, if needed, new reward primitives in :mod:`rewards` -- no trainer changes.

The only task implemented today is **gsm8k** (grade-school math word problems).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from datasets import Dataset, concatenate_datasets, load_dataset

from chimera.grpo.rewards import correctness_reward, countdown_reward

# A reward function scores one completion against its gold answer.
RewardFunc = Callable[[str, str], float]

# Instruction shared by math tasks: think, then emit the gradeable final-answer line.
MATH_SYSTEM_PROMPT = (
    "You are a careful math assistant. Reason step by step, then give the final answer "
    "on its own line in the exact form '#### <number>' with no extra text after it."
)


@dataclass
class Task:
    """A verifiable GRPO task.

    Attributes:
        name: registry key (e.g. ``"gsm8k"``).
        load_splits: ``(data_dir, val_size) -> (train_ds, val_ds)``; ``val_ds`` is a held-out
            slice of ``val_size`` rows used for periodic pass@1 during training.
        load_test: ``data_dir -> test_ds`` for the standalone :mod:`eval` script.
        build_prompt: ``example -> list[chat message]`` (a system + user turn) to be fed
            through the tokenizer's chat template.
        gold_of: ``example -> gold answer string`` (passed to the reward funcs).
        reward_funcs / reward_weights: parallel lists; total reward is the weighted sum.
        correctness: the single func used to score val/test pass@1 (subset of reward_funcs).
        question_key: dataset column shown to the model (for logging / prompt building).
    """

    name: str
    load_splits: Callable[[str, int], tuple[Dataset, Dataset]]
    load_test: Callable[[str], Dataset]
    build_prompt: Callable[[dict], list[dict]]
    gold_of: Callable[[dict], str]
    reward_funcs: list[RewardFunc]
    reward_weights: list[float]
    correctness: RewardFunc
    question_key: str = "question"


# --- gsm8k --------------------------------------------------------------------------------


def _gsm8k_splits(data_dir: str, val_size: int = 200) -> tuple[Dataset, Dataset]:
    """Load GSM8K ``main/train`` and carve off a deterministic ``val_size`` slice for pass@1.

    GSM8K ships only train/test, so we hold out the first ``val_size`` train rows as the
    in-loop validation set and train on the rest. The official ``test`` split is reserved for
    the standalone eval (see :func:`_gsm8k_test`) so it never leaks into training signal.
    """
    full = load_dataset("openai/gsm8k", "main", split="train")
    val = full.select(range(val_size))
    train = full.select(range(val_size, len(full)))
    return train, val


def _gsm8k_test(data_dir: str) -> Dataset:
    return load_dataset("openai/gsm8k", "main", split="test")


def _gsm8k_prompt(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]


def _gsm8k_gold(example: dict) -> str:
    # The gold answer string ("... #### 72"); reward funcs parse the number out of it.
    return example["answer"]


GSM8K = Task(
    name="gsm8k",
    load_splits=_gsm8k_splits,
    load_test=_gsm8k_test,
    build_prompt=_gsm8k_prompt,
    gold_of=_gsm8k_gold,
    # Correctness only. An additive format bonus was dropped: a bare "#### N" trivially
    # earns it, so under sampling GRPO collapsed to degenerate 4-token outputs that grabbed
    # the bonus while abandoning reasoning (pass@1 0.141 -> 0.031). Correctness is the true,
    # non-exploitable objective; the system prompt alone already yields ~100% format rate.
    reward_funcs=[correctness_reward],
    reward_weights=[1.0],
    correctness=correctness_reward,
)


# --- orca-math ----------------------------------------------------------------------------
# microsoft/orca-math-word-problems-200k: 200k synthetic grade/early-competition word
# problems with free-form `answer` text that (99.7% of rows) ends on the numeric result, so
# the same last-number extractor + numeric correctness reward grade it. More volume and a bit
# more variety/difficulty than GSM8K's 7.5k. We always evaluate on GSM8K (below), so orca only
# ever enters the *training* pool -- it widens the data without changing the yardstick.


def _orca_full(data_dir: str) -> Dataset:
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    # Match GSM8K's column layout so the two concatenate cleanly in the mix task.
    return ds.select_columns(["question", "answer"])


def _orca_splits(data_dir: str, val_size: int = 200) -> tuple[Dataset, Dataset]:
    """Train on orca-math, but validate on held-out GSM8K for cross-run comparability."""
    _, val = _gsm8k_splits(data_dir, val_size)
    return _orca_full(data_dir), val


ORCA_MATH = Task(
    name="orcamath",
    load_splits=_orca_splits,
    load_test=_gsm8k_test,  # report on GSM8K test, same yardstick as everything else
    build_prompt=_gsm8k_prompt,
    gold_of=_gsm8k_gold,
    reward_funcs=[correctness_reward],
    reward_weights=[1.0],
    correctness=correctness_reward,
)


# --- mathmix (gsm8k + orca-math) ----------------------------------------------------------


def _mathmix_splits(data_dir: str, val_size: int = 200) -> tuple[Dataset, Dataset]:
    """GSM8K train (minus the val slice) concatenated with all of orca-math; GSM8K val.

    Both datasets share ``question``/``answer`` columns and the same numeric reward, so the
    pool is a drop-in superset. The trainer shuffles and (optionally caps via ``train_size``)
    this combined pool, so each step mixes GSM8K and orca prompts.
    """
    gsm_train, val = _gsm8k_splits(data_dir, val_size)
    orca = _orca_full(data_dir)
    mixed = concatenate_datasets([gsm_train, orca]).shuffle(seed=0)
    return mixed, val


MATH_MIX = Task(
    name="mathmix",
    load_splits=_mathmix_splits,
    load_test=_gsm8k_test,
    build_prompt=_gsm8k_prompt,
    gold_of=_gsm8k_gold,
    reward_funcs=[correctness_reward],
    reward_weights=[1.0],
    correctness=correctness_reward,
)


# --- dapo-math (harder integer-answer math) -----------------------------------------------
# open-r1/DAPO-Math-17k-Processed: competition-style problems with a bare answer string in
# `solution`. Harder than GSM8K, so a larger fraction of groups land at intermediate pass
# rates (more GRPO signal, less small-integer guessing). We keep only rows whose gold parses
# as a number so the existing numeric reward grades them; eval stays on GSM8K.

from chimera.grpo.rewards import extract_final_answer as _extract  # noqa: E402


def _dapo_splits(data_dir: str, val_size: int = 200) -> tuple[Dataset, Dataset]:
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    ds = ds.filter(lambda r: _extract(r["solution"]) is not None)
    _, val = _gsm8k_splits(data_dir, val_size)
    return ds, val


def _dapo_prompt(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
    ]


DAPO_MATH = Task(
    name="dapomath",
    load_splits=_dapo_splits,
    load_test=_gsm8k_test,
    build_prompt=_dapo_prompt,
    gold_of=lambda ex: ex["solution"],
    reward_funcs=[correctness_reward],
    reward_weights=[1.0],
    correctness=correctness_reward,
    question_key="prompt",
)


# --- countdown (generate-and-check arithmetic; a non-math verifiable domain) ---------------
# Jiayi-Pan/Countdown-Tasks-3to4: reach `target` using each of `nums` exactly once with
# + - * / and parentheses. The reward (countdown_reward) is generate-and-check: no stored
# answer, ~0 luck floor -- the cleanest unhackable objective for a tiny model, and proof the
# pipeline generalizes beyond numeric-answer math. Evaluated on its own held-out split.

COUNTDOWN_SYSTEM_PROMPT = (
    "You are solving a Countdown puzzle. Using each of the given numbers exactly once and the "
    "operations + - * / and parentheses, construct an arithmetic expression equal to the "
    "target. Reason step by step, then give your expression on its own final line in the exact "
    "form 'Answer: <expression>' (for example 'Answer: (3 + 1) * 6')."
)


def _countdown_splits(data_dir: str, val_size: int = 200) -> tuple[Dataset, Dataset]:
    full = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    val = full.select(range(val_size))
    train = full.select(range(val_size, len(full)))
    return train, val


def _countdown_test(data_dir: str) -> Dataset:
    # No official test split; reuse the held-out 200-row val slice as the standalone test set.
    return _countdown_splits(data_dir, 200)[1]


def _countdown_prompt(example: dict) -> list[dict]:
    nums = ", ".join(str(n) for n in example["nums"])
    return [
        {"role": "system", "content": COUNTDOWN_SYSTEM_PROMPT},
        {"role": "user", "content": f"Numbers: {nums}\nTarget: {example['target']}"},
    ]


def _countdown_gold(example: dict) -> str:
    # Encode both target and the available numbers for the generate-and-check reward.
    return f"{example['target']}|{','.join(str(n) for n in example['nums'])}"


COUNTDOWN = Task(
    name="countdown",
    load_splits=_countdown_splits,
    load_test=_countdown_test,
    build_prompt=_countdown_prompt,
    gold_of=_countdown_gold,
    reward_funcs=[countdown_reward],
    reward_weights=[1.0],
    correctness=countdown_reward,
    question_key="target",
)


TASKS: dict[str, Task] = {
    t.name: t for t in (GSM8K, ORCA_MATH, MATH_MIX, DAPO_MATH, COUNTDOWN)
}


def get_task(name: str) -> Task:
    """Look up a registered task by name, with a helpful error listing the options."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]

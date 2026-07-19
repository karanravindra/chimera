"""Zero-shot benchmark runner: cached lm-eval over the standard task set + results table.

All default tasks are loglikelihood-ranking tasks (score candidate continuations, pick
the highest logprob), so models are only exposed through lm-eval's loglikelihood
interface (see ChimeraLM) — no generation needed.
"""

import hashlib
import os
import pickle
from pathlib import Path

import pandas as pd
import torch

from .lm_harness import ChimeraLM  # noqa: F401  (env setup + re-export for callers)

import lm_eval  # noqa: E402  (lm_harness sets HF_HOME/LM_HARNESS_CACHE_PATH first)
from lm_eval.tasks import TaskManager  # noqa: E402

TASKS = ["blimp", "lambada_openai", "piqa", "sciq", "arc_easy"]

# Random-chance baselines (%) and the GPT-2 small (124M) reference row.
CHANCE = {
    "blimp": 50.0,
    "lambada_openai": 0.0,
    "piqa": 50.0,
    "sciq": 25.0,
    "arc_easy": 25.0,
}
GPT2_SMALL = {
    "blimp": 82.29,
    "lambada_openai": 32.16,
    "piqa": 62.62,
    "sciq": 64.40,
    "arc_easy": 39.52,
}
METRIC_PREF = ["acc_norm", "acc", "perplexity"]  # one headline metric per task

# Loading the 71 task datasets (blimp = 67 subtasks) costs ~35s and is what
# simple_evaluate repeats on EVERY call. evaluate() takes a prebuilt task_dict, so we
# load once and keep it in this process-level dict — later runs (even after retraining)
# reuse it and skip straight to scoring.
_LOADED_TASKS: dict = {}


def get_loaded_tasks(tasks):
    key = tuple(tasks)
    if key not in _LOADED_TASKS:
        loaded = TaskManager().load(list(tasks))
        for _, task_obj in loaded["tasks"].items():
            if task_obj.get_config("num_fewshot") is None:
                task_obj.set_config(key="num_fewshot", value=0)  # 0-shot
        _LOADED_TASKS[key] = loaded
    return _LOADED_TASKS[key]


def model_fingerprint(model) -> str:
    """Content hash of the model weights, so the results cache invalidates iff the
    weights change (retraining -> re-score; re-running the same model -> instant)."""
    h = hashlib.md5()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def run_eval(model, lm, tasks=TASKS, force_rerun=False):
    """Cached lm-eval. Three independent caches, each keyed on what actually changes it:
      * results dict   -> model weights   (same model -> instant, skip everything below)
      * loaded tasks   -> task set        (in-process; skips the ~35s dataset reload)
      * tokenized reqs -> tokenizer       (in the adapter; skips re-encoding every run)
    So after a retrain only the ~19s scoring + bootstrap actually re-run.
    ``force_rerun=True`` ignores the results cache and re-scores."""
    cache_dir = Path(os.environ["LM_HARNESS_CACHE_PATH"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{model_fingerprint(model)}|{lm.tokenizer_name}|{','.join(sorted(tasks))}|0shot"
    cache_file = cache_dir / f"results-{hashlib.md5(key.encode()).hexdigest()}.pkl"
    if cache_file.exists() and not force_rerun:
        print(f"[eval] loaded cached results from {cache_file.name} (same weights)")
        return pickle.loads(cache_file.read_bytes())
    task_dict = get_loaded_tasks(tasks)  # in-process; loads datasets once per session
    out = lm_eval.evaluate(lm=lm, task_dict=task_dict, cache_requests=True)
    results = out["results"]
    cache_file.write_bytes(pickle.dumps(results))
    print(f"[eval] scored + cached results -> {cache_file.name}")
    return results


def headline(task_metrics):
    """Pick the preferred headline metric for a task -> (metric_name, value, stderr)."""
    flat = {}
    for key, val in task_metrics.items():
        name = key.split(",")[0]
        if name in ("alias",) or "_stderr" in key or not isinstance(val, (int, float)):
            continue
        flat.setdefault(name, (val, task_metrics.get(f"{name}_stderr,none")))
    for name in METRIC_PREF:
        if name in flat:
            return name, flat[name][0], flat[name][1]
    name = next(iter(flat))
    return name, flat[name][0], flat[name][1]


def results_table(results, tasks=TASKS):
    """Results as a pandas Styler: headline metric per task, chance + GPT-2-small
    reference columns, the better of {this model, GPT-2} bolded per row."""
    rows = []
    for task in tasks:
        name, val, stderr = headline(results[task])
        is_ppl = "perplex" in name or "bits_per_byte" in name
        rows.append(
            {
                "task": task,
                "metric": name,
                "this model": val if is_ppl else val * 100,
                "stderr": (stderr if stderr is None else (stderr if is_ppl else stderr * 100)),
                "chance": CHANCE.get(task),
                "GPT-2 small (124M)": GPT2_SMALL.get(task),
            }
        )

    df = pd.DataFrame(rows).set_index("task")

    def _bold_best(row):
        # Bold the better of {this model, GPT-2}: higher is better for acc metrics,
        # lower for perplexity. (Skips lambada perplexity rows — acc is the headline.)
        cols = ["this model", "GPT-2 small (124M)"]
        vals = {c: row[c] for c in cols if pd.notna(row[c])}
        if len(vals) < 2:
            return ["" for _ in row.index]
        best = min(vals, key=vals.get) if "perplex" in row["metric"] else max(vals, key=vals.get)
        return ["font-weight: bold" if c == best else "" for c in row.index]

    return df.style.apply(_bold_best, axis=1).format(
        {
            "this model": "{:.2f}",
            "stderr": "{:.2f}",
            "chance": "{:.1f}",
            "GPT-2 small (124M)": "{:.2f}",
        },
        na_rep="-",
    )

#!/usr/bin/env python3
"""Pull scalar metrics from a local wandb run.

By default reads the most recent run (via the `wandb/latest-run` symlink) and
prints its final scalar metrics from `files/wandb-summary.json`. The summary
file also contains huge per-layer gradient/parameter histograms (dict-valued
entries); those are filtered out so only scalars remain.

No network needed. The default (summary) path is stdlib-only -- the summary is
plain JSON. The --history path reads the run's `.wandb` datastore locally and
needs the wandb package (use `.venv/bin/python` or `uv run`); it still does NOT
hit the network.

Usage:
  python pull_metrics.py                 # latest run, final scalar metrics
  python pull_metrics.py --list [N]      # list N most recent runs (default 10)
  python pull_metrics.py RUN             # a run id (lrs7fuic) or run dir path
  python pull_metrics.py --json          # machine-readable JSON to stdout
  python pull_metrics.py --history        # all time-series: n/first/min/max/last
  python pull_metrics.py --history KEY     # full time-series for one key

Run from the repo root, or pass --wandb-dir to point elsewhere.
"""

import argparse
import glob
import json
import os
import sys


def find_wandb_dir(explicit):
    if explicit:
        return os.path.abspath(explicit)
    # walk up from cwd looking for a `wandb/` dir
    d = os.getcwd()
    while True:
        cand = os.path.join(d, "wandb")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    sys.exit("error: no wandb/ directory found (cd to repo root or pass --wandb-dir)")


def all_run_dirs(wandb_dir):
    """Run dirs newest-first. Matches run-* and offline-run-*."""
    runs = glob.glob(os.path.join(wandb_dir, "run-*")) + glob.glob(
        os.path.join(wandb_dir, "offline-run-*")
    )
    runs = [r for r in runs if os.path.isdir(r) and not os.path.islink(r)]
    runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return runs


def resolve_run(wandb_dir, run):
    if run is None:
        link = os.path.join(wandb_dir, "latest-run")
        if os.path.exists(link):
            return os.path.realpath(link)
        runs = all_run_dirs(wandb_dir)
        if not runs:
            sys.exit(f"error: no runs found in {wandb_dir}")
        return runs[0]
    # explicit: a path, a dir name, or a bare run id suffix
    if os.path.isdir(run):
        return os.path.abspath(run)
    for r in all_run_dirs(wandb_dir):
        if os.path.basename(r) == run or r.endswith("-" + run):
            return r
    sys.exit(f"error: run {run!r} not found in {wandb_dir}")


def load_summary(run_dir):
    path = os.path.join(run_dir, "files", "wandb-summary.json")
    if not os.path.exists(path):
        sys.exit(f"error: no summary at {path} (run may still be initializing)")
    with open(path) as f:
        return json.load(f)


def scalar_metrics(summary):
    """Keep only scalar values; drop histograms (dicts) and internal keys we
    surface separately."""
    out = {}
    for k, v in summary.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if k.startswith(("gradients/", "parameters/", "_")):
            continue
        out[k] = v
    return out


def run_meta(run_dir):
    meta = {}
    p = os.path.join(run_dir, "files", "wandb-metadata.json")
    if os.path.exists(p):
        with open(p) as f:
            meta = json.load(f)
    rid = os.path.basename(run_dir).split("-")[-1]
    return {
        "run_id": rid,
        "run_dir": run_dir,
        "started": meta.get("startedAt"),
        "host": meta.get("host"),
        "program": meta.get("codePathLocal") or meta.get("program"),
    }


def print_table(meta, summary):
    s = scalar_metrics(summary)
    print(f"run:     {meta['run_id']}  ({os.path.basename(meta['run_dir'])})")
    if meta.get("program"):
        print(f"program: {meta['program']}")
    if meta.get("started"):
        print(f"started: {meta['started']}  host: {meta.get('host')}")
    rt = summary.get("_runtime")
    if rt is not None:
        print(f"runtime: {rt:.0f}s ({rt/60:.1f} min)   step: {summary.get('_step')}"
              f"   global_step: {summary.get('trainer/global_step')}"
              f"   epoch: {summary.get('epoch')}")
    print()
    width = max((len(k) for k in s), default=10)
    for k in sorted(s):
        print(f"  {k:<{width}}  {s[k]}")
    if not s:
        print("  (no scalar metrics in summary)")


def hist_stats(hist):
    """Reconstruct magnitude stats from a wandb histogram dict
    ({'bins': edges, 'values': counts}). Returns abs-mean, std, mean,
    and the nonzero value range."""
    bins, vals = hist.get("bins"), hist.get("values")
    if not bins or not vals:
        return None
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(vals))]
    n = sum(vals)
    if n == 0:
        return None
    mean = sum(c * w for c, w in zip(centers, vals)) / n
    absmean = sum(abs(c) * w for c, w in zip(centers, vals)) / n
    var = sum(w * (c - mean) ** 2 for c, w in zip(centers, vals)) / n
    # range of bins that actually hold mass
    nz = [i for i, w in enumerate(vals) if w > 0]
    lo, hi = bins[nz[0]], bins[nz[-1] + 1]
    return {"absmean": absmean, "std": var ** 0.5, "mean": mean,
            "min": lo, "max": hi, "n": int(n)}


def load_hists(run_dir, prefix):
    """Return {tensor_name: {'bins','values'}} for a prefix ('gradients/' or
    'parameters/'). Prefers the run summary (clean nested-dict histograms);
    falls back to the live history datastore, where each histogram is split
    into flattened `<tensor>.bins/.values/._type` items (last step wins).
    Returns (hists, source)."""
    # 1) summary: nested dicts — works for finished runs, no wandb needed
    spath = os.path.join(run_dir, "files", "wandb-summary.json")
    if os.path.exists(spath):
        with open(spath) as f:
            summary = json.load(f)
        hists = {k[len(prefix):]: v for k, v in summary.items()
                 if k.startswith(prefix) and isinstance(v, dict) and "bins" in v}
        if hists:
            return hists, "summary"
    # 2) history datastore: needs wandb, reconstruct flattened histograms
    try:
        from wandb.sdk.internal import datastore
        from wandb.proto import wandb_internal_pb2 as pb
    except Exception as e:
        sys.exit(f"error: no {prefix.rstrip('/')} in summary and reading the "
                 f"live history needs the wandb package ({e}); use .venv/bin/python")
    wf = glob.glob(os.path.join(run_dir, "run-*.wandb"))
    if not wf:
        sys.exit(f"error: no summary and no .wandb datastore in {run_dir}")
    ds = datastore.DataStore()
    ds.open_for_scan(wf[0])
    parts = {}  # tensor -> {'bins':..., 'values':...}, last step wins
    while True:
        raw = ds.scan_data()
        if raw is None:
            break
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        for it in rec.history.item:
            k = it.key or ".".join(it.nested_key)
            if not k.startswith(prefix):
                continue
            for suf in (".bins", ".values"):
                if k.endswith(suf):
                    name = k[len(prefix):-len(suf)]
                    try:
                        parts.setdefault(name, {})[suf[1:]] = json.loads(it.value_json)
                    except Exception:
                        pass
    hists = {n: d for n, d in parts.items() if "bins" in d and "values" in d}
    return hists, "history (live run)"


def print_dist(run_dir, prefix, topn, as_json):
    """Per-tensor magnitude report for gradients/ or parameters/ histograms.
    Flags vanishing / exploding / dead tensors."""
    hists, source = load_hists(run_dir, prefix)
    rows = []
    for name, h in hists.items():
        st = hist_stats(h)
        if st:
            rows.append((name, st))
    if not rows:
        sys.exit(f"error: no {prefix.rstrip('/')} histograms found. The run must "
                 f"call wandb.watch(..., log='all'|'gradients'|'parameters') and "
                 f"have logged >=1 step.")
    rows.sort(key=lambda r: r[1]["absmean"], reverse=True)

    if as_json:
        print(json.dumps({k: st for k, st in rows}, indent=2))
        return

    kind = prefix.rstrip("/")
    absmeans = [st["absmean"] for _, st in rows]
    gmax = max(st["max"] for _, st in rows)
    gmin = min(st["min"] for _, st in rows)
    print(f"{kind} distribution for run {os.path.basename(run_dir).split('-')[-1]}"
          f"  ({len(rows)} tensors, source: {source})")
    print(f"  per-tensor |{kind[:-1] if kind.endswith('s') else kind}| mean:"
          f" max={max(absmeans):.3g}  min={min(absmeans):.3g}  "
          f"median={sorted(absmeans)[len(absmeans)//2]:.3g}")
    print(f"  value range across all tensors: [{gmin:.3g}, {gmax:.3g}]")

    # health flags
    dead = [k for k, st in rows if st["absmean"] == 0 or st["max"] == st["min"] == 0]
    vanish = [k for k, st in rows if 0 < st["absmean"] < 1e-6]
    explode = [k for k, st in rows if st["absmean"] > 1.0 or abs(st["max"]) > 10
               or abs(st["min"]) > 10]
    nanrisk = [k for k, st in rows
               if st["std"] != st["std"] or st["absmean"] != st["absmean"]]
    print("  flags:")
    print(f"    dead (all-zero):      {len(dead)}"
          + (f"  e.g. {dead[:3]}" if dead else ""))
    print(f"    vanishing (<1e-6):    {len(vanish)}"
          + (f"  e.g. {vanish[:3]}" if vanish else ""))
    print(f"    exploding (>1 or |v|>10): {len(explode)}"
          + (f"  e.g. {explode[:3]}" if explode else ""))
    if nanrisk:
        print(f"    NaN/Inf:              {len(nanrisk)}  e.g. {nanrisk[:3]}")

    def show(title, items):
        print(f"\n  {title}")
        w = max(len(k) for k, _ in items)
        print(f"    {'tensor':<{w}}  {'absmean':>10}  {'std':>10}  "
              f"{'min':>10}  {'max':>10}")
        for k, st in items:
            print(f"    {k:<{w}}  {st['absmean']:>10.3g}  {st['std']:>10.3g}  "
                  f"{st['min']:>10.3g}  {st['max']:>10.3g}")

    show(f"top {topn} by |{kind[:-1]}|:", rows[:topn])
    show(f"bottom {topn} by |{kind[:-1]}|:", rows[-topn:])


def read_history(run_dir):
    """Read all scalar time-series from the local `.wandb` datastore.
    Returns {key: [values...]}. Needs the wandb package (no network)."""
    try:
        from wandb.sdk.internal import datastore
        from wandb.proto import wandb_internal_pb2 as pb
    except Exception as e:
        sys.exit(f"error: --history needs the wandb package ({e}); "
                 f"use .venv/bin/python or `uv run`")
    wf = glob.glob(os.path.join(run_dir, "run-*.wandb"))
    if not wf:
        sys.exit(f"error: no .wandb datastore in {run_dir}")
    ds = datastore.DataStore()
    ds.open_for_scan(wf[0])
    series = {}
    while True:
        raw = ds.scan_data()
        if raw is None:
            break
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        for item in rec.history.item:
            k = item.key or ".".join(item.nested_key)
            if k.startswith(("gradients/", "parameters/", "_")):
                continue
            try:
                v = json.loads(item.value_json)
            except Exception:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                series.setdefault(k, []).append(v)
    return series


def print_history(run_dir, key, as_json):
    series = read_history(run_dir)
    if key:  # one key: dump the full series
        if key not in series:
            sys.exit(f"error: key {key!r} not in history. Available:\n  "
                     + "\n  ".join(sorted(series)))
        vals = series[key]
        if as_json:
            print(json.dumps({key: vals}))
        else:
            print(f"{key}  (n={len(vals)})")
            for i, v in enumerate(vals):
                print(f"  {i:4d}  {v}")
        return
    # no key: per-series summary
    if as_json:
        print(json.dumps({k: {"n": len(v), "first": v[0], "min": min(v),
                              "max": max(v), "last": v[-1]}
                          for k, v in series.items()}, indent=2))
        return
    width = max((len(k) for k in series), default=10)
    print(f"  {'metric':<{width}}  {'n':>5}  {'first':>12}  {'min':>12}  "
          f"{'max':>12}  {'last':>12}")
    for k in sorted(series):
        v = series[k]
        print(f"  {k:<{width}}  {len(v):>5}  {v[0]:>12.5g}  {min(v):>12.5g}  "
              f"{max(v):>12.5g}  {v[-1]:>12.5g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="run id or run dir (default: latest)")
    ap.add_argument("--wandb-dir", help="path to the wandb/ dir")
    ap.add_argument("--list", nargs="?", const=10, type=int, metavar="N",
                    help="list N most recent runs and exit")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--history", nargs="?", const="", metavar="KEY",
                    help="(needs wandb) time-series summary, or full series for KEY")
    ap.add_argument("--gradients", "--grad", nargs="?", const=10, type=int,
                    metavar="N", dest="gradients",
                    help="per-tensor gradient magnitude report + health flags "
                         "(top/bottom N, default 10)")
    ap.add_argument("--params", nargs="?", const=10, type=int, metavar="N",
                    help="same report for parameter (weight) distributions")
    args = ap.parse_args()

    wandb_dir = find_wandb_dir(args.wandb_dir)

    if args.list is not None:
        for r in all_run_dirs(wandb_dir)[: args.list]:
            m = run_meta(r)
            try:
                step = load_summary(r).get("_step")
            except SystemExit:
                step = "?"
            print(f"{m['run_id']:>10}  {os.path.basename(r):<34}  step={step}")
        return

    run_dir = resolve_run(wandb_dir, args.run)

    if args.history is not None:
        print_history(run_dir, args.history, args.json)
        return

    if args.gradients is not None:
        print_dist(run_dir, "gradients/", args.gradients, args.json)
        return

    if args.params is not None:
        print_dist(run_dir, "parameters/", args.params, args.json)
        return

    summary = load_summary(run_dir)
    meta = run_meta(run_dir)

    if args.json:
        print(json.dumps({"meta": meta, "metrics": scalar_metrics(summary)}, indent=2))
    else:
        print_table(meta, summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Round-6 run-level analysis: one row per (training run, schedule, sampler) cell, plus every
pre-registered run-level test.

Inputs are raw per-episode logs under the r6_* raw roots plus the round-5 `training_variants`
root (for the runs that already existed). Outputs:
  results/r6_runlevel_summary.csv  — one row per training run and schedule
  results/r6_runlevel_tests.csv    — every pre-registered test with n, p, difference, interval

Derived policy seconds per episode = mean policy calls x round-5 clean ms/call (labelled
derived); no new timing benchmark is taken in this brief.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from adaptive_replan import runlevel as S  # noqa: E402
from adaptive_replan import mcnemar_exact, wilson_interval  # noqa: E402

MS_PER_CALL = {("ddpm", 100): {"B0": 145.0, "V1": 142.8, "co": 145.0},
               ("ddpm", 20): {"B0": 28.7, "V1": 28.7, "co": 28.7},
               ("ddim", 5): {"B0": 7.2, "V1": 7.2, "co": 7.2}}

RAW_BASE = ROOT / "results" / "raw"
ROOT_NAMES = ("training_variants", "r6_train_pushcube", "r6_adaptive_pushcube",
              "r6_combo_pushcube", "r6_norm_other_tasks", "r6_cotrain_compare",
              "r6_adaptive_other_tasks", "r7_frontier_pick", "r7_frontier_pull",
              "r7_variants_pick", "r7_variants_pull", "r7_combo_pick", "r7_combo_pull",
              "r7_cotrain_runs", "r7_batch96", "r7_v1b_pick")
ROOTS = {name: (RAW_BASE / ("training_variants" if name == "r5_training_variants" else name))
         for name in ("r5_training_variants",) + ROOT_NAMES[1:]}


def schedule_key(rec):
    p = rec.get("schedule_params") or {}
    ps = ",".join(f"{k}={p[k]}" for k in sorted(p))
    return f"{rec['schedule']}({ps})" if ps else rec["schedule"]


def cell_meta(phase_dir: Path):
    metas = sorted(phase_dir.rglob("chunk_*.meta.json"))
    if not metas:
        return None
    m = json.loads(metas[0].read_text())
    raw = m.get("diffusion_steps", 100)
    try:
        steps = int(raw)
    except (TypeError, ValueError):
        steps = 100
    return {"sampler": m.get("sampler", "ddpm"), "steps": steps}


def parse_identity(phase: str):
    """Extract (arm, seed) from a phase name across round-5/6/7 roots."""
    import re
    m = re.search(r"(B0|V1|V2a|V2b|V3|B96)s(\d+)", phase)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r"co3_s(\d+)", phase)
    if m:
        return "CO", int(m.group(1))
    if phase.startswith("cotrain_") or phase.startswith("co_"):
        return "co_tm", 0
    if re.match(r"^(pick|pull)_", phase):
        return "co_tm", 0
    return None, None


def load_cells(roots):
    cells = {}
    for root_name, root in roots.items():
        if not root.is_dir():
            continue
        for phase_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if phase_dir.name == ".locks" or phase_dir.name.startswith("_"):
                continue
            meta = cell_meta(phase_dir)
            if meta is None:
                continue
            arm, seed = parse_identity(phase_dir.name)
            if arm is None:
                continue
            for task_dir in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
                task = task_dir.name
                for f in sorted(task_dir.rglob("chunk_*.jsonl")):
                    for line in open(f):
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        key = (root_name, phase_dir.name, task, schedule_key(r),
                               meta["sampler"], meta["steps"])
                        cells.setdefault(key, {"arm": arm, "seed": seed, "recs": {}})
                        cells[key]["recs"][int(r["seed"])] = r
    return cells


def summarize(key, cell):
    recs = cell["recs"]
    seeds = sorted(recs)
    n = len(seeds)
    k = sum(1 for s in seeds if recs[s].get("success_once"))
    ke = sum(1 for s in seeds if recs[s].get("success_at_end"))
    lo, hi = wilson_interval(k, n)
    calls = float(np.mean([recs[s]["n_inference"] for s in seeds]))
    arm = cell["arm"]
    ms = MS_PER_CALL.get((key[4], key[5]), {}).get(arm, 145.0)
    return {
        "root": key[0], "phase": key[1], "task": key[2], "schedule": key[3],
        "sampler": key[4], "steps": key[5], "arm": arm, "train_seed": cell["seed"],
        "n_episodes": n, "successes": k, "rate": k / n,
        "wilson_lo95": round(lo, 5), "wilson_hi95": round(hi, 5),
        "successes_at_end": ke, "rate_at_end": ke / n,
        "mean_calls": round(calls, 2), "mean_horizon": round(
            float(np.mean([recs[s]["mean_horizon"] for s in seeds])), 3),
        "derived_policy_s_per_ep": round(calls * ms / 1000.0, 3),
        "derived_ms_per_call": ms,
    }


def rate_by_run(rows, task, schedule, sampler, steps, arm):
    out = {}
    for r in rows:
        if (r["task"], r["schedule"], r["sampler"], r["steps"], r["arm"]) == (
                task, schedule, sampler, steps, arm) and r["train_seed"] is not None:
            out[int(r["train_seed"])] = float(r["rate"])
    return out


def paired_run_rates(rows, task, schedule, sampler, steps, arm, base_schedule):
    """Per-run (rate_A - rate_base) using identical evaluation seeds within each run."""
    a = {}
    b = {}
    for r in rows:
        if (r["arm"] != arm or r["task"] != task or r["sampler"] != sampler
                or r["steps"] != steps or r["train_seed"] is None):
            continue
        if r["schedule"] == schedule:
            a[int(r["train_seed"])] = float(r["rate"])
        elif r["schedule"] == base_schedule:
            b[int(r["train_seed"])] = float(r["rate"])
    common = sorted(set(a) & set(b))
    return common, [a[s] - b[s] for s in common], {s: a[s] for s in common}, {s: b[s] for s in common}


def add_test(tests, name, kind, arm_a, arm_b, schedule, n_runs, stat, p, diff, lo, hi,
             note="", n_a=None, n_b=None):
    tests.append({"test": name, "kind": kind, "arm_a": arm_a, "arm_b": arm_b,
                  "schedule": schedule, "n_a": n_a, "n_b": n_b,
                  "n_runs_per_arm": n_runs,
                  "complete": (n_a is not None and n_b is not None and n_a == n_b),
                  "statistic": round(float(stat), 5), "p_value": round(float(p), 6),
                  "difference": round(float(diff), 5),
                  "ci_lo95": round(float(lo), 5), "ci_hi95": round(float(hi), 5),
                  "note": note})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=str(RAW_BASE),
                    help="raw root holding training_variants/ and r6_*/r7_*/ (default: results/raw)")
    ap.add_argument("--out-prefix", default=str(ROOT / "results" / "r7_runlevel"))
    args = ap.parse_args(argv)
    global ROOTS
    base = Path(args.raw_root)
    ROOTS = {name: (base / "training_variants" if name == "r5_training_variants" else base / name)
             for name in ROOTS}
    cells = load_cells(ROOTS)
    if not cells:
        raise SystemExit("no cells found")
    rows = [summarize(k, c) for k, c in sorted(cells.items())]
    rows.sort(key=lambda r: (r["task"], str(r["arm"]), str(r["train_seed"]), r["schedule"],
                             r["sampler"], r["steps"]))
    with open(f"{args.out_prefix}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] {len(rows)} cells -> {args.out_prefix}_summary.csv")

    grouped = defaultdict(list)
    for (root_name, phase, task, sched, sampler, steps), cell in cells.items():
        recs = cell["recs"]
        rate = sum(1 for r in recs.values() if r.get("success_once")) / max(len(recs), 1)
        grouped[(task, cell["arm"], sched, sampler, steps)].append(rate)
    arms = []
    for (task, arm, sched, sampler, steps), rates in sorted(grouped.items(),
                                                            key=lambda x: (x[0][0], str(x[0][1]))):
        s = S.summarize_arm(rates)
        arms.append({"task": task, "arm": arm, "schedule": sched, "sampler": sampler,
                     "steps": steps, "n_runs": s["n_runs"], "mean": round(s["mean"], 5),
                     "sd": round(s["sd"], 5), "min": round(s["min"], 5),
                     "max": round(s["max"], 5), "range": round(s["range"], 5)})
    with open(f"{args.out_prefix}_arms.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(arms[0].keys()))
        w.writeheader()
        w.writerows(arms)
    print(f"[arms] {len(arms)} arm summaries -> {args.out_prefix}_arms.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

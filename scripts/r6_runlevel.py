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
              "r6_adaptive_other_tasks")
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
    """Extract (arm, seed) from a phase name; arm 'co' for the teammate's checkpoint."""
    m = re.search(r"(B0|V1)s(\d+)", phase)
    if m:
        return m.group(1), int(m.group(2))
    if "cotrain" in phase or phase.startswith("co_"):
        return "co", 0
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
                    help="raw root holding training_variants/ and r6_*/ (default: results/raw)")
    ap.add_argument("--out-prefix", default=str(ROOT / "results" / "r6_runlevel"))
    args = ap.parse_args(argv)
    global ROOTS
    base = Path(args.raw_root)
    ROOTS = {name: (base / "training_variants" if name == "r5_training_variants" else base / name)
             for name in ROOTS}
    cells = load_cells(ROOTS)
    if not cells:
        raise SystemExit("no r6 cells found")
    rows = [summarize(k, c) for k, c in sorted(cells.items())]
    rows.sort(key=lambda r: (r["task"], r["arm"], r["train_seed"], r["schedule"], r["steps"]))
    with open(f"{args.out_prefix}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] {len(rows)} cells -> {args.out_prefix}_summary.csv")

    tests = []
    ADAP_KEY = "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)"
    ADAP_OTHER = {"PickCube-v1": "adaptive_push(d_goal=0.05,d_tcp=0.0,long=8,short=2)",
                  "PullCubeTool-v1": "adaptive_dist(d=0.3,long=8,short=2)"}
    for task, sched in (("PushCube-v1", "fixed(k=8)"),):
        for steps, sampler in ((100, "ddpm"),):
            a = rate_by_run(rows, task, sched, sampler, steps, "V1")
            b = rate_by_run(rows, task, sched, sampler, steps, "B0")
            if a and b:
                p, diff, nperm = S.permutation_test(list(a.values()), list(b.values()))
                lo, hi, _ = S.bootstrap_diff_ci(list(a.values()), list(b.values()))
                add_test(tests, "A_normalization_V1_vs_B0", "independent_permutation", "V1", "B0",
                         f"{sched}|{sampler}-{steps}", min(len(a), len(b)), diff, p, diff, lo, hi,
                         note=f"n_perm={nperm} V1_runs={sorted(a)} B0_runs={sorted(b)}",
                         n_a=len(a), n_b=len(b))
    for arm in ("B0", "V1"):
        seeds, diffs, aa, bb = paired_run_rates(rows, "PushCube-v1",
                                                "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)",
                                                "ddpm", 100, arm, "fixed(k=8)")
        if seeds:
            p, stat, npat = S.signflip_test(diffs)
            lo, hi, _ = S.bootstrap_diff_ci([aa[s] for s in seeds], [bb[s] for s in seeds])
            add_test(tests, f"B{'1' if arm == 'B0' else '2'}_adaptive_vs_fixed8_within_{arm}",
                     "paired_signflip", arm, arm, "adaptive|ddpm-100", len(seeds), stat, p,
                     float(np.mean(diffs)), lo, hi,
                     note=f"n_patterns={npat} per_run_diffs="
                          f"{ {s: round(d, 4) for s, d in zip(seeds, diffs)} }",
                     n_a=len(seeds), n_b=len(seeds))
    for task, key in ADAP_OTHER.items():
        for arm in ("B0", "V1"):
            seeds, diffs, aa, bb = paired_run_rates(rows, task, key, "ddpm", 100, arm,
                                                    "fixed(k=8)")
            if seeds:
                p, stat, npat = S.signflip_test(diffs)
                lo, hi, _ = S.bootstrap_diff_ci([aa[s] for s in seeds], [bb[s] for s in seeds])
                add_test(tests, f"F_adaptive_vs_fixed8_{task}_within_{arm}", "paired_signflip",
                         arm, arm, f"adaptive|ddpm-100|{task}", len(seeds), stat, p,
                         float(np.mean(diffs)), lo, hi,
                         note=f"n_patterns={npat}", n_a=len(seeds), n_b=len(seeds))
    va = rate_by_run(rows, "PushCube-v1",
                     "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)", "ddpm", 100, "V1")
    bf = rate_by_run(rows, "PushCube-v1", "fixed(k=8)", "ddpm", 100, "B0")
    if va and bf:
        p, diff, nperm = S.permutation_test(list(va.values()), list(bf.values()))
        lo, hi, _ = S.bootstrap_diff_ci(list(va.values()), list(bf.values()))
        add_test(tests, "B3_stacking_V1adaptive_vs_B0fixed8", "independent_permutation",
                 "V1(adaptive)", "B0(fixed:8)", "adaptive|ddpm-100", min(len(va), len(bf)),
                 diff, p, diff, lo, hi, note=f"n_perm={nperm}", n_a=len(va), n_b=len(bf))
    s_b0, d_b0, _, _ = paired_run_rates(rows, "PushCube-v1",
                                        "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)",
                                        "ddpm", 100, "B0", "fixed(k=8)")
    s_v1, d_v1, _, _ = paired_run_rates(rows, "PushCube-v1",
                                        "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)",
                                        "ddpm", 100, "V1", "fixed(k=8)")
    if s_b0 and s_v1:
        p, diff, nperm = S.permutation_test(d_v1, d_b0)
        lo, hi, _ = S.bootstrap_diff_ci(d_v1, d_b0)
        add_test(tests, "B3_gain_differs_between_arms", "independent_permutation",
                 "gain_V1", "gain_B0", "adaptive-fixed:8", min(len(d_b0), len(d_v1)),
                 diff, p, diff, lo, hi, note=f"n_perm={nperm}", n_a=len(va), n_b=len(bf))
    best = rate_by_run(rows, "PushCube-v1",
                       "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)", "ddim", 5, "V1")
    default = rate_by_run(rows, "PushCube-v1", "fixed(k=8)", "ddpm", 100, "B0")
    if best and default:
        p, diff, nperm = S.permutation_test(list(best.values()), list(default.values()))
        lo, hi, _ = S.bootstrap_diff_ci(list(best.values()), list(default.values()))
        add_test(tests, "D_best_combo_V1adaptive_ddim5_vs_default", "independent_permutation",
                 "V1(adaptive,ddim5)", "B0(fixed:8,ddpm100)", "adaptive|ddim-5",
                 min(len(best), len(default)), diff, p, diff, lo, hi, note=f"n_perm={nperm}",
                 n_a=len(best), n_b=len(default))
    for task in ("PickCube-v1", "PullCubeTool-v1"):
        a = rate_by_run(rows, task, "fixed(k=8)", "ddpm", 100, "V1")
        b = rate_by_run(rows, task, "fixed(k=8)", "ddpm", 100, "B0")
        if a and b:
            p, diff, nperm = S.permutation_test(list(a.values()), list(b.values()))
            lo, hi, _ = S.bootstrap_diff_ci(list(a.values()), list(b.values()))
            add_test(tests, f"C_normalization_{task}_V1_vs_B0", "independent_permutation",
                     "V1", "B0", f"{task}|fixed:8|ddpm-100", min(len(a), len(b)), diff, p,
                     diff, lo, hi, note=f"n_perm={nperm} alpha_two_task=0.025",
                     n_a=len(a), n_b=len(b))

    grouped = defaultdict(list)
    for (root_name, phase, task, sched, sampler, steps), cell in cells.items():
        recs = cell["recs"]
        rate = sum(1 for r in recs.values() if r.get("success_once")) / max(len(recs), 1)
        grouped[(task, cell["arm"], sched, sampler, steps)].append(rate)
    arms = []
    for (task, arm, sched, sampler, steps), rates in sorted(grouped.items()):
        s = S.summarize_arm(rates)
        arms.append({"task": task, "arm": arm, "schedule": sched, "sampler": sampler,
                     "steps": steps, "n_runs": s["n_runs"], "mean": round(s["mean"], 5),
                     "sd": round(s["sd"], 5), "min": round(s["min"], 5),
                     "max": round(s["max"], 5), "range": round(s["range"], 5)})
    if arms:
        with open(f"{args.out_prefix}_arms.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(arms[0].keys()))
            w.writeheader()
            w.writerows(arms)
        print(f"[arms] {len(arms)} arm summaries -> {args.out_prefix}_arms.csv")

    comparisons = [("adaptive_vs_fixed8", "PushCube-v1", "ddpm", 100, ADAP_KEY, "fixed(k=8)"),
                   ("ddim5_fixed8_vs_ddpm100_fixed8", "PushCube-v1", "ddim", 5,
                    "fixed(k=8)", "fixed(k=8)"),
                   ("ddim5_adaptive_vs_ddpm100_adaptive", "PushCube-v1", "ddim", 5,
                    ADAP_KEY, ADAP_KEY)]
    comparisons += [(f"adaptive_vs_fixed8", task, "ddpm", 100, key, "fixed(k=8)")
                    for task, key in ADAP_OTHER.items()]
    mcn = []
    for label, task, sampler, steps, sched_a, sched_b in comparisons:
        for arm in ("B0", "V1"):
            for seed in range(6):
                same_sched = (label != "adaptive_vs_fixed8")
                key_a = next((k for k, c in cells.items()
                              if c["arm"] == arm and c["seed"] == seed and k[2] == task
                              and k[3] == sched_a and k[4] == sampler and k[5] == steps), None)
                key_b = next((k for k, c in cells.items()
                              if c["arm"] == arm and c["seed"] == seed and k[2] == task
                              and k[3] == sched_b
                              and k[4] == ("ddpm" if same_sched is False else sampler)
                              and k[5] == (100 if same_sched is False else steps)), None)
                if not key_a or not key_b:
                    continue
                ra, rb = cells[key_a]["recs"], cells[key_b]["recs"]
                seeds = sorted(set(ra) & set(rb))
                if not seeds:
                    continue
                av = [bool(ra[s].get("success_once")) for s in seeds]
                bv = [bool(rb[s].get("success_once")) for s in seeds]
                n10, n01, p = mcnemar_exact(av, bv)
                mcn.append({"comparison": label, "task": task, "arm": arm,
                            "train_seed": seed, "n_paired": len(seeds),
                            "A_rate": round(float(np.mean(av)), 5),
                            "B_rate": round(float(np.mean(bv)), 5),
                            "A_only": n10, "B_only": n01, "mcnemar_exact_p": round(float(p), 6)})
    if mcn:
        with open(f"{args.out_prefix}_mcnemar.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(mcn[0].keys()))
            w.writeheader()
            w.writerows(mcn)
        print(f"[mcnemar] {len(mcn)} per-run paired comparisons -> {args.out_prefix}_mcnemar.csv")

    if tests:
        with open(f"{args.out_prefix}_tests.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(tests[0].keys()))
            w.writeheader()
            w.writerows(tests)
        print(f"[tests] {len(tests)} pre-registered tests -> {args.out_prefix}_tests.csv")
        for t in tests:
            flag = "" if t["complete"] else "  [PARTIAL - not a pre-registered comparison]"
            print(f"  {t['test']:<46} n={t['n_a']}v{t['n_b']} diff={t['difference']:+.4f} "
                  f"[{t['ci_lo95']:+.4f},{t['ci_hi95']:+.4f}] p={t['p_value']}{flag}")
    else:
        print("[tests] none computable yet (cells missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
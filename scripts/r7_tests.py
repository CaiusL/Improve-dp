#!/usr/bin/env python
"""Round-7 pre-registered comparisons -> results/r7_tests.csv.

Every row records: the comparison, the kind of test, the two arms/schedules, n per side, the
number of permutations (or patterns) enumerated, the smallest attainable p for that design,
the statistic, the difference with a 95% interval, the p-value where the test is valid, and the
descriptive flag where it is not.

Design notes carried from the pre-registration:
  * training-time comparisons use the training run as the unit (exact permutation on run rates);
  * within-model comparisons on identical seeds use exact McNemar on episodes (valid);
  * 3 vs 4 runs gives C(7,3) = 35 permutations, smallest attainable two-sided p = 2/35 = 0.057,
    so no variant test can reach alpha = 0.05 by construction - those rows are flagged invalid
    and reported with the P2 descriptive flag instead;
  * 4 vs 4 -> 70 permutations (smallest p = 0.0286), 4 vs 6 -> 210 (0.0095), 6 vs 6 -> 924.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from adaptive_replan import runlevel as S  # noqa: E402
import r7_runlevel as R  # noqa: E402
from adaptive_replan import mcnemar_exact, newcombe_paired_diff  # noqa: E402

ADAP_PICK = "adaptive_push(d_goal=0.05,d_tcp=0.0,long=8,short=2)"
ADAP_PULL = "adaptive_dist(d=0.3,long=8,short=2)"
ADAP_PUSH = "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)"
FIXED = {"fixed:2": "fixed(k=2)", "fixed:4": "fixed(k=4)", "fixed:8": "fixed(k=8)"}
TASK_ADAP = {"PickCube-v1": ADAP_PICK, "PullCubeTool-v1": ADAP_PULL, "PushCube-v1": ADAP_PUSH}
ROWS = []


def smallest_p(n_a, n_b):
    total = comb(n_a + n_b, n_a)
    return 2.0 / total if total > 1 else 1.0


def _num(x):
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return ""


def add(test, kind, task, a, b, n_a, n_b, statistic, difference, lo, hi, p, valid, note=""):
    ROWS.append({"test": test, "kind": kind, "task": task, "arm_a": a[0], "sched_a": a[1],
                 "arm_b": b[0], "sched_b": b[1], "n_a": n_a, "n_b": n_b,
                 "permutations": int(comb(n_a + n_b, n_a)) if kind.startswith("run") else "",
                 "smallest_attainable_p": round(smallest_p(n_a, n_b), 5) if kind.startswith("run") else "",
                 "statistic": _num(statistic), "difference": _num(difference),
                 "ci_lo95": _num(lo), "ci_hi95": _num(hi),
                 "p_value": (_num(p) if valid else ""), "valid_test": valid,
                 "note": note})


def pick_recs(cells, task, arm, sched, sampler="ddpm", steps=100, root=None):
    out = []
    for k, c in cells.items():
        if (k[2] == task and c["arm"] == arm and k[3] == sched and k[4] == sampler
                and k[5] == steps and (root is None or k[0] == root)):
            out.append((c["seed"], c["recs"]))
    return out


def merge_recs(lst):
    merged = {}
    for _, recs in lst:
        merged.update(recs)
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=str(R.RAW_BASE))
    ap.add_argument("--out", default=str(HERE.parent / "results" / "r7_tests.csv"))
    args = ap.parse_args(argv)
    base = Path(args.raw_root)
    roots = {name: (base / "training_variants" if name == "r5_training_variants" else base / name)
             for name in R.ROOTS}
    R.ROOTS = roots
    cells = R.load_cells(roots)
    print(f"loaded {len(cells)} cells")

    for task, block in (("PickCube-v1", "r7_frontier_pick"), ("PullCubeTool-v1", "r7_frontier_pull")):
        adap = merge_recs(pick_recs(cells, task, "co_tm", TASK_ADAP[task], root=block))
        fixed = {k: merge_recs(pick_recs(cells, task, "co_tm", v, root=block))
                 for k, v in FIXED.items()}
        if not adap or not any(fixed.values()):
            add(f"P1a_{task}_adaptive_vs_best_fixed", "paired_episodes", task,
                ("co_tm", "adaptive"), ("co_tm", "best_fixed"), 0, 0, 0, 0, 0, 0, 0, False,
                "not available yet")
            continue
        best_name, best_rate = None, -1.0
        for k, recs in fixed.items():
            if recs and len(recs) >= 100:
                r = np.mean([bool(v.get("success_once")) for v in recs.values()])
                if r > best_rate:
                    best_name, best_rate = k, r
        for k, recs in list(fixed.items()) + [("best_fixed=" + str(best_name),
                                              fixed.get(best_name, {}))]:
            if not recs:
                continue
            seeds = sorted(set(adap) & set(recs))
            if not seeds:
                continue
            av = [bool(adap[s].get("success_once")) for s in seeds]
            bv = [bool(recs[s].get("success_once")) for s in seeds]
            n10, n01, p = mcnemar_exact(av, bv)
            diff, lo, hi = newcombe_paired_diff(av, bv)
            add(f"P1a_{task}_adaptive_vs_{k}", "paired_episodes", task, ("co_tm", "adaptive"),
                ("co_tm", k), len(seeds), len(seeds), diff, diff, lo, hi, p, True,
                f"max_compatible_loss={-lo*100:.1f}pp discordant={n10}v{n01}")

    for task, block in (("PickCube-v1", "r7_frontier_pick"), ("PullCubeTool-v1", "r7_frontier_pull")):
        scheds = [("fixed2", FIXED["fixed:2"]), ("fixed4", FIXED["fixed:4"]),
                  ("fixed8", FIXED["fixed:8"]), ("adaptive", TASK_ADAP[task])]
        for stag, sstr in scheds:
            base_recs = merge_recs(pick_recs(cells, task, "co_tm", sstr, "ddpm", 100, root=block))
            for tag, sampler, steps in (("ddpm20", "ddpm", 20), ("ddim20", "ddim", 20)):
                rec = merge_recs(pick_recs(cells, task, "co_tm", sstr, sampler, steps, root=block))
                if not base_recs or not rec:
                    continue
                seeds = sorted(set(base_recs) & set(rec))
                if len(seeds) < 50:
                    continue
                av = [bool(rec[s].get("success_once")) for s in seeds]
                bv = [bool(base_recs[s].get("success_once")) for s in seeds]
                n10, n01, p = mcnemar_exact(av, bv)
                diff, lo, hi = newcombe_paired_diff(av, bv)
                add(f"P1b_{task}_{stag}_{tag}_vs_ddpm100", "paired_episodes", task,
                    ("co_tm", f"{stag}_{tag}"), ("co_tm", f"{stag}_ddpm100"), len(seeds),
                    len(seeds), diff, diff, lo, hi, p, True,
                    f"largest_compatible_loss={-lo*100:.1f}pp discordant={n10}v{n01}")

    for task in ("PickCube-v1", "PullCubeTool-v1"):
        b0 = merge_recs(pick_recs(cells, task, "B0", "fixed(k=8)"))
        if not b0:
            continue
        base_rates = []
        for _, recs in pick_recs(cells, task, "B0", "fixed(k=8)"):
            base_rates.append(np.mean([bool(v.get("success_once")) for v in recs.values()]))
        base_rates = np.array(base_rates)
        for arm in ("V1", "V2a", "V2b", "V3"):
            rates = []
            for _, recs in pick_recs(cells, task, arm, "fixed(k=8)"):
                if recs:
                    rates.append(np.mean([bool(v.get("success_once")) for v in recs.values()]))
            if not rates:
                continue
            rates = np.array(rates)
            flag = S.descriptive_flag(rates, base_rates)
            p, diff, nperm = S.permutation_test(rates, base_rates)
            lo, hi, _ = S.bootstrap_diff_ci(rates, base_rates)
            add(f"P2_{task}_{arm}_vs_B0", "run_level_descriptive", task, (arm, "fixed:8"),
                ("B0", "fixed:8"), len(rates), len(base_rates), diff, diff, lo, hi, p, False,
                f"flag={flag} baseline_min={base_rates.min():.3f} baseline_max={base_rates.max():.3f} "
                f"n_perm={nperm} smallest_p={smallest_p(len(rates), len(base_rates)):.4f}")

    for task, block in (("PickCube-v1", "r7_combo_pick"), ("PullCubeTool-v1", "r7_combo_pull")):
        def rates_of(arm, sched, sampler, steps):
            """Per-run rates; no root filter, because the DDPM-100 cells come from the brief-6
            roots while the DDIM-5 cells are produced by this task's round-7 root."""
            out = {}
            for seed, recs in pick_recs(cells, task, arm, sched, sampler, steps):
                out[seed] = np.mean([bool(v.get("success_once")) for v in recs.values()])
            return out
        default = rates_of("B0", "fixed(k=8)", "ddpm", 100)
        prespec = rates_of("V1", ADAP_PICK if task == "PickCube-v1" else ADAP_PULL, "ddim", 5)
        cells_means = {}
        for arm in ("B0", "V1"):
            for sname, sstr in (("fixed:8", "fixed(k=8)"),
                                ("adaptive", ADAP_PICK if task == "PickCube-v1" else ADAP_PULL)):
                for sampler, steps in (("ddpm", 100), ("ddim", 5)):
                    r = rates_of(arm, sstr, sampler, steps)
                    if r:
                        cells_means[(arm, sname, f"{sampler}-{steps}")] = (float(np.mean(list(r.values()))), r)
        if default and prespec:
            p, diff, nperm = S.permutation_test(list(prespec.values()), list(default.values()))
            lo, hi, _ = S.bootstrap_diff_ci(list(prespec.values()), list(default.values()))
            add(f"P3_{task}_prespecified_V1adaptive_ddim5_vs_default", "run_level", task,
                ("V1", "adaptive|ddim-5"), ("B0", "fixed:8|ddpm-100"), len(prespec),
                len(default), diff, diff, lo, hi, p, True, f"n_perm={nperm}")
        if default and cells_means:
            best = max(cells_means.items(), key=lambda kv: kv[1][0])
            p, diff, nperm = S.permutation_test(list(best[1][1].values()), list(default.values()))
            lo, hi, _ = S.bootstrap_diff_ci(list(best[1][1].values()), list(default.values()))
            add(f"P3_{task}_posthoc_best_vs_default", "run_level", task, best[0],
                ("B0", "fixed:8|ddpm-100"), len(best[1][1]), len(default), diff, diff, lo, hi, p,
                False, f"POST HOC n_perm={nperm}")
        for arm in ("B0", "V1"):
            for sname, sstr in (("fixed:8", "fixed(k=8)"),
                                ("adaptive", ADAP_PICK if task == "PickCube-v1" else ADAP_PULL)):
                d5 = pick_recs(cells, task, arm, sstr, "ddim", 5, root=block)
                d100 = pick_recs(cells, task, arm, sstr, "ddpm", 100)
                lost = tot = 0
                for (s5, r5), (s100, r100) in zip(sorted(d5), sorted(d100)):
                    seeds = sorted(set(r5) & set(r100))
                    if not seeds:
                        continue
                    av = [bool(r5[s].get("success_once")) for s in seeds]
                    bv = [bool(r100[s].get("success_once")) for s in seeds]
                    _, _, p = mcnemar_exact(av, bv)
                    diff = np.mean(av) - np.mean(bv)
                    tot += 1
                    if p < 0.05 and diff < 0:
                        lost += 1
                    add(f"P3_{task}_{arm}_{sname}_run{s5}_ddim5_vs_ddpm100", "paired_episodes",
                        task, (arm, f"{sname}|ddim-5"), (arm, f"{sname}|ddpm-100"), len(seeds),
                        len(seeds), diff, diff, "", "", p, True,
                        f"run_loses_significantly={bool(p < 0.05 and diff < 0)}")
                if tot:
                    add(f"P3_{task}_{arm}_{sname}_ddim5_loss_count", "summary", task,
                        (arm, sname), (arm, sname), tot, tot, lost / tot, 0, 0, 0, 0, True,
                        f"{lost} of {tot} runs lose success significantly at DDIM-5")

    fields = ["test", "kind", "task", "arm_a", "sched_a", "arm_b", "sched_b", "n_a", "n_b",
              "permutations", "smallest_attainable_p", "statistic", "difference", "ci_lo95",
              "ci_hi95", "p_value", "valid_test", "note"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(ROWS)
    print(f"[tests] {len(ROWS)} rows -> {args.out}")
    for r in ROWS:
        if r["kind"] != "paired_episodes" or "run" in r["test"]:
            print(f"  {r['test']:<58} n={r['n_a']}v{r['n_b']} diff={r['difference']:+.4f} "
                  f"p={r['p_value']} valid={r['valid_test']} {r['note'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
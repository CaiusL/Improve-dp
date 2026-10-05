#!/usr/bin/env python
"""Round 5 Task 3 analysis: training-variant evaluations on seeds 500000-500199.

Reads results/raw/training_variants/<phase>/... where phase names are eval_<MODEL>
(optionally _<schedule>). Reports per-model Wilson summaries, per-variant mean+range
across training seeds, and the pre-registered decision rule:
variant beats baseline only if BOTH its seeds exceed the better B0 seed by more than
the B0 seed-to-seed spread AND McNemar p < 0.05 for each pair against both B0 seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(HERE.parent / "src"))
from adaptive_replan import mcnemar_exact, newcombe_paired_diff, wilson_interval  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "results" / "raw" / "training_variants"


def schedule_key(rec):
    p = rec.get("schedule_params") or {}
    ps = ",".join(f"{k}={p[k]}" for k in sorted(p))
    return f"{rec['schedule']}({ps})" if ps else rec["schedule"]


def load(schedules=None):
    store = defaultdict(dict)
    for ph in sorted(p for p in RAW.iterdir() if p.is_dir()):
        name = ph.name.split("_", 1)[1] if ph.name.startswith("eval_") else ph.name
        for f in ph.rglob("chunk_*.jsonl"):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                k = schedule_key(r)
                if schedules and k not in schedules:
                    continue
                store[(name, k)][int(r["seed"])] = r
    return store


def summ(recs):
    seeds = sorted(recs)
    n = len(seeds)
    k = sum(1 for s in seeds if recs[s].get("success_once"))
    ke = sum(1 for s in seeds if recs[s].get("success_at_end"))
    lo, hi = wilson_interval(k, n)
    return {"model": None, "n": n, "successes": k, "success_rate": k / n,
            "wilson_lo95": round(lo, 4), "wilson_hi95": round(hi, 4),
            "successes_at_end": ke, "rate_at_end": round(ke / n, 4),
            "mean_calls": round(float(np.mean([recs[s]["n_inference"] for s in seeds])), 2),
            "mean_horizon": round(float(np.mean([recs[s]["mean_horizon"] for s in seeds])), 2),
            "mean_inf_s_logged": round(float(np.mean([recs[s].get("inference_time_s") or np.nan
                                                 for s in seeds])), 3)}


def main(argv=None):
    global RAW
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=str(RAW.parent),
                    help="raw root containing training_variants/ (default: results/raw)")
    ap.add_argument("--schedule", default="fixed(k=8)")
    ap.add_argument("--out", default=str(ROOT / "results" / "r5_training"))
    args = ap.parse_args(argv)
    RAW = Path(args.raw_root) / "training_variants"
    store = load({args.schedule})
    if not store:
        raise SystemExit("no eval cells found")

    rows = []
    for (name, sched), recs in sorted(store.items()):
        s = summ(recs)
        s["model"] = name
        s["schedule"] = sched
        rows.append(s)
    with open(f"{args.out}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    b0 = [r for r in rows if r["model"].startswith("B0")]
    rates = {r["model"]: r["success_rate"] for r in rows}
    recs_by_model = {(n, s): recs for (n, s), recs in store.items()}
    spread = None
    if len(b0) >= 2:
        spread = abs(b0[0]["success_rate"] - b0[1]["success_rate"])
    print(f"B0 seed-to-seed spread: {spread if spread is None else round(spread * 100, 1)} pp")

    paired = []
    for (name, sched) in sorted(store):
        if name.startswith("B0"):
            continue
        for base in [b["model"] for b in b0]:
            seeds = sorted(set(recs_by_model[(name, sched)]) & set(recs_by_model[(base, sched)]))
            if not seeds:
                continue
            av = [bool(recs_by_model[(name, sched)][s].get("success_once")) for s in seeds]
            bv = [bool(recs_by_model[(base, sched)][s].get("success_once")) for s in seeds]
            n10, n01, p = mcnemar_exact(av, bv)
            diff, lo, hi = newcombe_paired_diff(av, bv)
            paired.append({"variant": name, "vs": base, "n_paired": len(seeds),
                           "diff": round(diff, 4), "diff_lo95": round(lo, 4),
                           "diff_hi95": round(hi, 4), "A_only": n10, "B_only": n01,
                           "mcnemar_exact_p": round(p, 6)})
    with open(f"{args.out}_paired.csv", "w", newline="") as fh:
        if paired:
            w = csv.DictWriter(fh, fieldnames=list(paired[0].keys()))
            w.writeheader()
            w.writerows(paired)

    print(f"{'model':<12}{'n':>5}{'succ':>6}{'rate':>7}{'95% CI':>17}{'calls':>7}{'hor':>6}")
    for r in rows:
        print(f"{r['model']:<12}{r['n']:>5}{r['successes']:>6}{r['success_rate']:>7.3f}"
              f"   [{r['wilson_lo95']:.3f},{r['wilson_hi95']:.3f}]{r['mean_calls']:>7.1f}"
              f"{r['mean_horizon']:>6.2f}")
    print()
    for p_ in paired:
        print(f"{p_['variant']:<8} vs {p_['vs']:<6} diff={p_['diff']:+.3f} "
              f"[{p_['diff_lo95']:+.3f},{p_['diff_hi95']:+.3f}] p={p_['mcnemar_exact_p']}")

    print("\nDecision rule check (grouped by VARIANT, trailing s<digit> removed):")
    import re
    def variant_of(name):
        return re.sub(r"s[0-9]$", "", name)
    prim = {r["model"]: r for r in rows if r["model"].endswith(("s0", "s1"))}
    b0_prim = [r for r in rows if variant_of(r["model"]) == "B0" and r["model"] in prim]
    if len(b0_prim) < 2:
        print("  B0 primary seeds incomplete; rule not computable yet")
        return 0
    better_base = max(b0_prim, key=lambda r: r["success_rate"])["model"]
    spread = better_base and abs(b0_prim[0]["success_rate"] - b0_prim[1]["success_rate"])
    b0_rates = sorted(r["success_rate"] for r in rows if variant_of(r["model"]) == "B0")
    print(f"  B0 primary spread |s0-s1| = {spread * 100:.1f} pp; better B0 seed = "
          f"{better_base} ({rates[better_base] * 100:.1f}%); rule threshold = "
          f"{(rates[better_base] + spread) * 100:.1f}%")
    for v in sorted({variant_of(r["model"]) for r in rows} - {"B0"}):
        vrows_prim = [r for r in rows if variant_of(r["model"]) == v
                      and r["model"].endswith(("s0", "s1"))]
        if not vrows_prim:
            continue
        vrows_all = [r for r in rows if variant_of(r["model"]) == v]
        exceed = all(r["success_rate"] > rates[better_base] + spread for r in vrows_prim)
        ps = [p_ for p_ in paired if p_["variant"] == better_base or True]
        psv = [p_ for p_ in paired
               if variant_of(p_["variant"]) == v and p_["vs"] == better_base]
        sig = len(psv) == len(vrows_prim) and all(p_["mcnemar_exact_p"] < 0.05 for p_ in psv)
        verdict = ("BETTER THAN BASELINE (rule satisfied)" if (exceed and sig)
                   else "NOT distinguishable from training noise by the rule")
        vr = sorted(r["success_rate"] for r in vrows_all)
        overlap = not (vr[0] > b0_rates[-1])
        print(f"  {v}: primary seeds {[round(r['success_rate'], 3) for r in vrows_prim]} "
              f"(all > {rates[better_base] + spread:.3f}: {exceed}); McNemar p<0.05 for every "
              f"primary seed vs {better_base}: {sig} -> {verdict}")
        print(f"      secondary (all seeds {[r['model'] for r in vrows_all]}): "
              f"range {vr[0] * 100:.1f}-{vr[-1] * 100:.1f}% vs B0 range "
              f"{b0_rates[0] * 100:.1f}-{b0_rates[-1] * 100:.1f}%; ranges overlap: {overlap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

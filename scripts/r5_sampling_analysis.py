#!/usr/bin/env python
"""Round 5 Task 2 analysis: sampler x steps sweep (2.1) and frontier (2.2).

Keys cells by (task, sampler, diffusion_steps) instead of schedule only, pairs every
reduced setting against DDPM-100 on identical seeds (exact McNemar + Newcombe 95%
interval), reports the largest compatible loss and the fastest setting whose interval
excludes a loss > 10 points. Writes CSVs next to the raw tree.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(HERE.parent / "src"))
from adaptive_replan import mcnemar_exact, newcombe_paired_diff, wilson_interval  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "results" / "raw" / "sampling_sweep"


def cell_meta(phase_dir):
    tag = list(Path(phase_dir).rglob("chunk_*.meta.json"))
    if tag:
        m = json.loads(tag[0].read_text())
        ds = m.get("diffusion_steps", 100)
        try:
            ds = int(ds)
        except (TypeError, ValueError):
            ds = 100
        return m.get("sampler", "ddpm"), ds, m.get("schedule")
    return None, None, None


def schedule_key(rec):
    prm = rec.get("schedule_params") or {}
    ps = ",".join(f"{k}={prm[k]}" for k in sorted(prm))
    return f"{rec['schedule']}({ps})" if ps else rec["schedule"]


def load(phases, schedule_filter=None):
    store = defaultdict(dict)
    for ph in phases:
        sampler, steps, sched = cell_meta(ph)
        if sampler is None:
            print(f"[skip {ph.name}: no meta.json]")
            continue
        for f in ph.rglob("chunk_*.jsonl"):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["task"], sampler, steps, schedule_key(r))
                if schedule_filter and key[3] != schedule_filter:
                    continue
                store[key][int(r["seed"])] = r
    return store


def summ(recs):
    seeds = sorted(recs)
    n = len(seeds)
    k = sum(1 for s in seeds if recs[s].get("success_once"))
    ke = sum(1 for s in seeds if recs[s].get("success_at_end"))
    lo, hi = wilson_interval(k, n)
    return {
        "n": n, "successes": k, "success_rate": k / n,
        "wilson_lo95": round(lo, 4), "wilson_hi95": round(hi, 4),
        "successes_at_end": ke, "rate_at_end": ke / n,
        "mean_calls": round(float(np.mean([recs[s]["n_inference"] for s in seeds])), 2),
        "mean_inf_s": round(float(np.mean([recs[s].get("inference_time_s") or np.nan
                                           for s in seeds])), 3),
        "mean_wall_s": round(float(np.mean([recs[s]["wall_time_s"] for s in seeds])), 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", default=None,
                    help="comma-separated phase dirs under sampling_sweep/ (default: all)")
    ap.add_argument("--schedule", default="fixed(k=8)",
                    help="canonical schedule key, e.g. fixed(k=8)")
    ap.add_argument("--raw-root", default=str(RAW.parent),
                    help="raw root containing sampling_sweep/ (default: results/raw)")
    ap.add_argument("--out", default=str(ROOT / "results" / "r5_sampling"))
    args = ap.parse_args(argv)
    base = Path(args.raw_root) / "sampling_sweep"
    phases = ([base / p for p in args.phases.split(",")] if args.phases
              else sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")))
    store = load(phases, args.schedule)
    if not store:
        raise SystemExit("no cells found")

    rows = []
    for (task, sampler, steps, sched), recs in sorted(store.items()):
        rows.append({"task": task, "sampler": sampler, "steps": steps, "schedule": sched,
                     **summ(recs)})
    Path(f"{args.out}_summary.csv").parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    paired = []
    for task in sorted({r["task"] for r in rows}):
        for (t, sampler, steps, sched), recs in sorted(store.items()):
            if t != task or (sampler, steps) == ("ddpm", 100):
                continue
            base_recs = store.get((task, "ddpm", 100, sched))
            if not base_recs:
                continue
            seeds = sorted(set(base_recs) & set(recs))
            if not seeds:
                continue
            av = [bool(recs[s].get("success_once")) for s in seeds]
            bv = [bool(base_recs[s].get("success_once")) for s in seeds]
            n10, n01, p = mcnemar_exact(av, bv)
            diff, lo, hi = newcombe_paired_diff(av, bv)
            paired.append({"task": task, "A": f"{sampler}-{steps} {sched}",
                           "B": f"ddpm-100 {sched}",
                           "n_paired": len(seeds), "A_rate": round(np.mean(av), 4),
                           "B_rate": round(np.mean(bv), 4), "diff": round(diff, 4),
                           "diff_lo95": round(lo, 4), "diff_hi95": round(hi, 4),
                           "A_only": n10, "B_only": n01, "mcnemar_exact_p": round(p, 6),
                           "max_compatible_loss_pp": round(-lo * 100, 1),
                           "excludes_loss_gt10pp": bool(lo > -0.10 - 1e-12)})
    if paired:
        with open(f"{args.out}_paired.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(paired[0].keys()))
            w.writeheader()
            w.writerows(paired)

    print(f"{'task':<18}{'sampler':<8}{'steps':>5}{'n':>4}{'succ':>6}{'rate':>7}"
          f"{'95% CI':>17}{'calls':>7}{'inf_s':>7}")
    for r in rows:
        print(f"{r['task']:<18}{r['sampler']:<8}{r['steps']:>5}{r['n']:>4}{r['successes']:>6}"
              f"{r['success_rate']:>7.3f}   [{r['wilson_lo95']:.3f},{r['wilson_hi95']:.3f}]"
              f"{r['mean_calls']:>7.1f}{r['mean_inf_s']:>7.2f}")
    print()
    for r in paired:
        print(f"{r['task']:<18}{r['A']:<10} vs ddpm-100: {r['diff']:+.3f} "
              f"[{r['diff_lo95']:+.3f},{r['diff_hi95']:+.3f}] p={r['mcnemar_exact_p']:<8} "
              f"maxloss={r['max_compatible_loss_pp']:>4}pp excl10={r['excludes_loss_gt10pp']}")
    print(f"\nwrote {args.out}_summary.csv ({len(rows)} cells), "
          f"{args.out}_paired.csv ({len(paired)} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Summarise raw per-episode logs into tables, paired tests and figures."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from adaptive_replan import mcnemar_exact, newcombe_paired_diff, wilson_interval  # noqa: E402

DEFAULT_RAW = HERE.parent / "results" / "raw"


def schedule_key(rec):
    p = rec.get("schedule_params") or {}
    ps = ",".join(f"{k}={p[k]}" for k in sorted(p))
    return f"{rec['schedule']}({ps})" if ps else rec["schedule"]


def family(key):
    return key.split("(")[0]


def load(phases, task, raw_root, seed_filter=None):
    """Collect episodes; a later phase overrides an earlier one for the same (spec, seed)."""
    store = {}
    for phase in phases:
        base = Path(raw_root) / phase / task
        if not base.exists():
            continue
        for tagdir in sorted(base.iterdir()):
            files = sorted(tagdir.glob("merged.jsonl")) or sorted(tagdir.glob("chunk_*.jsonl"))
            for f in files:
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if seed_filter is not None and int(r["seed"]) not in seed_filter:
                        continue
                    r["_phase"] = phase
                    store[(schedule_key(r), int(r["seed"]))] = r
    return store


def summarise(store):
    groups = defaultdict(dict)
    for (skey, seed), r in store.items():
        groups[skey][seed] = r
    rows = []
    for skey, recs in sorted(groups.items()):
        seeds = sorted(recs)
        n = len(seeds)
        k = sum(1 for s in seeds if recs[s].get("success_once"))
        ke = sum(1 for s in seeds if recs[s].get("success_at_end"))
        lo, hi = wilson_interval(k, n)
        s2 = s8 = tot = 0
        for s in seeds:
            for kk, c in (recs[s].get("horizon_counts") or {}).items():
                if int(kk) == 2:
                    s2 += c
                elif int(kk) == 8:
                    s8 += c
                tot += c
        rows.append({
            "schedule": skey, "n_episodes": n, "successes": k, "success_rate": round(k / n, 4),
            "wilson_lo95": round(lo, 4), "wilson_hi95": round(hi, 4),
            "successes_at_end": ke, "rate_at_end": round(ke / n, 4),
            "mean_inference_calls": round(float(np.mean([recs[s]["n_inference"] for s in seeds])), 2),
            "mean_env_steps": round(float(np.mean([recs[s]["n_env_steps"] for s in seeds])), 1),
            "mean_horizon": round(float(np.mean([recs[s]["mean_horizon"] for s in seeds])), 3),
            "frac_short_decisions": round(s2 / tot, 4) if tot else "",
            "frac_long_decisions": round(s8 / tot, 4) if tot else "",
            "mean_wall_time_s": round(float(np.mean([recs[s]["wall_time_s"] for s in seeds])), 1),
        })
    return rows, groups


def paired_table(groups, a_key, b_key, metric="success_once"):
    a, b = groups.get(a_key), groups.get(b_key)
    if not a or not b:
        return None
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    av = [bool(a[s].get(metric)) for s in seeds]
    bv = [bool(b[s].get(metric)) for s in seeds]
    n10, n01, p = mcnemar_exact(av, bv)
    diff, lo, hi = newcombe_paired_diff(av, bv)
    return {"metric": metric, "A": a_key, "B": b_key, "n_paired": len(seeds),
            "A_rate": round(sum(av) / len(av), 4), "B_rate": round(sum(bv) / len(bv), 4),
            "diff": round(diff, 4), "diff_lo95": round(lo, 4), "diff_hi95": round(hi, 4),
            "A_only": n10, "B_only": n01, "mcnemar_exact_p": round(p, 6),
            "method": "Newcombe hybrid score for paired proportions"}


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", required=True,
                    help="comma-separated phase directories; a later phase wins on conflicts")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", default=None, help="restrict to a balanced seed set, e.g. 300000-300199")
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW))
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--include-adaptive-pairs", action="store_true",
                    help="also pair adaptive-family schedules against each other")
    args = ap.parse_args(argv)

    seed_filter = set()
    if args.seeds:
        from adaptive_replan import parse_seeds
        seed_filter = set(parse_seeds(args.seeds))
    store = load([p.strip() for p in args.phases.split(",")], args.task,
                 args.raw_root, seed_filter or None)
    if not store:
        raise SystemExit("no episodes found")
    rows, groups = summarise(store)
    write_csv(f"{args.out}_summary.csv", rows)

    print(f"{'schedule':<58}{'n':>4}{'succ':>6}{'rate':>7}{'95% CI':>17}{'inf':>8}{'h':>7}")
    for r in rows:
        print(f"{r['schedule'][:58]:<58}{r['n_episodes']:>4}{r['successes']:>6}"
              f"{r['success_rate']:>7.3f}   [{r['wilson_lo95']:.3f},{r['wilson_hi95']:.3f}]"
              f"{r['mean_inference_calls']:>8.1f}{r['mean_horizon']:>7.2f}")

    adap = [k for k in groups if family(k).startswith("adaptive")]
    base = [k for k in groups if family(k) in ("fixed", "random")]
    if args.include_adaptive_pairs:
        base = base + adap
    paired = []
    seen = set()
    for a in adap:
        for b in base:
            if a == b or (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            row = paired_table(groups, a, b)
            if row:
                paired.append(row)
    write_csv(f"{args.out}_paired.csv", paired)
    if paired:
        print(f"\npaired comparisons written: {len(paired)}")
        for r in paired[:8]:
            print(f"  {r['A'][:40]:<40} vs {r['B'][:16]:<16} diff={r['diff']:+.3f} "
                  f"[{r['diff_lo95']:+.3f},{r['diff_hi95']:+.3f}] p={r['mcnemar_exact_p']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

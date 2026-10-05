#!/usr/bin/env python
"""Recompute the summary tables from the raw logs and check them against the released CSVs.

No checkpoint or teammate code is needed; only the per-episode raw logs (see
results/MANIFEST.md for their location and checksums). Exits non-zero if any
recomputed value disagrees.

Usage: python scripts/verify_results.py [--raw-root DIR] [--tolerance 5e-4]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from pack_final_results import collect, stats  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance", type=float, default=5e-4)
    ap.add_argument("--raw-root", default=str(ROOT / "results" / "raw"),
                    help="directory holding the per-episode raw logs")
    args = ap.parse_args(argv)

    final = ROOT / "results" / "FINAL_RESULTS.csv"
    if not final.exists():
        raise SystemExit(f"{final} not found; run scripts/pack_final_results.py first")

    recomputed = {}
    for key, recs in collect(args.raw_root).items():
        recomputed[key] = stats(recs)

    checked = mismatched = 0
    problems = []
    for row in csv.DictReader(open(final)):
        key = (row["task"], row["seed_block"], row["schedule"])
        actual = recomputed.get(key)
        if not actual:
            problems.append(f"no raw logs for {key}")
            continue
        for name, want in actual.items():
            if name not in row or row[name] == "" or want == "" or \
                    (isinstance(want, float) and np.isnan(want)):
                continue
            checked += 1
            tol = args.tolerance
            if name.startswith("wilson"):
                tol = max(tol, 0.002)
            if abs(float(row[name]) - float(want)) > tol:
                mismatched += 1
                problems.append(f"{key} {name}: reported {row[name]}, recomputed {want}")

    print(f"verify_results: {checked} values checked, {mismatched} mismatches, "
          f"{len(problems)} problems")
    for p in problems[:40]:
        print(f"  PROBLEM {p}")
    if mismatched or problems:
        return 1
    print("all summary values reproduce from the raw logs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

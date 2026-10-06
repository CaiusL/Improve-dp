#!/usr/bin/env python
"""Integrity check for round-5 raw logs: parse errors, duplicate seeds, unexpected counts.

Run before any analysis. A cell that fails must be deleted and re-run (never patched by
hand): the counts alone cannot tell a torn write from a legitimate duplicate.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "results" / "raw"


def check(patterns, expect_n=None):
    problems = []
    cells = 0
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat))):
            cells += 1
            seeds = []
            bad = 0
            for i, line in enumerate(open(f), 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception as exc:
                    bad += 1
                    if bad <= 2:
                        problems.append(f"PARSE {f}:{i} {str(exc)[:50]}")
                    continue
                seeds.append(int(rec["seed"]))
            dupes = [s for s, c in Counter(seeds).items() if c > 1]
            if bad:
                problems.append(f"BADLINES {f}: {bad} unparsable line(s)")
            if dupes:
                problems.append(f"DUPLICATE seeds {f}: {sorted(dupes)[:5]}")
            if expect_n and len(seeds) != expect_n:
                problems.append(f"COUNT {f}: {len(seeds)} episodes (expected {expect_n})")
    return cells, problems


def main(argv=None):
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=str(ROOT),
                    help="raw root (default: results/raw)")
    ap.add_argument("--expect-n", type=int, default=10,
                    help="episodes per chunk (chunk size; default 10)")
    args = ap.parse_args(argv)
    patterns = [
        "sampling_sweep/*/PushCube-v1/*/chunk_*.jsonl",
        "sampling_sweep/*/PickCube-v1/*/chunk_*.jsonl",
        "sampling_sweep/*/PullCubeTool-v1/*/chunk_*.jsonl",
        "sampling_sweep/front_*/PushCube-v1/*/chunk_*.jsonl",
        "training_variants/*/PushCube-v1/*/chunk_*.jsonl",
        "r6_*/*/PushCube-v1/*/chunk_*.jsonl",
        "r6_*/*/PickCube-v1/*/chunk_*.jsonl",
        "r6_*/*/PullCubeTool-v1/*/chunk_*.jsonl",
        "r7_*/*/PushCube-v1/*/chunk_*.jsonl",
        "r7_*/*/PickCube-v1/*/chunk_*.jsonl",
        "r7_*/*/PullCubeTool-v1/*/chunk_*.jsonl",
    ]
    ROOT = Path(args.raw_root)
    cells, problems = check(patterns, args.expect_n)
    print(f"checked {cells} chunk files; {len(problems)} problem(s)")
    for p in problems[:40]:
        print(" ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
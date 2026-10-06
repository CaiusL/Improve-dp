#!/usr/bin/env python
"""P5: failure-phase classification for the trained models (and the co-trained reference).

Thresholds are taken unchanged from round 3's classifier (REACH 0.05, LIFT 0.05,
GOAL_THRESH 0.025); PullCubeTool uses the harness phase labels (never grasped / grasped but
still far from the cube / reached the cube). Reads episodes logged with --decision-log.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REACH, LIFT, GOAL_THRESH = 0.05, 0.05, 0.025
NEAR_D, GRASP_EPS = 0.30, 0.06
PICK_PHASES = ["never reached the cube", "cube at goal, success not registered",
               "reached but never grasped", "grasped but never lifted",
               "lifted but never brought to the goal"]


def classify_pick(dec):
    dt = [d["d_tcp_obj"] for d in dec if d.get("d_tcp_obj") is not None]
    dg = [d["d_obj_goal"] for d in dec if d.get("d_obj_goal") is not None]
    cz = [d["cube_z"] for d in dec if d.get("cube_z") is not None]
    gr = [d["is_grasped"] for d in dec if d.get("is_grasped") is not None]
    if not dt or not dg:
        return "unclassified"
    if min(dt) > REACH:
        return PICK_PHASES[0]
    if min(dg) <= GOAL_THRESH:
        return PICK_PHASES[1]
    if not any(g and g >= 0.5 for g in gr):
        return PICK_PHASES[2]
    if not cz or max(cz) < LIFT:
        return PICK_PHASES[3]
    return PICK_PHASES[4]


def classify_pull(dec):
    dtl = [d["d_tcp_tool"] for d in dec if d.get("d_tcp_tool") is not None]
    dtc = [d["d_tool_cube"] for d in dec if d.get("d_tool_cube") is not None]
    if not dtl:
        return "unclassified"
    if min(dtl) >= GRASP_EPS:
        return "never grasped the tool"
    if not dtc or min(dtc) >= NEAR_D:
        return "grasped but never pulled to the cube"
    return "reached the cube but failed"


def episodes(phase_dir):
    for f in sorted(glob.glob(str(phase_dir / "*" / "*" / "chunk_*.jsonl"))):
        for line in open(f):
            if line.strip():
                yield json.loads(line)


def classify(root, task, label, max_failed=32):
    d = Path(root)
    recs = list(episodes(d))
    if not recs:
        return None
    fails = [r for r in recs if not r.get("success_once")]
    fails = fails[:max_failed]
    counts = Counter()
    for r in fails:
        dec = r.get("decisions") or []
        if not dec:
            counts["no decision log"] += 1
            continue
        counts[classify_pick(dec) if task == "PickCube-v1" else classify_pull(dec)] += 1
    n = len(recs)
    so = sum(1 for r in recs if r.get("success_once"))
    se = sum(1 for r in recs if r.get("success_at_end"))
    neither = sum(1 for r in recs if r.get("success_once") and not r.get("success_at_end"))
    return {"task": task, "arm": label, "n_episodes": n, "success_once": so, "success_at_end": se,
            "once_not_at_end": neither,
            "once_not_at_end_share_of_success": (neither / so) if so else 0.0,
            "n_failed_classified": sum(counts.values()),
            "phases": dict(counts)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="JSON list of {root, task, label}")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "r7_failure.csv"))
    args = ap.parse_args(argv)
    specs = json.loads(Path(args.spec).read_text())
    rows = []
    for s in specs:
        r = classify(s["root"], s["task"], s["label"], s.get("max_failed", 32))
        if r:
            rows.append(r)
            print(f"{s['label']:<28} n={r['n_episodes']:>3} once={r['success_once']:>3} "
                  f"at_end={r['success_at_end']:>3} once_not_at_end={r['once_not_at_end']:>3} "
                  f"| failures classified: {r['n_failed_classified']}")
            for k, v in sorted(r["phases"].items(), key=lambda kv: -kv[1]):
                print(f"    {v:>3}  {k}")
    if rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["task", "arm", "n_episodes", "success_once", "success_at_end",
                        "once_not_at_end", "share_of_success_once_not_at_end",
                        "n_failed_classified", "phase", "count"])
            for r in rows:
                for phase, cnt in sorted(r["phases"].items(), key=lambda kv: -kv[1]):
                    w.writerow([r["task"], r["arm"], r["n_episodes"], r["success_once"],
                                r["success_at_end"], r["once_not_at_end"],
                                round(r["once_not_at_end_share_of_success"], 4),
                                r["n_failed_classified"], phase, cnt])
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

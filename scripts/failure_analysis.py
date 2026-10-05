#!/usr/bin/env python
"""Task-specific failure analysis from per-decision logs."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

DEFAULT_RAW = HERE.parent / "results" / "raw"

GOAL_RADIUS = {"PushCube-v1": 0.100, "PickCube-v1": 0.025}
TCP_NEAR = 0.05
LIFT = 0.05
REACH = 0.05

PICK_PHASES = ["never reached the cube",
               "cube at goal, success not registered",
               "reached but never grasped",
               "grasped but never lifted",
               "lifted but never brought to the goal"]


def load_phase(phase, task, raw_root):
    eps = {}
    for f in glob.glob(str(Path(raw_root) / phase / task / "*" / "chunk_*.jsonl")):
        tag = Path(f).parent.name
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r["_tag"] = tag
                eps[(tag, r["seed"])] = r
    return list(eps.values())


def classify_pickcube(rec):
    dec = rec.get("decisions") or []
    if not dec:
        return None
    dt = [d["d_tcp_obj"] for d in dec if d.get("d_tcp_obj") is not None]
    dg = [d["d_obj_goal"] for d in dec if d.get("d_obj_goal") is not None]
    cz = [d.get("cube_z") for d in dec if d.get("cube_z") is not None]
    gr = [d.get("is_grasped") for d in dec if d.get("is_grasped") is not None]
    if not dt or not dg:
        return None
    if min(dt) > REACH:
        return PICK_PHASES[0]
    if min(dg) <= GOAL_RADIUS["PickCube-v1"]:
        return PICK_PHASES[1]
    if not any(g and g >= 0.5 for g in gr):
        return PICK_PHASES[2]
    if not cz or max(cz) < LIFT:
        return PICK_PHASES[3]
    return PICK_PHASES[4]


def classify_pushcube(rec):
    dec = rec.get("decisions") or []
    if not dec:
        return None
    dg = [d.get("d_obj_goal_xy") for d in dec if d.get("d_obj_goal_xy") is not None]
    dt = [d.get("d_tcp_obj") for d in dec if d.get("d_tcp_obj") is not None]
    if not dg:
        return None
    if min(dg) < GOAL_RADIUS["PushCube-v1"] + 0.01:
        return "pushed into the goal area but never registered success"
    if dt and min(dt) > 0.10:
        return "never approached the cube"
    return "pushed the cube but never reached the goal area"


def classify_pullcube(rec):
    dec = rec.get("decisions") or []
    ff = rec.get("final_features") or {}
    if dec:
        min_tcp_tool = min((d.get("d_tcp_tool") for d in dec if d.get("d_tcp_tool") is not None),
                           default=np.inf)
        min_tool_cube = min((d.get("d_tool_cube") for d in dec if d.get("d_tool_cube") is not None),
                            default=np.inf)
    else:
        min_tcp_tool = ff.get("d_tcp_tool", np.inf)
        min_tool_cube = ff.get("d_tool_cube", np.inf)
    if not np.isfinite(min_tcp_tool) or not np.isfinite(min_tool_cube):
        return "unclassified (no features)"
    if min_tcp_tool > 0.06:
        return "never grasped the tool"
    if min_tool_cube > 0.20:
        return "grasped but never brought the tool to the cube"
    return "reached the cube but did not complete the pull"


CLASSIFIERS = {
    "PickCube-v1": (classify_pickcube, PICK_PHASES),
    "PushCube-v1": (classify_pushcube, ["never approached the cube",
                                        "pushed the cube but never reached the goal area",
                                        "pushed into the goal area but never registered success"]),
    "PullCubeTool-v1": (classify_pullcube, ["never grasped the tool",
                                            "grasped but never brought the tool to the cube",
                                            "reached the cube but did not complete the pull",
                                            "unclassified (no features)"]),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--task", required=True, choices=sorted(CLASSIFIERS))
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW))
    args = ap.parse_args(argv)

    eps = load_phase(args.phase, args.task, args.raw_root)
    classified = [r for r in eps if r.get("decisions")]
    if not classified:
        raise SystemExit("no --decision-log episodes found for this phase")

    classify, phases = CLASSIFIERS[args.task]
    by = defaultdict(list)
    for r in classified:
        by[r["_tag"]].append(r)

    n_all = mid = 0
    for r in eps:
        so, se = r.get("success_once"), r.get("success_at_end")
        if so is None or se is None:
            continue
        n_all += 1
        if so and not se:
            mid += 1

    print(f"{args.task} failure analysis ({len(classified)} traced episodes, phase {args.phase})")
    print(f"success = {GOAL_RADIUS.get(args.task, '')} threshold task condition\n")
    print(f"{'schedule':<46}{'n':>4}{'succ':>6}{'fail':>6}  " + "".join(f"{p[:26]:>28}" for p in phases))
    for tag, recs in sorted(by.items()):
        counts = defaultdict(int)
        fails = [r for r in recs if not r.get("success_once")]
        for r in fails:
            counts[classify(r) or "unclassified"] += 1
        print(f"{tag[:46]:<46}{len(recs):>4}{len(recs)-len(fails):>6}{len(fails):>6}  "
              + "".join(f"{counts.get(p, 0):>28}" for p in phases))

    if n_all:
        print(f"\nsuccess_once vs success_at_end: {mid}/{n_all} "
              f"({100*mid/n_all:.2f}%) succeeded mid-episode but not at the end")
    return 0


if __name__ == "__main__":
    sys.exit(main())

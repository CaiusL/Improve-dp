#!/usr/bin/env python
"""Task B verification: the adaptive rule must read RAW observations, never V1's normalized ones.

Two independent checks on real episodes (a V1 checkpoint, i.e. a model whose rollout applies an
`obs_norm` affine to the model input):

1. recompute `d_tcp` and `d_goal` from the RAW observation row that the harness logged, using
   the release package's feature implementation (a separate code path from the working
   harness), and compare with the values the rule actually used;
2. re-derive the rule's horizon from those recomputed raw features and compare with the logged
   horizon — and confirm that the *normalized* row would have produced different features (so
   the check is not vacuous).

Writes a JSON summary and exits non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from adaptive_replan import extract_features  # noqa: E402

TOL = 1e-3


def rule_from_features(feats, short=2, long=8, d_tcp_thr=0.08, d_goal_thr=0.10):
    d_tcp = feats.get("d_tcp_obj", np.inf)
    d_goal = feats.get("d_obj_goal_xy", feats.get("d_obj_goal", np.inf))
    return short if (d_tcp < d_tcp_thr or d_goal < d_goal_thr) else long


def check_file(path: Path, task: str, expect_fixed=None):
    rec = json.loads(path.read_text().strip().splitlines()[-1])
    decisions = rec.get("decisions") or []
    rows = [d for d in decisions if "feature_check" in d]
    mism_feat, mism_rule, nonvacuous = [], [], 0
    for d in rows:
        fc = d["feature_check"]
        raw = np.asarray(fc["raw_row"], dtype=np.float64)
        feats = extract_features(task, raw)
        dtc = float(feats.get("d_tcp_obj"))
        dgl = float(feats.get("d_obj_goal_xy"))
        if d.get("d_tcp_obj") is None or abs(dtc - float(d["d_tcp_obj"])) > TOL:
            mism_feat.append((d["decision"], d.get("d_tcp_obj"), dtc))
        if d.get("d_obj_goal_xy") is None or abs(dgl - float(d["d_obj_goal_xy"])) > TOL:
            mism_feat.append((d["decision"], d.get("d_obj_goal_xy"), dgl))
        k_re = expect_fixed if expect_fixed is not None else rule_from_features(feats)
        if int(d["k"]) != int(k_re):
            mism_rule.append((d["decision"], d["k"], k_re))
        if fc.get("obs_norm_present"):
            if (fc.get("d_tcp_norm") is not None
                    and abs(float(fc["d_tcp_norm"]) - dtc) > 1e-6):
                nonvacuous += 1
    return {
        "file": str(path),
        "task": task,
        "decisions": len(decisions),
        "with_payload": len(rows),
        "obs_norm_present": bool(rows and rows[0]["feature_check"]["obs_norm_present"]),
        "feature_mismatches": len(mism_feat),
        "rule_mismatches": len(mism_rule),
        "decisions_where_normalized_differs": nonvacuous,
        "first_mismatches": (mism_feat[:3] + mism_rule[:3]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixed-jsonl", required=True)
    ap.add_argument("--adaptive-jsonl", required=True)
    ap.add_argument("--task", default="PushCube-v1")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "r6_rawfeat_check.json"))
    args = ap.parse_args(argv)
    out = {
        "fixed_schedule_episode": check_file(Path(args.fixed_jsonl), args.task, expect_fixed=2),
        "adaptive_schedule_episode": check_file(Path(args.adaptive_jsonl), args.task),
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for key, res in out.items():
        print(f"[{key}] decisions={res['decisions']} obs_norm_present={res['obs_norm_present']} "
              f"feature_mismatches={res['feature_mismatches']} "
              f"rule_mismatches={res['rule_mismatches']} "
              f"normalized_differs={res['decisions_where_normalized_differs']}")
        for m in res["first_mismatches"]:
            print("   mismatch:", m)
    bad = sum(r["feature_mismatches"] + r["rule_mismatches"] for r in out.values())
    nonvac = sum(r["decisions_where_normalized_differs"] for r in out.values())
    ok = bad == 0 and nonvac > 0
    print(f"\nVERIFICATION {'PASSED' if ok else 'FAILED'}: mismatches={bad}, "
          f"non-vacuous decisions={nonvac} (must be > 0)")
    print(f"summary -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
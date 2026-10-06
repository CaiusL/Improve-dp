#!/usr/bin/env python
"""P0(a): per-dimension statistics of the training observations that V1 normalizes.

For each task, using exactly the 200 demonstration trajectories the runs train on, this reports
per-dimension mean and sd, the ten smallest sds with their dimension names, the median sd, how
many dimensions fall below 1% of the median, and the scale factor V1 actually applies
(the training copy replaces any sd < 1e-6 by 1.0, so only dimensions with sd slightly above
1e-6 are amplified).

Writes results/r7_p0_dimstats.json and prints a compact table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from adaptive_replan import OBS_FIELDS  # noqa: E402

V1_FLOOR = 1e-6


def dim_names(task):
    names = []
    for fname, start, end in OBS_FIELDS[task]:
        for i in range(start, end):
            suffix = f"[{i - start}]" if end - start > 1 else ""
            names.append(f"{fname}{suffix}")
    return names


def stats_for(task, config):
    cfg = json.loads(Path(config).read_text())
    entry = cfg["tasks"][task]
    keys = entry["trajectory_keys"]
    with h5py.File(entry["demo_path"], "r") as f:
        obs = np.concatenate([np.asarray(f[k]["obs"], dtype=np.float64) for k in keys], axis=0)
    mean = obs.mean(axis=0)
    sd = obs.std(axis=0)
    names = dim_names(task)
    assert len(names) == obs.shape[1], (len(names), obs.shape)
    eff = np.where(sd < V1_FLOOR, 1.0, sd)
    scale = 1.0 / eff
    med = float(np.median(sd))
    below = [i for i in range(len(sd)) if sd[i] < 0.01 * med]
    order = np.argsort(sd)[:10]
    return {
        "task": task, "obs_dim": int(obs.shape[1]), "n_transitions": int(obs.shape[0]),
        "median_sd": med,
        "n_dims_below_1pct_of_median": len(below),
        "dims_below_1pct_of_median": [names[i] for i in below],
        "smallest_10_sd": [{"dim": names[i], "sd": float(sd[i]),
                            "grade_scale_factor": float(scale[i])} for i in order],
        "largest_scale_factor": float(scale.max()),
        "largest_scale_dim": names[int(np.argmax(scale))],
        "n_dims_sd_lt_1e6": int((sd < V1_FLOOR).sum()),
        "dims_sd_lt_1e6": [names[i] for i in range(len(sd)) if sd[i] < V1_FLOOR],
        "mean_range": [float(mean.min()), float(mean.max())],
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", required=True,
                    metavar="TASK=PATH",
                    help="task and the training data manifest (config JSON) it used; repeatable")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "r7_p0_dimstats.json"))
    args = ap.parse_args(argv)
    out = {"v1_floor_rule": "sd < 1e-6 replaced by 1.0 (from the training copy)",
           "tasks": {}}
    for item in args.manifest:
        task, cfg = item.split("=", 1)
        r = stats_for(task, cfg)
        out["tasks"][task] = r
        print(f"\n=== {task} (obs_dim {r['obs_dim']}, {r['n_transitions']} transitions)")
        print(f"  median sd across dims: {r['median_sd']:.6g}")
        print(f"  dims with sd < 1% of median: {r['n_dims_below_1pct_of_median']}"
              f" {r['dims_below_1pct_of_median'][:6]}")
        print(f"  largest scale factor applied: {r['largest_scale_factor']:.3g}"
              f" on {r['largest_scale_dim']}")
        print("  10 smallest sds:")
        for d in r["smallest_10_sd"]:
            print(f"    {d['dim']:<30} sd={d['sd']:.6g}  x{d['grade_scale_factor']:.3g}")
    p = Path(args.out)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {p}")
    pk = out["tasks"]["PickCube-v1"]
    print(f"\nTRIGGER part 1 (PickCube dims with sd < 1% median): "
          f"{pk['n_dims_below_1pct_of_median']} -> "
          f"{'TRIGGERED' if pk['n_dims_below_1pct_of_median'] > 0 else 'not triggered'}")


if __name__ == "__main__":
    main()
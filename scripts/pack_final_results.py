#!/usr/bin/env python
"""Build the final packaged tables: FINAL_RESULTS.csv, the paired table and MANIFEST.md."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from adaptive_replan import mcnemar_exact, newcombe_paired_diff, wilson_interval  # noqa: E402

DEFAULT_RAW = ROOT / "results" / "raw"

BLOCKS = [
    ("pullcubetool_block1", "PullCubeTool-v1", "test"),
    ("pullcubetool_block2", "PullCubeTool-v1", "confirm_pullcubetool"),
    ("pushcube_block1", "PushCube-v1", "pushcube_test"),
    ("pushcube_block1_controls", "PushCube-v1", "pushcube_controls"),
    ("pushcube_block2", "PushCube-v1", "pushcube_fresh"),
    ("pushcube_steps50", "PushCube-v1", "pushcube_steps_fresh_50"),
    ("pushcube_steps20", "PushCube-v1", "pushcube_steps_fresh_20"),
    ("pickcube_test", "PickCube-v1", "pickcube_test"),
    ("pickcube_val_baselines", "PickCube-v1", "pickcube_val"),
    ("pickcube_stillness", "PickCube-v1", "pickcube_stillness"),
    ("pushcube_window_validation", "PushCube-v1", "pushcube_win_val"),
]


def schedule_key(rec):
    p = rec.get("schedule_params") or {}
    ps = ",".join(f"{k}={p[k]}" for k in sorted(p))
    return f"{rec['schedule']}({ps})" if ps else rec["schedule"]


def collect(raw_root):
    store = defaultdict(dict)
    for label, task, phase in BLOCKS:
        for f in glob.glob(str(Path(raw_root) / phase / task / "*" / "chunk_*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                store[(task, label, schedule_key(r))][int(r["seed"])] = r
    return store


def stats(recs):
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
    its = [recs[s].get("inference_time_s") for s in seeds]
    return {
        "n_episodes": n, "successes": k, "success_rate": k / n,
        "wilson_lo95": lo, "wilson_hi95": hi,
        "successes_at_end": ke, "rate_at_end": ke / n,
        "mean_inference_calls": float(np.mean([recs[s]["n_inference"] for s in seeds])),
        "mean_horizon": float(np.mean([recs[s]["mean_horizon"] for s in seeds])),
        "frac_short_decisions": (s2 / tot if tot else np.nan),
        "frac_long_decisions": (s8 / tot if tot else np.nan),
        "mean_wall_time_s": float(np.mean([recs[s]["wall_time_s"] for s in seeds])),
        "inference_time_s": (float(np.mean([t for t in its if t is not None]))
                             if any(t is not None for t in its) else ""),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW))
    ap.add_argument("--with-manifest", action="store_true")
    args = ap.parse_args(argv)
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)

    store = collect(args.raw_root)
    rows = []
    for (task, block, spec), recs in sorted(store.items()):
        st = stats(recs)
        rows.append({"task": task, "seed_block": block, "schedule": spec,
                     **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in st.items()}})
    if not rows:
        raise SystemExit("no raw logs found; check --raw-root")

    fields = sorted(rows[0], key=lambda c: (["task", "seed_block", "schedule"].index(c)
                                            if c in ("task", "seed_block", "schedule") else 99))
    rows.sort(key=lambda r: (r["task"], r["seed_block"], r["schedule"]))
    with open(out / "FINAL_RESULTS.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: ("" if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in r.items()} for r in rows])
    print(f"[FINAL_RESULTS.csv] {len(rows)} rows")

    if args.with_manifest:
        lines = ["# Raw log manifest", "",
                 "One row per raw per-episode log file. Checksums are SHA-256 (first 16 hex chars).",
                 "", "| phase | task | schedule tag | episodes | seed range | sha256 |",
                 "|---|---|---|---|---|---|"]
        files = sorted(glob.glob(str(Path(args.raw_root) / "*" / "*" / "*" / "chunk_*.jsonl")))
        total = 0
        for f in files:
            p = Path(f)
            tag = p.parent.name
            task = p.parent.parent.name
            phase = p.parent.parent.parent.name
            if phase.endswith("AMBIGUOUS_provenance"):
                continue
            seeds = [json.loads(line)["seed"] for line in open(f) if line.strip()]
            if not seeds:
                continue
            total += len(seeds)
            lines.append(f"| {phase} | {task} | {tag} | {len(seeds)} | "
                         f"{min(seeds)}-{max(seeds)} | `{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}` |")
        lines += ["", f"**Total files:** {len(files)}. **Total episodes:** {total}.", ""]
        (out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
        print(f"[MANIFEST.md] {len(files)} files, {total} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

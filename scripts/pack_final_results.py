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

# --- Round 5 (Addendum 4): sampler sweep, execution-horizon frontier, training variants ---
R5_TASKS = ("PushCube-v1", "PickCube-v1", "PullCubeTool-v1")
R5_SWEEP_PHASES = ("ddpm100", "ddpm50", "ddpm20", "ddpm10", "ddpm5",
                   "ddim50", "ddim20", "ddim10", "ddim5")
R5_FRONT_PHASES = ("front_ddpm100", "front_ddpm20", "front_ddim5")
BLOCKS += [(f"r5_sweep_{p}", t, f"sampling_sweep/{p}") for p in R5_SWEEP_PHASES for t in R5_TASKS]
BLOCKS += [("r5_" + p, "PushCube-v1", f"sampling_sweep/{p}") for p in R5_FRONT_PHASES]
# frontier label -> sweep phase whose fixed:8 half it reused (seeds 400000-400099)
R5_FRONT_REUSE = {"r5_front_ddpm100": "sampling_sweep/ddpm100",
                  "r5_front_ddpm20": "sampling_sweep/ddpm20",
                  "r5_front_ddim5": "sampling_sweep/ddim5"}


def schedule_key(rec):
    p = rec.get("schedule_params") or {}
    ps = ",".join(f"{k}={p[k]}" for k in sorted(p))
    return f"{rec['schedule']}({ps})" if ps else rec["schedule"]


def discover_blocks(raw_root):
    raw_root = Path(raw_root)
    blocks = list(BLOCKS)
    tv = raw_root / "training_variants"
    if tv.is_dir():  # round-5 training variants, discovered so partial runs still pack
        blocks += [(d.name.replace("eval_", "r5_train_"), "PushCube-v1",
                    f"training_variants/{d.name}")
                   for d in sorted(tv.iterdir()) if d.is_dir() and d.name.startswith("eval_")]
    if raw_root.is_dir():  # round-6/7 roots: rN_<name>/<phase>/<task>/<schedule-tag>/
        for root in sorted(d for d in raw_root.iterdir()
                          if d.is_dir() and d.name.startswith(("r6_", "r7_"))):
            for phase in sorted(d for d in root.iterdir()
                                if d.is_dir() and not d.name.startswith("_")):
                for task_dir in sorted(d for d in phase.iterdir() if d.is_dir()):
                    blocks.append((f"{root.name}/{phase.name}", task_dir.name,
                                   f"{root.name}/{phase.name}"))
    return blocks


def collect(raw_root):
    raw_root = Path(raw_root)
    store = defaultdict(dict)
    for label, task, phase in discover_blocks(raw_root):
        for f in glob.glob(str(raw_root / phase / task / "*" / "chunk_*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                store[(task, label, schedule_key(r))][int(r["seed"])] = r
    # The round-5 frontier reused the first 100 sweep episodes (seeds 400000-400099) for its
    # fixed:8 cells at the same three samplers, so those cells are only complete once the reused
    # half is folded in; the round-5 analysis did this and printed n=200. Fold it in here too,
    # keyed by seed, so the packaged table matches the analysis and the report.
    for label, phase in R5_FRONT_REUSE.items():
        for f in glob.glob(str(raw_root / phase / "PushCube-v1" / "*" / "chunk_*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                store[("PushCube-v1", label, schedule_key(r))][int(r["seed"])] = r
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
        raw_root = Path(args.raw_root)
        blocks = discover_blocks(raw_root)
        lines = ["# Raw log manifest", "",
                 "One row per raw per-episode log file. Checksums are SHA-256 (first 16 hex chars).",
                 "", "| phase | task | schedule tag | episodes | seed range | sha256 |",
                 "|---|---|---|---|---|---|"]
        labelled = {}
        for label, task, phase in blocks:
            for f in glob.glob(str(raw_root / phase / task / "*" / "chunk_*.jsonl")):
                labelled[Path(f).resolve()] = (phase, task)
        files = sorted(q for q in raw_root.rglob("chunk_*.jsonl")
                       if "AMBIGUOUS_provenance" not in str(q))
        total = 0
        n_lab = 0
        for p in files:
            key = p.resolve()
            if key in labelled:
                phase, task = labelled[key]
                n_lab += 1
            else:
                phase = "/".join(p.parts[-5:-3]) if len(p.parts) >= 5 else p.parent.parent.parent.name
                task = p.parent.parent.name
            tag = p.parent.name
            seeds = [json.loads(line)["seed"] for line in open(p) if line.strip()]
            if not seeds:
                continue
            total += len(seeds)
            lines.append(f"| {phase} | {task} | {tag} | {len(seeds)} | "
                         f"{min(seeds)}-{max(seeds)} | `{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}` |")
        lines += ["", f"**Total files:** {len(files)} (round-5 labelled: {n_lab}). "
                      f"**Total episodes:** {total}.", ""]
        (out / "MANIFEST.md").write_text("\n".join(lines) + "\n")
        print(f"[MANIFEST.md] {len(files)} files, {total} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

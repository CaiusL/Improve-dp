#!/usr/bin/env python
"""Draw the success-versus-inference-cost figures (per task and the combined panel)."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from adaptive_replan import wilson_interval  # noqa: E402

DEFAULT_RAW = HERE.parent / "results" / "raw"

STYLE = {
    "fixed": ("o", "#1f77b4"),
    "random": ("s", "#ff7f0e"),
    "adaptive_push": ("^", "#2ca02c"),
    "adaptive_dist": ("^", "#2ca02c"),
    "adaptive_push_inv": ("X", "#9467bd"),
    "adaptive_time": ("P", "#8c564b"),
    "adaptive_thresh": ("D", "#e377c2"),
    "adaptive_phase": ("v", "#d62728"),
}


def schedule_key(rec):
    p = rec.get("schedule_params") or {}
    ps = ",".join(f"{k}={p[k]}" for k in sorted(p))
    return f"{rec['schedule']}({ps})" if ps else rec["schedule"]


def family(key):
    return key.split("(")[0]


def collect(phase, task, raw_root):
    groups = defaultdict(dict)
    for f in glob.glob(str(Path(raw_root) / phase / task / "*" / "chunk_*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                groups[schedule_key(r)][int(r["seed"])] = r
    out = {}
    for key, recs in groups.items():
        seeds = sorted(recs)
        n = len(seeds)
        k = sum(1 for s in seeds if recs[s].get("success_once"))
        lo, hi = wilson_interval(k, n)
        out[key] = {"rate": k / n, "lo": lo, "hi": hi, "n": n,
                    "calls": float(np.mean([recs[s]["n_inference"] for s in seeds]))}
    return out


def annotate_panel(ax, stats, title):
    seen, placed = set(), []
    for key in sorted(stats, key=lambda k: stats[k]["calls"]):
        st = stats[key]
        marker, color = STYLE.get(family(key), ("x", "gray"))
        ax.errorbar(st["calls"], st["rate"],
                    yerr=[[st["rate"] - st["lo"]], [st["hi"] - st["rate"]]],
                    marker=marker, color=color, capsize=4, ms=9, ls="none",
                    label=family(key) if family(key) not in seen else None)
        seen.add(family(key))
        k = sum(1 for (px, _) in placed if abs(px - st["calls"]) < 12)
        placed.append((st["calls"], st["rate"]))
        ax.annotate(key if len(key) < 38 else family(key), (st["calls"], st["rate"]),
                    textcoords="offset points",
                    xytext=(7, [8, -14, 20, -26, 32, -38][k % 6]), fontsize=6,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.6))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("mean policy inference calls per episode")
    ax.grid(alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), fontsize=7, loc="best")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW))
    ap.add_argument("--outdir", default=str(HERE.parent / "results"))
    args = ap.parse_args(argv)

    panels = [
        ("PickCube-v1", "pickcube_test", "PickCube-v1 (test seeds 300000-300199)"),
        ("PushCube-v1", "pushcube_fresh", "PushCube-v1 (fresh seeds 300000-300199)"),
        ("PullCubeTool-v1", "test", "PullCubeTool-v1 (test seeds 200000-200199)"),
    ]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, (task, phase, title) in zip(axes, panels):
        stats = collect(phase, task, args.raw_root)
        if not stats:
            ax.set_axis_off()
            ax.set_title(f"{task}: no data for phase {phase}")
            continue
        annotate_panel(ax, stats, f"{task}\n({title.split('(')[1].rstrip(')')})")
        single = plt.figure(figsize=(9, 6))
        annotate_panel(single.add_subplot(111), stats, f"{title}: success vs inference cost")
        for ext in ("png", "pdf"):
            single.savefig(outdir / f"fig_{task}_success_vs_cost.{ext}", dpi=150)
        plt.close(single)
    axes[0].set_ylabel("success rate (success_once)")
    fig.suptitle("Adaptive execution horizon: success versus inference cost "
                 "(95% Wilson intervals)", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_combined_success_vs_cost.{ext}", dpi=150)
    plt.close(fig)
    print(f"[figures written to {outdir}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

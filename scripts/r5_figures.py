#!/usr/bin/env python
"""Round 5 figures: (a) PushCube success-vs-seconds frontier, (b) success-vs-steps
by task and sampler family, (c) training-variant per-run points and means."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

SCHED_STYLE = {
    "fixed(k=2)": ("o", "#1f77b4", "fixed:2"),
    "fixed(k=4)": ("s", "#ff7f0e", "fixed:4"),
    "fixed(k=8)": ("^", "#2ca02c", "fixed:8"),
}
ADAP_LABEL = "adaptive (0.08/0.10)"


def sched_style(key):
    if key in SCHED_STYLE:
        return SCHED_STYLE[key]
    if key.startswith("adaptive_push"):
        return ("D", "#9467bd", ADAP_LABEL)
    return ("x", "#7f7f7f", key)


def frontier_figure(rows, ms_per_call, path_stem):
    fig, ax = plt.subplots(figsize=(9, 6))
    seen = {}
    for r in rows:
        if r["task"] != "PushCube-v1" or r["schedule"] not in ("fixed(k=2)", "fixed(k=4)",
                                                               "fixed(k=8)") and not \
                r["schedule"].startswith("adaptive_push"):
            continue
        key = f"{r['sampler']}{r['steps']}"
        s = 100.0 * float(r["success_rate"])
        lo = 100.0 * (float(r["success_rate"]) - float(r["wilson_lo95"]))
        hi = 100.0 * (float(r["wilson_hi95"]) - float(r["success_rate"]))
        secs = float(r["mean_calls"]) * ms_per_call[key] / 1000.0
        marker, colour, label = sched_style(r["schedule"])
        mstyle = {"ddpm100": "-", "ddpm20": "--", "ddim5": ":"}[key]
        ax.errorbar([secs], [s], yerr=[[lo], [hi]], marker=marker, color=colour,
                    linestyle="None", capsize=3, ms=8)
        ax.annotate(f"{key}", (secs, s), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=colour)
        seen.setdefault(label, colour)
    for label, colour in seen.items():
        ax.scatter([], [], color=colour, marker=sched_style_inverse(label), label=label)
    ax.set_xlabel("policy inference seconds per episode (calls x clean ms/call)")
    ax.set_ylabel("success_once (%) with 95% Wilson")
    ax.set_xscale("log")
    ax.set_title("PushCube-v1: execution horizon x sampler frontier (seeds 400000-400199)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=150)


def sched_style_inverse(label):
    for key, (marker, colour, lbl) in SCHED_STYLE.items():
        if lbl == label:
            return marker
    return "D"


def steps_figure(rows, path_stem):
    tasks = ["PushCube-v1", "PickCube-v1", "PullCubeTool-v1"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, task in zip(axes, tasks):
        for sampler, colour, ls in (("ddpm", "#1f77b4", "-"), ("ddim", "#d62728", "--")):
            pts = sorted([r for r in rows if r["task"] == task and r["sampler"] == sampler
                          and r["schedule"] == "fixed(k=8)"], key=lambda r: int(r["steps"]))
            if not pts:
                continue
            x = [int(r["steps"]) for r in pts]
            y = [100.0 * float(r["success_rate"]) for r in pts]
            ylo = [100.0 * (float(r["success_rate"]) - float(r["wilson_lo95"])) for r in pts]
            yhi = [100.0 * (float(r["wilson_hi95"]) - float(r["success_rate"])) for r in pts]
            ax.errorbar(x, y, yerr=[ylo, yhi], color=colour, linestyle=ls, marker="o",
                        capsize=3, label=sampler.upper())
        base = [r for r in rows if r["task"] == task and r["sampler"] == "ddpm"
                and int(r["steps"]) == 100]
        if base:
            ax.axhline(100.0 * float(base[0]["success_rate"]), color="k", lw=0.8, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("diffusion sampling steps")
        ax.set_ylabel("success_once (%)")
        ax.set_title(task)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("DDPM vs DDIM: success versus sampling steps (fixed:8, seeds 400000-400099)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=150)


def variants_figure(path_stem):
    p = RESULTS / "r5_training_summary.csv"
    if not p.exists():
        print("[skip variants figure: no r5_training_summary.csv]")
        return
    import re
    rows = list(csv.DictReader(open(p)))
    order, means = [], {}
    for r in rows:
        v = re.sub(r"s[0-9]$", "", r["model"])
        if v not in [o[0] for o in order]:
            order.append((v, []))
        for o in order:
            if o[0] == v:
                o[1].append(r)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs, colours = [], {"B0": "#7f7f7f", "V1": "#1f77b4", "V2a": "#2ca02c",
                       "V2b": "#ff7f0e", "V3": "#9467bd"}.copy()
    pos = []
    for i, (v, rr) in enumerate(order):
        vals = [100.0 * float(r["success_rate"]) for r in rr]
        for j, r in enumerate(rr):
            lo = 100.0 * (float(r["success_rate"]) - float(r["wilson_lo95"]))
            hi = 100.0 * (float(r["wilson_hi95"]) - float(r["success_rate"]))
            x = i + (j - (len(rr) - 1) / 2) * 0.18
            ax.errorbar([x], [100.0 * float(r["success_rate"])], yerr=[[lo], [hi]],
                        marker="o", ls="None", color=colours.get(v, "k"), capsize=3)
        mean = np.mean(vals)
        ax.plot([i - 0.28, i + 0.28], [mean, mean], color=colours.get(v, "k"), lw=2,
                alpha=0.6)
        pos.append(f"{v}\n(n={len(rr)})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(pos)
    ax.set_ylabel("success_once (%) with 95% Wilson")
    ax.set_title("Training variants, PushCube-v1, fixed:8, seeds 500000-500199 "
                 "(bars = mean across training seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=150)
    print(f"[variants figure: {len(order)} variants]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timing-csv", default=str(RESULTS / "r5_timing_clean.csv"))
    args = ap.parse_args(argv)
    ms = {}
    agg = {}
    for r in csv.DictReader(open(args.timing_csv)):
        key = r["setting"].split("_")[0]
        agg.setdefault(key, []).append(float(r["ms_per_call_mean"]))
    for k, v in agg.items():
        ms[k] = float(np.mean(v))
    sweep = list(csv.DictReader(open(RESULTS / "r5_sampling_summary.csv")))
    steps_figure(sweep, str(RESULTS / "fig_r5_steps"))
    fpath = RESULTS / "r5_frontier_summary.csv"
    if fpath.exists():
        frontier_figure(list(csv.DictReader(open(fpath))), ms, str(RESULTS / "fig_r5_frontier"))
        print("[frontier figure]")
    variants_figure(str(RESULTS / "fig_r5_variants"))
    print("[r5 figures written]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

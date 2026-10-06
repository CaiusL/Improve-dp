#!/usr/bin/env python
"""Round-6 figures, all driven by results/r6_runlevel_summary.csv.

(a) PushCube per-run success for B0 and V1 under fixed:8 and adaptive (strip plot + run means)
(b) the eight-cell combination chart: run-level success vs derived policy seconds per episode
(c) V1 vs B0 per-run success on each of the three tasks (fixed:8, DDPM-100)
(d) the co-trained checkpoint vs the single-task B0 runs, per task

Every figure is written as PNG and PDF. Cells that are missing are skipped with a message, so
the script is safe to run at any point during the round.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
ADAP = "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)"
FIXED8 = "fixed(k=8)"
COLOUR = {"B0": "#7f7f7f", "V1": "#1f77b4"}


def load():
    rows = list(csv.DictReader(open(RESULTS / "r6_runlevel_summary.csv")))
    for r in rows:
        r["rate"] = float(r["rate"])
        r["steps"] = int(r["steps"])
        r["train_seed"] = int(r["train_seed"]) if r["train_seed"] not in ("", "None") else None
        r["derived_policy_s_per_ep"] = float(r["derived_policy_s_per_ep"])
        r["mean_calls"] = float(r["mean_calls"])
    return rows


def pick(rows, task, arm, schedule, sampler="ddpm", steps=100):
    return sorted([r for r in rows if r["task"] == task and r["arm"] == arm
                   and r["schedule"] == schedule and r["sampler"] == sampler
                   and r["steps"] == steps], key=lambda r: r["train_seed"])


def strip_panel(ax, rows, task, schedules, title=None, ylabel=True):
    xs, labels = [], []
    for i, (arm, sched) in enumerate(schedules):
        rs = pick(rows, task, arm, sched)
        if not rs:
            continue
        vals = [100 * r["rate"] for r in rs]
        jitter = np.linspace(-0.09, 0.09, len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter([i + j for j in jitter], vals, color=COLOUR[arm], zorder=3, s=42)
        ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, color=COLOUR[arm], lw=2.5, zorder=4)
        ax.annotate(f"mean {np.mean(vals):.1f}", (i + 0.24, np.mean(vals)), fontsize=8,
                    color=COLOUR[arm], va="center")
        xs.append(i)
        labels.append(f"{arm}\n{'adaptive' if sched == ADAP else 'fixed:8'}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(70, 102)
    ax.grid(True, axis="y", alpha=0.3)
    if ylabel:
        ax.set_ylabel("success_once (%), one point per training run")
    if title:
        ax.set_title(title)
    return len(xs)


def fig_a(rows, stem):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    n = strip_panel(ax, rows, "PushCube-v1", [("B0", FIXED8), ("V1", FIXED8),
                                              ("B0", ADAP), ("V1", ADAP)],
                    title="PushCube-v1: observation normalization and adaptive replanning\n"
                          "(n=200 episodes per run, seeds 500000-500199)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"[fig a] {n} groups -> {stem}.png/.pdf")


def fig_b(rows, stem):
    cells = [("B0", FIXED8, "ddpm", 100, "default DP"),
             ("V1", FIXED8, "ddpm", 100, "V1 fixed:8"),
             ("B0", ADAP, "ddpm", 100, "B0 adaptive"),
             ("V1", ADAP, "ddpm", 100, "V1 adaptive"),
             ("B0", FIXED8, "ddim", 5, "B0 fixed:8 DDIM-5"),
             ("V1", FIXED8, "ddim", 5, "V1 fixed:8 DDIM-5"),
             ("B0", ADAP, "ddim", 5, "B0 adaptive DDIM-5"),
             ("V1", ADAP, "ddim", 5, "V1 adaptive DDIM-5")]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    plotted = 0
    for arm, sched, sampler, steps, label in cells:
        rs = pick(rows, "PushCube-v1", arm, sched, sampler, steps)
        if not rs:
            continue
        plotted += 1
        rate = 100 * float(np.mean([r["rate"] for r in rs]))
        secs = float(np.mean([r["derived_policy_s_per_ep"] for r in rs]))
        calls = float(np.mean([r["mean_calls"] for r in rs]))
        lo = 100 * min(r["rate"] for r in rs)
        hi = 100 * max(r["rate"] for r in rs)
        ax.errorbar([secs], [rate], yerr=[[rate - lo], [hi - rate]], marker="o", ms=9,
                    color=COLOUR[arm], capsize=4, ls="None",
                    fillstyle="full" if sampler == "ddpm" else "none", mew=2)
        ax.annotate(f"{label}\n{calls:.0f} calls, {secs:.2f}s", (secs, rate),
                    textcoords="offset points", xytext=(8, -14), fontsize=8,
                    color=COLOUR[arm])
    ax.set_xscale("log")
    ax.set_xlabel("derived policy seconds per episode (mean calls x round-5 ms/call)")
    ax.set_ylabel("run-level success_once (%) — mean, whiskers = run min/max")
    ax.set_ylim(60, 102)
    ax.grid(True, alpha=0.3)
    ax.set_title("PushCube-v1 combination frontier (filled = DDPM-100, open = DDIM-5)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"[fig b] {plotted}/8 cells -> {stem}.png/.pdf")


def fig_c(rows, stem):
    tasks = [("PushCube-v1", "seeds 500000-500199"), ("PickCube-v1", "seeds 600000-600199"),
             ("PullCubeTool-v1", "seeds 700000-700199")]
    have = [t for t in tasks if pick(rows, t[0], "B0", FIXED8) and pick(rows, t[0], "V1", FIXED8)]
    if not have:
        print("[fig c] skipped: no completed V1/B0 pairs yet")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 5), squeeze=False)
    for ax, (task, note) in zip(axes[0], have):
        strip_panel(ax, rows, task, [("B0", FIXED8), ("V1", FIXED8)],
                    title=f"{task}\n{note} (fixed:8, DDPM-100)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"[fig c] {len(have)} tasks -> {stem}.png/.pdf")


def fig_d(rows, stem):
    tasks = ["PushCube-v1", "PickCube-v1", "PullCubeTool-v1"]
    have = [t for t in tasks if pick(rows, t, "B0", FIXED8)
            or [r for r in rows if r["arm"] == "co" and r["task"] == t]]
    if not have:
        print("[fig d] skipped: no cells yet")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 5), squeeze=False)
    for ax, task in zip(axes[0], have):
        b0 = pick(rows, task, "B0", FIXED8)
        co = [r for r in rows if r["arm"] == "co" and r["task"] == task and r["schedule"] == FIXED8]
        vals = [100 * r["rate"] for r in b0]
        if vals:
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter([0 + j for j in jitter], vals, color=COLOUR["B0"], s=42, zorder=3)
            ax.plot([-0.2, 0.2], [np.mean(vals)] * 2, color=COLOUR["B0"], lw=2.5, zorder=4)
        if co:
            c = 100 * co[0]["rate"]
            ax.scatter([1], [c], color="#d62728", marker="D", s=60, zorder=3)
            ax.plot([0.8, 1.2], [c] * 2, color="#d62728", lw=2.5, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"single-task B0\n({len(vals)} runs)", "co-trained\n(1 checkpoint)"],
                           fontsize=9)
        ax.set_ylim(0, 102)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(f"{task} (fixed:8, DDPM-100)")
        if ax is axes[0][0]:
            ax.set_ylabel("success_once (%)")
    fig.suptitle("Co-trained checkpoint versus single-task runs — descriptive, confounded "
                 "(different trainer, batch composition and run counts)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"[fig d] {len(have)} tasks -> {stem}.png/.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default=str(RESULTS / "fig_r6"))
    args = ap.parse_args(argv)
    rows = load()
    print(f"loaded {len(rows)} run-level cells")
    fig_a(rows, f"{args.prefix}_a_pushcube_runs")
    fig_b(rows, f"{args.prefix}_b_combination")
    fig_c(rows, f"{args.prefix}_c_three_tasks")
    fig_d(rows, f"{args.prefix}_d_cotrain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
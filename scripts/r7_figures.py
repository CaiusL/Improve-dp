#!/usr/bin/env python
"""Round-7 figures, driven by results/r7_runlevel_summary.csv (plus r7_tests.csv for flags).

(a) frontier per task: success versus derived policy seconds per episode, co-trained checkpoint,
    fresh blocks, every schedule x sampler cell labelled;
(b) per-run strip plots per task: B0, V1, V2a, V2b, V3 (and V1b if present) with run means;
(c) combination charts per task: the eight cells {B0,V1} x {fixed:8, adaptive} x {DDPM-100,
    DDIM-5}, success versus derived seconds;
(d) co-trained versus single-task per task: our own co-trained runs and the teammate's checkpoint
    against the single-task B0 runs.

PNG + PDF for every figure; missing cells are skipped so the script is safe at any time.
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
ADAP = {"PickCube-v1": "adaptive_push(d_goal=0.05,d_tcp=0.0,long=8,short=2)",
        "PullCubeTool-v1": "adaptive_dist(d=0.3,long=8,short=2)",
        "PushCube-v1": "adaptive_push(d_goal=0.1,d_tcp=0.08,long=8,short=2)"}
FIXED = {"fixed(k=2)": "fixed:2", "fixed(k=4)": "fixed:4", "fixed(k=8)": "fixed:8"}
COLOUR = {"B0": "#7f7f7f", "V1": "#1f77b4", "V2a": "#2ca02c", "V2b": "#ff7f0e",
          "V3": "#9467bd", "V1b": "#8c564b", "CO": "#d62728", "co_tm": "#d62728"}


def load():
    rows = list(csv.DictReader(open(RESULTS / "r7_runlevel_summary.csv")))
    for r in rows:
        for k in ("rate", "rate_at_end", "mean_calls", "derived_policy_s_per_ep", "mean_horizon"):
            r[k] = float(r[k])
        r["steps"] = int(r["steps"])
    return rows


def sel(rows, **kw):
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def fig_frontier(rows, task, stem):
    cells = sel(rows, task=task)
    cells = [r for r in cells if r["root"].startswith("r7_frontier")]
    if not cells:
        print(f"[fig frontier {task}] skipped")
        return
    grouped = defaultdict(list)
    for r in cells:
        grouped[(r["schedule"], r["sampler"], r["steps"])].append(r)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for (sched, sampler, steps), rs in sorted(grouped.items()):
        rate = 100 * float(np.mean([r["rate"] for r in rs]))
        secs = float(np.mean([r["derived_policy_s_per_ep"] for r in rs]))
        label = FIXED.get(sched, "adaptive")
        marker = {"fixed:2": "o", "fixed:4": "s", "fixed:8": "^"}.get(label, "D")
        filled = sampler == "ddpm" and steps == 100
        ax.scatter([secs], [rate], marker=marker, s=90, color="#1f77b4" if label != "adaptive"
                   else "#9467bd", facecolors="#1f77b4" if label != "adaptive" else "#9467bd",
                   edgecolors="black", linewidths=0.5, alpha=1.0 if filled else 0.45, zorder=3)
        ax.annotate(f"{label} {sampler}-{steps}", (secs, rate), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("derived policy seconds per episode")
    ax.set_ylabel("success_once (%)")
    ax.set_title(f"{task}: sampler x horizon frontier (co-trained checkpoint, fresh block)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"[fig frontier {task}] {len(grouped)} cells -> {stem}")


def fig_variants(rows, task, stem):
    arms = ["B0", "V1", "V2a", "V2b", "V3", "V1b"]
    present = [a for a in arms if sel(rows, task=task, arm=a, schedule="fixed(k=8)",
                                      sampler="ddpm", steps=100)]
    if not present:
        print(f"[fig variants {task}] skipped")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, arm in enumerate(present):
        rs = sel(rows, task=task, arm=arm, schedule="fixed(k=8)", sampler="ddpm", steps=100)
        rates = [100 * r["rate"] for r in rs]
        jit = np.linspace(-0.09, 0.09, len(rates)) if len(rates) > 1 else [0.0]
        ax.scatter([i + j for j in jit], rates, color=COLOUR.get(arm, "k"), s=44, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(rates)] * 2, color=COLOUR.get(arm, "k"), lw=2.5,
                zorder=4)
        ax.annotate(f"{np.mean(rates):.1f}", (i + 0.24, np.mean(rates)), fontsize=8, va="center")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"{a}\n(n={len(sel(rows, task=task, arm=a, schedule='fixed(k=8)', sampler='ddpm', steps=100))})"
                        for a in present])
    ax.set_ylabel("success_once (%) — one point per training run")
    ax.set_title(f"{task}: trained variants at fixed:8, DDPM-100 (B0/V1 are brief-6 baselines)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"[fig variants {task}] {len(present)} arms -> {stem}")


def fig_combination(rows, task, stem):
    cells = []
    for arm in ("B0", "V1"):
        for tag, sched in (("fixed:8", "fixed(k=8)"), ("adaptive", ADAP[task])):
            for sampler, steps in (("ddpm", 100), ("ddim", 5)):
                rs = sel(rows, task=task, arm=arm, schedule=sched, sampler=sampler, steps=steps)
                rs = [r for r in rs if r["root"].startswith(("r6_combo", "r7_combo"))] or rs
                if rs:
                    cells.append((arm, tag, f"{sampler}-{steps}", rs))
    if not cells:
        print(f"[fig combination {task}] skipped")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for arm, tag, samp, rs in cells:
        rate = 100 * float(np.mean([r["rate"] for r in rs]))
        secs = float(np.mean([r["derived_policy_s_per_ep"] for r in rs]))
        lo = 100 * min(r["rate"] for r in rs)
        hi = 100 * max(r["rate"] for r in rs)
        ax.errorbar([secs], [rate], yerr=[[rate - lo], [hi - rate]], marker="o", ms=9,
                    color=COLOUR[arm], capsize=4, ls="None",
                    fillstyle="full" if samp.startswith("ddpm") else "none", mew=2)
        ax.annotate(f"{arm} {tag} {samp}\n{float(np.mean([r['mean_calls'] for r in rs])):.0f} calls, "
                    f"{secs:.2f}s", (secs, rate), textcoords="offset points", xytext=(7, -12),
                    fontsize=7.5, color=COLOUR[arm])
    ax.set_xscale("log")
    ax.set_xlabel("derived policy seconds per episode")
    ax.set_ylabel("run-level success_once (%) — mean, whiskers = run min/max")
    ax.set_title(f"{task}: combination cells (filled = DDPM-100, open = DDIM-5)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"[fig combination {task}] {len(cells)} cells -> {stem}")


def fig_cotrain(rows, stem):
    tasks = ["PushCube-v1", "PickCube-v1", "PullCubeTool-v1"]
    have = [t for t in tasks if sel(rows, task=t, arm="B0", schedule="fixed(k=8)", sampler="ddpm",
                                    steps=100)
            or sel(rows, task=t, arm="CO", schedule="fixed(k=8)", sampler="ddpm", steps=100)]
    if not have:
        print("[fig cotrain] skipped")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.4 * len(have), 5), squeeze=False)
    for ax, task in zip(axes[0], have):
        series = [("B0 single-task", "B0", "#7f7f7f"), ("our co-trained", "CO", "#d62728"),
                  ("teammate CUDA", "co_tm", "#ff7f0e")]
        for i, (label, arm, colour) in enumerate(series):
            rs = sel(rows, task=task, arm=arm, schedule="fixed(k=8)", sampler="ddpm", steps=100)
            if not rs:
                continue
            rates = [100 * r["rate"] for r in rs]
            jit = np.linspace(-0.08, 0.08, len(rates)) if len(rates) > 1 else [0.0]
            ax.scatter([i + j for j in jit], rates, color=colour, s=42, zorder=3)
            ax.plot([i - 0.2, i + 0.2], [np.mean(rates)] * 2, color=colour, lw=2.5, zorder=4)
            ax.annotate(f"{np.mean(rates):.1f}", (i + 0.22, np.mean(rates)), fontsize=8, va="center")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([s[0] for s in series], fontsize=8)
        ax.set_ylim(0, 102)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(task)
        if ax is axes[0][0]:
            ax.set_ylabel("success_once (%), fixed:8 / DDPM-100")
    fig.suptitle("Co-training versus single-task: run-level rates (descriptive - confounds listed "
                 "in the report)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print(f"[fig cotrain] {len(have)} tasks -> {stem}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default=str(RESULTS / "fig_r7"))
    args = ap.parse_args(argv)
    rows = load()
    print(f"loaded {len(rows)} cells")
    for task, tag in (("PickCube-v1", "pick"), ("PullCubeTool-v1", "pull")):
        fig_frontier(rows, task, f"{args.prefix}_frontier_{tag}")
        fig_variants(rows, task, f"{args.prefix}_variants_{tag}")
        fig_combination(rows, task, f"{args.prefix}_combination_{tag}")
    print("[fig cotrain] skipped: the co-training comparison was cut by the user "
          "(amendment 7a)")
    fig_variants(rows, "PushCube-v1", f"{args.prefix}_variants_pushcube")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
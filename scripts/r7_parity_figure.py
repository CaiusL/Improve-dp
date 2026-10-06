#!/usr/bin/env python
"""P8 parity figure: three task columns, one row per experiment type, each cell marked
supported / not established / contradicted, with the headline numbers as text.

The status matrix is built from the round-7 test table (results/r7_tests.csv) plus the brief-6
results; the mapping from tests to cells is explicit below so the figure cannot silently change
meaning when new data lands.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

COLOUR = {"supported": "#2ca02c", "not established": "#ffbf00",
          "contradicted": "#d62728", "n/a": "#dddddd", "descriptive": "#1f77b4"}

ROWS = [
    ("best fixed vs adaptive (co-trained checkpoint)",
     "fixed/adaptive on the teammate's checkpoint",
     {"PushCube-v1": "supported", "PickCube-v1": "contradicted", "PullCubeTool-v1": "not established"}),
    ("sampler x horizon frontier (fresh blocks)",
     "DDPM-20 / DDIM-20 vs DDPM-100 per schedule",
     {"PushCube-v1": "supported", "PickCube-v1": "spoiler", "PullCubeTool-v1": "spoiler"}),
    ("observation normalization (V1 vs B0)",
     "run-level exact permutation test",
     {"PushCube-v1": "supported", "PickCube-v1": "contradicted", "PullCubeTool-v1": "supported"}),
    ("V2a shorter prediction horizon (8)",
     "descriptive flag, 3 runs vs 4",
     {"PushCube-v1": "not established", "PickCube-v1": "spoiler", "PullCubeTool-v1": "spoiler"}),
    ("V2b longer prediction horizon (32)",
     "descriptive flag, 3 runs vs 4",
     {"PushCube-v1": "not established", "PickCube-v1": "spoiler", "PullCubeTool-v1": "spoiler"}),
    ("V3 wider encoder (256)",
     "descriptive flag, 3 runs vs 4",
     {"PushCube-v1": "not established", "PickCube-v1": "spoiler", "PullCubeTool-v1": "spoiler"}),
    ("adaptive replanning on our trained models",
     "run-level sign-flip / permutation, 4-6 runs",
     {"PushCube-v1": "contradicted", "PickCube-v1": "contradicted", "PullCubeTool-v1": "contradicted"}),
    ("combination incl. DDIM-5 on trained models",
     "best cell vs default DP, run level",
     {"PushCube-v1": "supported", "PickCube-v1": "spoiler", "PullCubeTool-v1": "spoiler"}),
    ("co-training vs single-task (multi-run)",
     "cut by the user (amendment 7a) - not analysed",
     {"PushCube-v1": "n/a", "PickCube-v1": "n/a", "PullCubeTool-v1": "n/a"}),
]
TASKS = ["PushCube-v1", "PickCube-v1", "PullCubeTool-v1"]


def load_notes():
    """Pull short numeric notes for the cells that are already computable."""
    notes = {}
    p = RESULTS / "r7_tests.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["test"].startswith("P2_") and r["valid_test"] == "False":
                notes[(r["task"], r["arm_a"])] = f"{float(r['difference'])*100:+.1f} pp ({r['note'].split('flag=')[1].split()[0]})"
            if r["test"].startswith("P4a_") and r["valid_test"] == "True":
                notes[(r["task"], "CO")] = f"{float(r['difference'])*100:+.1f} pp, p={r['p_value']}"
    return notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default=str(RESULTS / "fig_r7_parity"))
    args = ap.parse_args(argv)
    notes = load_notes()
    fig, ax = plt.subplots(figsize=(13, 7.5))
    n_rows = len(ROWS)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.axis("off")
    for i, (title, sub, status) in enumerate(ROWS):
        ax.text(-0.02, i + 0.35, title, ha="right", va="center", fontsize=9.5, weight="bold")
        ax.text(-0.02, i + 0.62, sub, ha="right", va="center", fontsize=7.5, color="#555555")
        for j, task in enumerate(TASKS):
            st = status[task]
            colour = COLOUR.get("not established" if st == "spoiler" else st, "#dddddd")
            label = {"spoiler": "pending / to be filled", "n/a": "n/a"}.get(st, st)
            ax.add_patch(Rectangle((j + 0.02, i + 0.12), 0.96, 0.76, facecolor=colour,
                                   edgecolor="white", alpha=0.85))
            ax.text(j + 0.5, i + 0.42, label, ha="center", va="center", fontsize=8.5,
                    color="white" if st in ("supported", "contradicted") else "black",
                    weight="bold")
            n = notes.get((task, "V2a" if "V2a" in title else
                           "V2b" if "V2b" in title else
                           "V3" if "V3" in title else "CO"), "")
            if n:
                ax.text(j + 0.5, i + 0.68, n, ha="center", va="center", fontsize=6.5,
                        color="white" if st in ("supported", "contradicted") else "#333333")
    for j, task in enumerate(TASKS):
        ax.text(j + 0.5, -0.35, task, ha="center", va="center", fontsize=11, weight="bold")
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c in
               ("#2ca02c", "#ffbf00", "#d62728", "#1f77b4")]
    ax.legend(handles, ["supported", "not established", "contradicted", "descriptive only"],
              loc="lower center", ncol=4, fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.04))
    ax.set_title("Parity across the three tasks: what the evidence supports, does not establish, "
                 "or contradicts", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.prefix}.{ext}", dpi=150, bbox_inches="tight")
    print(f"[parity] wrote {args.prefix}.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
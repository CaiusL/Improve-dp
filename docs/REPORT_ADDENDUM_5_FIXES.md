# REPORT_ADDENDUM_5_FIXES — consistency pass on `REPORT_ADDENDUM_5.md`

Written 2026-10-06 13:35 CST (round 7, Task P8). `REPORT_ADDENDUM_5.md` is **not edited**; this
file lists each stale statement, the corrected text, and where the corrected numbers come from.
Corrected numbers are taken from the round-6/7 run-level tables, which are regenerated from the
raw logs (`results/r7_runlevel_summary.csv`, `_arms.csv`, `r6_runlevel_mcnemar.csv`).

---

## 1. A duplicated block sits at the very top of the file (before the title)

**Stale text (lines 1-22 of the file).** The file begins with the `PickCube-v1`/`PullCubeTool-v1`
result table and its four bullets, and only then the title
`# REPORT_ADDENDUM_5.md — Round 6: ...`. Cause: a slice bug in an editing step
(`t[t.index(A):t.index(B)]` with `B` earlier than `A`) inserted the block at offset 0.

**Correction.** Delete lines 1-22; the canonical Section C carries the same table (item 5 below).
The content of the prepended block is *substantively correct* except for the two numbers in item
5, so nothing is lost.

---

## 2. Section E (co-training) — table still says "pending", compares one baseline run

**Stale text.**
```
**RESULT (co-trained side complete; single-task side partly pending).** ...
| `PickCube-v1` | 42.5% | pending (Task C) | — |
| `PullCubeTool-v1` | 7.5% | pending (Task C) | — |
```
and in the prose, `PickCube-v1` 42.5% vs "75.5% for B0s0".

**Corrected text.**
```
**RESULT (complete).** Co-trained `panda3_shared_a0`, fixed:8, DDPM-100, n = 200 per task:
| task | co-trained | single-task B0 (4 runs, mean) | B0 range |
| PushCube-v1     | 77.0% | 85.25% (6 runs) | 79.5-92.5% |
| PickCube-v1     | 42.5% | 60.38% (4 runs) | 46.0-75.5% |
| PullCubeTool-v1 |  7.5% | 11.62% (4 runs) |  9.0-15.0% |
```
Source: `results/r7_runlevel_arms.csv` (B0 means), brief-6 co-trained cells in
`results/raw/r6_cotrain_compare/`. The descriptive caveat (one co-trained checkpoint, one training
seed, different batch composition and device) is unchanged and still applies.

---

## 3. Section G.1 — co-training row uses a single baseline seed

**Stale text.** `co-trained 77.0% (PushCube-v1, vs single-task B0 mean 85.25%) and 42.5%
(PickCube-v1, vs single-task 75.5% for B0s0); PullCubeTool-v1 7.5%`

**Corrected text.** `co-trained 77.0% (PushCube-v1, vs single-task B0 mean 85.25% over 6 runs),
42.5% (PickCube-v1, vs 60.38% over 4 runs) and 7.5% (PullCubeTool-v1, vs 11.62% over 4 runs) —
single-task is ahead on all three tasks; descriptive only.`

---

## 4. Section D — two DDPM-100 adaptive rows use 4 runs, and a note says cells are unfinished

**Stale text (table rows).**
```
| B0 | adaptive | DDPM-100 | 4 | 82.2% | 75.5 | 89.0 | 89.0 | 12.90 |
| V1 | adaptive | DDPM-100 | 4 | 94.5% | 90.5 | 98.0 | 87.5 | 12.50 |
```
**Stale note.** `adaptive cells for the three round-6 B0 runs and the two round-6 V1 runs are
still being evaluated`.

**Corrected rows (all 12 adaptive cells complete).**
```
| B0 | adaptive | DDPM-100 | 6 | 83.42% | 79.5 | 89.0 | 89.0 | 12.90 |
| V1 | adaptive | DDPM-100 | 6 | 95.17% | 90.5 | 98.0 | 87.5 | 12.50 |
```
**Corrected note.** `all 24 combination cells and all 12 adaptive cells are complete; the
headline test below uses the complete 6-vs-6 cells.` Source: `r7_runlevel_arms.csv`.

---

## 5. Section C — heading and PullCubeTool numbers

**Stale text.** Heading `**RESULT — PickCube-v1 complete (4 + 4 runs); PullCubeTool-v1
pending.**`; the `PullCubeTool-v1` row `(all four above every B0 run) | 27.44% | ... |
**+15.82 pp** [+9.57, +22.07] | **0.028571**`; and the closing sentence `... and (pending) has
whatever effect PullCubeTool-v1 shows.`

**Corrected text.** Heading `**RESULT (complete, 4 + 4 runs per task).**`; the PullCubeTool row
becomes `27.50% ... **+15.88 pp** [+9.63, +22.13] | **0.028571**` (the exact permutation p is
unchanged: 2/70, the two arms separate completely); closing sentence becomes `... and on
PullCubeTool-v1 it *doubles* success, +15.88 pp at p = 0.0286 (which does not survive the
two-task alpha = 0.025)`.
Source: `results/r7_runlevel_arms.csv` (V1 mean 27.50% over 4 runs) and `results/r7_tests.csv`
row `P2_PullCubeTool-v1_V1_vs_B0` (difference +0.1588, CI [+0.0963, +0.2213]).

---

## 6. Section D — the DDIM-5 per-run cost paragraph quoted 4 runs

**Stale text.** `B0s0 +1.0 pp (p = 0.87), B0s1 +4.0 pp (p = 0.096), B0s2 -8.5 pp (p = 0.0137),
V1s0 +1.5 pp (p = 0.25)`.

**Corrected text (all 12 runs at fixed:8, DDIM-5 vs DDPM-100, exact McNemar per run).**
`B0: +1.0 (p = 0.87), +4.0 (0.096), **-8.5 (0.0137)**, +2.5 (0.52), +5.0 (0.17), +2.5 (0.53);
V1: +1.5 (0.25), +1.0 (0.73), -2.0 (0.22), +1.0 (0.84), +1.5 (0.58), +1.5 (0.51).
So 11 of 12 runs are within +-5 pp and **1 of 12 loses success significantly** (B0s2),
while the cell means are +1.1 pp (B0: 86.3 vs 85.2) and +0.8 pp (V1: 96.8 vs 96.1).`
Source: `results/r6_runlevel_mcnemar.csv` (regenerated after fixing the baseline-matching bug
described in item 8).

---

## 7. Section F — verified complete

The brief asked to check a paragraph that appeared cut off. The section is **complete and
consistent** with its table: it ends with `... its one positive result remains the teammate's
co-trained checkpoint on PushCube-v1.` The statement `smallest attainable two-sided p is
2/16 = 0.125` is correct for four paired differences (2^4 sign patterns).

---

## 8. Bug found in the analysis script while preparing this pass (not in the report)

`r6_runlevel.py` (shipped in release 1.2.0) matched the DDPM-100 baseline for the DDIM-5
comparisons with the **same** sampler/step setting, so those McNemar rows compared each cell with
itself and reported differences of exactly 0.0 pp (p = 1.0) for all 12 runs. The bug was
introduced by a refactor during round 6 (an inverted `same_sched` condition). Fixed here: the
baseline is always the same run's DDPM-100 cell. The corrected rows reproduce the values quoted
in `REPORT_ADDENDUM_5.md` §D (e.g. B0s2 -8.5 pp, p = 0.0137) and extend them to all 12 runs.
The fixed script is shipped in release 1.3.0.

---

## 9. Section B — the last two per-run differences in each row are stale

**Stale text (table, §B "RESULT").**
```
| B0 | +5.0, -11.0, -6.5, -5.5, +1.1, -0.5 pp | -1.83 pp | 0.5625 | ... |
| V1 | -1.5, -5.0, -1.5, -1.0, -0.5, +1.0 pp | -0.92 pp | 0.5000 | ... |
```
The printed seed-4/5 values do not even reproduce the row's own mean: B0 +5.0-11.0-6.5-5.5+1.1-0.5
gives -2.90 pp, and V1 gives -1.42 pp, against the -1.83 / -0.92 pp printed in the same row.
Cause: the first four entries were recomputed after a data fix but the last two were left from an
earlier pass.

**Corrected text (per-run adaptive-minus-`fixed:8`, n = 200 per run, exact McNemar).**
```
| B0 | +5.0, -11.0, -6.5, -5.5, +3.5, +3.5 pp | -1.83 pp [-6.17, +2.42] | 0.5625 | B0s1, B0s2 | none |
| V1 | -1.5, -5.0, -1.5, -1.0, +3.0, +0.5 pp | -0.92 pp [-4.08, +2.17] | 0.5000 | V1s1 | none |
```
Source: `results/r7_runlevel_summary.csv` (per-run rates, same runs and seeds 500000-500199) —
B0 `fixed:8` seeds 0-5 = 84.0/92.5/89.5/85.0/81.0/79.5 vs adaptive 89.0/81.5/83.0/79.5/84.5/83.0;
V1 `fixed:8` = 98.5/97.5/99.5/91.5/94.0/95.5 vs adaptive 97.0/92.5/98.0/90.5/97.0/96.0. Every
other §B number (means, sign-flip p = 0.5625/0.5000, the two significant losses, B3 +9.92 pp
p = 0.006494, interaction +0.92 pp p = 0.772727) is unchanged and reproduces exactly.

---

## 10. Section F — one mean difference is rounded down by 0.01 pp

**Stale text.** `| PickCube-v1 | V1 | 4 | **-5.87 pp** | ... |`
**Corrected text.** `-5.88 pp` (exact -5.875 pp; the other three rows are unchanged at the
precision printed: -14.75, -0.13, -7.62/-7.63 pp). Addendum 6 states its rounding convention
(half-up); the earlier reports are mixed, which is why two tie values differ by 0.01 pp.

---

## 11. Round-7 script defects found in a later review (not addendum-5 statements)

Two defects in the round-7 pipeline were found after this pass and are fixed in release 1.3.2:
`r7_runlevel.py` looked the clean per-call time up under the arm name `co` instead of `co_tm`, so
every co-trained frontier cell used 145.0 ms and the DDPM-20/DDIM-20 `derived s/ep` values were
~4x too large; and `pack_final_results.py` did not fold the round-5 frontier's reused sweep half
into its three `r5_front_*` `fixed:8` rows, which shipped at n = 100 instead of 200. Details and
the regenerated values are in `REPORT_ADDENDUM_6.md` (revision note 2) and `NOTES.md`
(`CORRECTION`, round 7 post-release).
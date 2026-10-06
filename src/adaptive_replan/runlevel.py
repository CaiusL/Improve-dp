#!/usr/bin/env python
"""Run-level statistics for round 6: the training run is the unit of analysis.

Three tools, all exact where an exact test is possible:

* ``permutation_test`` — exact two-sided permutation test on two independent groups of
  run-level rates. Enumerates every way of labelling ``n1`` of the ``n1+n2`` runs as group A
  (``C(n1+n2, n1)`` assignments) and counts how many give a difference at least as extreme as
  the observed one. 6 vs 6 -> 924 assignments; 4 vs 4 -> 70 (smallest attainable p = 2/70).
* ``signflip_test`` — exact two-sided sign-flip (paired) permutation test on per-run
  differences: all ``2**n`` sign patterns, statistic = mean of signed differences.
* ``bootstrap_diff_ci`` — percentile bootstrap interval for the difference in means
  (resample each group with replacement), fixed default seed for reproducibility.

Episodes are never pooled across training runs here: the inputs are one rate per run.
"""
from __future__ import annotations

import itertools

import numpy as np

DEFAULT_BOOTSTRAP_SEED = 20261006
DEFAULT_BOOTSTRAP_N = 20000


def _as_float(a):
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty group")
    return arr


def permutation_test(group_a, group_b, tol: float = 1e-12):
    """Exact two-sided permutation test on two independent groups of run-level rates.

    Returns (p_value, observed_difference, n_permutations).
    """
    a, b = _as_float(group_a), _as_float(group_b)
    n1, n2 = a.size, b.size
    combined = np.concatenate([a, b])
    observed = float(a.mean() - b.mean())
    total = 0
    hits = 0
    for idx in itertools.combinations(range(n1 + n2), n1):
        mask = np.zeros(n1 + n2, dtype=bool)
        mask[list(idx)] = True
        diff = float(combined[mask].mean() - combined[~mask].mean())
        total += 1
        if abs(diff) >= abs(observed) - tol:
            hits += 1
    return hits / total, observed, total


def signflip_test(differences, tol: float = 1e-12):
    """Exact two-sided sign-flip permutation test on per-run paired differences.

    Returns (p_value, observed_mean_difference, n_patterns).
    """
    d = _as_float(differences)
    n = d.size
    observed = float(d.mean())
    total = 0
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        stat = float(np.mean(np.asarray(signs) * d))
        total += 1
        if abs(stat) >= abs(observed) - tol:
            hits += 1
    return hits / total, observed, total


def bootstrap_diff_ci(group_a, group_b, n_boot: int = DEFAULT_BOOTSTRAP_N,
                      seed: int = DEFAULT_BOOTSTRAP_SEED, alpha: float = 0.05):
    """Percentile bootstrap interval for mean(A) - mean(B). Method: resample each group with
    replacement ``n_boot`` times at fixed seed ``seed`` and take the alpha/2 and 1-alpha/2
    percentiles."""
    a, b = _as_float(group_a), _as_float(group_b)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(a.mean() - b.mean())


def summarize_arm(rates):
    """Mean, sd (ddof=1), min, max and range of an arm's run-level rates."""
    r = _as_float(rates)
    return {
        "n_runs": int(r.size),
        "mean": float(r.mean()),
        "sd": float(r.std(ddof=1)) if r.size > 1 else float("nan"),
        "min": float(r.min()),
        "max": float(r.max()),
        "range": float(r.max() - r.min()),
    }

def descriptive_flag(rates, baseline_rates):
    """Descriptive flag used when a design cannot reach significance by construction.

    "suggestive" iff the arm's mean run-level rate lies outside the baseline's min-max range
    *and* every one of the arm's runs falls on the same side of the baseline mean; otherwise
    "within training noise".
    """
    r = _as_float(rates)
    b = _as_float(baseline_rates)
    outside = bool(r.mean() > b.max() or r.mean() < b.min())
    same_side = bool(np.all(r > b.mean()) or np.all(r < b.mean()))
    return "suggestive" if (outside and same_side) else "within training noise"


"""Statistics used throughout the study: intervals, paired tests, power."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, norm

Z95 = 1.959963984540054


def wilson_interval(k, n, z=Z95):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def mcnemar_exact(a_success, b_success):
    """Exact McNemar test on paired booleans.

    Returns (A_only, B_only, p) where A_only counts pairs where A succeeded and B did not.
    """
    n10 = sum(1 for x, y in zip(a_success, b_success) if x and not y)
    n01 = sum(1 for x, y in zip(a_success, b_success) if (not x) and y)
    if n10 + n01 == 0:
        return n10, n01, 1.0
    return n10, n01, float(binomtest(n10, n10 + n01, 0.5).pvalue)


def newcombe_paired_diff(a_success, b_success, z=Z95):
    """95% Newcombe hybrid-score interval for a paired difference of proportions.

    Returns (difference, lo, hi).
    """
    n = len(a_success)
    p1 = sum(a_success) / n
    p2 = sum(b_success) / n
    l1, u1 = wilson_interval(sum(a_success), n, z)
    l2, u2 = wilson_interval(sum(b_success), n, z)
    lo = p1 - p2 - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = p1 - p2 + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return p1 - p2, float(max(-1.0, lo)), float(min(1.0, hi))


def bonferroni_alpha(n_tests, family_alpha=0.05):
    """Bonferroni-corrected significance level for a family of tests."""
    return family_alpha / n_tests if n_tests else float("nan")


def n_unpaired(p1, p2, alpha=0.05, power=0.80):
    """Episodes per arm for a two-proportion z-test."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def n_paired(psi, p1, p2, alpha=0.05, power=0.80):
    """Seed pairs for an exact McNemar test given the discordance rate psi."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    delta = abs(p2 - p1)
    var = psi - delta ** 2
    if var <= 0:
        return None
    return int(np.ceil((za * np.sqrt(psi) + zb * np.sqrt(var)) ** 2 / delta ** 2))

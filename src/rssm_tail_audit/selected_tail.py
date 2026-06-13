"""Finite selected-tail estimator used by this RSSM latent-dynamics repo.

This module is the intentionally abstract part shared with the prior WAM line
of work: given a finite candidate pool with score ``S`` and measured utility
``R``, top-score candidate selection has an exact empirical expectation. The
scientific object in this repository is different: the scores are RSSM-style
latent imagination scores while utilities are measured by executing selected
actions in hidden-mode dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TieGroup:
    """One score tie group in ascending score order."""

    score: float
    start: int
    stop: int
    r_min: int
    r_max: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def _as_1d_float(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_n_values(n_values: Iterable[int]) -> list[int]:
    out = [int(n) for n in n_values]
    if not out:
        raise ValueError("n_values must be non-empty")
    if any(n < 1 for n in out):
        raise ValueError("all N values must be >= 1")
    return out


def sorted_tie_groups(scores: Iterable[float]) -> tuple[np.ndarray, list[TieGroup]]:
    """Return a stable ascending score order and its tie groups."""

    scores_arr = _as_1d_float(scores, "scores")
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    groups: list[TieGroup] = []
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        groups.append(TieGroup(float(sorted_scores[i]), i, j, i + 1, j))
        i = j
    return order, groups


def utility_tail_selection_finite(
    scores: Iterable[float],
    utilities: Iterable[float],
    n_values: Iterable[int],
) -> dict[int, float]:
    """Expected real-valued utility of top-score candidate selection from a finite pool.

    Sampling is with replacement. If several sampled candidates share the
    maximum score, the selected candidate is drawn uniformly from the tied
    sampled top-score candidates, which equals using that score group's mean
    utility in the rank-interval formula.
    """

    scores_arr = _as_1d_float(scores, "scores")
    utilities_arr = _as_1d_float(utilities, "utilities")
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same length")
    ns = _as_n_values(n_values)

    m = len(scores_arr)
    order, groups = sorted_tie_groups(scores_arr)
    sorted_utilities = utilities_arr[order]
    out: dict[int, float] = {}
    for N in ns:
        expected = 0.0
        for group in groups:
            mass = (group.r_max / m) ** N - ((group.r_min - 1) / m) ** N
            expected += float(np.mean(sorted_utilities[group.start : group.stop])) * mass
        out[N] = float(expected)
    return out


def binary_tail_selection_finite(
    scores: Iterable[float],
    success: Iterable[bool | int | float],
    n_values: Iterable[int],
) -> dict[int, float]:
    """Exact finite candidate-budget selection success probability for binary utility."""

    success_arr = _as_1d_float(success, "success")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must be binary")
    return utility_tail_selection_finite(scores, success_arr, n_values)


def auc_kappa(scores: Iterable[float], success: Iterable[bool | int | float]) -> float:
    """Tie-aware AUC: P(S+ > S-) + 0.5 P(S+ = S-)."""

    scores_arr = _as_1d_float(scores, "scores")
    success_arr = _as_1d_float(success, "success")
    if scores_arr.shape != success_arr.shape:
        raise ValueError("scores and success must have the same length")
    if not np.all((success_arr == 0.0) | (success_arr == 1.0)):
        raise ValueError("success must be binary")
    pos = scores_arr[success_arr == 1.0]
    neg = scores_arr[success_arr == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for s in pos:
        wins += float(np.sum(s > neg)) + 0.5 * float(np.sum(s == neg))
    return float(wins / (len(pos) * len(neg)))


def n2_auc_identity(p: float, kappa: float) -> float:
    """The binary N=2 identity f_2 = p^2 + 2p(1-p)kappa."""

    p = float(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite for 0 < p < 1")
    return float(p * p + 2.0 * p * (1.0 - p) * float(kappa))


def tie_rate(scores: Iterable[float]) -> float:
    """Fraction of score pairs that are tied."""

    scores_arr = _as_1d_float(scores, "scores")
    n = len(scores_arr)
    if n < 2:
        return 0.0
    _, counts = np.unique(scores_arr, return_counts=True)
    tied_pairs = sum(int(c) * (int(c) - 1) / 2 for c in counts)
    return float(tied_pairs / (n * (n - 1) / 2))


def simulate_tail_selection(
    scores: Iterable[float],
    utilities: Iterable[float],
    N: int,
    n_trials: int = 10_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo validation helper with explicit uniform tie handling."""

    scores_arr = _as_1d_float(scores, "scores")
    utilities_arr = _as_1d_float(utilities, "utilities")
    if scores_arr.shape != utilities_arr.shape:
        raise ValueError("scores and utilities must have the same length")
    if int(N) < 1:
        raise ValueError("N must be >= 1")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores_arr), size=(int(n_trials), int(N)))
    drawn_scores = scores_arr[idx]
    max_scores = np.max(drawn_scores, axis=1)
    selected = np.empty(int(n_trials), dtype=int)
    for row in range(int(n_trials)):
        tied_positions = np.flatnonzero(drawn_scores[row] == max_scores[row])
        selected_position = rng.choice(tied_positions)
        selected[row] = idx[row, selected_position]
    return float(np.mean(utilities_arr[selected]))

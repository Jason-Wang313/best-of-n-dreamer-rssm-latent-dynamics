"""Metrics for selected latent value, real utility, and tail diagnostics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .envs import RolloutRecord
from .theorem import utility_best_of_n_finite


def as_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def bootstrap_ci(
    values: Iterable[float],
    seed: int = 0,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval."""

    arr = as_array(values)
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": float(np.mean(arr)),
        "lo": float(lo),
        "hi": float(hi),
        "n": float(len(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def selection_curves(
    records: list[RolloutRecord],
    scores: Iterable[float],
    n_values: Iterable[int],
) -> list[dict[str, float | int]]:
    """Compute exact finite selected curves for several measured quantities."""

    scores_arr = as_array(scores)
    real = [r.real_utility for r in records]
    latent = [r.latent_value for r in records]
    pred = [r.value_pred for r in records]
    risk = [r.risk for r in records]
    real_curve = utility_best_of_n_finite(scores_arr, real, n_values)
    latent_curve = utility_best_of_n_finite(scores_arr, latent, n_values)
    pred_curve = utility_best_of_n_finite(scores_arr, pred, n_values)
    risk_curve = utility_best_of_n_finite(scores_arr, risk, n_values)
    return [
        {
            "N": int(N),
            "selected_real_utility": float(real_curve[int(N)]),
            "selected_latent_value": float(latent_curve[int(N)]),
            "selected_value_pred": float(pred_curve[int(N)]),
            "selected_risk": float(risk_curve[int(N)]),
            "latent_real_gap": float(latent_curve[int(N)] - real_curve[int(N)]),
        }
        for N in n_values
    ]


def top_tail_diagnostics(
    records: list[RolloutRecord],
    scores: Iterable[float],
    top_fraction: float = 0.10,
) -> dict[str, float]:
    """Diagnostics on the selected-score upper tail."""

    scores_arr = as_array(scores)
    k = max(1, int(np.ceil(len(scores_arr) * float(top_fraction))))
    order = np.argsort(scores_arr, kind="mergesort")
    tail_idx = order[-k:]
    real = np.asarray([records[i].real_utility for i in tail_idx], dtype=float)
    latent = np.asarray([records[i].latent_value for i in tail_idx], dtype=float)
    uncertainty = np.asarray([records[i].uncertainty for i in tail_idx], dtype=float)
    blocked = np.asarray([records[i].hidden_mode != "free" for i in tail_idx], dtype=float)
    imagined_free = np.asarray([records[i].imagined_free_prob for i in tail_idx], dtype=float)
    all_real = np.asarray([r.real_utility for r in records], dtype=float)
    all_latent = np.asarray([r.latent_value for r in records], dtype=float)
    return {
        "top_fraction": float(top_fraction),
        "tail_real_mean": float(np.mean(real)),
        "tail_latent_mean": float(np.mean(latent)),
        "tail_gap": float(np.mean(latent) - np.mean(real)),
        "tail_uncertainty_mean": float(np.mean(uncertainty)),
        "tail_blocked_rate": float(np.mean(blocked)),
        "tail_imagined_free_prob": float(np.mean(imagined_free)),
        "population_real_mean": float(np.mean(all_real)),
        "population_latent_mean": float(np.mean(all_latent)),
    }


def high_n_delta(curves: list[dict[str, float | int]], key: str) -> float:
    """Return last minus first for a curve key."""

    if not curves:
        raise ValueError("curves must be non-empty")
    return float(curves[-1][key]) - float(curves[0][key])

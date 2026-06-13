"""Experiment J: posterior-prior belief intervention stress.

This experiment asks whether the RSSM-specific belief diagnostics are doing
mechanistic work, rather than merely decorating a generic selected-tail curve.
It tests high-risk hidden-mode regimes where posterior-prior drift and optimistic
free-mode belief should identify raw high-N failures before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rssm_tail_audit.envs import HiddenModeConfig, HiddenModeToyEnv, RolloutRecord
from rssm_tail_audit.metrics import bootstrap_ci, selection_curves
from rssm_tail_audit.scorers import PilotCalibrator, fit_pilot_calibrator, score_records

from experiments.common import (
    ensure_dirs,
    figures_dir,
    results_dir,
    root_from_file,
    smoke_argparser,
    tables_dir,
    write_json,
    write_rows,
)


N_GRID = [1, 4, 16, 64]


@dataclass(frozen=True)
class InterventionRegime:
    name: str
    blocked_prob: float
    clue_strength: float
    horizon: int


REGIMES = [
    InterventionRegime("ambiguous_blocked", blocked_prob=0.70, clue_strength=0.08, horizon=5),
    InterventionRegime("long_horizon_slip", blocked_prob=0.72, clue_strength=0.10, horizon=8),
    InterventionRegime("high_ambiguity", blocked_prob=0.76, clue_strength=0.04, horizon=6),
    InterventionRegime("decoder_drift", blocked_prob=0.68, clue_strength=0.06, horizon=7),
]


def _arr(records: Iterable[RolloutRecord], attr: str) -> np.ndarray:
    return np.asarray([float(getattr(r, attr)) for r in records], dtype=float)


def _tail_corr(records: list[RolloutRecord], raw_scores: np.ndarray) -> float:
    k = max(1, int(np.ceil(0.10 * len(records))))
    tail_idx = np.argsort(raw_scores, kind="mergesort")[-k:]
    drift = _arr(records, "posterior_prior_kl")[tail_idx] + _arr(records, "belief_error")[tail_idx]
    gap = (_arr(records, "latent_value") - _arr(records, "real_utility"))[tail_idx]
    if float(np.std(drift)) == 0.0 or float(np.std(gap)) == 0.0:
        return 0.0
    return float(np.corrcoef(drift, gap)[0, 1])


def intervention_scores(
    records: list[RolloutRecord],
    calibrator: PilotCalibrator,
) -> dict[str, np.ndarray]:
    """Scores for the raw, diagnostic-only, full repair, and oracle interventions."""

    raw = score_records(records, "raw_value", calibrator=calibrator)
    posterior_prior = _arr(records, "posterior_prior_kl")
    belief_error = _arr(records, "belief_error")
    return {
        "raw_value": raw,
        "belief_drift_penalty": raw - 2.75 * (posterior_prior + 0.5 * belief_error),
        "pilot_plus_diagnostics": score_records(records, "combined_repair", calibrator=calibrator),
        "oracle": score_records(records, "oracle", calibrator=calibrator),
    }


def _plot(rows: list[dict[str, object]], unit_rows: list[dict[str, object]], output) -> None:
    df = pd.DataFrame(rows)
    units = pd.DataFrame(unit_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), dpi=160)

    n64 = df[df["N"] == 64].copy()
    order = ["raw_value", "belief_drift_penalty", "pilot_plus_diagnostics", "oracle"]
    labels = ["raw", "belief\ndrift", "pilot+\ndiag", "oracle"]
    means = [float(n64[n64["scorer"] == scorer]["selected_real_utility"].mean()) for scorer in order]
    lows = []
    highs = []
    for scorer in order:
        ci = bootstrap_ci(n64[n64["scorer"] == scorer]["selected_real_utility"], seed=200 + len(scorer), n_boot=600)
        lows.append(means[order.index(scorer)] - ci["lo"])
        highs.append(ci["hi"] - means[order.index(scorer)])
    axes[0].bar(labels, means, yerr=[lows, highs], color=["#a23b3b", "#bc6c25", "#1a8f5a", "#202020"], alpha=0.88)
    axes[0].set_ylabel("N=64 selected real utility")
    axes[0].set_title("Belief interventions")
    axes[0].grid(True, axis="y", color="#dddddd", linewidth=0.7)

    effects = [
        ("raw high-N\nchange", units["raw_real_delta_high_n"]),
        ("belief drift\nminus raw", units["belief_penalty_minus_raw_n64"]),
        ("pilot+diag\nminus raw", units["full_minus_raw_n64"]),
        ("oracle\nminus full", units["oracle_minus_full_n64"]),
    ]
    effect_means = [float(np.mean(values)) for _, values in effects]
    effect_lows = []
    effect_highs = []
    for i, (_, values) in enumerate(effects):
        ci = bootstrap_ci(values, seed=300 + i, n_boot=600)
        effect_lows.append(effect_means[i] - ci["lo"])
        effect_highs.append(ci["hi"] - effect_means[i])
    axes[1].bar(
        [label for label, _ in effects],
        effect_means,
        yerr=[effect_lows, effect_highs],
        color=["#a23b3b", "#bc6c25", "#1a8f5a", "#4a4a4a"],
        alpha=0.88,
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.9)
    axes[1].set_ylabel("Seed-regime effect")
    axes[1].set_title("Exact selected-tail effects")
    axes[1].grid(True, axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run(smoke: bool = False, seed: int = 9):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    regimes = REGIMES[:2] if smoke else REGIMES
    seeds = range(2) if smoke else range(5)
    n_candidates = 220 if smoke else 600
    pilot_size = 70 if smoke else 180

    rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    for seed_idx in seeds:
        for regime in regimes:
            env = HiddenModeToyEnv(
                HiddenModeConfig(
                    blocked_prob=regime.blocked_prob,
                    clue_strength=regime.clue_strength,
                    observation_noise=0.08,
                )
            )
            pool_seed = int(seed + 101 + 17 * seed_idx + len(regime.name))
            records = env.generate_candidate_pool(
                n=n_candidates,
                horizon=regime.horizon,
                seed=pool_seed,
                model_flavor="belief_collapsed",
                state_id=seed_idx,
            )
            rng = np.random.default_rng(seed + seed_idx)
            pilot_idx = rng.permutation(len(records))[: min(pilot_size, len(records))]
            calibrator = fit_pilot_calibrator([records[i] for i in pilot_idx])
            scores = intervention_scores(records, calibrator)
            raw_corr = _tail_corr(records, scores["raw_value"])

            curves_by_scorer: dict[str, list[dict[str, float | int]]] = {}
            for scorer, scorer_scores in scores.items():
                curves = selection_curves(records, scorer_scores, N_GRID)
                curves_by_scorer[scorer] = curves
                for curve in curves:
                    rows.append(
                        {
                            "seed": seed_idx,
                            "regime": regime.name,
                            "scorer": scorer,
                            "tail_drift_gap_corr": raw_corr,
                            **curve,
                        }
                    )

            raw = curves_by_scorer["raw_value"]
            belief = curves_by_scorer["belief_drift_penalty"]
            full = curves_by_scorer["pilot_plus_diagnostics"]
            oracle = curves_by_scorer["oracle"]
            unit_rows.append(
                {
                    "seed": seed_idx,
                    "regime": regime.name,
                    "horizon": regime.horizon,
                    "blocked_prob": regime.blocked_prob,
                    "clue_strength": regime.clue_strength,
                    "tail_drift_gap_corr": raw_corr,
                    "raw_real_delta_high_n": float(raw[-1]["selected_real_utility"] - raw[0]["selected_real_utility"]),
                    "raw_latent_delta_high_n": float(raw[-1]["selected_latent_value"] - raw[0]["selected_latent_value"]),
                    "belief_penalty_minus_raw_n64": float(
                        belief[-1]["selected_real_utility"] - raw[-1]["selected_real_utility"]
                    ),
                    "full_minus_raw_n64": float(full[-1]["selected_real_utility"] - raw[-1]["selected_real_utility"]),
                    "oracle_minus_full_n64": float(
                        oracle[-1]["selected_real_utility"] - full[-1]["selected_real_utility"]
                    ),
                    "pilot_size": len(pilot_idx),
                    "n_candidates": len(records),
                }
            )

    write_rows(tables_dir(root, smoke) / "experiment_j_belief_interventions.csv", rows)
    write_rows(tables_dir(root, smoke) / "experiment_j_belief_intervention_units.csv", unit_rows)
    _plot(rows, unit_rows, figures_dir(root, smoke) / "figure10_belief_interventions.png")

    def ci_for(key: str, offset: int) -> dict[str, float]:
        return bootstrap_ci([float(row[key]) for row in unit_rows], seed=seed + offset, n_boot=1000)

    raw_harmful_count = sum(1 for row in unit_rows if float(row["raw_real_delta_high_n"]) < 0.0)
    summary = {
        "experiment": "J_belief_intervention_stress",
        "smoke": smoke,
        "n_seed_regime_units": len(unit_rows),
        "n_records_per_seed_regime": n_candidates,
        "pilot_size": pilot_size,
        "n_values": N_GRID,
        "regimes": [regime.__dict__ for regime in regimes],
        "claim": "Posterior-prior and belief-collapse diagnostics identify raw selected-tail failure before execution and recover utility under bounded CPU experiments.",
        "key_result": {
            "raw_harmful_seed_regime_count": raw_harmful_count,
            "raw_harmful_fraction": raw_harmful_count / max(1, len(unit_rows)),
            "raw_real_delta_high_n_ci": ci_for("raw_real_delta_high_n", 1),
            "raw_latent_delta_high_n_ci": ci_for("raw_latent_delta_high_n", 2),
            "belief_penalty_minus_raw_n64_ci": ci_for("belief_penalty_minus_raw_n64", 3),
            "full_minus_raw_n64_ci": ci_for("full_minus_raw_n64", 4),
            "oracle_minus_full_n64_ci": ci_for("oracle_minus_full_n64", 5),
            "tail_drift_gap_corr_ci": ci_for("tail_drift_gap_corr", 6),
        },
    }
    write_json(results_dir(root, smoke) / "experiment_j_belief_interventions.json", summary)
    return summary


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

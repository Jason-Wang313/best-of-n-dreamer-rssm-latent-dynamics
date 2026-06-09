"""Experiment H: OOD hidden-mode stress grid."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latent_dynamics_best_of_n.envs import HiddenModeConfig, HiddenModeToyEnv
from latent_dynamics_best_of_n.leakage import deterministic_split
from latent_dynamics_best_of_n.metrics import selection_curves
from latent_dynamics_best_of_n.scorers import fit_pilot_calibrator, score_records

from experiments.common import ensure_dirs, figures_dir, results_dir, root_from_file, smoke_argparser, tables_dir, write_json


P_GRID = [0.45, 0.60, 0.75, 0.85]
CLUE_GRID = [0.05, 0.10, 0.18]
HORIZON_GRID = [3, 5, 8]
N_GRID = [1, 8, 64]


def classify_regime(raw_delta_n64_vs_n1: float) -> str:
    if raw_delta_n64_vs_n1 < -0.25:
        return "harm"
    if raw_delta_n64_vs_n1 > 0.25:
        return "helpful"
    return "neutral"


def _plot(regime_rows: list[dict[str, object]], output) -> None:
    df = pd.DataFrame(regime_rows)
    fig, axes = plt.subplots(1, len(HORIZON_GRID), figsize=(10.8, 3.8), dpi=150, sharey=True)
    colors = {"harm": "#b23b3b", "neutral": "#777777", "helpful": "#1a8f5a"}
    for ax, horizon in zip(axes, HORIZON_GRID):
        sub = df[df["horizon"] == horizon]
        for regime, group in sub.groupby("regime"):
            ax.scatter(
                group["clue_strength"],
                group["hidden_mode_probability"],
                s=95,
                marker="s",
                color=colors[str(regime)],
                label=str(regime),
                alpha=0.88,
            )
        ax.set_title(f"H={horizon}")
        ax.set_xlabel("Clue strength")
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Hidden-mode probability")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run(smoke: bool = False, seed: int = 8):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    p_grid = [0.45, 0.75] if smoke else P_GRID
    clue_grid = [0.05, 0.18] if smoke else CLUE_GRID
    horizon_grid = [3, 5] if smoke else HORIZON_GRID
    rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    for p in p_grid:
        for clue in clue_grid:
            for horizon in horizon_grid:
                combo_seed = seed + int(p * 1000) + int(clue * 10000) + horizon * 17
                env = HiddenModeToyEnv(
                    HiddenModeConfig(blocked_prob=float(p), clue_strength=float(clue), observation_noise=0.08)
                )
                records = env.generate_candidate_pool(
                    n=360 if smoke else 920,
                    horizon=horizon,
                    seed=combo_seed,
                    model_flavor="belief_collapsed",
                    state_id=horizon,
                )
                pilot_idx, eval_idx = deterministic_split(
                    len(records),
                    pilot_size=110 if smoke else 320,
                    seed=combo_seed + 13,
                )
                calibrator = fit_pilot_calibrator([records[int(i)] for i in pilot_idx])
                eval_records = [records[int(i)] for i in eval_idx]
                curve_by_scorer = {}
                for scorer in ["raw_value", "combined_repair", "oracle"]:
                    scores = score_records(eval_records, scorer, calibrator=calibrator)
                    curves = selection_curves(eval_records, scores, N_GRID)
                    curve_by_scorer[scorer] = curves
                    for curve in curves:
                        rows.append(
                            {
                                "hidden_mode_probability": float(p),
                                "clue_strength": float(clue),
                                "horizon": int(horizon),
                                "scorer": scorer,
                                **curve,
                            }
                        )
                raw_n1 = float(curve_by_scorer["raw_value"][0]["selected_real_utility"])
                raw_n64 = float(curve_by_scorer["raw_value"][-1]["selected_real_utility"])
                repair_n64 = float(curve_by_scorer["combined_repair"][-1]["selected_real_utility"])
                regime = classify_regime(raw_n64 - raw_n1)
                high_risk = bool(p >= 0.75 and clue <= 0.10)
                regime_rows.append(
                    {
                        "hidden_mode_probability": float(p),
                        "clue_strength": float(clue),
                        "horizon": int(horizon),
                        "regime": regime,
                        "raw_n64_minus_n1_real": float(raw_n64 - raw_n1),
                        "combined_repair_n64_minus_raw_n64": float(repair_n64 - raw_n64),
                        "high_risk": high_risk,
                    }
                )

    pd.DataFrame(rows).to_csv(tables_dir(root, smoke) / "experiment_h_ood_stress_grid.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(tables_dir(root, smoke) / "experiment_h_ood_regimes.csv", index=False)
    counts = {name: sum(1 for row in regime_rows if row["regime"] == name) for name in ["harm", "neutral", "helpful"]}
    high_risk_gains = [
        float(row["combined_repair_n64_minus_raw_n64"]) for row in regime_rows if bool(row["high_risk"])
    ]
    payload = {
        "experiment": "H_ood_stress_grid",
        "smoke": bool(smoke),
        "hidden_mode_probability_grid": p_grid,
        "clue_strength_grid": clue_grid,
        "horizon_grid": horizon_grid,
        "n_values": N_GRID,
        "regime_counts": counts,
        "regimes": regime_rows,
        "key_result": {
            "raw_high_n_harm_regions": int(counts["harm"]),
            "raw_high_n_neutral_or_helpful_regions": int(counts["neutral"] + counts["helpful"]),
            "combined_repair_mean_gain_high_risk_regions": float(np.mean(high_risk_gains)) if high_risk_gains else 0.0,
        },
    }
    write_json(results_dir(root, smoke) / "experiment_h_ood_stress_grid.json", payload)
    _plot(regime_rows, figures_dir(root, smoke) / "figure8_ood_stress_grid.png")
    return payload


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

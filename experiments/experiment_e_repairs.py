"""Experiment E: repair comparisons for selected-tail real utility."""

from __future__ import annotations

from latent_dynamics_best_of_n.envs import HiddenModeConfig, HiddenModeToyEnv

from experiments.common import (
    N_GRID,
    ensure_dirs,
    evaluate_scorers,
    figures_dir,
    plot_curve,
    results_dir,
    root_from_file,
    smoke_argparser,
    tables_dir,
    write_json,
)


def run(smoke: bool = False, seed: int = 4):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.70, clue_strength=0.10, observation_noise=0.09))
    records = env.generate_candidate_pool(
        n=560 if smoke else 1800,
        horizon=5,
        seed=seed,
        model_flavor="belief_collapsed",
    )
    scorers = [
        "raw_value",
        "uncertainty_pessimism",
        "ensemble_uncertainty_repair",
        "belief_consistency",
        "decoder_consistency",
        "pilot_calibrated",
        "combined_repair",
        "random",
        "oracle",
    ]
    rows, summary = evaluate_scorers(records, scorers, N_GRID, pilot_size=220 if smoke else 1000, seed=seed)
    import pandas as pd

    pd.DataFrame(rows).to_csv(tables_dir(root, smoke) / "experiment_e_repairs.csv", index=False)
    raw = summary["scorers"]["raw_value"]
    repair = summary["scorers"]["combined_repair"]
    oracle = summary["scorers"]["oracle"]
    summary.update(
        {
            "experiment": "E_strong_repair",
            "n_records": len(records),
            "key_result": {
                "combined_repair_n64_real_improvement_over_raw": repair["N64_real"] - raw["N64_real"],
                "combined_repair_oracle_gap_n64": oracle["N64_real"] - repair["N64_real"],
                "combined_repair_fraction_of_oracle_gap_closed": (repair["N64_real"] - raw["N64_real"])
                / max(1e-9, oracle["N64_real"] - raw["N64_real"]),
            },
        }
    )
    write_json(results_dir(root, smoke) / "experiment_e_repairs.json", summary)
    plot_curve(
        rows,
        figures_dir(root, smoke) / "figure2_repair_comparison.png",
        [
            "raw_value",
            "uncertainty_pessimism",
            "ensemble_uncertainty_repair",
            "belief_consistency",
            "decoder_consistency",
            "pilot_calibrated",
            "combined_repair",
            "oracle",
        ],
        ["selected_real_utility"],
        "Repair scorers recover high-N executed utility",
        "Expected selected real utility",
    )
    return summary


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

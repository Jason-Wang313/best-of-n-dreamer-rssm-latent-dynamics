"""Experiment D: horizon and selection budget mismatch."""

from __future__ import annotations

from latent_dynamics_best_of_n.envs import HiddenModeConfig, HiddenModeToyEnv

from experiments.common import N_GRID, ensure_dirs, evaluate_scorers, plot_curve, root_from_file, smoke_argparser, write_json


def run(smoke: bool = False, seed: int = 3):
    root = root_from_file()
    ensure_dirs(root)
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.68, clue_strength=0.10, observation_noise=0.08))
    horizons = [2, 4, 6] if smoke else [2, 4, 6, 8]
    all_rows = []
    horizon_summary = {}
    for h in horizons:
        records = env.generate_candidate_pool(
            n=420 if smoke else 1100,
            horizon=h,
            seed=seed + h,
            model_flavor="value_optimistic",
            state_id=h,
        )
        rows, summary = evaluate_scorers(records, ["belief_collapsed", "combined_repair", "oracle"], N_GRID, seed=seed + h)
        for row in rows:
            row["horizon"] = h
            row["scorer"] = f"H{h}_{row['scorer']}"
        all_rows.extend(rows)
        horizon_summary[str(h)] = summary
    import pandas as pd

    pd.DataFrame(all_rows).to_csv(root / "results" / "tables" / "experiment_d_horizon_budget.csv", index=False)
    summary = {
        "experiment": "D_horizon_and_selection_budget",
        "horizons": horizons,
        "n_grid": N_GRID,
        "summary_by_horizon": horizon_summary,
        "claim": "Longer horizons and larger N amplify latent-real mismatch under optimistic latent scoring.",
    }
    write_json(root / "results" / "experiment_d_horizon_budget.json", summary)
    plot_curve(
        all_rows,
        root / "figures" / "figure4_horizon_budget.png",
        [f"H{h}_belief_collapsed" for h in horizons],
        ["selected_real_utility"],
        "Real utility under horizon and Best-of-N scaling",
        "Expected selected real utility",
    )
    return summary


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

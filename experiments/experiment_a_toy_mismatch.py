"""Experiment A: controlled hidden-mode latent mismatch."""

from __future__ import annotations

from rssm_tail_audit.envs import HiddenModeConfig, HiddenModeToyEnv, records_to_frame

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


def run(smoke: bool = False, seed: int = 0):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.68, clue_strength=0.12, observation_noise=0.08))
    records = env.generate_candidate_pool(
        n=550 if smoke else 1800,
        horizon=5,
        seed=seed,
        model_flavor="belief_collapsed",
    )
    scorers = ["random", "good", "overconfident", "value_optimistic", "belief_collapsed", "combined_repair", "oracle"]
    rows, summary = evaluate_scorers(records, scorers, N_GRID, pilot_size=120 if smoke else 360, seed=seed)
    records_to_frame(records).to_csv(tables_dir(root, smoke) / "experiment_a_candidate_pool.csv", index=False)
    import pandas as pd

    pd.DataFrame(rows).to_csv(tables_dir(root, smoke) / "experiment_a_curves.csv", index=False)
    summary.update(
        {
            "experiment": "A_controlled_latent_dynamics_toy",
            "n_records": len(records),
            "claim": "Increasing N raises imagined latent value while selected real utility stagnates or worsens for raw/collapsed latent scoring.",
        }
    )
    raw = summary["scorers"]["belief_collapsed"]
    repair = summary["scorers"]["combined_repair"]
    oracle = summary["scorers"]["oracle"]
    summary["key_result"] = {
        "raw_latent_delta_high_n": raw["latent_delta_high_n"],
        "raw_real_delta_high_n": raw["real_delta_high_n"],
        "repair_n64_real_improvement_over_raw": repair["N64_real"] - raw["N64_real"],
        "oracle_gap_after_repair_n64": oracle["N64_real"] - repair["N64_real"],
    }
    write_json(results_dir(root, smoke) / "experiment_a_toy_mismatch.json", summary)
    plot_curve(
        rows,
        figures_dir(root, smoke) / "figure1_latent_mismatch.png",
        ["belief_collapsed"],
        ["selected_latent_value", "selected_real_utility"],
        "Latent value inflates while executed utility falls",
        "Expected selected quantity",
    )
    return summary


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

"""Experiment C: hidden-mode belief collapse across ambiguous modes."""

from __future__ import annotations

from rssm_tail_audit.envs import HiddenModeConfig, HiddenModeToyEnv

from experiments.common import N_GRID, ensure_dirs, evaluate_scorers, results_dir, root_from_file, smoke_argparser, tables_dir, write_json


def run(smoke: bool = False, seed: int = 2):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    env = HiddenModeToyEnv(
        HiddenModeConfig(
            modes=("free", "blocked", "slip", "heavy"),
            blocked_prob=0.74,
            clue_strength=0.08,
            observation_noise=0.10,
        )
    )
    records = env.generate_candidate_pool(
        n=500 if smoke else 1600,
        horizon=6,
        seed=seed,
        model_flavor="belief_collapsed",
    )
    rows, summary = evaluate_scorers(records, ["belief_collapsed", "combined_repair", "random", "oracle"], N_GRID, seed=seed)
    import pandas as pd

    pd.DataFrame(rows).to_csv(tables_dir(root, smoke) / "experiment_c_curves.csv", index=False)
    mode_tail = summary["scorers"]["belief_collapsed"]["tail"]
    summary.update(
        {
            "experiment": "C_hidden_mode_belief_collapse",
            "n_records": len(records),
            "key_result": {
                "raw_tail_blocked_rate": mode_tail["tail_blocked_rate"],
                "raw_tail_imagined_free_prob": mode_tail["tail_imagined_free_prob"],
                "raw_real_delta_high_n": summary["scorers"]["belief_collapsed"]["real_delta_high_n"],
            },
        }
    )
    write_json(results_dir(root, smoke) / "experiment_c_belief_collapse.json", summary)
    return summary


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

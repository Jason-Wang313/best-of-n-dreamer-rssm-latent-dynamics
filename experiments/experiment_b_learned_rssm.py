"""Experiment B: learned small RSSM-style model smoke/full run."""

from __future__ import annotations

import numpy as np
import torch

from latent_dynamics_best_of_n.envs import records_to_frame
from latent_dynamics_best_of_n.rssm import RSSMTrainConfig, learned_rssm_candidate_pool, train_rssm

from experiments.common import N_GRID, ensure_dirs, evaluate_scorers, root_from_file, smoke_argparser, write_json


def run(smoke: bool = False, seed: int = 1):
    root = root_from_file()
    ensure_dirs(root)
    cfg = RSSMTrainConfig(
        num_sequences=40 if smoke else 96,
        epochs=3 if smoke else 8,
        seq_len=6 if smoke else 8,
        seed=seed,
    )
    model, losses, data = train_rssm(cfg)
    records = learned_rssm_candidate_pool(model, n=260 if smoke else 760, horizon=5, seed=seed + 100)
    rows, summary = evaluate_scorers(
        records,
        ["raw_value", "uncertainty_pessimism", "pilot_calibrated", "combined_repair", "random", "oracle"],
        N_GRID,
        pilot_size=80 if smoke else 220,
        seed=seed,
    )
    records_to_frame(records).to_csv(root / "results" / "tables" / "experiment_b_learned_pool.csv", index=False)
    import pandas as pd

    pd.DataFrame(rows).to_csv(root / "results" / "tables" / "experiment_b_curves.csv", index=False)
    artifact_path = root / "results" / "learned_tiny_rssm.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "losses": losses}, artifact_path)
    np.savez_compressed(
        root / "results" / "learned_rssm_dataset_snapshot.npz",
        obs=data["obs"][:8],
        actions=data["actions"][:8],
        rewards=data["rewards"][:8],
        returns=data["returns"][:8],
    )
    summary.update(
        {
            "experiment": "B_learned_small_rssm_style_model",
            "train_losses": losses,
            "model_artifact": str(artifact_path.relative_to(root)),
            "n_records": len(records),
        }
    )
    raw = summary["scorers"]["raw_value"]
    repair = summary["scorers"]["combined_repair"]
    summary["key_result"] = {
        "raw_latent_delta_high_n": raw["latent_delta_high_n"],
        "raw_real_delta_high_n": raw["real_delta_high_n"],
        "repair_n64_real_improvement_over_raw": repair["N64_real"] - raw["N64_real"],
    }
    write_json(root / "results" / "experiment_b_learned_rssm.json", summary)
    return summary


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

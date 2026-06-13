"""Multi-seed evidence summary for strong claim support."""

from __future__ import annotations

import numpy as np

from rssm_tail_audit.envs import HiddenModeConfig, HiddenModeToyEnv
from rssm_tail_audit.metrics import selection_curves, top_tail_diagnostics
from rssm_tail_audit.rssm import RSSMTrainConfig, learned_rssm_candidate_pool, train_rssm
from rssm_tail_audit.scorers import fit_pilot_calibrator, score_records

from experiments.common import N_GRID, ensure_dirs, results_dir, root_from_file, smoke_argparser, write_json


def ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": float(len(arr)),
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
    }


def n64_real(records, scorer: str, seed: int, pilot_size: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    pilot = [records[i] for i in idx[: min(pilot_size, len(records))]]
    calibrator = fit_pilot_calibrator(pilot)
    scores = score_records(records, scorer, calibrator=calibrator)
    curves = selection_curves(records, scores, [1, 64])
    return curves, top_tail_diagnostics(records, scores)


def summarize_pair(records, raw_scorer: str, seed: int, pilot_size: int) -> dict[str, float]:
    raw_curves, raw_tail = n64_real(records, raw_scorer, seed, pilot_size)
    repair_curves, _ = n64_real(records, "combined_repair", seed, pilot_size)
    oracle_curves, _ = n64_real(records, "oracle", seed, pilot_size)
    raw_n64 = float(raw_curves[-1]["selected_real_utility"])
    repair_n64 = float(repair_curves[-1]["selected_real_utility"])
    oracle_n64 = float(oracle_curves[-1]["selected_real_utility"])
    gap = max(1e-9, oracle_n64 - raw_n64)
    return {
        "raw_real_delta_high_n": float(raw_curves[-1]["selected_real_utility"] - raw_curves[0]["selected_real_utility"]),
        "raw_latent_delta_high_n": float(raw_curves[-1]["selected_latent_value"] - raw_curves[0]["selected_latent_value"]),
        "repair_n64_real_improvement_over_raw": float(repair_n64 - raw_n64),
        "repair_fraction_of_oracle_gap_closed": float((repair_n64 - raw_n64) / gap),
        "oracle_gap_after_repair_n64": float(oracle_n64 - repair_n64),
        "tail_blocked_rate": float(raw_tail["tail_blocked_rate"]),
        "tail_imagined_free_prob": float(raw_tail["tail_imagined_free_prob"]),
        "tail_gap": float(raw_tail["tail_gap"]),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(rows[0])
    return {key: ci([float(row[key]) for row in rows]) for key in keys}


def run_toy_family(smoke: bool, seed_base: int, experiment: str) -> dict[str, object]:
    if experiment == "A":
        config = HiddenModeConfig(blocked_prob=0.68, clue_strength=0.12, observation_noise=0.08)
        flavor = "belief_collapsed"
        horizon = 5
        raw = "belief_collapsed"
        n = 360 if smoke else 1000
        pilot = 140 if smoke else 480
    elif experiment == "C":
        config = HiddenModeConfig(
            modes=("free", "blocked", "slip", "heavy"),
            blocked_prob=0.74,
            clue_strength=0.08,
            observation_noise=0.10,
        )
        flavor = "belief_collapsed"
        horizon = 6
        raw = "belief_collapsed"
        n = 360 if smoke else 900
        pilot = 140 if smoke else 420
    elif experiment == "E":
        config = HiddenModeConfig(blocked_prob=0.70, clue_strength=0.10, observation_noise=0.09)
        flavor = "belief_collapsed"
        horizon = 5
        raw = "raw_value"
        n = 420 if smoke else 1800
        pilot = 220 if smoke else 1000
    else:
        raise ValueError(experiment)
    env = HiddenModeToyEnv(config)
    seeds = [seed_base + i for i in range(2 if smoke else 5)]
    rows = []
    for seed in seeds:
        records = env.generate_candidate_pool(n=n, horizon=horizon, seed=seed, model_flavor=flavor)
        rows.append({"seed": float(seed), **summarize_pair(records, raw, seed, pilot)})
    return {"seeds": seeds, "aggregate": aggregate([{k: v for k, v in row.items() if k != "seed"} for row in rows]), "per_seed": rows}


def run_learned_family(smoke: bool, seed_base: int) -> dict[str, object]:
    seeds = [seed_base] if smoke else [seed_base + i for i in range(3)]
    rows = []
    for seed in seeds:
        cfg = RSSMTrainConfig(
            num_sequences=24 if smoke else 56,
            epochs=2 if smoke else 5,
            seq_len=5 if smoke else 6,
            seed=seed,
        )
        model, losses, _ = train_rssm(cfg)
        records = learned_rssm_candidate_pool(model, n=180 if smoke else 420, horizon=5, seed=seed + 100)
        row = summarize_pair(records, "raw_value", seed, 80 if smoke else 180)
        row["train_loss"] = float(losses["loss"])
        rows.append({"seed": float(seed), **row})
    return {"seeds": seeds, "aggregate": aggregate([{k: v for k, v in row.items() if k != "seed"} for row in rows]), "per_seed": rows}


def run_horizon_family(smoke: bool, seed_base: int) -> dict[str, object]:
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.68, clue_strength=0.10, observation_noise=0.08))
    seeds = [seed_base + i for i in range(2 if smoke else 5)]
    rows = []
    for seed in seeds:
        low = env.generate_candidate_pool(n=320 if smoke else 820, horizon=2, seed=seed, model_flavor="value_optimistic")
        high = env.generate_candidate_pool(n=320 if smoke else 820, horizon=8, seed=seed + 1000, model_flavor="value_optimistic")
        low_row = summarize_pair(low, "belief_collapsed", seed, 120 if smoke else 360)
        high_row = summarize_pair(high, "belief_collapsed", seed + 1000, 120 if smoke else 360)
        rows.append(
            {
                "seed": float(seed),
                "h8_minus_h2_raw_real_delta": high_row["raw_real_delta_high_n"] - low_row["raw_real_delta_high_n"],
                "h8_minus_h2_raw_latent_delta": high_row["raw_latent_delta_high_n"] - low_row["raw_latent_delta_high_n"],
                "h8_repair_improvement": high_row["repair_n64_real_improvement_over_raw"],
                "h8_repair_fraction_closed": high_row["repair_fraction_of_oracle_gap_closed"],
            }
        )
    return {"seeds": seeds, "aggregate": aggregate([{k: v for k, v in row.items() if k != "seed"} for row in rows]), "per_seed": rows}


def run(smoke: bool = False, seed: int = 50):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    payload = {
        "experiment": "multiseed_strong_evidence",
        "smoke": bool(smoke),
        "families": {
            "controlled_latent_dynamics": run_toy_family(smoke, seed, "A"),
            "learned_rssm": run_learned_family(smoke, seed + 20),
            "belief_collapse": run_toy_family(smoke, seed + 40, "C"),
            "horizon_budget": run_horizon_family(smoke, seed + 60),
            "repair": run_toy_family(smoke, seed + 80, "E"),
        },
    }
    write_json(results_dir(root, smoke) / "multiseed_strong_evidence.json", payload)
    return payload


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

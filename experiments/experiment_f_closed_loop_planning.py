"""Experiment F: closed-loop receding-horizon latent planning."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latent_dynamics_best_of_n.envs import HiddenModeConfig, HiddenModeToyEnv, RolloutRecord
from latent_dynamics_best_of_n.rssm import RSSMTrainConfig, TinyRSSM, learned_rssm_candidate_pool, train_rssm
from latent_dynamics_best_of_n.scorers import fit_pilot_calibrator, score_records

from experiments.common import ensure_dirs, figures_dir, results_dir, root_from_file, smoke_argparser, tables_dir, write_json


N_VALUES = [1, 8, 64]
SCORERS = ["raw_value", "combined_repair", "random", "oracle"]


def _controlled_candidates(
    env: HiddenModeToyEnv,
    mode: str,
    obs: np.ndarray,
    n: int,
    plan_horizon: int,
    seed: int,
    state_id: int,
) -> list[RolloutRecord]:
    rng = np.random.default_rng(seed)
    records: list[RolloutRecord] = []
    obs_clue = float(obs[0])
    posterior_free = float(np.clip(0.50 + 1.20 * (obs_clue - 0.5), 0.04, 0.96))
    for i in range(int(n)):
        risk0 = float(rng.beta(1.55, 1.05))
        if rng.random() < 0.14:
            risk0 = float(rng.uniform(0.0, 0.22))
        actions = np.clip(risk0 + rng.normal(0.0, 0.11, size=plan_horizon), 0.0, 1.0)
        risk = float(np.mean(actions))
        real = env.execute(mode, actions)
        imagined_free = float(np.clip(0.80 + 0.20 * risk + rng.normal(0.0, 0.025), 0.0, 1.0))
        reward_pred = plan_horizon * (0.18 + 1.18 * risk + 0.62 * imagined_free)
        latent_value = reward_pred + 0.78 * plan_horizon * risk**2
        value_pred = latent_value + rng.normal(0.0, 0.10)
        pp_kl = float(abs(imagined_free - posterior_free) * (0.85 + risk))
        ambiguity = float(1.0 - min(1.0, abs(obs_clue - 0.5) / 0.24))
        uncertainty = float(0.10 + 1.05 * risk * ambiguity + 0.45 * pp_kl)
        decoder_error = float(max(0.0, 0.04 + risk * max(0.0, imagined_free - obs_clue) + rng.normal(0.0, 0.02)))
        belief_error = float(risk * max(0.0, imagined_free - posterior_free))
        records.append(
            RolloutRecord(
                seed=int(seed),
                state_id=int(state_id),
                candidate_id=i,
                hidden_mode=mode,
                horizon=int(plan_horizon),
                actions=actions.astype(float),
                observation=obs.astype(float),
                latent_value=float(latent_value),
                value_pred=float(value_pred),
                reward_pred=float(reward_pred),
                real_utility=float(real),
                uncertainty=uncertainty,
                posterior_prior_kl=pp_kl,
                decoder_error=decoder_error,
                belief_error=belief_error,
                risk=risk,
                imagined_free_prob=imagined_free,
                posterior_free_prob=posterior_free,
                diagnostics={
                    "oracle_score": float(real),
                    "random_score": float(rng.normal()),
                    "ensemble_std": float(0.55 * uncertainty + 0.25 * decoder_error + 0.20 * pp_kl),
                },
            )
        )
    return records


def _learned_candidates(
    model: TinyRSSM,
    env: HiddenModeToyEnv,
    mode: str,
    obs: np.ndarray,
    n: int,
    plan_horizon: int,
    seed: int,
    state_id: int,
) -> list[RolloutRecord]:
    rng = np.random.default_rng(seed)
    records: list[RolloutRecord] = []
    for i in range(int(n)):
        risk0 = float(rng.beta(1.55, 1.10))
        if rng.random() < 0.12:
            risk0 = float(rng.uniform(0.0, 0.25))
        actions = np.clip(risk0 + rng.normal(0.0, 0.12, size=plan_horizon), 0.0, 1.0)
        risk = float(np.mean(actions))
        real = env.execute(mode, actions)
        imagined = model.imagine_score(obs, actions)
        value_pred = float(imagined["score"] + 0.48 * plan_horizon * risk)
        posterior_free = float(np.clip(0.50 + 1.1 * (float(obs[0]) - 0.5), 0.04, 0.96))
        imagined_free = float(np.clip(0.62 + 0.32 * risk, 0.0, 1.0))
        pp_kl = float(abs(imagined_free - posterior_free) * (0.72 + risk))
        uncertainty = float(imagined["uncertainty"] + 0.38 * risk + 0.25 * pp_kl)
        decoder_error = float(imagined["decoder_error"] + risk * max(0.0, imagined_free - float(obs[0])))
        records.append(
            RolloutRecord(
                seed=int(seed),
                state_id=int(state_id),
                candidate_id=i,
                hidden_mode=mode,
                horizon=int(plan_horizon),
                actions=actions.astype(float),
                observation=obs.astype(float),
                latent_value=value_pred,
                value_pred=value_pred,
                reward_pred=float(imagined["score"]),
                real_utility=float(real),
                uncertainty=uncertainty,
                posterior_prior_kl=pp_kl,
                decoder_error=decoder_error,
                belief_error=float(risk * max(0.0, imagined_free - posterior_free)),
                risk=risk,
                imagined_free_prob=imagined_free,
                posterior_free_prob=posterior_free,
                diagnostics={
                    "oracle_score": float(real),
                    "random_score": float(rng.normal()),
                    "ensemble_std": float(0.55 * uncertainty + 0.25 * decoder_error + 0.20 * pp_kl),
                },
            )
        )
    return records


def _make_episode_pools(
    env: HiddenModeToyEnv,
    make_candidates: Callable[[str, np.ndarray, int, int, int], list[RolloutRecord]],
    episode_seed: int,
    episode_horizon: int,
) -> tuple[str, list[list[RolloutRecord]]]:
    rng = np.random.default_rng(episode_seed)
    mode = env.sample_mode(rng)
    pools: list[list[RolloutRecord]] = []
    for t in range(episode_horizon):
        obs = env.observe(mode, rng)
        pools.append(make_candidates(mode, obs, max(N_VALUES), t, episode_seed * 1009 + t * 9173))
    return mode, pools


def _score_episode_pools(
    env: HiddenModeToyEnv,
    mode: str,
    pools: list[list[RolloutRecord]],
    scorer: str,
    n_select: int,
    calibrator,
) -> dict[str, float | str]:
    total = 0.0
    risks: list[float] = []
    for t, records in enumerate(pools):
        subset = records[:n_select]
        scores = score_records(subset, scorer, calibrator=calibrator)
        choice = int(np.argmax(scores))
        action = float(subset[choice].actions[0])
        risks.append(action)
        total += (env.config.discount**t) * env.real_step_reward(mode, action, t)
    return {"return": float(total), "mode": mode, "mean_action": float(np.mean(risks))}


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["scorer"]), int(row["N"]))].append(float(row["return"]))
    out: dict[str, object] = {}
    for (family, scorer, n_value), vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        out.setdefault(family, {}).setdefault(scorer, {})[str(n_value)] = {
            "mean_return": float(np.mean(arr)),
            "std_return": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        }
    return out


def _plot(rows: list[dict[str, object]], output) -> None:
    df = pd.DataFrame(rows)
    means = df.groupby(["family", "scorer", "N"], as_index=False)["return"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=150, sharey=False)
    colors = {"raw_value": "#2f6fbb", "combined_repair": "#1a8f5a", "random": "#777777", "oracle": "#202020"}
    for ax, family in zip(axes, ["controlled", "learned_rssm"]):
        sub = means[means["family"] == family]
        for scorer in SCORERS:
            g = sub[sub["scorer"] == scorer].sort_values("N")
            ax.plot(g["N"], g["return"], marker="o", linewidth=2.0, color=colors[scorer], label=scorer)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(family.replace("_", " "))
        ax.set_xlabel("Best-of-N candidates")
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Closed-loop return")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run(smoke: bool = False, seed: int = 6):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    rows: list[dict[str, object]] = []

    controlled_env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.76, clue_strength=0.08, observation_noise=0.09))
    controlled_pilot = controlled_env.generate_candidate_pool(
        n=260 if smoke else 950,
        horizon=4,
        seed=seed + 1000,
        model_flavor="belief_collapsed",
    )
    controlled_calibrator = fit_pilot_calibrator(controlled_pilot)
    controlled_seeds = [seed + i for i in range(2 if smoke else 8)]
    controlled_episodes = 8 if smoke else 30
    for family_seed in controlled_seeds:
        for episode in range(controlled_episodes):
            episode_seed = family_seed * 1000 + episode
            mode, pools = _make_episode_pools(
                controlled_env,
                lambda mode, obs, n, state_id, cand_seed: _controlled_candidates(
                    controlled_env, mode, obs, n, 4, cand_seed, state_id
                ),
                episode_seed,
                episode_horizon=5,
            )
            for n_value in N_VALUES:
                for scorer in SCORERS:
                    result = _score_episode_pools(
                        controlled_env,
                        mode,
                        pools,
                        scorer,
                        n_value,
                        controlled_calibrator,
                    )
                    rows.append(
                        {
                            "family": "controlled",
                            "seed": family_seed,
                            "episode": episode,
                            "N": n_value,
                            "scorer": scorer,
                            **result,
                        }
                    )

    learned_seeds = [seed + 20 + i for i in range(1 if smoke else 3)]
    learned_episodes = 5 if smoke else 12
    for learned_seed in learned_seeds:
        cfg = RSSMTrainConfig(
            num_sequences=24 if smoke else 52,
            epochs=2 if smoke else 4,
            seq_len=5 if smoke else 6,
            seed=learned_seed,
        )
        model, losses, _ = train_rssm(cfg)
        learned_env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.74, clue_strength=0.09, observation_noise=0.09))
        learned_pilot = learned_rssm_candidate_pool(model, n=120 if smoke else 360, horizon=4, seed=learned_seed + 2000)
        learned_calibrator = fit_pilot_calibrator(learned_pilot)
        for episode in range(learned_episodes):
            episode_seed = learned_seed * 1000 + episode
            mode, pools = _make_episode_pools(
                learned_env,
                lambda mode, obs, n, state_id, cand_seed, m=model, e=learned_env: _learned_candidates(
                    m, e, mode, obs, n, 4, cand_seed, state_id
                ),
                episode_seed,
                episode_horizon=5,
            )
            for n_value in N_VALUES:
                for scorer in SCORERS:
                    result = _score_episode_pools(
                        learned_env,
                        mode,
                        pools,
                        scorer,
                        n_value,
                        learned_calibrator,
                    )
                    rows.append(
                        {
                            "family": "learned_rssm",
                            "seed": learned_seed,
                            "episode": episode,
                            "N": n_value,
                            "scorer": scorer,
                            "train_loss": float(losses["loss"]),
                            **result,
                        }
                    )

    pd.DataFrame(rows).to_csv(tables_dir(root, smoke) / "experiment_f_closed_loop_planning.csv", index=False)
    summary = _summarize(rows)
    controlled = summary["controlled"]  # type: ignore[index]
    learned = summary["learned_rssm"]  # type: ignore[index]
    key_result = {
        "controlled_raw_n1_mean_return": controlled["raw_value"]["1"]["mean_return"],  # type: ignore[index]
        "controlled_raw_n64_mean_return": controlled["raw_value"]["64"]["mean_return"],  # type: ignore[index]
        "controlled_oracle_n64_mean_return": controlled["oracle"]["64"]["mean_return"],  # type: ignore[index]
        "controlled_combined_repair_n64_improvement_over_raw": controlled["combined_repair"]["64"]["mean_return"]  # type: ignore[index]
        - controlled["raw_value"]["64"]["mean_return"],  # type: ignore[index]
        "learned_raw_n1_mean_return": learned["raw_value"]["1"]["mean_return"],  # type: ignore[index]
        "learned_raw_n64_mean_return": learned["raw_value"]["64"]["mean_return"],  # type: ignore[index]
        "learned_oracle_n64_mean_return": learned["oracle"]["64"]["mean_return"],  # type: ignore[index]
        "learned_combined_repair_n64_improvement_over_raw": learned["combined_repair"]["64"]["mean_return"]  # type: ignore[index]
        - learned["raw_value"]["64"]["mean_return"],  # type: ignore[index]
    }
    payload = {
        "experiment": "F_closed_loop_receding_horizon_planning",
        "smoke": bool(smoke),
        "n_values": N_VALUES,
        "scorers": SCORERS,
        "summary": summary,
        "key_result": key_result,
    }
    write_json(results_dir(root, smoke) / "experiment_f_closed_loop_planning.json", payload)
    _plot(rows, figures_dir(root, smoke) / "figure6_closed_loop_planning.png")
    return payload


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

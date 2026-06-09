"""Lightweight Gymnasium tabular benchmarks for Best-of-N selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import gymnasium as gym
import numpy as np

from .envs import RolloutRecord


@dataclass(frozen=True)
class GymBenchmarkSpec:
    name: str
    env_id: str
    horizon: int
    discount: float
    raw_scale: float = 1.0


BENCHMARKS = {
    "FrozenLake-v1": GymBenchmarkSpec("FrozenLake-v1", "FrozenLake-v1", horizon=12, discount=0.97, raw_scale=4.0),
    "CliffWalkingSlippery-v1": GymBenchmarkSpec(
        "CliffWalkingSlippery-v1", "CliffWalkingSlippery-v1", horizon=14, discount=0.99, raw_scale=1.0
    ),
    "Taxi-v3": GymBenchmarkSpec("Taxi-v3", "Taxi-v3", horizon=16, discount=0.97, raw_scale=1.0),
}


def make_env(env_id: str):
    if env_id == "FrozenLake-v1":
        return gym.make(env_id, is_slippery=True)
    return gym.make(env_id)


def seeded_start_state(env_id: str, seed: int) -> int:
    env = make_env(env_id)
    try:
        obs, _ = env.reset(seed=seed)
        return int(obs)
    finally:
        env.close()


@lru_cache(maxsize=8)
def transition_model(env_id: str):
    env = make_env(env_id)
    try:
        return env.unwrapped.P, int(env.action_space.n), int(env.observation_space.n)
    finally:
        env.close()


def valid_action_sequence(env_id: str, actions: Iterable[int]) -> bool:
    _, n_actions, _ = transition_model(env_id)
    return all(0 <= int(a) < n_actions for a in actions)


def expected_return(env_id: str, start_state: int, actions: Iterable[int], discount: float) -> float:
    """Exact finite-horizon expected return under the Gymnasium transition table."""

    action_list = [int(a) for a in actions]
    p_model, _, _ = transition_model(env_id)
    dist = {int(start_state): 1.0}
    total = 0.0
    for t, action in enumerate(action_list):
        next_dist: dict[int, float] = {}
        for state, state_prob in dist.items():
            for prob, next_state, reward, terminated in p_model[int(state)][int(action)]:
                mass = float(state_prob) * float(prob)
                total += (float(discount) ** t) * mass * float(reward)
                if not terminated:
                    next_dist[int(next_state)] = next_dist.get(int(next_state), 0.0) + mass
        dist = next_dist
        if not dist:
            break
    return float(total)


def _to_row_col(env_id: str, state: int) -> tuple[int, int]:
    if env_id == "FrozenLake-v1":
        return divmod(int(state), 4)
    if env_id == "CliffWalkingSlippery-v1":
        return divmod(int(state), 12)
    raise ValueError(env_id)


@lru_cache(maxsize=512)
def _taxi_decode(state: int) -> tuple[int, int, int, int]:
    env = make_env("Taxi-v3")
    try:
        return tuple(int(x) for x in env.unwrapped.decode(int(state)))  # type: ignore[return-value]
    finally:
        env.close()


@lru_cache(maxsize=2048)
def _potential(env_id: str, state: int) -> float:
    if env_id == "FrozenLake-v1":
        row, col = _to_row_col(env_id, state)
        goal = (3, 3)
        return 1.0 - 0.08 * (abs(goal[0] - row) + abs(goal[1] - col))
    if env_id == "CliffWalkingSlippery-v1":
        row, col = _to_row_col(env_id, state)
        goal = (3, 11)
        return -0.18 * (abs(goal[0] - row) + abs(goal[1] - col))
    if env_id == "Taxi-v3":
        taxi_row, taxi_col, passenger, dest = _taxi_decode(state)
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        if passenger < 4:
            target = locs[passenger]
            onboard_bonus = 0.0
        else:
            target = locs[dest]
            onboard_bonus = 5.0
        dist = abs(target[0] - taxi_row) + abs(target[1] - taxi_col)
        return onboard_bonus - 0.25 * dist
    raise ValueError(env_id)


def optimistic_model_score(env_id: str, start_state: int, actions: Iterable[int], discount: float) -> float:
    """Score a sequence with a deliberately optimistic one-transition model."""

    action_list = [int(a) for a in actions]
    p_model, _, _ = transition_model(env_id)
    state = int(start_state)
    total = 0.0
    for t, action in enumerate(action_list):
        transitions = p_model[state][int(action)]

        def optimistic_value(item):
            _, next_state, reward, _ = item
            mod_reward = float(reward)
            if env_id == "CliffWalkingSlippery-v1" and mod_reward <= -100.0:
                mod_reward = -2.0
            if env_id == "Taxi-v3" and mod_reward <= -10.0:
                mod_reward = -1.2
            if env_id == "FrozenLake-v1" and bool(item[3]) and mod_reward <= 0.0:
                mod_reward = -0.05
            return mod_reward + (float(discount) ** 0.5) * _potential(env_id, int(next_state))

        best = max(transitions, key=optimistic_value)
        _, next_state, reward, terminated = best
        mod_reward = float(reward)
        if env_id == "CliffWalkingSlippery-v1" and mod_reward <= -100.0:
            mod_reward = -2.0
        if env_id == "Taxi-v3" and mod_reward <= -10.0:
            mod_reward = -1.2
        if env_id == "FrozenLake-v1" and terminated and mod_reward <= 0.0:
            mod_reward = -0.05
        total += (float(discount) ** t) * mod_reward
        state = int(next_state)
        if terminated:
            break
    total += (float(discount) ** len(action_list)) * _potential(env_id, state)
    return float(total)


def transition_diagnostics(env_id: str, start_state: int, actions: Iterable[int], discount: float) -> dict[str, float]:
    """Return entropy/risk diagnostics from the exact transition table."""

    action_list = [int(a) for a in actions]
    p_model, _, _ = transition_model(env_id)
    dist = {int(start_state): 1.0}
    entropy = 0.0
    catastrophe = 0.0
    illegal = 0.0
    for t, action in enumerate(action_list):
        next_dist: dict[int, float] = {}
        for state, state_prob in dist.items():
            transitions = p_model[int(state)][int(action)]
            probs = np.asarray([float(item[0]) for item in transitions], dtype=float)
            probs = probs[probs > 0]
            entropy += float(state_prob) * float(-np.sum(probs * np.log(probs))) / max(1, len(action_list))
            for prob, next_state, reward, terminated in transitions:
                mass = float(state_prob) * float(prob)
                reward = float(reward)
                if env_id == "FrozenLake-v1" and terminated and reward <= 0.0:
                    catastrophe += (float(discount) ** t) * mass
                if env_id == "CliffWalkingSlippery-v1" and reward <= -100.0:
                    catastrophe += (float(discount) ** t) * mass
                if env_id == "Taxi-v3" and reward <= -10.0:
                    illegal += (float(discount) ** t) * mass
                if not terminated:
                    next_dist[int(next_state)] = next_dist.get(int(next_state), 0.0) + mass
        dist = next_dist
        if not dist:
            break
    risk = float(catastrophe + 0.35 * illegal)
    return {"entropy": float(entropy), "catastrophe": float(catastrophe), "illegal": float(illegal), "risk": risk}


def _sample_actions(env_id: str, rng: np.random.Generator, horizon: int) -> np.ndarray:
    if env_id == "FrozenLake-v1":
        probs = np.asarray([0.08, 0.40, 0.44, 0.08], dtype=float)
        actions = rng.choice(4, size=horizon, p=probs)
        if rng.random() < 0.30:
            prefix = np.asarray([2, 2, 1, 1, 1, 2], dtype=int)
            actions[: min(horizon, len(prefix))] = prefix[: min(horizon, len(prefix))]
        return actions.astype(int)
    if env_id == "CliffWalkingSlippery-v1":
        probs = np.asarray([0.32, 0.48, 0.04, 0.16], dtype=float)
        actions = rng.choice(4, size=horizon, p=probs)
        if rng.random() < 0.36:
            prefix = np.asarray([0] + [1] * max(0, horizon - 1), dtype=int)
            actions[: len(prefix)] = prefix[:horizon]
        return actions.astype(int)
    if env_id == "Taxi-v3":
        probs = np.asarray([0.18, 0.18, 0.18, 0.18, 0.14, 0.14], dtype=float)
        actions = rng.choice(6, size=horizon, p=probs)
        if rng.random() < 0.45:
            spots = rng.choice(horizon, size=max(1, horizon // 4), replace=False)
            actions[spots] = rng.choice([4, 5], size=len(spots))
        return actions.astype(int)
    raise ValueError(env_id)


def generate_benchmark_records(
    spec: GymBenchmarkSpec,
    n: int,
    seed: int,
    start_state: int | None = None,
    state_id: int = 0,
) -> list[RolloutRecord]:
    """Generate candidate action sequences with exact expected real utility."""

    rng = np.random.default_rng(seed)
    start = int(seeded_start_state(spec.env_id, seed) if start_state is None else start_state)
    records: list[RolloutRecord] = []
    for i in range(int(n)):
        actions_i = _sample_actions(spec.env_id, rng, spec.horizon)
        real = expected_return(spec.env_id, start, actions_i, spec.discount)
        raw = optimistic_model_score(spec.env_id, start, actions_i, spec.discount)
        diag = transition_diagnostics(spec.env_id, start, actions_i, spec.discount)
        optimism_gap = max(0.0, raw - real)
        uncertainty = float(0.35 * diag["entropy"] + 0.65 * diag["risk"] + 0.08 * optimism_gap)
        decoder_error = float(0.40 * diag["risk"] + 0.05 * diag["entropy"])
        pp_kl = float(0.18 * optimism_gap + 0.25 * diag["entropy"])
        risk = float(diag["risk"])
        value_pred = float(spec.raw_scale * raw + 0.35 * optimism_gap)
        records.append(
            RolloutRecord(
                seed=int(seed),
                state_id=int(state_id),
                candidate_id=i,
                hidden_mode=spec.name,
                horizon=int(spec.horizon),
                actions=actions_i.astype(float),
                observation=np.asarray([start, spec.horizon, risk, diag["entropy"]], dtype=float),
                latent_value=value_pred,
                value_pred=value_pred,
                reward_pred=float(raw),
                real_utility=float(real),
                uncertainty=uncertainty,
                posterior_prior_kl=pp_kl,
                decoder_error=decoder_error,
                belief_error=float(risk * (0.5 + diag["entropy"])),
                risk=risk,
                imagined_free_prob=float(np.clip(1.0 - 0.20 * risk + 0.05 * optimism_gap, 0.0, 1.0)),
                posterior_free_prob=float(np.clip(1.0 - risk, 0.0, 1.0)),
                diagnostics={
                    "oracle_score": float(real),
                    "random_score": float(rng.normal()),
                    "ensemble_std": float(0.50 * uncertainty + 0.30 * decoder_error + 0.20 * pp_kl),
                    "transition_entropy": float(diag["entropy"]),
                    "catastrophe": float(diag["catastrophe"]),
                    "illegal": float(diag["illegal"]),
                    "start_state": float(start),
                },
            )
        )
    return records

"""Gymnasium classic-control benchmark records for RSSM tail audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np

from .envs import RolloutRecord


@dataclass(frozen=True)
class ClassicControlSpec:
    name: str
    env_id: str
    horizon: int
    discount: float
    actions: tuple[int, ...]
    action_scale: tuple[float, ...]


CLASSIC_CONTROL_BENCHMARKS = {
    "CartPole-v1": ClassicControlSpec(
        name="CartPole-v1",
        env_id="CartPole-v1",
        horizon=80,
        discount=0.99,
        actions=(0, 1),
        action_scale=(-1.0, 1.0),
    ),
    "MountainCar-v0": ClassicControlSpec(
        name="MountainCar-v0",
        env_id="MountainCar-v0",
        horizon=70,
        discount=0.99,
        actions=(0, 1, 2),
        action_scale=(-1.0, 0.0, 1.0),
    ),
    "Acrobot-v1": ClassicControlSpec(
        name="Acrobot-v1",
        env_id="Acrobot-v1",
        horizon=70,
        discount=0.99,
        actions=(0, 1, 2),
        action_scale=(-1.0, 0.0, 1.0),
    ),
}


def start_state(env_id: str, seed: int) -> np.ndarray:
    env = gym.make(env_id)
    try:
        env.reset(seed=int(seed))
        return np.asarray(env.unwrapped.state, dtype=float).copy()
    finally:
        env.close()


def valid_classic_action_sequence(spec: ClassicControlSpec, actions: Iterable[int]) -> bool:
    allowed = set(int(a) for a in spec.actions)
    return all(int(a) in allowed for a in actions)


def _set_start_state(env, state: np.ndarray) -> None:
    env.unwrapped.state = np.asarray(state, dtype=float).copy()


def _cartpole_margin(state: np.ndarray) -> float:
    x, _, theta, _ = np.asarray(state, dtype=float)
    x_margin = max(0.0, 1.0 - abs(float(x)) / 2.4)
    theta_margin = max(0.0, 1.0 - abs(float(theta)) / 0.2095)
    return 0.5 * x_margin + 0.5 * theta_margin


def _mountaincar_progress(states: list[np.ndarray], final_state: np.ndarray) -> float:
    if states:
        max_position = max(float(s[0]) for s in states)
    else:
        max_position = float(final_state[0])
    final_position = float(final_state[0])
    final_velocity = float(final_state[1])
    return 35.0 * (max_position + 0.5) + 15.0 * max(0.0, final_position + 0.5) + 8.0 * final_velocity


def _acrobot_tip_height(state: np.ndarray) -> float:
    theta1, theta2 = float(state[0]), float(state[1])
    return -np.cos(theta1) - np.cos(theta1 + theta2)


def _acrobot_progress(states: list[np.ndarray], final_state: np.ndarray) -> float:
    heights = [_acrobot_tip_height(s) for s in states] or [_acrobot_tip_height(final_state)]
    return 14.0 * max(heights) + 8.0 * _acrobot_tip_height(final_state)


def rollout_utility(spec: ClassicControlSpec, initial_state: np.ndarray, actions: Iterable[int]) -> tuple[float, dict[str, float]]:
    """Execute an action sequence in the real Gymnasium environment."""

    env = gym.make(spec.env_id)
    try:
        env.reset(seed=0)
        _set_start_state(env, initial_state)
        total_reward = 0.0
        states: list[np.ndarray] = []
        terminated_at = spec.horizon
        final_state = np.asarray(initial_state, dtype=float).copy()
        for t, action in enumerate(actions):
            _, reward, terminated, truncated, _ = env.step(int(action))
            final_state = np.asarray(env.unwrapped.state, dtype=float).copy()
            states.append(final_state)
            total_reward += (spec.discount**t) * float(reward)
            if terminated or truncated:
                terminated_at = t + 1
                break
    finally:
        env.close()

    if spec.name == "CartPole-v1":
        shaped = total_reward + 8.0 * _cartpole_margin(final_state)
        success = float(terminated_at >= spec.horizon)
    elif spec.name == "MountainCar-v0":
        shaped = total_reward + _mountaincar_progress(states, final_state)
        success = float(float(final_state[0]) >= 0.5)
    elif spec.name == "Acrobot-v1":
        shaped = total_reward + _acrobot_progress(states, final_state)
        success = float(_acrobot_tip_height(final_state) > 1.0)
    else:
        raise ValueError(spec.name)
    return float(shaped), {
        "discounted_env_return": float(total_reward),
        "terminated_at": float(terminated_at),
        "success": success,
        "final_state_norm": float(np.linalg.norm(final_state)),
    }


def _sample_actions(spec: ClassicControlSpec, rng: np.random.Generator) -> np.ndarray:
    horizon = int(spec.horizon)
    if spec.name == "CartPole-v1":
        if rng.random() < 0.36:
            action = int(rng.choice(spec.actions))
            return np.full(horizon, action, dtype=int)
        if rng.random() < 0.55:
            switch = int(rng.integers(max(2, horizon // 5), max(3, 4 * horizon // 5)))
            first = int(rng.choice(spec.actions))
            out = np.full(horizon, first, dtype=int)
            out[switch:] = 1 - first
            return out
        p_right = float(rng.beta(0.75, 0.75))
        return rng.choice(spec.actions, size=horizon, p=[1.0 - p_right, p_right]).astype(int)
    if spec.name == "MountainCar-v0":
        if rng.random() < 0.32:
            return np.full(horizon, 2, dtype=int)
        if rng.random() < 0.52:
            period = int(rng.integers(6, 14))
            phase = int(rng.integers(0, period))
            return np.asarray([0 if ((t + phase) % period) < period // 2 else 2 for t in range(horizon)], dtype=int)
        return rng.choice(spec.actions, size=horizon, p=[0.32, 0.10, 0.58]).astype(int)
    if spec.name == "Acrobot-v1":
        if rng.random() < 0.34:
            action = int(rng.choice([0, 2]))
            return np.full(horizon, action, dtype=int)
        if rng.random() < 0.55:
            period = int(rng.integers(5, 12))
            return np.asarray([0 if (t % period) < period // 2 else 2 for t in range(horizon)], dtype=int)
        return rng.choice(spec.actions, size=horizon, p=[0.42, 0.10, 0.48]).astype(int)
    raise ValueError(spec.name)


def _action_features(spec: ClassicControlSpec, actions: np.ndarray) -> dict[str, float]:
    scale = {int(a): float(v) for a, v in zip(spec.actions, spec.action_scale)}
    signed = np.asarray([scale[int(a)] for a in actions], dtype=float)
    switches = np.abs(np.diff(signed)) > 1e-9
    switch_rate = float(np.mean(switches)) if switches.size else 0.0
    runs = []
    run_len = 1
    for changed in switches:
        if changed:
            runs.append(run_len)
            run_len = 1
        else:
            run_len += 1
    runs.append(run_len)
    max_run_fraction = float(max(runs) / max(1, len(actions)))
    return {
        "action_bias": float(abs(np.mean(signed))),
        "action_energy": float(np.mean(np.abs(signed))),
        "switch_rate": switch_rate,
        "max_run_fraction": max_run_fraction,
        "signed_action_mean": float(np.mean(signed)),
    }


def _optimistic_latent_score(spec: ClassicControlSpec, features: dict[str, float], initial_state: np.ndarray) -> float:
    bias = features["action_bias"]
    energy = features["action_energy"]
    switch_rate = features["switch_rate"]
    run = features["max_run_fraction"]
    if spec.name == "CartPole-v1":
        theta = abs(float(initial_state[2]))
        return 55.0 + 34.0 * bias + 12.0 * energy + 10.0 * run - 6.0 * switch_rate - 12.0 * theta
    if spec.name == "MountainCar-v0":
        pos = float(initial_state[0])
        return -45.0 + 48.0 * max(0.0, features["signed_action_mean"]) + 18.0 * energy + 8.0 * run + 20.0 * (pos + 0.5)
    if spec.name == "Acrobot-v1":
        return -42.0 + 36.0 * energy + 22.0 * bias + 8.0 * run - 4.0 * switch_rate
    raise ValueError(spec.name)


def generate_classic_control_records(
    spec: ClassicControlSpec,
    n: int,
    seed: int,
    initial_state: np.ndarray | None = None,
    state_id: int = 0,
) -> list[RolloutRecord]:
    """Generate candidate sequences scored by a biased latent proxy and executed in Gymnasium."""

    rng = np.random.default_rng(int(seed))
    init = start_state(spec.env_id, int(seed)) if initial_state is None else np.asarray(initial_state, dtype=float)
    records: list[RolloutRecord] = []
    for candidate_id in range(int(n)):
        actions = _sample_actions(spec, rng)
        real_utility, rollout_diag = rollout_utility(spec, init, actions)
        features = _action_features(spec, actions)
        optimistic_score = _optimistic_latent_score(spec, features, init)
        risk = float(0.55 * features["max_run_fraction"] + 0.30 * features["action_bias"] + 0.15 * features["action_energy"])
        posterior_prior_kl = float(0.10 + 1.10 * risk + 0.25 * max(0.0, 0.30 - features["switch_rate"]))
        uncertainty = float(0.15 + 0.75 * risk + 0.20 * features["action_energy"])
        decoder_error = float(0.05 + 0.65 * features["max_run_fraction"] + 0.20 * features["action_bias"])
        belief_error = float(0.10 + risk * (1.0 - min(0.85, features["switch_rate"] + 0.15)))
        latent_value = float(optimistic_score + rng.normal(0.0, 0.35))
        value_pred = float(latent_value + 0.25 * posterior_prior_kl + rng.normal(0.0, 0.20))
        imagined_free = float(np.clip(0.82 + 0.18 * risk, 0.0, 1.0))
        posterior_free = float(np.clip(0.88 - 0.60 * risk, 0.0, 1.0))
        records.append(
            RolloutRecord(
                seed=int(seed),
                state_id=int(state_id),
                candidate_id=int(candidate_id),
                hidden_mode=spec.name,
                horizon=int(spec.horizon),
                actions=actions.astype(float),
                observation=init.astype(float),
                latent_value=latent_value,
                value_pred=value_pred,
                reward_pred=latent_value,
                real_utility=float(real_utility),
                uncertainty=uncertainty,
                posterior_prior_kl=posterior_prior_kl,
                decoder_error=decoder_error,
                belief_error=belief_error,
                risk=risk,
                imagined_free_prob=imagined_free,
                posterior_free_prob=posterior_free,
                diagnostics={
                    "oracle_score": float(real_utility),
                    "random_score": float(rng.normal()),
                    "ensemble_std": float(0.50 * uncertainty + 0.25 * posterior_prior_kl + 0.25 * decoder_error),
                    "action_bias": features["action_bias"],
                    "action_energy": features["action_energy"],
                    "switch_rate": features["switch_rate"],
                    "max_run_fraction": features["max_run_fraction"],
                    **rollout_diag,
                },
            )
        )
    return records

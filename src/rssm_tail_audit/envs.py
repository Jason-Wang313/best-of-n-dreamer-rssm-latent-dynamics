"""Hidden-mode toy dynamics for latent-imagination candidate-budget selection experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HiddenModeConfig:
    """Configuration for ambiguous hidden physical modes."""

    modes: tuple[str, ...] = ("free", "blocked", "slip", "heavy")
    blocked_prob: float = 0.62
    clue_strength: float = 0.18
    observation_noise: float = 0.07
    discount: float = 0.97


@dataclass
class RolloutRecord:
    """One imagined action sequence with latent diagnostics and executed utility."""

    seed: int
    state_id: int
    candidate_id: int
    hidden_mode: str
    horizon: int
    actions: np.ndarray
    observation: np.ndarray
    latent_value: float
    value_pred: float
    reward_pred: float
    real_utility: float
    uncertainty: float
    posterior_prior_kl: float
    decoder_error: float
    belief_error: float
    risk: float
    imagined_free_prob: float
    posterior_free_prob: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "seed": self.seed,
            "state_id": self.state_id,
            "candidate_id": self.candidate_id,
            "hidden_mode": self.hidden_mode,
            "horizon": self.horizon,
            "latent_value": self.latent_value,
            "value_pred": self.value_pred,
            "reward_pred": self.reward_pred,
            "real_utility": self.real_utility,
            "uncertainty": self.uncertainty,
            "posterior_prior_kl": self.posterior_prior_kl,
            "decoder_error": self.decoder_error,
            "belief_error": self.belief_error,
            "risk": self.risk,
            "imagined_free_prob": self.imagined_free_prob,
            "posterior_free_prob": self.posterior_free_prob,
        }
        row.update({f"diag_{k}": float(v) for k, v in self.diagnostics.items()})
        return row


class HiddenModeToyEnv:
    """Small CPU toy where latent optimism can disagree with real execution."""

    def __init__(self, config: HiddenModeConfig | None = None):
        self.config = config or HiddenModeConfig()

    def sample_mode(self, rng: np.random.Generator) -> str:
        bad_mass = float(np.clip(self.config.blocked_prob, 0.0, 1.0))
        if rng.random() > bad_mass:
            return "free"
        return str(rng.choice([m for m in self.config.modes if m != "free"]))

    def observe(self, mode: str, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return an intentionally ambiguous observation vector."""

        rng = rng or np.random.default_rng(0)
        free_signal = 1.0 if mode == "free" else 0.0
        clue = 0.5 + self.config.clue_strength * (free_signal - 0.5)
        clue += rng.normal(0.0, self.config.observation_noise)
        return np.array([np.clip(clue, 0.0, 1.0), 1.0, 0.0, free_signal * 0.0], dtype=float)

    def real_step_reward(self, mode: str, action: float, t: int = 0) -> float:
        """Executed utility, not a model score."""

        risk = float(np.clip(action, 0.0, 1.0))
        safe_reward = 0.20
        if mode == "free":
            return safe_reward + 1.08 * risk - 0.08 * risk * risk
        if mode == "slip":
            return safe_reward - 1.15 * risk - 0.26 * risk * risk - 0.02 * t
        if mode == "heavy":
            return safe_reward - 0.80 * risk - 0.42 * risk * risk
        return safe_reward - 1.35 * risk - 0.35 * risk * risk

    def execute(self, mode: str, actions: Iterable[float]) -> float:
        total = 0.0
        for t, action in enumerate(actions):
            total += (self.config.discount**t) * self.real_step_reward(mode, float(action), t)
        return float(total)

    def generate_candidate_pool(
        self,
        n: int = 1500,
        horizon: int = 5,
        seed: int = 0,
        model_flavor: str = "belief_collapsed",
        state_id: int = 0,
    ) -> list[RolloutRecord]:
        """Generate imagined candidates with hidden real utilities.

        The model flavor changes the latent score fields only. Real utilities
        are always produced by executing the action sequence in the hidden mode.
        """

        rng = np.random.default_rng(seed)
        records: list[RolloutRecord] = []
        for i in range(int(n)):
            mode = self.sample_mode(rng)
            obs = self.observe(mode, rng)
            base_risk = float(rng.beta(1.45, 1.15))
            if rng.random() < 0.16:
                base_risk = float(rng.uniform(0.0, 0.25))
            actions = np.clip(base_risk + rng.normal(0.0, 0.10, size=int(horizon)), 0.0, 1.0)
            risk = float(np.mean(actions))
            real_utility = self.execute(mode, actions)

            obs_clue = float(obs[0])
            posterior_free = float(np.clip(0.50 + 1.20 * (obs_clue - 0.5), 0.04, 0.96))
            if model_flavor in {"belief_collapsed", "overconfident", "value_optimistic"}:
                imagined_free = float(np.clip(0.78 + 0.20 * risk + rng.normal(0.0, 0.03), 0.0, 1.0))
            elif model_flavor == "good":
                imagined_free = posterior_free
            else:
                imagined_free = float(np.clip(0.62 + 0.20 * risk + rng.normal(0.0, 0.05), 0.0, 1.0))

            reward_pred = horizon * (0.18 + 1.20 * risk + 0.60 * imagined_free)
            optimism = {"value_optimistic": 1.25, "belief_collapsed": 0.72, "overconfident": 0.45}.get(
                model_flavor, 0.25
            )
            latent_value = reward_pred + optimism * horizon * (risk**2)
            value_pred = latent_value + rng.normal(0.0, 0.12)
            if model_flavor == "good":
                value_pred = real_utility + rng.normal(0.0, 0.12)
                latent_value = value_pred

            posterior_prior_kl = float(abs(imagined_free - posterior_free) * (0.8 + risk))
            ambiguity = float(1.0 - min(1.0, abs(obs_clue - 0.5) / 0.24))
            uncertainty = float(0.10 + 0.95 * risk * ambiguity + 0.45 * posterior_prior_kl)
            decoder_error = float(0.04 + risk * max(0.0, imagined_free - obs_clue) + rng.normal(0.0, 0.02))
            decoder_error = float(max(0.0, decoder_error))
            belief_error = float(risk * max(0.0, imagined_free - posterior_free))

            diagnostics = {
                "good_score": real_utility + rng.normal(0.0, 0.08),
                "overconfident_score": reward_pred + 0.35 * horizon * risk,
                "value_optimistic_score": reward_pred + 1.40 * horizon * (risk**2),
                "belief_collapsed_score": latent_value,
                "oracle_score": real_utility,
                "random_score": rng.normal(0.0, 1.0),
                "ensemble_std": 0.55 * uncertainty + 0.25 * decoder_error + 0.20 * posterior_prior_kl,
            }
            records.append(
                RolloutRecord(
                    seed=int(seed),
                    state_id=int(state_id),
                    candidate_id=i,
                    hidden_mode=mode,
                    horizon=int(horizon),
                    actions=actions.astype(float),
                    observation=obs.astype(float),
                    latent_value=float(latent_value),
                    value_pred=float(value_pred),
                    reward_pred=float(reward_pred),
                    real_utility=float(real_utility),
                    uncertainty=uncertainty,
                    posterior_prior_kl=posterior_prior_kl,
                    decoder_error=decoder_error,
                    belief_error=belief_error,
                    risk=risk,
                    imagined_free_prob=imagined_free,
                    posterior_free_prob=posterior_free,
                    diagnostics=diagnostics,
                )
            )
        return records


def records_to_frame(records: list[RolloutRecord]):
    """Convert records to a pandas DataFrame without making pandas a core import."""

    import pandas as pd

    return pd.DataFrame([r.as_dict() for r in records])

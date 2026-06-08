"""A compact CPU-friendly RSSM-style latent dynamics model.

The goal is not to reproduce Dreamer. The model contains the architectural
pieces needed for the experiments: encoder, recurrent deterministic state,
stochastic prior/posterior, decoder, reward head, and value head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .envs import HiddenModeConfig, HiddenModeToyEnv, RolloutRecord


@dataclass(frozen=True)
class RSSMTrainConfig:
    obs_dim: int = 4
    action_dim: int = 1
    hidden_dim: int = 24
    latent_dim: int = 8
    seq_len: int = 8
    num_sequences: int = 96
    epochs: int = 8
    lr: float = 3e-3
    seed: int = 0


class TinyRSSM(nn.Module):
    """Minimal recurrent state-space model with Gaussian latent state."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 1, hidden_dim: int = 24, latent_dim: int = 8):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.prior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.posterior = nn.Linear(hidden_dim + hidden_dim, 2 * latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def initial(self, batch_size: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    @staticmethod
    def split_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_scale = torch.chunk(stats, 2, dim=-1)
        scale = F.softplus(raw_scale) + 0.05
        return mean, scale

    @staticmethod
    def gaussian_kl(mean_q: torch.Tensor, scale_q: torch.Tensor, mean_p: torch.Tensor, scale_p: torch.Tensor) -> torch.Tensor:
        var_q = scale_q.pow(2)
        var_p = scale_p.pow(2)
        return 0.5 * ((var_q + (mean_q - mean_p).pow(2)) / var_p - 1.0 + 2.0 * (torch.log(scale_p) - torch.log(scale_q)))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, deterministic: bool = False) -> dict[str, torch.Tensor]:
        """Observe a sequence.

        ``obs`` has shape [B, T, obs_dim], ``actions`` has shape
        [B, T, action_dim]. Outputs keep the [B, T, ...] convention.
        """

        batch, steps, _ = obs.shape
        h, z = self.initial(batch, obs.device)
        recons: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        prior_means: list[torch.Tensor] = []
        prior_scales: list[torch.Tensor] = []
        post_means: list[torch.Tensor] = []
        post_scales: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []
        hiddens: list[torch.Tensor] = []
        for t in range(steps):
            h = self.gru(torch.cat([z, actions[:, t]], dim=-1), h)
            prior_mean, prior_scale = self.split_stats(self.prior(h))
            embed = self.encoder(obs[:, t])
            post_mean, post_scale = self.split_stats(self.posterior(torch.cat([h, embed], dim=-1)))
            if deterministic:
                z = post_mean
            else:
                z = post_mean + torch.randn_like(post_scale) * post_scale
            hz = torch.cat([h, z], dim=-1)
            recons.append(self.decoder(hz))
            rewards.append(self.reward_head(hz))
            values.append(self.value_head(hz))
            prior_means.append(prior_mean)
            prior_scales.append(prior_scale)
            post_means.append(post_mean)
            post_scales.append(post_scale)
            latents.append(z)
            hiddens.append(h)
        stack = lambda xs: torch.stack(xs, dim=1)
        return {
            "recon": stack(recons),
            "reward": stack(rewards),
            "value": stack(values),
            "prior_mean": stack(prior_means),
            "prior_scale": stack(prior_scales),
            "post_mean": stack(post_means),
            "post_scale": stack(post_scales),
            "z": stack(latents),
            "h": stack(hiddens),
        }

    @torch.no_grad()
    def imagine_score(self, obs0: np.ndarray, actions: np.ndarray) -> dict[str, float]:
        """Score an action sequence by latent reward plus terminal value."""

        self.eval()
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs0[None, :], dtype=torch.float32, device=device)
        h, z = self.initial(1, device)
        embed = self.encoder(obs)
        post_mean, post_scale = self.split_stats(self.posterior(torch.cat([h, embed], dim=-1)))
        z = post_mean
        total = 0.0
        uncertainty = 0.0
        decoder_error = 0.0
        for t, action in enumerate(np.asarray(actions, dtype=float).reshape(-1, 1)):
            a = torch.as_tensor(action[None, :], dtype=torch.float32, device=device)
            h = self.gru(torch.cat([z, a], dim=-1), h)
            prior_mean, prior_scale = self.split_stats(self.prior(h))
            z = prior_mean
            hz = torch.cat([h, z], dim=-1)
            reward = self.reward_head(hz).item()
            total += (0.97**t) * reward
            uncertainty += float(torch.mean(prior_scale).item())
            recon = self.decoder(hz).detach().cpu().numpy()[0]
            decoder_error += float(np.mean((recon - obs0) ** 2))
        terminal = self.value_head(torch.cat([h, z], dim=-1)).item()
        return {
            "score": float(total + 0.97 ** len(actions) * terminal),
            "uncertainty": float(uncertainty / max(1, len(actions))),
            "decoder_error": float(decoder_error / max(1, len(actions))),
            "posterior_scale": float(torch.mean(post_scale).item()),
        }


def make_sequence_dataset(config: RSSMTrainConfig) -> dict[str, np.ndarray]:
    """Generate small hidden-mode trajectories for RSSM training."""

    rng = np.random.default_rng(config.seed)
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.45, clue_strength=0.12, observation_noise=0.08))
    obs = np.zeros((config.num_sequences, config.seq_len, config.obs_dim), dtype=np.float32)
    actions = np.zeros((config.num_sequences, config.seq_len, config.action_dim), dtype=np.float32)
    rewards = np.zeros((config.num_sequences, config.seq_len, 1), dtype=np.float32)
    returns = np.zeros_like(rewards)
    modes: list[str] = []
    for b in range(config.num_sequences):
        mode = env.sample_mode(rng)
        modes.append(mode)
        position = 0.0
        for t in range(config.seq_len):
            action = float(np.clip(rng.beta(1.8, 1.4) + rng.normal(0.0, 0.08), 0.0, 1.0))
            reward = env.real_step_reward(mode, action, t)
            position += (0.10 + action) * (1.0 if mode == "free" else 0.35)
            clue_obs = env.observe(mode, rng)
            obs[b, t] = np.array([position / (t + 1.0), clue_obs[0], action, 1.0], dtype=np.float32)
            actions[b, t, 0] = action
            rewards[b, t, 0] = reward
        running = 0.0
        for t in reversed(range(config.seq_len)):
            running = float(rewards[b, t, 0]) + 0.97 * running
            returns[b, t, 0] = running
    return {"obs": obs, "actions": actions, "rewards": rewards, "returns": returns, "modes": np.asarray(modes)}


def train_rssm(config: RSSMTrainConfig | None = None) -> tuple[TinyRSSM, dict[str, float], dict[str, np.ndarray]]:
    """Train the compact RSSM for a few CPU epochs."""

    config = config or RSSMTrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    data = make_sequence_dataset(config)
    model = TinyRSSM(config.obs_dim, config.action_dim, config.hidden_dim, config.latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)
    rewards = torch.as_tensor(data["rewards"], dtype=torch.float32)
    returns = torch.as_tensor(data["returns"], dtype=torch.float32)
    last_losses: dict[str, float] = {}
    for _ in range(config.epochs):
        out = model(obs, actions, deterministic=False)
        recon_loss = F.mse_loss(out["recon"], obs)
        reward_loss = F.mse_loss(out["reward"], rewards)
        value_loss = F.mse_loss(out["value"], returns)
        kl = model.gaussian_kl(out["post_mean"], out["post_scale"], out["prior_mean"], out["prior_scale"]).mean()
        loss = recon_loss + reward_loss + 0.35 * value_loss + 0.02 * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        last_losses = {
            "loss": float(loss.item()),
            "recon_loss": float(recon_loss.item()),
            "reward_loss": float(reward_loss.item()),
            "value_loss": float(value_loss.item()),
            "kl": float(kl.item()),
        }
    return model, last_losses, data


def learned_rssm_candidate_pool(
    model: TinyRSSM,
    n: int = 700,
    horizon: int = 5,
    seed: int = 0,
) -> list[RolloutRecord]:
    """Evaluate latent RSSM scores against separately executed utility."""

    rng = np.random.default_rng(seed)
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.68, clue_strength=0.10, observation_noise=0.09))
    records: list[RolloutRecord] = []
    for i in range(int(n)):
        mode = env.sample_mode(rng)
        obs0 = env.observe(mode, rng)
        risk = float(rng.beta(1.55, 1.15))
        actions = np.clip(risk + rng.normal(0.0, 0.12, size=int(horizon)), 0.0, 1.0)
        risk = float(np.mean(actions))
        real = env.execute(mode, actions)
        imagined = model.imagine_score(obs0, actions)
        # The evaluation distribution is more blocked than the training data,
        # creating a controlled prior-shift optimism in the learned value tail.
        value_pred = float(imagined["score"] + 0.45 * horizon * risk)
        posterior_free = float(np.clip(0.50 + 1.1 * (float(obs0[0]) - 0.5), 0.04, 0.96))
        imagined_free = float(np.clip(0.60 + 0.32 * risk, 0.0, 1.0))
        pp_kl = float(abs(imagined_free - posterior_free) * (0.7 + risk))
        uncertainty = float(imagined["uncertainty"] + 0.35 * risk + 0.25 * pp_kl)
        decoder_error = float(imagined["decoder_error"] + risk * max(0.0, imagined_free - float(obs0[0])))
        diagnostics = {
            "good_score": real + rng.normal(0.0, 0.10),
            "oracle_score": real,
            "random_score": rng.normal(),
            "belief_collapsed_score": value_pred,
            "overconfident_score": value_pred + 0.25 * horizon * risk,
            "value_optimistic_score": value_pred + 0.75 * horizon * risk * risk,
        }
        records.append(
            RolloutRecord(
                seed=int(seed),
                state_id=0,
                candidate_id=i,
                hidden_mode=mode,
                horizon=int(horizon),
                actions=actions.astype(float),
                observation=obs0.astype(float),
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
                diagnostics=diagnostics,
            )
        )
    return records

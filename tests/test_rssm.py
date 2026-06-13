from __future__ import annotations

import numpy as np
import torch

from rssm_tail_audit.rssm import RSSMTrainConfig, TinyRSSM, learned_rssm_candidate_pool, train_rssm


def test_rssm_tensor_shapes():
    model = TinyRSSM(obs_dim=4, action_dim=1, hidden_dim=12, latent_dim=5)
    obs = torch.zeros(3, 4, 4)
    actions = torch.zeros(3, 4, 1)
    out = model(obs, actions, deterministic=True)
    assert out["recon"].shape == (3, 4, 4)
    assert out["reward"].shape == (3, 4, 1)
    assert out["value"].shape == (3, 4, 1)
    assert out["prior_mean"].shape == (3, 4, 5)
    assert torch.isfinite(out["post_scale"]).all()


def test_rssm_training_and_learned_pool_smoke():
    cfg = RSSMTrainConfig(num_sequences=12, epochs=1, seq_len=4, hidden_dim=12, latent_dim=5, seed=8)
    model, losses, _ = train_rssm(cfg)
    assert np.isfinite(list(losses.values())).all()
    records = learned_rssm_candidate_pool(model, n=20, horizon=3, seed=9)
    assert len(records) == 20
    assert np.isfinite([r.value_pred for r in records]).all()
    assert np.isfinite([r.real_utility for r in records]).all()

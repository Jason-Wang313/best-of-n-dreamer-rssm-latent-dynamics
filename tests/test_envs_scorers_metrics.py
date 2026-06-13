from __future__ import annotations

import numpy as np

from rssm_tail_audit.envs import HiddenModeConfig, HiddenModeToyEnv
from rssm_tail_audit.gate import deployment_gate
from rssm_tail_audit.metrics import bootstrap_ci, high_n_delta, selection_curves, top_tail_diagnostics
from rssm_tail_audit.scorers import fit_pilot_calibrator, score_records


def test_toy_environment_deterministic_hidden_ambiguity():
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.7))
    records_a = env.generate_candidate_pool(n=40, horizon=4, seed=3)
    records_b = env.generate_candidate_pool(n=40, horizon=4, seed=3)
    assert [r.hidden_mode for r in records_a] == [r.hidden_mode for r in records_b]
    assert np.allclose([r.real_utility for r in records_a], [r.real_utility for r in records_b])
    modes = {r.hidden_mode for r in records_a}
    assert "free" in modes and any(m != "free" for m in modes)
    obs_by_mode = {}
    for r in records_a:
        obs_by_mode.setdefault(r.hidden_mode, []).append(r.observation[0])
    assert max(np.mean(v) for v in obs_by_mode.values()) - min(np.mean(v) for v in obs_by_mode.values()) < 0.35


def test_scorer_repairs_penalize_hallucinated_tail():
    env = HiddenModeToyEnv()
    records = env.generate_candidate_pool(n=160, horizon=5, seed=5, model_flavor="belief_collapsed")
    calibrator = fit_pilot_calibrator(records[:64])
    raw = score_records(records, "raw_value")
    repaired = score_records(records, "combined_repair", calibrator=calibrator)
    ensemble = score_records(records, "ensemble_uncertainty_repair", calibrator=calibrator)
    risky = np.argsort([r.uncertainty + r.decoder_error + r.posterior_prior_kl for r in records])[-20:]
    assert np.mean(repaired[risky] - raw[risky]) < -0.25
    assert ensemble.shape == raw.shape
    assert np.isfinite(ensemble).all()
    oracle = score_records(records, "oracle")
    assert np.allclose(oracle, [r.real_utility for r in records])


def test_metrics_and_gate_outputs():
    env = HiddenModeToyEnv()
    records = env.generate_candidate_pool(n=120, horizon=4, seed=6)
    scores = score_records(records, "belief_collapsed")
    curves = selection_curves(records, scores, [1, 4, 16])
    tail = top_tail_diagnostics(records, scores, top_fraction=0.2)
    assert len(curves) == 3
    assert "tail_gap" in tail
    assert bootstrap_ci([1.0, 2.0, 3.0], seed=1)["n"] == 3.0
    decision = deployment_gate(
        real_delta_high_n=high_n_delta(curves, "selected_real_utility"),
        latent_delta_high_n=high_n_delta(curves, "selected_latent_value"),
        tail_real_minus_population=tail["tail_real_mean"] - tail["population_real_mean"],
        tail_uncertainty=tail["tail_uncertainty_mean"],
        pilot_labels=80,
    )
    assert decision in {"allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"}
    assert (
        deployment_gate(
            real_delta_high_n=-0.2,
            latent_delta_high_n=1.0,
            tail_real_minus_population=-0.3,
            tail_uncertainty=0.1,
            pilot_labels=100,
        )
        == "block_high_n"
    )

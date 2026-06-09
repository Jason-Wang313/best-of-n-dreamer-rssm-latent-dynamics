from __future__ import annotations

import numpy as np

from experiments.common import figures_dir, results_dir, root_from_file, tables_dir
from experiments.experiment_f_closed_loop_planning import (
    _controlled_candidates,
    _make_episode_pools,
    _score_episode_pools,
)
from experiments.experiment_h_ood_stress_grid import classify_regime
from latent_dynamics_best_of_n.envs import HiddenModeConfig, HiddenModeToyEnv
from latent_dynamics_best_of_n.gym_benchmarks import (
    BENCHMARKS,
    expected_return,
    generate_benchmark_records,
    seeded_start_state,
    valid_action_sequence,
)
from latent_dynamics_best_of_n.leakage import audit_calibration_split, build_leakage_report, deterministic_split
from latent_dynamics_best_of_n.scorers import fit_pilot_calibrator


def test_leakage_audit_sentinel_detection():
    pilot, eval_idx = deterministic_split(20, 6, seed=1)
    clean = audit_calibration_split(pilot, eval_idx, labels_used_indices=pilot, scored_indices=np.arange(20))
    leaky = audit_calibration_split(pilot, eval_idx, labels_used_indices=np.r_[pilot, eval_idx[:2]])
    report = build_leakage_report(40, 10, seed=2)
    assert clean.passed
    assert not leaky.passed
    assert report["passed"] is True


def test_closed_loop_deterministic_and_finite_returns():
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.75, clue_strength=0.08))
    pilot = env.generate_candidate_pool(n=80, horizon=4, seed=3)
    calibrator = fit_pilot_calibrator(pilot)
    make = lambda mode, obs, n, state_id, cand_seed: _controlled_candidates(env, mode, obs, n, 4, cand_seed, state_id)
    mode_a, pools_a = _make_episode_pools(env, make, episode_seed=44, episode_horizon=3)
    mode_b, pools_b = _make_episode_pools(env, make, episode_seed=44, episode_horizon=3)
    result_a = _score_episode_pools(env, mode_a, pools_a, "combined_repair", 8, calibrator)
    result_b = _score_episode_pools(env, mode_b, pools_b, "combined_repair", 8, calibrator)
    assert mode_a == mode_b
    assert np.isfinite(result_a["return"])
    assert result_a == result_b


def test_ood_regime_classification_boundaries():
    assert classify_regime(-0.30) == "harm"
    assert classify_regime(0.30) == "helpful"
    assert classify_regime(0.0) == "neutral"


def test_smoke_artifact_paths_are_separate():
    root = root_from_file()
    assert results_dir(root, smoke=True) != results_dir(root, smoke=False)
    assert tables_dir(root, smoke=True).as_posix().endswith("results/smoke/tables")
    assert figures_dir(root, smoke=True).as_posix().endswith("figures/smoke")


def test_gymnasium_benchmarks_seeded_valid_and_finite():
    for name, spec in BENCHMARKS.items():
        start = seeded_start_state(spec.env_id, seed=5)
        records = generate_benchmark_records(spec, n=4, seed=6, start_state=start)
        assert len(records) == 4
        for record in records:
            actions = [int(a) for a in record.actions]
            assert valid_action_sequence(spec.env_id, actions)
            assert np.isfinite(expected_return(spec.env_id, start, actions, spec.discount))
            assert np.isfinite(record.value_pred)
            assert np.isfinite(record.real_utility)

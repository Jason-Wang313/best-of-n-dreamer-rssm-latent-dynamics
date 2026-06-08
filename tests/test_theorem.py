from __future__ import annotations

import itertools

import numpy as np

from latent_dynamics_best_of_n.theorem import (
    auc_kappa,
    binary_best_of_n_finite,
    n2_auc_identity,
    simulate_best_of_n,
    utility_best_of_n_finite,
)


def brute_force(scores, utility, N):
    scores = np.asarray(scores, dtype=float)
    utility = np.asarray(utility, dtype=float)
    total = 0.0
    count = 0
    for tup in itertools.product(range(len(scores)), repeat=N):
        drawn = scores[list(tup)]
        max_score = drawn.max()
        tied = np.flatnonzero(drawn == max_score)
        total += float(np.mean(utility[[tup[i] for i in tied]]))
        count += 1
    return total / count


def test_finite_law_matches_bruteforce_with_ties():
    scores = [0.0, 1.0, 1.0, 2.0]
    utility = [0.0, 1.0, 3.0, -2.0]
    exact = utility_best_of_n_finite(scores, utility, [1, 2, 3])
    assert exact[1] == np.mean(utility)
    assert np.isclose(exact[2], brute_force(scores, utility, 2))
    assert np.isclose(exact[3], brute_force(scores, utility, 3))


def test_binary_law_auc_identity_and_edges():
    scores = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    success = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    curve = binary_best_of_n_finite(scores, success, [1, 2])
    assert np.isclose(curve[1], np.mean(success))
    assert np.isclose(curve[2], n2_auc_identity(np.mean(success), auc_kappa(scores, success)))
    assert binary_best_of_n_finite(scores, np.ones_like(scores), [1, 64])[64] == 1.0
    assert binary_best_of_n_finite(scores, np.zeros_like(scores), [1, 64])[64] == 0.0


def test_oracle_anti_aligned_and_constant_cases():
    utility = np.linspace(-1.0, 2.0, 8)
    oracle = utility_best_of_n_finite(utility, utility, [1, 2, 8])
    anti = utility_best_of_n_finite(-utility, utility, [1, 2, 8])
    const = utility_best_of_n_finite(utility, np.ones_like(utility) * 4.2, [1, 8])
    assert oracle[8] > oracle[1]
    assert anti[8] < anti[1]
    assert const == {1: 4.2, 8: 4.2}


def test_monte_carlo_agrees_with_exact_law():
    rng = np.random.default_rng(7)
    utility = rng.normal(size=80)
    scores = np.round(utility + rng.normal(scale=0.4, size=80), 1)
    exact = utility_best_of_n_finite(scores, utility, [8])[8]
    mc = simulate_best_of_n(scores, utility, 8, n_trials=12000, seed=9)
    assert abs(exact - mc) < 0.08

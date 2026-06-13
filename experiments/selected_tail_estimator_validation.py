"""Validate the finite selected-tail estimator against Monte Carlo simulation."""

from __future__ import annotations

import numpy as np

from rssm_tail_audit.selected_tail import (
    auc_kappa,
    binary_tail_selection_finite,
    n2_auc_identity,
    simulate_tail_selection,
    utility_tail_selection_finite,
)

from experiments.common import (
    N_GRID,
    ensure_dirs,
    figures_dir,
    plot_curve,
    results_dir,
    root_from_file,
    smoke_argparser,
    tables_dir,
    write_json,
)


def run(smoke: bool = False, seed: int = 5):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    rng = np.random.default_rng(seed)
    n = 160 if smoke else 480
    utility = rng.normal(0.0, 1.0, size=n)
    score = np.round(utility + rng.normal(0.0, 0.6, size=n), 1)
    success = (utility > np.median(utility)).astype(float)
    trials = 4000 if smoke else 14000
    rows = []
    max_abs_error = 0.0
    for N in N_GRID:
        exact = utility_tail_selection_finite(score, utility, [N])[N]
        mc = simulate_tail_selection(score, utility, N=N, n_trials=trials, seed=seed + N)
        err = abs(exact - mc)
        max_abs_error = max(max_abs_error, err)
        rows.append(
            {
                "scorer": "finite_law",
                "N": N,
                "selected_real_utility": exact,
                "selected_latent_value": mc,
                "selected_value_pred": mc,
                "selected_risk": err,
                "latent_real_gap": mc - exact,
            }
        )
    p = float(np.mean(success))
    kappa = auc_kappa(score, success)
    binary_n2 = binary_tail_selection_finite(score, success, [2])[2]
    summary = {
        "experiment": "exact_finite_law_validation",
        "n_records": n,
        "n_trials": trials,
        "max_abs_error": max_abs_error,
        "binary_n2_exact": binary_n2,
        "binary_n2_auc_identity": n2_auc_identity(p, kappa),
        "constant_utility_curve": utility_tail_selection_finite(score, np.ones_like(score) * 3.5, [1, 16, 64]),
        "oracle_curve": utility_tail_selection_finite(utility, utility, [1, 16, 64]),
        "anti_aligned_curve": utility_tail_selection_finite(-utility, utility, [1, 16, 64]),
    }
    write_json(results_dir(root, smoke) / "selected_tail_estimator_validation.json", summary)
    import pandas as pd

    pd.DataFrame(rows).to_csv(tables_dir(root, smoke) / "selected_tail_estimator_validation.csv", index=False)
    plot_curve(
        rows,
        figures_dir(root, smoke) / "figure5_selected_tail_estimator.png",
        ["finite_law"],
        ["selected_real_utility", "selected_latent_value"],
        "finite selected-tail estimator matches Monte Carlo",
        "Selected utility",
    )
    return summary


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

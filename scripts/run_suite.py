"""Run the experiment suite in one Python process."""

from __future__ import annotations

import argparse

from experiments.exact_law_validation import run as run_exact
from experiments.experiment_a_toy_mismatch import run as run_a
from experiments.experiment_b_learned_rssm import run as run_b
from experiments.experiment_c_belief_collapse import run as run_c
from experiments.experiment_d_horizon_budget import run as run_d
from experiments.experiment_e_repairs import run as run_e
from experiments.multiseed_evidence import run as run_multiseed
from experiments.tail_diagnostics import run as run_tail
from scripts.run_claim_audit import main as run_claim_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed-base", type=int, default=20)
    args = parser.parse_args()
    seeds = {
        "exact": args.seed_base + 1,
        "a": args.seed_base + 2,
        "b": args.seed_base + 3,
        "c": args.seed_base + 4,
        "d": args.seed_base + 5,
        "e": args.seed_base + 6,
    }
    steps = [
        ("experiments/exact_law_validation.py", lambda: run_exact(smoke=args.smoke, seed=seeds["exact"])),
        ("experiments/experiment_a_toy_mismatch.py", lambda: run_a(smoke=args.smoke, seed=seeds["a"])),
        ("experiments/experiment_b_learned_rssm.py", lambda: run_b(smoke=args.smoke, seed=seeds["b"])),
        ("experiments/experiment_c_belief_collapse.py", lambda: run_c(smoke=args.smoke, seed=seeds["c"])),
        ("experiments/experiment_d_horizon_budget.py", lambda: run_d(smoke=args.smoke, seed=seeds["d"])),
        ("experiments/experiment_e_repairs.py", lambda: run_e(smoke=args.smoke, seed=seeds["e"])),
        ("experiments/tail_diagnostics.py", lambda: run_tail(smoke=args.smoke, seed=seeds["a"])),
        ("experiments/multiseed_evidence.py", lambda: run_multiseed(smoke=args.smoke, seed=args.seed_base + 100)),
    ]
    for label, fn in steps:
        print(f"[suite] running {label}", flush=True)
        fn()
    print("[suite] running scripts/run_claim_audit.py", flush=True)
    run_claim_audit()


if __name__ == "__main__":
    main()

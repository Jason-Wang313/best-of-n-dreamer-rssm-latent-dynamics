# Final Audit

The repository is designed around reproducible local commands:

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

Expected generated artifacts:

- `results/selected_tail_estimator_validation.json`
- `results/experiment_a_toy_mismatch.json`
- `results/experiment_b_learned_rssm.json`
- `results/experiment_c_belief_collapse.json`
- `results/experiment_d_horizon_budget.json`
- `results/experiment_e_repairs.json`
- `results/experiment_j_belief_interventions.json`
- `results/multiseed_strong_evidence.json`
- `results/claims_status.json`
- ten required figures under `figures/`, including `figures/figure10_belief_interventions.png`

The claim audit is intentionally conservative. Full runs fail if a paper-level claim lacks strong multi-seed support, if a forbidden overclaim appears in docs or paper files, or if any claim is unsupported.

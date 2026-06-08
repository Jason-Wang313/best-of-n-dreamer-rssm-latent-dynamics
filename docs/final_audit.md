# Final Audit

The repository is designed around reproducible local commands:

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

Expected generated artifacts:

- `results/exact_law_validation.json`
- `results/experiment_a_toy_mismatch.json`
- `results/experiment_b_learned_rssm.json`
- `results/experiment_c_belief_collapse.json`
- `results/experiment_d_horizon_budget.json`
- `results/experiment_e_repairs.json`
- `results/claims_status.json`
- five required figures under `figures/`

The claim audit is intentionally conservative. It allows partial repair or learned-model claims when evidence exists but thresholds are not strong enough for a full support label.

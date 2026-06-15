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
- `results/experiment_f_closed_loop_planning.json`
- `results/experiment_g_label_budget_ablation.json`
- `results/experiment_h_ood_stress_grid.json`
- `results/experiment_i_gymnasium_benchmarks.json`
- `results/experiment_j_belief_interventions.json`
- `results/experiment_k_classic_control_benchmarks.json`
- `results/multiseed_strong_evidence.json`
- `results/claims_status.json`
- eleven required figures under `figures/`, including `figures/figure10_belief_interventions.png` and `figures/figure11_classic_control_benchmarks.png`

The claim audit is intentionally conservative. Full runs fail if a paper-level claim lacks strong multi-seed support, if a forbidden overclaim appears in docs or paper files, or if any claim is unsupported.

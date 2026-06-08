# Claim Status

- Project: When Latent Imagination Lies
- Unsupported count: 0
- Weak strong-evidence checks: 0
- Full multi-seed evidence: True

| Category | Status | Strength | Claim | Evidence |
| --- | --- | --- | --- | --- |
| theorem claims | SUPPORTED | STRONG | The finite tie-aware Best-of-N law exactly predicts selected utility from an empirical score/utility pool; Monte Carlo agrees within sampling error. | results/exact_law_validation.json |
| controlled latent dynamics claims | SUPPORTED | STRONG | In the controlled hidden-mode toy, high N can inflate selected imagined latent value while selected executed real utility stagnates or drops. | results/experiment_a_toy_mismatch.json; figures/figure1_latent_mismatch.png |
| learned RSSM-style claims | SUPPORTED | STRONG | A CPU-trained small RSSM-style model with encoder, recurrent state, stochastic prior/posterior, decoder, reward head, and value head reproduces the latent-real selection mismatch in a smoke-scale learned setting. | results/experiment_b_learned_rssm.json; results/learned_tiny_rssm.pt |
| hidden-mode belief-collapse claims | SUPPORTED | STRONG | Under ambiguous hidden modes, high-N latent selection concentrates in optimistic imagined modes even when many selected candidates execute in blocked/slip/heavy modes. | results/experiment_c_belief_collapse.json |
| horizon/selection-budget claims | SUPPORTED | STRONG | Varying N and H shows where longer or fatter imagination worsens latent-real mismatch. | results/experiment_d_horizon_budget.json; figures/figure4_horizon_budget.png |
| repair claims | SUPPORTED | STRONG | RSSM-specific repairs can substantially recover selected-tail real utility over raw latent value selection; near-oracle is reported only when achieved. | results/experiment_e_repairs.json; figures/figure2_repair_comparison.png |
| optional benchmark claims | SUPPORTED | STRONG | No external benchmark, full Dreamer benchmark, or real-world robotic benchmark is implemented in this repo. | Repository scope and scripts are toy/RSSM-style CPU experiments only. |
| unsupported robotics claims | SUPPORTED | STRONG | Real robot validation is unsupported and must not be claimed. | No robot datasets, robot execution adapters, or robot result artifacts exist. |
| forbidden overclaims | SUPPORTED | STRONG | Forbidden universal or real-robot claims are blocked. | {"blocked_phrases": ["We solve Dreamer.", "We solve model-based RL.", "Best-of-N always hurts.", "More imagination always helps.", "Uncertainty always fixes the issue.", "We validate on real robots.", "This is not toy evidence", "This is just a renamed WAM project."], "forbidden_hits": []} |

## Blocked Overclaims

The following phrases are explicitly treated as forbidden publication claims:
- `We solve Dreamer.`
- `We solve model-based RL.`
- `Best-of-N always hurts.`
- `More imagination always helps.`
- `Uncertainty always fixes the issue.`
- `We validate on real robots.`
- `This is not toy evidence`
- `This is just a renamed WAM project.`

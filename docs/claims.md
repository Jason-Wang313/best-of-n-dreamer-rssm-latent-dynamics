# Claims

Claims are regenerated into `results/claims_status.md` by:

```bash
bash scripts/run_claim_audit.sh
```

The audit categories are:

- measurement-estimator claims
- controlled latent dynamics claims
- learned RSSM-style claims
- hidden-mode belief-collapse claims
- belief-intervention mechanism claims
- horizon/selection-budget claims
- repair claims
- closed-loop planning claims
- label-budget repair claims
- leakage-free calibration claims
- OOD stress-grid claims
- lightweight Gymnasium benchmark claims
- scope boundary claims
- unsupported robotics claims
- forbidden overclaim checks

Each claim is marked `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`.

For full runs, the audit also requires `STRONG` evidence for the paper-level claims. The strict checks cover seed-level latent-value inflation, real-utility drops or stagnation, repair gains, belief-collapse tail diagnostics, posterior-prior belief-intervention recovery, horizon/budget amplification margins, closed-loop repair recovery, label-budget recovery, leakage-free calibration, OOD regime diversity, and scoped Gymnasium benchmark evidence.

The audit explicitly blocks universal or out-of-scope statements, including claims that the work solves Dreamer, solves model-based RL, proves monotone harm or monotone benefit from more imagination, proves uncertainty always repairs the issue, validates real robots, upgrades lightweight toy-text evidence into broad RL validation, or merely renames the WAM project.

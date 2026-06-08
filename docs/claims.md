# Claims

Claims are regenerated into `results/claims_status.md` by:

```bash
bash scripts/run_claim_audit.sh
```

The audit categories are:

- theorem claims
- controlled latent dynamics claims
- learned RSSM-style claims
- hidden-mode belief-collapse claims
- horizon/selection-budget claims
- repair claims
- optional benchmark claims
- unsupported robotics claims
- forbidden overclaim checks

Each claim is marked `SUPPORTED`, `PARTIAL`, or `UNSUPPORTED`.

For full runs, the audit also requires `STRONG` evidence for the paper-level claims. The strict checks are based on `results/multiseed_strong_evidence.json`, including seed-level latent-value inflation, real-utility drops or stagnation, repair gains, belief-collapse tail diagnostics, and horizon/budget amplification margins.

The audit explicitly blocks universal or out-of-scope statements, including claims that the work solves Dreamer, solves model-based RL, proves monotone harm or monotone benefit from more imagination, proves uncertainty always repairs the issue, validates real robots, upgrades toy evidence into benchmark evidence, or merely renames the WAM project.

# When Latent Imagination Lies

This repository studies Best-of-N selection in Dreamer/RSSM-style latent world-model planning. The central question is simple: when a learned latent model imagines many futures and picks the highest latent value, does the selected action sequence actually execute well in the real hidden dynamics?

The supported thesis is intentionally bounded. In controlled CPU toy settings and a small learned RSSM-style model, increasing `N` can raise selected imagined latent value while selected executed utility stagnates or worsens. RSSM-specific diagnostics and repairs can recover much of the selected-tail real utility. This is toy and learned-small evidence, not a full Dreamer benchmark and not robot validation.

## Quickstart

From this directory:

```bash
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/run_claim_audit.sh
pytest
```

The scripts write results under `results/` and figures under `figures/`.

## Key Artifacts

- `figures/figure1_latent_mismatch.png`: selected latent value rises with `N` while executed utility falls or stalls.
- `figures/figure2_repair_comparison.png`: raw value, repair scorers, random, and oracle comparison.
- `figures/figure3_tail_diagnostics.png`: selected-tail latent-real gaps.
- `figures/figure4_horizon_budget.png`: horizon and selection-budget sweep.
- `figures/figure5_exact_law_validation.png`: exact finite law versus Monte Carlo.
- `results/claims_status.md`: claim audit with `SUPPORTED`, `PARTIAL`, and `UNSUPPORTED` statuses.
- `results/multiseed_strong_evidence.json`: seed-level effect-size evidence used by the strict claim audit.
- `results/learned_tiny_rssm.pt`: small trained RSSM-style PyTorch artifact.

## What Is Reused From WAM

Only the abstract finite tie-aware Best-of-N selection law is reused: for a finite pool of candidates with score `S` and measured utility `R`, the exact expected utility of the top-score Best-of-N selection is determined by score tie groups and their utility means.

Everything scientific here is different: RSSM-like belief states, stochastic latent imagination, learned latent reward/value scoring, posterior-prior diagnostics, decoder/state consistency, hidden-mode belief collapse, pilot calibration, and utility measured only after executing selected actions in the toy dynamics.

## Scope Boundaries

This repo does not claim to solve Dreamer or model-based reinforcement learning. It does not claim that larger `N` is always harmful or always helpful. It does not claim uncertainty is a universal repair. It does not include real-robot evidence, full Dreamer benchmarks, or external RL benchmark suites.

The intended use is as a compact research scaffold for studying selected-tail failures in latent imagination and for testing whether architecture-aware diagnostics can make high-`N` planning safer.

The full claim audit requires every paper-level claim to clear strong evidence checks: multi-seed selected-tail effect sizes, repair margins, explicit scope boundaries, and forbidden-overclaim scanning. Smoke runs use smaller artifacts for speed; run `bash scripts/run_all.sh` before publication-style inspection.

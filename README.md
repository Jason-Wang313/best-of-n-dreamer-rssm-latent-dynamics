# Belief-Tail Audits for RSSM World Models

This repository studies candidate-budget selection in Dreamer/RSSM-style latent world-model planning. The central question is simple: when a learned latent model imagines many futures and picks the highest latent value, does the selected action sequence actually execute well in the real hidden dynamics?

The supported thesis is intentionally bounded. In controlled CPU toy settings, a small learned RSSM-style model, belief-intervention stress tests, three lightweight Gymnasium toy-text benchmarks, and three standard Gymnasium classic-control tasks, increasing `N` can raise selected imagined latent value while selected executed utility stagnates or worsens in diagnosed regimes. RSSM-specific diagnostics and repairs can recover much of the selected-tail real utility. This is controlled and standard low-dimensional benchmark evidence, not a full Dreamer benchmark and not robot validation.

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

- `paper/main.tex`: anonymous ICLR-style manuscript source.
- `paper/references.bib`: manuscript bibliography.
- `paper/build_submission.ps1`: local PDF build helper using `pdflatex` and `bibtex`.
- `figures/figure1_latent_mismatch.png`: selected latent value rises with `N` while executed utility falls or stalls.
- `figures/figure2_repair_comparison.png`: raw value, repair scorers, random, and oracle comparison.
- `figures/figure3_tail_diagnostics.png`: selected-tail latent-real gaps.
- `figures/figure4_horizon_budget.png`: horizon and selection-budget sweep.
- `figures/figure5_selected_tail_estimator.png`: finite selected-tail estimator versus Monte Carlo.
- `figures/figure6_closed_loop_planning.png`: receding-horizon controlled and learned planning.
- `figures/figure7_label_budget_repair.png`: repair recovery versus pilot-label budget.
- `figures/figure8_ood_stress_grid.png`: OOD hidden-mode regime classification.
- `figures/figure9_gymnasium_benchmarks.png`: lightweight Gymnasium toy-text benchmark results.
- `figures/figure10_belief_interventions.png`: posterior-prior and belief-collapse intervention stress.
- `figures/figure11_classic_control_benchmarks.png`: standard Gymnasium classic-control rollout stress tests.
- `results/claims_status.md`: claim audit with `SUPPORTED`, `PARTIAL`, and `UNSUPPORTED` statuses.
- `results/experiment_j_belief_interventions.json`: mechanism stress test for RSSM belief diagnostics.
- `results/leakage_audit.json`: pilot/eval split audit with a deliberately leaky sentinel.
- `results/multiseed_strong_evidence.json`: seed-level effect-size evidence used by the strict claim audit.
- `results/learned_tiny_rssm.pt`: small trained RSSM-style PyTorch artifact.

## What Is Reused From WAM

Only the abstract finite selected-tail estimator is reused: for a finite pool of candidates with score `S` and measured utility `R`, the exact expected utility of the top-score candidate selection is determined by score tie groups and their utility means.

Everything scientific here is different: RSSM-like belief states, stochastic latent imagination, learned latent reward/value scoring, posterior-prior diagnostics, decoder/state consistency, hidden-mode belief collapse, pilot calibration, and utility measured only after executing selected actions in the toy dynamics.

## Scope Boundaries

This repo does not claim to solve Dreamer or model-based reinforcement learning. It does not claim that larger `N` is always harmful or always helpful. It does not claim uncertainty is a universal repair. It does not include real-robot evidence, full Dreamer benchmarks, high-dimensional visual-control results, or broad external RL benchmark suites.

The intended use is as a compact research scaffold for studying selected-tail failures in latent imagination and for testing whether architecture-aware diagnostics can make high-`N` planning safer.

The full claim audit requires every paper-level claim to clear strong evidence checks: multi-seed selected-tail effect sizes, repair margins, explicit scope boundaries, and forbidden-overclaim scanning. Smoke runs use smaller artifacts for speed; run `bash scripts/run_all.sh` before publication-style inspection.

## Paper Build

From the repository root:

```powershell
pwsh paper/build_submission.ps1
```

The default final artifact is `paper/final/best of n dreamer rssm latent dynamics-v4.pdf`. Desktop copying is intentionally opt-in and should happen only after the source is committed and pushed.

Before using the PDF as a review artifact, run:

```bash
bash scripts/run_claim_audit.sh
pytest
```

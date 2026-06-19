# Final V4 Audit

Paper: `best of n dreamer rssm latent dynamics-v4.pdf`

Source folder: `C:\Users\wangz\best of n dreamer rssm latent dynamics`

GitHub remote: `https://github.com/Jason-Wang313/best-of-n-dreamer-rssm-latent-dynamics.git`

Verified on: 2026-06-19

## Final Artifact

- Repository PDF: `paper/final/best of n dreamer rssm latent dynamics-v4.pdf`
- Desktop PDF: `C:\Users\wangz\OneDrive\Desktop\best of n dreamer rssm latent dynamics-v4.pdf`
- SHA256: `190019F7906D9DD5A91C57F4CC4738DF5816FB35CB940350A2BB9621892FDB0E`
- Page count: 11
- Repo/Desktop hash match: yes

## Verification Commands

All checks below were run against the v4 source tree and final PDF.

```powershell
python -m compileall src experiments scripts tests -q
python -m pytest -q
bash scripts/run_claim_audit.sh
powershell -ExecutionPolicy Bypass -File paper\build_submission.ps1 -DesktopCopy "C:\Users\wangz\OneDrive\Desktop\best of n dreamer rssm latent dynamics-v4.pdf"
rg -n "undefined|Citation.*undefined|Reference.*undefined|Rerun to get|Overfull|LaTeX Warning|Package natbib Warning" "paper\main.log"
pdfinfo "paper\final\best of n dreamer rssm latent dynamics-v4.pdf"
pdftoppm -png "paper\final\best of n dreamer rssm latent dynamics-v4.pdf" "tmp\pdfs\rssm_v4\page"
```

Results:

- Compile check: passed.
- Unit tests: 19 passed.
- Claim audit: passed with 0 unsupported claims and 0 weak strong-evidence checks.
- LaTeX log scan: no unresolved citations, unresolved references, rerun warnings, overfull boxes, or natbib warnings.
- PDF render: all 11 pages rendered.
- Visual QA: pages 1, 5, 7, 10, and 11 inspected for layout, citations, tables, figures, clipping, and appendix/provenance readability.

## Evidence Scope

The v4 paper is scoped as an RSSM-style belief-tail audit paper, not a generic Best-of-N wrapper. Its evidence includes:

- finite selected-tail estimator validation;
- controlled hidden-mode latent-real mismatch;
- a CPU-trained small RSSM-style model with encoder, recurrent state, stochastic prior/posterior, decoder, reward head, and value head;
- belief-collapse and posterior-prior drift interventions;
- horizon and candidate-budget stress sweeps;
- RSSM-aware repair comparisons against raw, oracle, and diagnostic baselines;
- receding-horizon closed-loop planning;
- label-budget repair ablations with leakage audit;
- OOD stress-grid analysis;
- three Gymnasium toy-text benchmarks;
- three standard Gymnasium classic-control tasks: CartPole-v1, MountainCar-v0, and Acrobot-v1.

## Claim Gates

The claim ledger reports:

- Unsupported count: 0.
- Weak strong-evidence checks: 0.
- Full multi-seed evidence: true.
- Standard benchmark scope: limited to lightweight Gymnasium toy-text and classic-control tasks.
- Explicit non-claims: no full Dreamer validation, no broad model-based RL coverage, and no robotics validation.

Forbidden overclaims are blocked by the claim audit, including universal statements about Dreamer/model-based RL and real-robot validation.

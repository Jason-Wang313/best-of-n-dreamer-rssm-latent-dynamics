# Reviewer Attacks

## "This is just order statistics."

The finite estimator is order-statistic reasoning, and the repo says so. The contribution is the RSSM-specific belief-tail audit and controlled evidence around latent imagination tails.

## "The experiments are toy."

Correct. The claim audit keeps the evidence bounded to toy and learned-small settings. The repo does not promote toy evidence into full benchmark evidence.

## "Repairs are hand-designed."

Correct. They are diagnostic repair baselines: uncertainty, posterior-prior consistency, decoder consistency, and pilot calibration. They test whether architecture-aware signals recover selected-tail utility in controlled settings.

## "Are the diagnostics mechanistic or just extra score features?"

Experiment J isolates posterior-prior and belief-collapse signals in high-risk seed-regime units. A diagnostic-only belief penalty improves high-budget selected real utility without pilot labels, and the raw selected tail's belief drift correlates with latent-real gap. The full repair still trails oracle selection, so the paper claims a useful RSSM mechanism rather than a complete fix.

## "Why not evaluate full Dreamer?"

That is outside this scaffold. The learned experiment is RSSM-style and intentionally small enough to run on CPU in CI-like conditions.

## "Does high N always hurt?"

No. Oracle scoring improves with `N`; bad or misaligned latent scoring can hurt. The audit blocks universal monotone claims.

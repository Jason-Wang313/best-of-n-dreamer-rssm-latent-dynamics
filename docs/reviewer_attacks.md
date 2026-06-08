# Reviewer Attacks

## "This is just order statistics."

The finite law is order-statistic reasoning, and the repo says so. The contribution is the RSSM-specific failure analysis and controlled evidence around latent imagination tails.

## "The experiments are toy."

Correct. The claim audit keeps the evidence bounded to toy and learned-small settings. The repo does not promote toy evidence into full benchmark evidence.

## "Repairs are hand-designed."

Correct. They are diagnostic repair baselines: uncertainty, posterior-prior consistency, decoder consistency, and pilot calibration. They test whether architecture-aware signals recover selected-tail utility in controlled settings.

## "Why not evaluate full Dreamer?"

That is outside this scaffold. The learned experiment is RSSM-style and intentionally small enough to run on CPU in CI-like conditions.

## "Does high N always hurt?"

No. Oracle scoring improves with `N`; bad or misaligned latent scoring can hurt. The audit blocks universal monotone claims.

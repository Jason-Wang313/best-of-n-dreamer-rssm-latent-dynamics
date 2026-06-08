# Method

For each planning state we generate a finite pool of action sequences. Each candidate receives latent diagnostics and a score from an RSSM-style model or analytic latent proxy. The selected utility curve for a scorer is computed exactly with the finite tie-aware Best-of-N law.

Repair scorers subtract penalties for uncertainty, posterior-prior disagreement, decoder inconsistency, and hidden-mode belief collapse. A pilot-calibrated scorer fits a small linear model from latent diagnostics to executed utility labels. The combined repair uses pilot calibration plus conservative penalties.

# Differentiation From Prior Projects

This repo is distinct from generic latent-world-model, diffusion-world-model, JEPA-style, and WAM-style projects.

- Compared with diffusion world-model work, this repo is about recurrent RSSM-like latent belief and stochastic prior/posterior dynamics, not denoising diffusion rollouts.
- Compared with JEPA-style latent prediction, this repo includes explicit action-sequence selection, reward/value scoring, and execution-only real utility.
- Compared with generic latent planning papers, this repo centers the selected high-score tail induced by Best-of-N sampling.
- Compared with WAM work, only the finite abstract selection law is reused.

The experiments are CPU-first controlled probes, not broad benchmark claims.

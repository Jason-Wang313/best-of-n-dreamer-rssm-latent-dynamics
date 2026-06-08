# Differentiation From Best-of-N WAM

The WAM project studied inference laws and rollout-selection value for world-action model planning. This repository studies RSSM-style latent world models.

Shared component:

- the abstract finite tie-aware Best-of-N law over a candidate pool with score `S` and measured utility `R`.

Different scientific object:

- RSSM-like recurrent belief state `h_t` and stochastic latent state `z_t`;
- prior/posterior disagreement;
- decoder and state consistency;
- learned latent reward and value heads;
- hidden-mode belief collapse;
- utility measured by executing the selected action sequence in hidden toy dynamics;
- architecture-specific repair scorers and a conservative deployment gate.

The law is used as an evaluation identity, not as a WAM-specific training or deployment recipe.

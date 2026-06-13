# Theory

## RSSM-Style Setup

At time `t`, the agent observes `o_t` and considers candidate action sequences `a_{t:t+H-1}`. An RSSM-style model maintains a deterministic recurrent belief state `h_t` and stochastic latent state `z_t`.

- Encoder: `e_t = enc(o_t)`.
- Recurrent state: `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`.
- Prior: `p_phi(z_t | h_t)`.
- Posterior: `q_phi(z_t | h_t, e_t)`.
- Decoder: `p_phi(o_t | h_t, z_t)`.
- Reward head: `r_phi(h_t, z_t, a_t)`.
- Value head: `v_phi(h_t, z_t)`.

For each candidate action sequence, the model imagines a latent rollout by sampling or taking means under the prior and accumulates a score `S`, often an imagined reward/value sum. Separately, the environment executes the selected action sequence and returns real utility `R`. The mismatch studied here is between selected latent score/value and selected executed utility.

## Candidate-Budget Selection

Given an empirical pool of `m` candidates, sample `N` candidates with replacement and select one with maximum score `S`. If several sampled candidates tie for maximum score, break the tie uniformly.

Sort the pool by score in ascending order. Let a score tie group `g` occupy 1-indexed ranks `[r_min_g, r_max_g]` and have mean measured utility `mean_R_g`. The exact expected selected utility is:

```text
E[R_selected(N)] =
  sum_g mean_R_g * ((r_max_g / m)^N - ((r_min_g - 1) / m)^N)
```

Binary success is the special case `R in {0, 1}`. Constant utility remains constant for every `N`. Oracle scoring `S = R` improves selected utility with `N` in the finite pool, while anti-aligned scoring `S = -R` worsens it. Monte Carlo validation is included as an experiment rather than treated as a new theorem.

For binary utility and `N=2`, with success rate `p` and tie-aware AUC `kappa = P(S+ > S-) + 0.5 P(S+ = S-)`, the success probability is:

```text
f_2 = p^2 + 2 p (1 - p) kappa
```

## RSSM-Specific Failure Definitions

Latent imagination-value mismatch is a positive selected latent value trend with flat or negative selected executed utility. The latent-real utility gap is `E[V_latent_selected] - E[R_selected]`. Selected latent value inflation is an increase in selected latent score with `N` that is not matched by executed utility.

Belief overconfidence occurs when posterior or prior uncertainty is low despite ambiguous observations. Posterior-prior divergence measures disagreement between the observation-conditioned posterior and the imagined prior. Epistemic uncertainty is approximated by prior scale or ensemble/dropout-style score spread. Value hallucination is high predicted value in candidates with poor executed utility. Decoder-consistency failure is high reconstruction or state-consistency error in selected imagined states. Hidden-mode belief collapse occurs when ambiguous physical modes are mapped into an optimistic latent mode. Large-budget latent regret is the oracle selected utility minus the selected utility under a latent scorer.

Repairs in this repo are diagnostic scorers: uncertainty pessimism, posterior-prior/belief consistency, decoder/state consistency, pilot-calibrated value, and combined conservative repair. A deployment gate can return exactly one of `allow_high_n`, `stop_early`, `collect_pilot_labels`, or `block_high_n`.

## Difference From WAM

The finite selected-tail estimator is abstract and applies to any finite score/utility pool. This repo uses it only as a measurement estimator. The modeled object is RSSM-style latent imagination, with hidden-mode execution and architecture-specific diagnostics. No WAM training recipe, WAM rollout benchmark, or WAM-specific claim is made here.

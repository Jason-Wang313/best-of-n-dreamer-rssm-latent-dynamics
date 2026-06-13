# Introduction

Large-candidate imagination is attractive because it turns extra model samples into apparently better plans. In latent world models, however, the selected object is the model's latent score, not real executed utility. If the high-score latent tail is hallucinated or overconfident, additional samples can select increasingly unrealistic candidates.

This project studies that issue in RSSM-style latent dynamics. The experiments are small enough to run locally while preserving the architectural ingredients needed for the failure: recurrent belief, stochastic latents, prior/posterior mismatch, decoder consistency, learned reward/value scoring, and execution-only utility.

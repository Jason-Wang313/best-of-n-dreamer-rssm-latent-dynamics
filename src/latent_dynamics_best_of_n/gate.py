"""Conservative deployment gate for high-N latent imagination selection."""

from __future__ import annotations

from typing import Literal


GateDecision = Literal["allow_high_n", "stop_early", "collect_pilot_labels", "block_high_n"]


def deployment_gate(
    *,
    real_delta_high_n: float,
    latent_delta_high_n: float,
    tail_real_minus_population: float,
    tail_uncertainty: float,
    pilot_labels: int,
    min_pilot_labels: int = 64,
    harm_margin: float = 0.05,
    uncertainty_threshold: float = 1.10,
) -> GateDecision:
    """Return exactly one high-N deployment decision."""

    if real_delta_high_n < -abs(harm_margin) or tail_real_minus_population < -abs(harm_margin):
        return "block_high_n"
    if int(pilot_labels) < int(min_pilot_labels) or tail_uncertainty > uncertainty_threshold:
        return "collect_pilot_labels"
    if latent_delta_high_n > 0.0 and real_delta_high_n <= harm_margin:
        return "stop_early"
    return "allow_high_n"

"""Belief-tail audits for RSSM-style latent imagination mismatch."""

from .selected_tail import binary_tail_selection_finite, simulate_tail_selection, utility_tail_selection_finite

__all__ = [
    "binary_tail_selection_finite",
    "simulate_tail_selection",
    "utility_tail_selection_finite",
]

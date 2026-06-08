"""Checkout-local import shim.

The installable package lives under ``src/``. This shim lets Windows Python run
``python.exe -m experiments...`` from a Bash/WSL checkout without relying on
environment-variable propagation.
"""

from __future__ import annotations

from pathlib import Path

_REAL_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "latent_dynamics_best_of_n"
__path__ = [str(_REAL_PACKAGE)]

from .theorem import binary_best_of_n_finite, simulate_best_of_n, utility_best_of_n_finite

__all__ = [
    "binary_best_of_n_finite",
    "simulate_best_of_n",
    "utility_best_of_n_finite",
]

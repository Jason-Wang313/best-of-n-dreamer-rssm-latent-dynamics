"""Checkout-local import shim.

The installable package lives under ``src/``. This shim lets Windows Python run
``python.exe -m experiments...`` from a Bash/WSL checkout without relying on
environment-variable propagation.
"""

from __future__ import annotations

from pathlib import Path

_REAL_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "rssm_tail_audit"
__path__ = [str(_REAL_PACKAGE)]

from .selected_tail import binary_tail_selection_finite, simulate_tail_selection, utility_tail_selection_finite

__all__ = [
    "binary_tail_selection_finite",
    "simulate_tail_selection",
    "utility_tail_selection_finite",
]

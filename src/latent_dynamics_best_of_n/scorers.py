"""Latent value scorers and RSSM-specific repair penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .envs import RolloutRecord


SCORER_NAMES = {
    "raw_value",
    "good",
    "overconfident",
    "value_optimistic",
    "belief_collapsed",
    "random",
    "oracle",
    "uncertainty_pessimism",
    "belief_consistency",
    "decoder_consistency",
    "ensemble_uncertainty_repair",
    "pilot_calibrated",
    "combined_repair",
}


def _arr(records: Iterable[RolloutRecord], attr: str) -> np.ndarray:
    return np.asarray([float(getattr(r, attr)) for r in records], dtype=float)


def feature_matrix(records: list[RolloutRecord]) -> np.ndarray:
    """Features available to the pilot calibration repair."""

    value = _arr(records, "value_pred")
    uncertainty = _arr(records, "uncertainty")
    pp_kl = _arr(records, "posterior_prior_kl")
    decoder = _arr(records, "decoder_error")
    belief = _arr(records, "belief_error")
    risk = _arr(records, "risk")
    posterior_free = _arr(records, "posterior_free_prob")
    imagined_free = _arr(records, "imagined_free_prob")
    hallucinated_free = np.maximum(0.0, imagined_free - posterior_free)
    return np.column_stack(
        [
            np.ones(len(records)),
            value,
            uncertainty,
            pp_kl,
            decoder,
            belief,
            risk,
            risk**2,
            posterior_free,
            imagined_free,
            posterior_free * risk,
            hallucinated_free * risk,
            uncertainty * risk,
            decoder * risk,
            pp_kl * risk,
            posterior_free * risk**2,
            (1.0 - posterior_free) * risk**2,
        ]
    )


@dataclass(frozen=True)
class PilotCalibrator:
    """Small linear pilot-label calibrator for selected-tail real utility."""

    weights: np.ndarray
    ridge: float

    def predict(self, records: list[RolloutRecord]) -> np.ndarray:
        return feature_matrix(records) @ self.weights


def fit_pilot_calibrator(records: list[RolloutRecord], ridge: float = 1.0) -> PilotCalibrator:
    """Fit real utility from latent diagnostics on a small pilot set."""

    x = feature_matrix(records)
    y = _arr(records, "real_utility")
    reg = float(ridge) * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
    return PilotCalibrator(weights=weights.astype(float), ridge=float(ridge))


def score_records(
    records: list[RolloutRecord],
    scorer: str = "raw_value",
    calibrator: PilotCalibrator | None = None,
    penalty_scale: float = 1.0,
) -> np.ndarray:
    """Return a finite score array for the requested selection rule."""

    if scorer not in SCORER_NAMES:
        raise ValueError(f"unknown scorer: {scorer}")
    if scorer == "raw_value":
        return _arr(records, "value_pred")
    if scorer == "oracle":
        return _arr(records, "real_utility")
    if scorer == "random":
        return np.asarray([float(r.diagnostics.get("random_score", 0.0)) for r in records], dtype=float)
    if scorer in {"good", "overconfident", "value_optimistic", "belief_collapsed"}:
        key = f"{scorer}_score" if scorer != "good" else "good_score"
        return np.asarray([float(r.diagnostics.get(key, r.value_pred)) for r in records], dtype=float)

    raw = _arr(records, "value_pred")
    uncertainty = _arr(records, "uncertainty")
    pp_kl = _arr(records, "posterior_prior_kl")
    decoder = _arr(records, "decoder_error")
    belief = _arr(records, "belief_error")
    ensemble_std = np.asarray(
        [
            float(
                r.diagnostics.get(
                    "ensemble_std",
                    0.55 * r.uncertainty + 0.25 * r.decoder_error + 0.20 * r.posterior_prior_kl,
                )
            )
            for r in records
        ],
        dtype=float,
    )
    if scorer == "uncertainty_pessimism":
        return raw - penalty_scale * 2.25 * uncertainty
    if scorer == "belief_consistency":
        return raw - penalty_scale * 2.75 * (pp_kl + 0.5 * belief)
    if scorer == "decoder_consistency":
        return raw - penalty_scale * 3.10 * decoder
    if scorer == "ensemble_uncertainty_repair":
        base = calibrator.predict(records) if calibrator is not None else raw
        return base - penalty_scale * (0.65 * ensemble_std + 0.20 * uncertainty + 0.20 * pp_kl + 0.15 * decoder)
    if scorer == "pilot_calibrated":
        if calibrator is None:
            raise ValueError("pilot_calibrated scorer requires a calibrator")
        return calibrator.predict(records)
    if scorer == "combined_repair":
        base = calibrator.predict(records) if calibrator is not None else raw
        return base - penalty_scale * (0.30 * uncertainty + 0.30 * pp_kl + 0.25 * decoder + 0.15 * belief)
    raise AssertionError("unreachable scorer")

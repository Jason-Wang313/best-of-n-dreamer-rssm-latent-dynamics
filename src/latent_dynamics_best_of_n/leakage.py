"""Calibration leakage checks for pilot-label experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LeakageAudit:
    """Structured result for a pilot/eval split audit."""

    passed: bool
    pilot_size: int
    eval_size: int
    overlap_count: int
    leaked_label_count: int
    scored_eval_count: int
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": bool(self.passed),
            "pilot_size": int(self.pilot_size),
            "eval_size": int(self.eval_size),
            "overlap_count": int(self.overlap_count),
            "leaked_label_count": int(self.leaked_label_count),
            "scored_eval_count": int(self.scored_eval_count),
            "reasons": list(self.reasons),
        }


def deterministic_split(n: int, pilot_size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint pilot/eval indices with a deterministic permutation."""

    if n <= 0:
        raise ValueError("n must be positive")
    pilot_size = int(np.clip(pilot_size, 0, n))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    pilot = np.sort(order[:pilot_size])
    eval_idx = np.sort(order[pilot_size:])
    return pilot.astype(int), eval_idx.astype(int)


def audit_calibration_split(
    pilot_indices: Iterable[int],
    eval_indices: Iterable[int],
    labels_used_indices: Iterable[int],
    scored_indices: Iterable[int] | None = None,
) -> LeakageAudit:
    """Verify calibration labels are confined to the pilot split.

    Scoring features may be computed for evaluation candidates, but real
    utility labels used for fitting must be a subset of the pilot indices and
    pilot/eval indices must be disjoint.
    """

    pilot = {int(i) for i in pilot_indices}
    eval_set = {int(i) for i in eval_indices}
    labels = {int(i) for i in labels_used_indices}
    scored = {int(i) for i in scored_indices} if scored_indices is not None else set()

    overlap = pilot & eval_set
    leaked_labels = labels - pilot
    reasons: list[str] = []
    if overlap:
        reasons.append("pilot/eval splits overlap")
    if leaked_labels:
        reasons.append("calibrator labels include non-pilot eval candidates")
    if not labels:
        reasons.append("calibrator received no pilot labels")
    if labels and not labels.issubset(pilot):
        reasons.append("label provenance is not a subset of the pilot split")
    return LeakageAudit(
        passed=not reasons,
        pilot_size=len(pilot),
        eval_size=len(eval_set),
        overlap_count=len(overlap),
        leaked_label_count=len(leaked_labels),
        scored_eval_count=len(scored & eval_set),
        reasons=tuple(reasons),
    )


def build_leakage_report(n: int, pilot_size: int, seed: int = 0) -> dict[str, object]:
    """Build a positive audit plus a deliberately leaky sentinel."""

    pilot, eval_idx = deterministic_split(n=n, pilot_size=pilot_size, seed=seed)
    clean = audit_calibration_split(pilot, eval_idx, labels_used_indices=pilot, scored_indices=np.arange(n))
    sentinel_labels = np.concatenate([pilot, eval_idx[: min(3, len(eval_idx))]])
    sentinel = audit_calibration_split(pilot, eval_idx, labels_used_indices=sentinel_labels, scored_indices=np.arange(n))
    return {
        "schema_version": 1,
        "passed": bool(clean.passed and not sentinel.passed),
        "clean_audit": clean.as_dict(),
        "leaky_sentinel": sentinel.as_dict(),
        "policy": "Real-utility labels used to fit calibrators must be drawn only from the pilot split; evaluation labels may only be used for reporting after scoring.",
    }

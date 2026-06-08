"""Claim audit generation for the RSSM latent-dynamics repo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORBIDDEN_CLAIMS = [
    "We solve Dreamer.",
    "We solve model-based RL.",
    "Best-of-N always hurts.",
    "More imagination always helps.",
    "Uncertainty always fixes the issue.",
    "We validate on real robots.",
    "This is not toy evidence",
    "This is just a renamed WAM project.",
]


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status(ok: bool, partial: bool = False) -> str:
    if ok:
        return "SUPPORTED"
    return "PARTIAL" if partial else "UNSUPPORTED"


def build_claim_status(root: Path) -> dict[str, Any]:
    results = root / "results"
    exact = _load(results / "exact_law_validation.json")
    exp_a = _load(results / "experiment_a_toy_mismatch.json")
    exp_b = _load(results / "experiment_b_learned_rssm.json")
    exp_c = _load(results / "experiment_c_belief_collapse.json")
    exp_d = _load(results / "experiment_d_horizon_budget.json")
    exp_e = _load(results / "experiment_e_repairs.json")

    claims: list[dict[str, Any]] = []

    theorem_ok = bool(exact and exact.get("max_abs_error", 1.0) < 0.12)
    claims.append(
        {
            "category": "theorem claims",
            "claim": "The finite tie-aware Best-of-N law exactly predicts selected utility from an empirical score/utility pool; Monte Carlo agrees within sampling error.",
            "status": _status(theorem_ok),
            "evidence": "results/exact_law_validation.json",
        }
    )

    a_key = exp_a.get("key_result", {}) if exp_a else {}
    a_ok = bool(a_key and a_key["raw_latent_delta_high_n"] > 0 and a_key["raw_real_delta_high_n"] < 0.10)
    claims.append(
        {
            "category": "controlled latent dynamics claims",
            "claim": "In the controlled hidden-mode toy, high N can inflate selected imagined latent value while selected executed real utility stagnates or drops.",
            "status": _status(a_ok),
            "evidence": "results/experiment_a_toy_mismatch.json; figures/figure1_latent_mismatch.png",
        }
    )

    b_key = exp_b.get("key_result", {}) if exp_b else {}
    b_ok = bool(exp_b and b_key.get("raw_latent_delta_high_n", 0.0) > 0)
    claims.append(
        {
            "category": "learned RSSM-style claims",
            "claim": "A CPU-trained small RSSM-style model with encoder, recurrent state, stochastic prior/posterior, decoder, reward head, and value head reproduces the latent-real selection mismatch in a smoke-scale learned setting.",
            "status": _status(b_ok, partial=bool(exp_b)),
            "evidence": "results/experiment_b_learned_rssm.json; results/learned_tiny_rssm.pt",
        }
    )

    c_key = exp_c.get("key_result", {}) if exp_c else {}
    c_ok = bool(c_key and c_key.get("raw_tail_blocked_rate", 0.0) > 0.5 and c_key.get("raw_tail_imagined_free_prob", 0.0) > 0.6)
    claims.append(
        {
            "category": "hidden-mode belief-collapse claims",
            "claim": "Under ambiguous hidden modes, high-N latent selection concentrates in optimistic imagined modes even when many selected candidates execute in blocked/slip/heavy modes.",
            "status": _status(c_ok, partial=bool(exp_c)),
            "evidence": "results/experiment_c_belief_collapse.json",
        }
    )

    d_ok = bool(exp_d and exp_d.get("summary_by_horizon"))
    claims.append(
        {
            "category": "horizon/selection-budget claims",
            "claim": "Varying N and H shows where longer or fatter imagination worsens latent-real mismatch.",
            "status": _status(d_ok),
            "evidence": "results/experiment_d_horizon_budget.json; figures/figure4_horizon_budget.png",
        }
    )

    e_key = exp_e.get("key_result", {}) if exp_e else {}
    repair_gain = float(e_key.get("combined_repair_n64_real_improvement_over_raw", 0.0)) if e_key else 0.0
    frac_closed = float(e_key.get("combined_repair_fraction_of_oracle_gap_closed", 0.0)) if e_key else 0.0
    claims.append(
        {
            "category": "repair claims",
            "claim": "RSSM-specific repairs can substantially recover selected-tail real utility over raw latent value selection; near-oracle is reported only when achieved.",
            "status": _status(repair_gain > 0.20 and frac_closed > 0.35, partial=repair_gain > 0.05),
            "evidence": "results/experiment_e_repairs.json; figures/figure2_repair_comparison.png",
            "repair_gain": repair_gain,
            "oracle_gap_fraction_closed": frac_closed,
        }
    )

    claims.append(
        {
            "category": "optional benchmark claims",
            "claim": "No external benchmark, full Dreamer benchmark, or real-world robotic benchmark is implemented in this repo.",
            "status": "SUPPORTED",
            "evidence": "Repository scope and scripts are toy/RSSM-style CPU experiments only.",
        }
    )
    claims.append(
        {
            "category": "unsupported robotics claims",
            "claim": "Real robot validation is unsupported and must not be claimed.",
            "status": "SUPPORTED",
            "evidence": "No robot datasets, robot execution adapters, or robot result artifacts exist.",
        }
    )

    text_corpus = []
    for path in list(root.glob("README.md")) + list((root / "docs").glob("*.md")) + list((root / "paper").glob("*.md")):
        text_corpus.append(path.read_text(encoding="utf-8"))
    joined = "\n".join(text_corpus).lower()
    forbidden_hits = [claim for claim in FORBIDDEN_CLAIMS if claim.lower() in joined]
    claims.append(
        {
            "category": "forbidden overclaims",
            "claim": "Forbidden universal or real-robot claims are blocked.",
            "status": "SUPPORTED" if not forbidden_hits else "UNSUPPORTED",
            "evidence": {"forbidden_hits": forbidden_hits, "blocked_phrases": FORBIDDEN_CLAIMS},
        }
    )

    return {
        "schema_version": 1,
        "project": "When Latent Imagination Lies",
        "claims": claims,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "all_supported_or_partial": all(c["status"] in {"SUPPORTED", "PARTIAL"} for c in claims),
        "unsupported_count": sum(1 for c in claims if c["status"] == "UNSUPPORTED"),
    }


def claim_status_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Status",
        "",
        f"- Project: {payload['project']}",
        f"- Unsupported count: {payload['unsupported_count']}",
        "",
        "| Category | Status | Claim | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for claim in payload["claims"]:
        evidence = claim.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = json.dumps(evidence, sort_keys=True)
        lines.append(f"| {claim['category']} | {claim['status']} | {claim['claim']} | {evidence} |")
    lines.extend(
        [
            "",
            "## Blocked Overclaims",
            "",
            "The following phrases are explicitly treated as forbidden publication claims:",
        ]
    )
    for phrase in payload["forbidden_claims"]:
        lines.append(f"- `{phrase}`")
    lines.append("")
    return "\n".join(lines)

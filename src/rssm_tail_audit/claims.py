"""Claim audit generation for the RSSM latent-dynamics repo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORBIDDEN_CLAIMS = [
    "We solve Dreamer.",
    "We solve model-based RL.",
    "candidate-budget selection always hurts.",
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


def _metric(payload: dict[str, Any] | None, family: str, metric: str, field: str = "mean") -> float | None:
    try:
        value = payload["families"][family]["aggregate"][metric][field]  # type: ignore[index]
    except Exception:
        return None
    return float(value)


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def build_claim_status(root: Path) -> dict[str, Any]:
    results = root / "results"
    exact = _load(results / "selected_tail_estimator_validation.json")
    exp_a = _load(results / "experiment_a_toy_mismatch.json")
    exp_b = _load(results / "experiment_b_learned_rssm.json")
    exp_c = _load(results / "experiment_c_belief_collapse.json")
    exp_d = _load(results / "experiment_d_horizon_budget.json")
    exp_e = _load(results / "experiment_e_repairs.json")
    exp_f = _load(results / "experiment_f_closed_loop_planning.json")
    exp_g = _load(results / "experiment_g_label_budget_ablation.json")
    exp_h = _load(results / "experiment_h_ood_stress_grid.json")
    exp_i = _load(results / "experiment_i_gymnasium_benchmarks.json")
    exp_j = _load(results / "experiment_j_belief_interventions.json")
    leakage = _load(results / "leakage_audit.json")
    multi = _load(results / "multiseed_strong_evidence.json")
    full_multiseed = bool(multi and not multi.get("smoke"))

    claims: list[dict[str, Any]] = []
    weak_reasons: list[str] = []

    estimator_ok = bool(exact and exact.get("max_abs_error", 1.0) < 0.05)
    estimator_strength = "STRONG" if estimator_ok else "WEAK"
    if not estimator_ok:
        weak_reasons.append("finite selected-tail estimator Monte Carlo error is above the strong threshold")
    claims.append(
        {
            "category": "measurement-estimator claims",
            "claim": "The finite selected-tail estimator exactly predicts selected utility from an empirical score/utility pool; Monte Carlo agrees within sampling error.",
            "status": _status(estimator_ok),
            "evidence_strength": estimator_strength,
            "evidence": "results/selected_tail_estimator_validation.json",
        }
    )

    a_key = exp_a.get("key_result", {}) if exp_a else {}
    a_single = bool(a_key and a_key["raw_latent_delta_high_n"] > 0 and a_key["raw_real_delta_high_n"] < 0.10)
    a_strong = (
        _gt(_metric(multi, "controlled_latent_dynamics", "raw_latent_delta_high_n", "lo"), 2.5)
        and _lt(_metric(multi, "controlled_latent_dynamics", "raw_real_delta_high_n", "mean"), -1.0)
        and _lt(_metric(multi, "controlled_latent_dynamics", "raw_real_delta_high_n", "hi"), 0.15)
        and _gt(_metric(multi, "controlled_latent_dynamics", "repair_n64_real_improvement_over_raw", "lo"), 2.0)
    )
    a_ok = a_strong if full_multiseed else a_single
    if full_multiseed and not a_strong:
        weak_reasons.append("controlled latent dynamics multi-seed margins are not strong")
    claims.append(
        {
            "category": "controlled latent dynamics claims",
            "claim": "In the controlled hidden-mode toy, large candidate budgets can inflate selected imagined latent value while selected executed real utility stagnates or drops.",
            "status": _status(a_ok),
            "evidence_strength": "STRONG" if a_strong else ("SMOKE" if a_single else "WEAK"),
            "evidence": "results/experiment_a_toy_mismatch.json; figures/figure1_latent_mismatch.png",
        }
    )

    b_key = exp_b.get("key_result", {}) if exp_b else {}
    b_single = bool(exp_b and b_key.get("raw_latent_delta_high_n", 0.0) > 0 and b_key.get("raw_real_delta_high_n", 1.0) < 0)
    b_strong = (
        _gt(_metric(multi, "learned_rssm", "raw_latent_delta_high_n", "mean"), 0.25)
        and _lt(_metric(multi, "learned_rssm", "raw_real_delta_high_n", "mean"), -0.25)
        and _gt(_metric(multi, "learned_rssm", "repair_n64_real_improvement_over_raw", "mean"), 1.0)
    )
    b_ok = b_strong if full_multiseed else b_single
    if full_multiseed and not b_strong:
        weak_reasons.append("learned RSSM multi-seed mismatch or repair margin is not strong")
    claims.append(
        {
            "category": "learned RSSM-style claims",
            "claim": "A CPU-trained small RSSM-style model with encoder, recurrent state, stochastic prior/posterior, decoder, reward head, and value head reproduces the latent-real selection mismatch in a smoke-scale learned setting.",
            "status": _status(b_ok, partial=bool(exp_b)),
            "evidence_strength": "STRONG" if b_strong else ("SMOKE" if b_single else "WEAK"),
            "evidence": "results/experiment_b_learned_rssm.json; results/learned_tiny_rssm.pt",
        }
    )

    c_key = exp_c.get("key_result", {}) if exp_c else {}
    c_single = bool(c_key and c_key.get("raw_tail_blocked_rate", 0.0) > 0.5 and c_key.get("raw_tail_imagined_free_prob", 0.0) > 0.6)
    c_strong = (
        _gt(_metric(multi, "belief_collapse", "tail_blocked_rate", "lo"), 0.60)
        and _gt(_metric(multi, "belief_collapse", "tail_imagined_free_prob", "lo"), 0.85)
        and _lt(_metric(multi, "belief_collapse", "raw_real_delta_high_n", "hi"), 0.0)
    )
    c_ok = c_strong if full_multiseed else c_single
    if full_multiseed and not c_strong:
        weak_reasons.append("belief-collapse tail diagnostics are not strong across seeds")
    claims.append(
        {
            "category": "hidden-mode belief-collapse claims",
            "claim": "Under ambiguous hidden modes, large-budget latent selection concentrates in optimistic imagined modes even when many selected candidates execute in blocked/slip/heavy modes.",
            "status": _status(c_ok, partial=bool(exp_c)),
            "evidence_strength": "STRONG" if c_strong else ("SMOKE" if c_single else "WEAK"),
            "evidence": "results/experiment_c_belief_collapse.json",
        }
    )

    j_key = exp_j.get("key_result", {}) if exp_j else {}
    j_raw_delta = j_key.get("raw_real_delta_high_n_ci", {}) if j_key else {}
    j_belief_gain = j_key.get("belief_penalty_minus_raw_n64_ci", {}) if j_key else {}
    j_full_gain = j_key.get("full_minus_raw_n64_ci", {}) if j_key else {}
    j_corr = j_key.get("tail_drift_gap_corr_ci", {}) if j_key else {}
    j_strong = bool(
        exp_j
        and not exp_j.get("smoke")
        and exp_j.get("n_seed_regime_units", 0) >= 16
        and j_key.get("raw_harmful_fraction", 0.0) >= 0.80
        and j_raw_delta.get("hi", 1.0) < -0.50
        and j_belief_gain.get("lo", -1.0) > 0.25
        and j_full_gain.get("lo", -1.0) > 2.0
        and j_corr.get("lo", -1.0) > 0.0
    )
    if full_multiseed and not j_strong:
        weak_reasons.append(
            "belief-intervention stress does not show harmful raw high-N, positive belief-diagnostic recovery, and positive drift-gap correlation"
        )
    claims.append(
        {
            "category": "belief-intervention mechanism claims",
            "claim": "Posterior-prior and belief-collapse diagnostics are mechanistic RSSM signals: in high-risk seed-regime units, raw high-N selection is harmful, a diagnostic-only belief penalty improves N=64 utility, and drift correlates with selected-tail latent-real gap.",
            "status": _status(j_strong),
            "evidence_strength": "STRONG" if j_strong else "WEAK",
            "evidence": "results/experiment_j_belief_interventions.json; figures/figure10_belief_interventions.png",
        }
    )

    d_single = bool(exp_d and exp_d.get("summary_by_horizon"))
    d_strong = (
        _lt(_metric(multi, "horizon_budget", "h8_minus_h2_raw_real_delta", "mean"), -1.0)
        and _gt(_metric(multi, "horizon_budget", "h8_minus_h2_raw_latent_delta", "lo"), 5.0)
        and _gt(_metric(multi, "horizon_budget", "h8_repair_improvement", "lo"), 4.0)
    )
    d_ok = d_strong if full_multiseed else d_single
    if full_multiseed and not d_strong:
        weak_reasons.append("horizon/budget multi-seed amplification margins are not strong")
    claims.append(
        {
            "category": "horizon/selection-budget claims",
            "claim": "Varying candidate budget and horizon shows where longer imagination worsens latent-real mismatch.",
            "status": _status(d_ok),
            "evidence_strength": "STRONG" if d_strong else ("SMOKE" if d_single else "WEAK"),
            "evidence": "results/experiment_d_horizon_budget.json; figures/figure4_horizon_budget.png",
        }
    )

    e_key = exp_e.get("key_result", {}) if exp_e else {}
    repair_gain = float(e_key.get("combined_repair_n64_real_improvement_over_raw", 0.0)) if e_key else 0.0
    frac_closed = float(e_key.get("combined_repair_fraction_of_oracle_gap_closed", 0.0)) if e_key else 0.0
    e_strong = (
        _gt(_metric(multi, "repair", "repair_n64_real_improvement_over_raw", "lo"), 3.0)
        and _gt(_metric(multi, "repair", "repair_fraction_of_oracle_gap_closed", "lo"), 0.55)
    )
    e_single = repair_gain > 0.20 and frac_closed > 0.35
    e_ok = e_strong if full_multiseed else e_single
    if full_multiseed and not e_strong:
        weak_reasons.append("repair multi-seed gain or oracle-gap closure is not strong")
    claims.append(
        {
            "category": "repair claims",
            "claim": "RSSM-specific repairs can substantially recover selected-tail real utility over raw latent value selection; near-oracle is reported only when achieved.",
            "status": _status(e_ok, partial=repair_gain > 0.05),
            "evidence_strength": "STRONG" if e_strong else ("SMOKE" if e_single else "WEAK"),
            "evidence": "results/experiment_e_repairs.json; figures/figure2_repair_comparison.png",
            "repair_gain": repair_gain,
            "oracle_gap_fraction_closed": frac_closed,
        }
    )

    f_key = exp_f.get("key_result", {}) if exp_f else {}
    f_strong = bool(
        exp_f
        and not exp_f.get("smoke")
        and f_key.get("controlled_raw_n64_mean_return", 1e9) <= f_key.get("controlled_raw_n1_mean_return", -1e9) + 0.25
        and f_key.get("controlled_oracle_n64_mean_return", -1e9) > f_key.get("controlled_raw_n64_mean_return", 1e9) + 1.0
        and f_key.get("controlled_combined_repair_n64_improvement_over_raw", 0.0) >= 1.0
        and f_key.get("learned_raw_n64_mean_return", 1e9) <= f_key.get("learned_raw_n1_mean_return", -1e9) + 0.35
        and f_key.get("learned_oracle_n64_mean_return", -1e9) > f_key.get("learned_raw_n64_mean_return", 1e9) + 0.5
        and f_key.get("learned_combined_repair_n64_improvement_over_raw", 0.0) > 0.25
    )
    if full_multiseed and not f_strong:
        weak_reasons.append("closed-loop planning evidence does not meet raw failure and repair recovery margins")
    claims.append(
        {
            "category": "closed-loop planning claims",
            "claim": "In receding-horizon hidden-mode planning, raw large-budget selection underperforms oracle and repair recovers executed return in controlled and learned RSSM-style settings.",
            "status": _status(f_strong),
            "evidence_strength": "STRONG" if f_strong else "WEAK",
            "evidence": "results/experiment_f_closed_loop_planning.json; figures/figure6_closed_loop_planning.png",
        }
    )

    g_key = exp_g.get("key_result", {}) if exp_g else {}
    g_strong = bool(
        exp_g
        and not exp_g.get("smoke")
        and exp_g.get("audit_passed")
        and g_key.get("combined_repair_128_improvement_over_raw", 0.0) >= 2.0
        and g_key.get("combined_repair_1000_fraction_of_oracle_gap_closed", 0.0) >= 0.55
    )
    if full_multiseed and not g_strong:
        weak_reasons.append("label-budget ablation does not meet high-label repair recovery margins")
    claims.append(
        {
            "category": "label-budget repair claims",
            "claim": "A small pilot-label budget can calibrate selected-tail repairs, with high-label settings closing a majority of the raw-to-oracle gap.",
            "status": _status(g_strong),
            "evidence_strength": "STRONG" if g_strong else "WEAK",
            "evidence": "results/experiment_g_label_budget_ablation.json; figures/figure7_label_budget_repair.png",
        }
    )

    leakage_ok = bool(
        leakage
        and not leakage.get("smoke")
        and leakage.get("passed")
        and leakage.get("clean_audit", {}).get("passed")
        and not leakage.get("leaky_sentinel", {}).get("passed", True)
    )
    claims.append(
        {
            "category": "leakage-free calibration claims",
            "claim": "Pilot-label calibration is audited as eval-disjoint, and a deliberately leaky sentinel is caught by the same audit.",
            "status": _status(leakage_ok),
            "evidence_strength": "STRONG" if leakage_ok else "WEAK",
            "evidence": "results/leakage_audit.json",
        }
    )

    h_key = exp_h.get("key_result", {}) if exp_h else {}
    h_strong = bool(
        exp_h
        and not exp_h.get("smoke")
        and h_key.get("raw_high_n_harm_regions", 0) >= 1
        and h_key.get("raw_high_n_neutral_or_helpful_regions", 0) >= 1
        and h_key.get("combined_repair_mean_gain_high_risk_regions", 0.0) > 0.0
    )
    if full_multiseed and not h_strong:
        weak_reasons.append("OOD stress grid does not show both harmful and non-harmful raw large-budget regimes with repair mitigation")
    claims.append(
        {
            "category": "OOD stress-grid claims",
            "claim": "OOD stress sweeps identify regimes where raw large-budget selection hurts and regimes where it is neutral or helpful; repair reduces average harm in high-risk regions.",
            "status": _status(h_strong),
            "evidence_strength": "STRONG" if h_strong else "WEAK",
            "evidence": "results/experiment_h_ood_stress_grid.json; figures/figure8_ood_stress_grid.png",
        }
    )

    i_key = exp_i.get("key_result", {}) if exp_i else {}
    i_strong = bool(
        exp_i
        and not exp_i.get("smoke")
        and i_key.get("selected_tail_mismatch_benchmark_count", 0) >= 2
        and i_key.get("combined_repair_improvement_benchmark_count", 0) >= 2
    )
    if full_multiseed and not i_strong:
        weak_reasons.append("Gymnasium benchmark evidence does not meet the two-of-three mismatch and repair-support threshold")
    claims.append(
        {
            "category": "lightweight Gymnasium benchmark claims",
            "claim": "Exactly three lightweight Gymnasium toy-text benchmarks provide scoped standard stochastic-benchmark evidence for selected-tail mismatch and repair, not broad RL validation.",
            "status": _status(i_strong),
            "evidence_strength": "STRONG" if i_strong else "WEAK",
            "evidence": "results/experiment_i_gymnasium_benchmarks.json; figures/figure9_gymnasium_benchmarks.png",
        }
    )

    claims.append(
        {
            "category": "scope boundary claims",
            "claim": "The benchmark scope is limited to controlled RSSM-style evidence plus lightweight Gymnasium toy-text benchmarks; it does not claim full Dreamer, broad model-based RL, or robotics validation.",
            "status": "SUPPORTED",
            "evidence_strength": "STRONG",
            "evidence": "Repository scope, scripts, and generated benchmark artifacts.",
        }
    )
    claims.append(
        {
            "category": "unsupported robotics claims",
            "claim": "Real robot validation is unsupported and must not be claimed.",
            "status": "SUPPORTED",
            "evidence_strength": "STRONG",
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
            "evidence_strength": "STRONG" if not forbidden_hits else "WEAK",
            "evidence": {"forbidden_hits": forbidden_hits, "blocked_phrases": FORBIDDEN_CLAIMS},
        }
    )

    if not leakage_ok:
        weak_reasons.append("leakage audit is missing or did not catch the leaky sentinel")
    weak_count = len(weak_reasons) if full_multiseed else 0
    return {
        "schema_version": 1,
        "project": "Belief-Tail Audits for RSSM World Models",
        "claims": claims,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "full_multiseed_evidence": full_multiseed,
        "weak_count": weak_count,
        "weak_reasons": weak_reasons,
        "all_supported_or_partial": all(c["status"] in {"SUPPORTED", "PARTIAL"} for c in claims),
        "unsupported_count": sum(1 for c in claims if c["status"] == "UNSUPPORTED"),
    }


def claim_status_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Status",
        "",
        f"- Project: {payload['project']}",
        f"- Unsupported count: {payload['unsupported_count']}",
        f"- Weak strong-evidence checks: {payload.get('weak_count', 0)}",
        f"- Full multi-seed evidence: {payload.get('full_multiseed_evidence', False)}",
        "",
        "| Category | Status | Strength | Claim | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for claim in payload["claims"]:
        evidence = claim.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = json.dumps(evidence, sort_keys=True)
        lines.append(
            f"| {claim['category']} | {claim['status']} | {claim.get('evidence_strength', 'NA')} | {claim['claim']} | {evidence} |"
        )
    if payload.get("weak_reasons"):
        lines.extend(["", "## Weak Evidence Reasons", ""])
        for reason in payload["weak_reasons"]:
            lines.append(f"- {reason}")
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

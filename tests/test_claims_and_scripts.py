from __future__ import annotations

from pathlib import Path

from rssm_tail_audit.claims import FORBIDDEN_CLAIMS, build_claim_status


ROOT = Path(__file__).resolve().parents[1]


def test_script_contracts_reference_required_experiments():
    smoke = (ROOT / "scripts" / "run_smoke.sh").read_text(encoding="utf-8")
    full = (ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    suite = (ROOT / "scripts" / "run_suite.py").read_text(encoding="utf-8")
    for text in [suite]:
        assert "experiments/selected_tail_estimator_validation.py" in text
        assert "experiments/experiment_a_toy_mismatch.py" in text
        assert "experiments/experiment_b_learned_rssm.py" in text
        assert "experiments/multiseed_evidence.py" in text
        assert "scripts/run_claim_audit.py" in text
    for text in [smoke, full]:
        assert "scripts.run_suite" in text
        assert "PYTHONPATH" in text
    assert "experiments/experiment_d_horizon_budget.py" in suite
    assert "experiments/experiment_e_repairs.py" in suite
    assert "experiments/experiment_f_closed_loop_planning.py" in suite
    assert "experiments/experiment_g_label_budget_ablation.py" in suite
    assert "experiments/experiment_h_ood_stress_grid.py" in suite
    assert "experiments/experiment_i_gymnasium_stochastic_benchmarks.py" in suite
    assert "experiments/experiment_j_belief_interventions.py" in suite
    assert "experiments/experiment_k_classic_control_benchmarks.py" in suite
    assert "experiments/leakage_audit.py" in suite
    assert "if not args.smoke" in suite


def test_claim_audit_schema_without_forbidden_doc_hits():
    payload = build_claim_status(ROOT)
    assert "claims" in payload
    assert set(FORBIDDEN_CLAIMS) == set(payload["forbidden_claims"])
    categories = {c["category"] for c in payload["claims"]}
    assert "forbidden overclaims" in categories
    assert "leakage-free calibration claims" in categories
    assert "lightweight Gymnasium benchmark claims" in categories
    assert "classic-control Gymnasium benchmark claims" in categories
    assert "belief-intervention mechanism claims" in categories
    assert all(c["status"] in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"} for c in payload["claims"])


def test_claim_audit_rejects_missing_leakage_artifact(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "paper").mkdir()
    (tmp_path / "README.md").write_text("Scoped toy evidence only.\n", encoding="utf-8")
    payload = build_claim_status(tmp_path)
    leakage_claim = [c for c in payload["claims"] if c["category"] == "leakage-free calibration claims"][0]
    assert leakage_claim["status"] == "UNSUPPORTED"
    assert payload["unsupported_count"] >= 1

from __future__ import annotations

from pathlib import Path

from latent_dynamics_best_of_n.claims import FORBIDDEN_CLAIMS, build_claim_status


ROOT = Path(__file__).resolve().parents[1]


def test_script_contracts_reference_required_experiments():
    smoke = (ROOT / "scripts" / "run_smoke.sh").read_text(encoding="utf-8")
    full = (ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    suite = (ROOT / "scripts" / "run_suite.py").read_text(encoding="utf-8")
    for text in [suite]:
        assert "experiments/exact_law_validation.py" in text
        assert "experiments/experiment_a_toy_mismatch.py" in text
        assert "experiments/experiment_b_learned_rssm.py" in text
        assert "scripts/run_claim_audit.py" in text
    for text in [smoke, full]:
        assert "scripts.run_suite" in text
        assert "PYTHONPATH" in text
    assert "experiments/experiment_d_horizon_budget.py" in suite
    assert "experiments/experiment_e_repairs.py" in suite


def test_claim_audit_schema_without_forbidden_doc_hits():
    payload = build_claim_status(ROOT)
    assert "claims" in payload
    assert set(FORBIDDEN_CLAIMS) == set(payload["forbidden_claims"])
    categories = {c["category"] for c in payload["claims"]}
    assert "forbidden overclaims" in categories
    assert all(c["status"] in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"} for c in payload["claims"])

"""Regenerate claim status artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rssm_tail_audit.claims import build_claim_status, claim_status_markdown


def main() -> None:
    root = ROOT
    (root / "results").mkdir(parents=True, exist_ok=True)
    payload = build_claim_status(root)
    (root / "results" / "claims_status.json").write_text(
        __import__("json").dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "results" / "claims_status.md").write_text(claim_status_markdown(payload), encoding="utf-8")
    if payload["unsupported_count"] > 0:
        raise SystemExit(f"claim audit found {payload['unsupported_count']} unsupported claim(s)")
    if payload.get("full_multiseed_evidence") and payload.get("weak_count", 0) > 0:
        raise SystemExit(f"claim audit found {payload['weak_count']} weak strong-evidence check(s)")


if __name__ == "__main__":
    main()

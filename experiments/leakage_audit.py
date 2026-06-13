"""Generate the calibration leakage audit artifact."""

from __future__ import annotations

from rssm_tail_audit.leakage import build_leakage_report

from experiments.common import ensure_dirs, results_dir, root_from_file, smoke_argparser, write_json


def run(smoke: bool = False, seed: int = 0):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    payload = build_leakage_report(n=400 if smoke else 2400, pilot_size=80 if smoke else 1000, seed=seed)
    payload["experiment"] = "leakage_audit"
    payload["smoke"] = bool(smoke)
    write_json(results_dir(root, smoke) / "leakage_audit.json", payload)
    return payload


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

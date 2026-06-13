"""Experiment G: pilot-label budget ablation for repair calibration."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rssm_tail_audit.envs import HiddenModeConfig, HiddenModeToyEnv
from rssm_tail_audit.leakage import audit_calibration_split, deterministic_split
from rssm_tail_audit.metrics import selection_curves
from rssm_tail_audit.scorers import fit_pilot_calibrator, score_records

from experiments.common import ensure_dirs, figures_dir, results_dir, root_from_file, smoke_argparser, tables_dir, write_json


FULL_BUDGETS = [16, 32, 64, 128, 256, 512, 1000]


def _n64_real(records, scorer: str, calibrator=None) -> float:
    scores = score_records(records, scorer, calibrator=calibrator)
    return float(selection_curves(records, scores, [64])[0]["selected_real_utility"])


def _aggregate(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    df = pd.DataFrame(rows)
    out: dict[str, dict[str, float]] = {}
    for budget, group in df.groupby("label_budget"):
        raw = float(group["raw_n64_real"].mean())
        oracle = float(group["oracle_n64_real"].mean())
        for scorer in ["pilot_calibrated", "combined_repair"]:
            sub = group[group["scorer"] == scorer]
            mean_real = float(sub["selected_real_utility"].mean())
            gap = max(1e-9, oracle - raw)
            out[f"{int(budget)}_{scorer}"] = {
                "label_budget": float(budget),
                "mean_n64_real": mean_real,
                "improvement_over_raw": float(mean_real - raw),
                "fraction_of_oracle_gap_closed": float((mean_real - raw) / gap),
                "raw_n64_real": raw,
                "oracle_n64_real": oracle,
                "n_seeds": float(sub["seed"].nunique()),
            }
    return out


def _plot(rows: list[dict[str, float | int | str]], output) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.3, 4.3), dpi=150)
    colors = {"pilot_calibrated": "#44633f", "combined_repair": "#1a8f5a"}
    for scorer in ["pilot_calibrated", "combined_repair"]:
        sub = df[df["scorer"] == scorer].groupby("label_budget", as_index=False).mean(numeric_only=True)
        ax.plot(
            sub["label_budget"],
            sub["fraction_of_oracle_gap_closed"],
            marker="o",
            linewidth=2.0,
            color=colors[scorer],
            label=scorer,
        )
    ax.axhline(0.55, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["label_budget"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Pilot real-utility labels")
    ax.set_ylabel("Raw-to-oracle gap closed")
    ax.set_title("Repair label-budget efficiency")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run(smoke: bool = False, seed: int = 7):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    budgets = [16, 64, 128, 256] if smoke else FULL_BUDGETS
    max_budget = max(budgets)
    seeds = [seed + i for i in range(2 if smoke else 5)]
    env = HiddenModeToyEnv(HiddenModeConfig(blocked_prob=0.72, clue_strength=0.09, observation_noise=0.09))
    rows: list[dict[str, float | int | str]] = []
    audits = []
    for family_seed in seeds:
        records = env.generate_candidate_pool(
            n=900 if smoke else 2800,
            horizon=5,
            seed=family_seed,
            model_flavor="belief_collapsed",
        )
        pilot_all, eval_idx = deterministic_split(len(records), max_budget, seed=family_seed + 99)
        eval_records = [records[int(i)] for i in eval_idx]
        raw_n64 = _n64_real(eval_records, "raw_value")
        oracle_n64 = _n64_real(eval_records, "oracle")
        for budget in budgets:
            pilot_idx = pilot_all[:budget]
            audit = audit_calibration_split(pilot_idx, eval_idx, labels_used_indices=pilot_idx, scored_indices=eval_idx)
            audits.append({"seed": int(family_seed), "label_budget": int(budget), **audit.as_dict()})
            calibrator = fit_pilot_calibrator([records[int(i)] for i in pilot_idx])
            for scorer in ["pilot_calibrated", "combined_repair"]:
                selected = _n64_real(eval_records, scorer, calibrator=calibrator)
                gap = max(1e-9, oracle_n64 - raw_n64)
                rows.append(
                    {
                        "seed": int(family_seed),
                        "label_budget": int(budget),
                        "scorer": scorer,
                        "selected_real_utility": selected,
                        "raw_n64_real": raw_n64,
                        "oracle_n64_real": oracle_n64,
                        "improvement_over_raw": float(selected - raw_n64),
                        "fraction_of_oracle_gap_closed": float((selected - raw_n64) / gap),
                    }
                )

    pd.DataFrame(rows).to_csv(tables_dir(root, smoke) / "experiment_g_label_budget_ablation.csv", index=False)
    aggregate = _aggregate(rows)
    key128 = aggregate.get("128_combined_repair", {})
    key1000 = aggregate.get("1000_combined_repair", aggregate.get(f"{max_budget}_combined_repair", {}))
    label_efficiency = None
    for budget in budgets:
        item = aggregate[f"{budget}_combined_repair"]
        if item["improvement_over_raw"] >= 2.0 and item["fraction_of_oracle_gap_closed"] >= 0.55:
            label_efficiency = budget
            break
    payload = {
        "experiment": "G_repair_label_budget_ablation",
        "smoke": bool(smoke),
        "label_budgets": budgets,
        "summary": aggregate,
        "audit_passed": all(bool(a["passed"]) for a in audits),
        "audit_rows": audits,
        "label_efficiency_budget": label_efficiency,
        "key_result": {
            "combined_repair_128_improvement_over_raw": float(key128.get("improvement_over_raw", 0.0)),
            "combined_repair_1000_fraction_of_oracle_gap_closed": float(
                key1000.get("fraction_of_oracle_gap_closed", 0.0)
            ),
            "combined_repair_1000_improvement_over_raw": float(key1000.get("improvement_over_raw", 0.0)),
        },
    }
    write_json(results_dir(root, smoke) / "experiment_g_label_budget_ablation.json", payload)
    _plot(rows, figures_dir(root, smoke) / "figure7_label_budget_repair.png")
    return payload


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

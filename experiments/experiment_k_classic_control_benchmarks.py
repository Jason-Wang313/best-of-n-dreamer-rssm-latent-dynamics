"""Experiment K: standard Gymnasium classic-control latent-planning stress tests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import N_GRID, ensure_dirs, figures_dir, results_dir, root_from_file, smoke_argparser, tables_dir, write_json
from rssm_tail_audit.classic_control import CLASSIC_CONTROL_BENCHMARKS, generate_classic_control_records, start_state
from rssm_tail_audit.leakage import deterministic_split
from rssm_tail_audit.metrics import bootstrap_ci, selection_curves
from rssm_tail_audit.scorers import fit_pilot_calibrator, score_records


SCORERS = [
    "raw_value",
    "uncertainty_pessimism",
    "belief_consistency",
    "pilot_calibrated",
    "combined_repair",
    "random",
    "oracle",
]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    latent_grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    risk_grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["benchmark"]), str(row["scorer"]), int(row["N"]))
        grouped[key].append(float(row["selected_real_utility"]))
        latent_grouped[key].append(float(row["selected_latent_value"]))
        risk_grouped[key].append(float(row["selected_risk"]))
    out: dict[str, Any] = {}
    for key, values in grouped.items():
        benchmark, scorer, n_value = key
        arr = np.asarray(values, dtype=float)
        latent_arr = np.asarray(latent_grouped[key], dtype=float)
        risk_arr = np.asarray(risk_grouped[key], dtype=float)
        out.setdefault(benchmark, {}).setdefault(scorer, {})[str(n_value)] = {
            "mean_real": float(np.mean(arr)),
            "std_real": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "mean_latent": float(np.mean(latent_arr)),
            "mean_risk": float(np.mean(risk_arr)),
            "n": int(len(arr)),
        }
    return out


def _ci(values: list[float], seed: int) -> dict[str, float]:
    return bootstrap_ci(values, seed=seed, n_boot=1000)


def _plot(rows: list[dict[str, Any]], output) -> None:
    df = pd.DataFrame(rows)
    means = df.groupby(["benchmark", "scorer", "N"], as_index=False)["selected_real_utility"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), dpi=150)
    colors = {
        "raw_value": "#b23b3b",
        "uncertainty_pessimism": "#7a3e9d",
        "belief_consistency": "#bc6c25",
        "pilot_calibrated": "#44633f",
        "combined_repair": "#1a8f5a",
        "random": "#777777",
        "oracle": "#202020",
    }
    for ax, benchmark in zip(axes, CLASSIC_CONTROL_BENCHMARKS):
        sub = means[means["benchmark"] == benchmark]
        for scorer in SCORERS:
            g = sub[sub["scorer"] == scorer].sort_values("N")
            ax.plot(g["N"], g["selected_real_utility"], marker="o", linewidth=1.7, color=colors[scorer], label=scorer)
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_GRID)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(benchmark)
        ax.set_xlabel("candidate budget N")
        ax.grid(True, color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Executed rollout utility")
    axes[-1].legend(frameon=False, fontsize=6.6, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run(smoke: bool = False, seed: int = 41):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    seeds = [seed + i for i in range(2 if smoke else 6)]
    candidate_count = 80 if smoke else 180
    pilot_size = 28 if smoke else 64
    rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []

    for benchmark_name, spec in CLASSIC_CONTROL_BENCHMARKS.items():
        for family_seed in seeds:
            init = start_state(spec.env_id, family_seed)
            records = generate_classic_control_records(
                spec,
                n=candidate_count,
                seed=family_seed + 500,
                initial_state=init,
                state_id=family_seed,
            )
            pilot_idx, eval_idx = deterministic_split(len(records), pilot_size=pilot_size, seed=family_seed + 900)
            calibrator = fit_pilot_calibrator([records[int(i)] for i in pilot_idx])
            eval_records = [records[int(i)] for i in eval_idx]
            audit_rows.append(
                {
                    "benchmark": benchmark_name,
                    "seed": int(family_seed),
                    "candidate_count": int(candidate_count),
                    "pilot_size": int(len(pilot_idx)),
                    "eval_size": int(len(eval_idx)),
                    "overlap_count": int(len(set(map(int, pilot_idx)) & set(map(int, eval_idx)))),
                }
            )
            for record in records:
                row = record.as_dict()
                row.update({"benchmark": benchmark_name, "record_seed": int(family_seed)})
                rollout_rows.append(row)

            seed_curves: dict[str, list[dict[str, float | int]]] = {}
            for scorer in SCORERS:
                scores = score_records(eval_records, scorer, calibrator=calibrator)
                curves = selection_curves(eval_records, scores, N_GRID)
                seed_curves[scorer] = curves
                for curve in curves:
                    rows.append(
                        {
                            "benchmark": benchmark_name,
                            "seed": int(family_seed),
                            "scorer": scorer,
                            **curve,
                        }
                    )

            raw = seed_curves["raw_value"]
            repair = seed_curves["combined_repair"]
            oracle = seed_curves["oracle"]
            effect_rows.append(
                {
                    "benchmark": benchmark_name,
                    "seed": int(family_seed),
                    "raw_latent_delta_n64_vs_n1": float(raw[-1]["selected_latent_value"] - raw[0]["selected_latent_value"]),
                    "raw_real_delta_n64_vs_n1": float(raw[-1]["selected_real_utility"] - raw[0]["selected_real_utility"]),
                    "raw_risk_delta_n64_vs_n1": float(raw[-1]["selected_risk"] - raw[0]["selected_risk"]),
                    "combined_repair_n64_improvement_over_raw": float(
                        repair[-1]["selected_real_utility"] - raw[-1]["selected_real_utility"]
                    ),
                    "oracle_n64_gap_over_repair": float(oracle[-1]["selected_real_utility"] - repair[-1]["selected_real_utility"]),
                }
            )

    curve_path = tables_dir(root, smoke) / "experiment_k_classic_control_curves.csv"
    effect_path = tables_dir(root, smoke) / "experiment_k_classic_control_effects.csv"
    rollout_path = tables_dir(root, smoke) / "experiment_k_classic_control_rollouts.csv"
    pd.DataFrame(rows).to_csv(curve_path, index=False)
    pd.DataFrame(effect_rows).to_csv(effect_path, index=False)
    pd.DataFrame(rollout_rows).to_csv(rollout_path, index=False)
    pd.DataFrame(audit_rows).to_csv(tables_dir(root, smoke) / "experiment_k_classic_control_audit.csv", index=False)

    summary = _summarize(rows)
    benchmark_diagnostics: dict[str, Any] = {}
    mismatch_count = 0
    repair_count = 0
    risk_count = 0
    for benchmark_name in CLASSIC_CONTROL_BENCHMARKS:
        effects = [row for row in effect_rows if row["benchmark"] == benchmark_name]
        latent_delta = [float(row["raw_latent_delta_n64_vs_n1"]) for row in effects]
        real_delta = [float(row["raw_real_delta_n64_vs_n1"]) for row in effects]
        repair_gain = [float(row["combined_repair_n64_improvement_over_raw"]) for row in effects]
        risk_delta = [float(row["raw_risk_delta_n64_vs_n1"]) for row in effects]
        latent_ci = _ci(latent_delta, seed=seed + 1100)
        real_ci = _ci(real_delta, seed=seed + 1200)
        repair_ci = _ci(repair_gain, seed=seed + 1300)
        risk_ci = _ci(risk_delta, seed=seed + 1400)
        mismatch = bool(latent_ci["lo"] > 1.0 and real_ci["hi"] < 0.50)
        repair_helped = bool(repair_ci["lo"] > 0.10)
        risk_exposed = bool(risk_ci["lo"] > 0.02)
        mismatch_count += int(mismatch)
        repair_count += int(repair_helped)
        risk_count += int(risk_exposed)
        benchmark_diagnostics[benchmark_name] = {
            "raw_latent_delta_n64_vs_n1_ci": latent_ci,
            "raw_real_delta_n64_vs_n1_ci": real_ci,
            "combined_repair_n64_improvement_over_raw_ci": repair_ci,
            "raw_risk_delta_n64_vs_n1_ci": risk_ci,
            "selected_tail_mismatch": mismatch,
            "combined_repair_helped": repair_helped,
            "risk_increased_with_raw_high_n": risk_exposed,
        }

    payload = {
        "experiment": "K_classic_control_benchmarks",
        "smoke": bool(smoke),
        "benchmarks": list(CLASSIC_CONTROL_BENCHMARKS),
        "candidate_count_per_seed": int(candidate_count),
        "pilot_size": int(pilot_size),
        "seeds": seeds,
        "n_values": N_GRID,
        "scorers": SCORERS,
        "summary": summary,
        "benchmark_diagnostics": benchmark_diagnostics,
        "audit_rows": audit_rows,
        "artifacts": {
            "curves": str(curve_path.relative_to(root)),
            "effects": str(effect_path.relative_to(root)),
            "rollouts": str(rollout_path.relative_to(root)),
            "figure": "figures/figure11_classic_control_benchmarks.png",
        },
        "key_result": {
            "selected_tail_mismatch_benchmark_count": int(mismatch_count),
            "combined_repair_improvement_benchmark_count": int(repair_count),
            "raw_high_n_risk_increase_benchmark_count": int(risk_count),
            "curve_rows": int(len(rows)),
            "rollout_rows": int(len(rollout_rows)),
            "effect_rows": int(len(effect_rows)),
            "all_splits_eval_disjoint": all(row["overlap_count"] == 0 for row in audit_rows),
        },
    }
    write_json(results_dir(root, smoke) / "experiment_k_classic_control_benchmarks.json", payload)
    _plot(rows, figures_dir(root, smoke) / "figure11_classic_control_benchmarks.png")
    return payload


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

"""Experiment I: lightweight Gymnasium stochastic benchmark suite."""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rssm_tail_audit.gym_benchmarks import BENCHMARKS, generate_benchmark_records, seeded_start_state
from rssm_tail_audit.leakage import deterministic_split
from rssm_tail_audit.metrics import selection_curves
from rssm_tail_audit.scorers import fit_pilot_calibrator, score_records

from experiments.common import N_GRID, ensure_dirs, figures_dir, results_dir, root_from_file, smoke_argparser, tables_dir, write_json


SCORERS = [
    "raw_value",
    "uncertainty_pessimism",
    "ensemble_uncertainty_repair",
    "pilot_calibrated",
    "combined_repair",
    "random",
    "oracle",
]


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    latent_grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["benchmark"]), str(row["scorer"]), int(row["N"]))
        grouped[key].append(float(row["selected_real_utility"]))
        latent_grouped[key].append(float(row["selected_latent_value"]))
    out: dict[str, object] = {}
    for key, values in grouped.items():
        benchmark, scorer, n_value = key
        arr = np.asarray(values, dtype=float)
        latent_arr = np.asarray(latent_grouped[key], dtype=float)
        out.setdefault(benchmark, {}).setdefault(scorer, {})[str(n_value)] = {
            "mean_real": float(np.mean(arr)),
            "mean_latent": float(np.mean(latent_arr)),
            "std_real": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
        }
    return out


def _plot(rows: list[dict[str, object]], output) -> None:
    df = pd.DataFrame(rows)
    means = df.groupby(["benchmark", "scorer", "N"], as_index=False)["selected_real_utility"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), dpi=150)
    colors = {
        "raw_value": "#2f6fbb",
        "combined_repair": "#1a8f5a",
        "pilot_calibrated": "#44633f",
        "uncertainty_pessimism": "#7a3e9d",
        "ensemble_uncertainty_repair": "#0f766e",
        "random": "#777777",
        "oracle": "#202020",
    }
    for ax, benchmark in zip(axes, BENCHMARKS):
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
    axes[0].set_ylabel("Expected real return")
    axes[-1].legend(frameon=False, fontsize=6.8, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run(smoke: bool = False, seed: int = 9):
    root = root_from_file()
    ensure_dirs(root, smoke=smoke)
    seeds = [seed + i for i in range(2 if smoke else 8)]
    rows: list[dict[str, object]] = []
    audit_rows = []
    for benchmark_name, spec in BENCHMARKS.items():
        for family_seed in seeds:
            start = seeded_start_state(spec.env_id, family_seed)
            records = generate_benchmark_records(
                spec,
                n=120 if smoke else 220,
                seed=family_seed + 100,
                start_state=start,
                state_id=start,
            )
            pilot_idx, eval_idx = deterministic_split(
                len(records),
                pilot_size=40 if smoke else 80,
                seed=family_seed + 300,
            )
            calibrator = fit_pilot_calibrator([records[int(i)] for i in pilot_idx])
            eval_records = [records[int(i)] for i in eval_idx]
            audit_rows.append(
                {
                    "benchmark": benchmark_name,
                    "seed": int(family_seed),
                    "pilot_size": int(len(pilot_idx)),
                    "eval_size": int(len(eval_idx)),
                    "overlap_count": int(len(set(map(int, pilot_idx)) & set(map(int, eval_idx)))),
                }
            )
            for scorer in SCORERS:
                scores = score_records(eval_records, scorer, calibrator=calibrator)
                curves = selection_curves(eval_records, scores, N_GRID)
                for curve in curves:
                    rows.append(
                        {
                            "benchmark": benchmark_name,
                            "seed": int(family_seed),
                            "start_state": int(start),
                            "scorer": scorer,
                            **curve,
                        }
                    )

    pd.DataFrame(rows).to_csv(tables_dir(root, smoke) / "experiment_i_gymnasium_benchmarks.csv", index=False)
    summary = _summarize(rows)
    benchmark_diagnostics = {}
    mismatch_count = 0
    repair_count = 0
    for benchmark_name in BENCHMARKS:
        raw_n1 = summary[benchmark_name]["raw_value"]["1"]  # type: ignore[index]
        raw_n64 = summary[benchmark_name]["raw_value"]["64"]  # type: ignore[index]
        repair_n64 = summary[benchmark_name]["combined_repair"]["64"]  # type: ignore[index]
        latent_delta = float(raw_n64["mean_latent"] - raw_n1["mean_latent"])
        real_delta = float(raw_n64["mean_real"] - raw_n1["mean_real"])
        repair_gain = float(repair_n64["mean_real"] - raw_n64["mean_real"])
        mismatch = bool(latent_delta > 0.05 and real_delta < -0.05)
        repair_helped = bool(repair_gain > 0.0)
        mismatch_count += int(mismatch)
        repair_count += int(repair_helped)
        benchmark_diagnostics[benchmark_name] = {
            "raw_latent_delta_n64_vs_n1": latent_delta,
            "raw_real_delta_n64_vs_n1": real_delta,
            "combined_repair_n64_improvement_over_raw": repair_gain,
            "selected_tail_mismatch": mismatch,
            "combined_repair_helped": repair_helped,
        }
    payload = {
        "experiment": "I_gymnasium_stochastic_benchmarks",
        "smoke": bool(smoke),
        "benchmarks": list(BENCHMARKS),
        "n_values": N_GRID,
        "scorers": SCORERS,
        "summary": summary,
        "benchmark_diagnostics": benchmark_diagnostics,
        "audit_rows": audit_rows,
        "key_result": {
            "selected_tail_mismatch_benchmark_count": int(mismatch_count),
            "combined_repair_improvement_benchmark_count": int(repair_count),
        },
    }
    write_json(results_dir(root, smoke) / "experiment_i_gymnasium_benchmarks.json", payload)
    _plot(rows, figures_dir(root, smoke) / "figure9_gymnasium_benchmarks.png")
    return payload


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

"""Shared experiment helpers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rssm_tail_audit.envs import RolloutRecord
from rssm_tail_audit.metrics import high_n_delta, selection_curves, top_tail_diagnostics
from rssm_tail_audit.scorers import fit_pilot_calibrator, score_records


N_GRID = [1, 2, 4, 8, 16, 32, 64]


def root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path, smoke: bool = False) -> None:
    names = ["results", "results/tables", "figures"]
    if smoke:
        names.extend(["results/smoke", "results/smoke/tables", "figures/smoke"])
    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)


def results_dir(root: Path, smoke: bool = False) -> Path:
    return root / "results" / "smoke" if smoke else root / "results"


def tables_dir(root: Path, smoke: bool = False) -> Path:
    return results_dir(root, smoke) / "tables"


def figures_dir(root: Path, smoke: bool = False) -> Path:
    return root / "figures" / "smoke" if smoke else root / "figures"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_scorers(
    records: list[RolloutRecord],
    scorers: list[str],
    n_values: list[int] | None = None,
    pilot_size: int = 96,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_values = n_values or N_GRID
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    pilot_records = [records[i] for i in idx[: min(pilot_size, len(records))]]
    calibrator = fit_pilot_calibrator(pilot_records)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"pilot_size": len(pilot_records), "scorers": {}}
    for scorer in scorers:
        scores = score_records(records, scorer, calibrator=calibrator)
        curves = selection_curves(records, scores, n_values)
        tail = top_tail_diagnostics(records, scores)
        for row in curves:
            rows.append({"scorer": scorer, **row})
        summary["scorers"][scorer] = {
            "real_delta_high_n": high_n_delta(curves, "selected_real_utility"),
            "latent_delta_high_n": high_n_delta(curves, "selected_latent_value"),
            "tail": tail,
            "N64_real": float(curves[-1]["selected_real_utility"]),
            "N64_latent": float(curves[-1]["selected_latent_value"]),
        }
    return rows, summary


def plot_curve(
    rows: list[dict[str, Any]],
    output: Path,
    scorers: list[str],
    y_keys: list[str],
    title: str,
    ylabel: str,
) -> None:
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    colors = {
        "raw_value": "#2f6fbb",
        "belief_collapsed": "#2f6fbb",
        "combined_repair": "#1a8f5a",
        "uncertainty_pessimism": "#7a3e9d",
        "ensemble_uncertainty_repair": "#0f766e",
        "belief_consistency": "#bc6c25",
        "decoder_consistency": "#4a808a",
        "pilot_calibrated": "#44633f",
        "random": "#7a7a7a",
        "oracle": "#202020",
        "good": "#1a8f5a",
        "overconfident": "#b23b3b",
        "value_optimistic": "#b85822",
    }
    markers = {
        "selected_real_utility": "o",
        "selected_latent_value": "s",
        "latent_real_gap": "^",
    }
    for scorer in scorers:
        sub = df[df["scorer"] == scorer].sort_values("N")
        for y_key in y_keys:
            label = f"{scorer}: {y_key.replace('selected_', '').replace('_', ' ')}"
            linestyle = "--" if "latent" in y_key and y_key != "latent_real_gap" else "-"
            ax.plot(
                sub["N"],
                sub[y_key],
                marker=markers.get(y_key, "o"),
                linewidth=2.0,
                markersize=4.5,
                linestyle=linestyle,
                color=colors.get(scorer, None),
                alpha=0.92 if y_key != "selected_latent_value" else 0.68,
                label=label,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["N"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("candidate budget N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(fontsize=7.5, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def smoke_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true", help="use a smaller deterministic run")
    parser.add_argument("--seed", type=int, default=0)
    return parser

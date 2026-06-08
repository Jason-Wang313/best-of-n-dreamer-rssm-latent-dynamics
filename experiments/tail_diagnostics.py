"""Generate tail diagnostic figure from Experiment A artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import ensure_dirs, root_from_file, smoke_argparser


def run(smoke: bool = False, seed: int = 0):
    root = root_from_file()
    ensure_dirs(root)
    curve_path = root / "results" / "tables" / "experiment_a_curves.csv"
    if not curve_path.exists():
        from experiments.experiment_a_toy_mismatch import run as run_a

        run_a(smoke=smoke, seed=seed)
    df = pd.read_csv(curve_path)
    wanted = ["belief_collapsed", "combined_repair", "oracle"]
    sub = df[df["scorer"].isin(wanted)].copy()
    sub["gap"] = sub["latent_real_gap"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=150)
    colors = {"belief_collapsed": "#b23b3b", "combined_repair": "#1a8f5a", "oracle": "#202020"}
    for scorer in wanted:
        g = sub[sub["scorer"] == scorer]
        ax.plot(g["N"], g["gap"], marker="o", linewidth=2.0, color=colors[scorer], label=scorer)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sub["N"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Best-of-N candidates")
    ax.set_ylabel("Selected latent-real gap")
    ax.set_title("Upper-tail latent diagnostics")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = root / "figures" / "figure3_tail_diagnostics.png"
    fig.savefig(out)
    plt.close(fig)
    return {"figure": str(out.relative_to(root))}


def main() -> None:
    parser = smoke_argparser(__doc__ or "")
    args = parser.parse_args()
    run(smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()

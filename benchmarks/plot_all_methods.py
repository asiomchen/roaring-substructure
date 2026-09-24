"""Plot all serial 50k-query methods with 95% confidence intervals.

Run from the repository root:
    uv run benchmarks/plot_all_methods.py
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
METHODS = (
    ("Naive", "01_naive", "#9AA8B7"),
    ("Pattern", "02_pattern", "#61748A"),
    ("PostgreSQL GiST", "postgres18", "#8063A6"),
    ("Roaring scan", "03_roaring_approach", "#D88A0B"),
    ("Roaring postings", "05_roaring_postings", "#C56A17"),
    ("NumPy postings", "05a_numpy_postings", "#0875A8"),
)
FAST_METHODS = (2, 3, 4, 5)
T_95_DF_2 = 4.302652729911275  # Two-sided 95% Student's t critical value, df=2.


def load_timings(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    rows = []
    for label, benchmark, color in METHODS:
        runs = frame.loc[
            (frame["benchmark"] == benchmark) & (frame["phase"] == "match")
        ].sort_values("run")
        if runs["run"].tolist() != [1, 2, 3]:
            raise ValueError(f"Expected three match runs for {benchmark}")
        seconds = runs["seconds"].to_numpy(dtype=float)
        mean = float(seconds.mean())
        ci_half_width = T_95_DF_2 * float(seconds.std(ddof=1)) / np.sqrt(len(seconds))
        rows.append(
            {
                "label": label,
                "color": color,
                "mean": mean,
                "ci_low": mean - ci_half_width,
                "ci_high": mean + ci_half_width,
            }
        )
    return rows


def bars(ax, rows: list[dict], *, limit: float, ticks: list[int], title: str) -> None:
    y = np.arange(len(rows))
    means = np.array([row["mean"] for row in rows])
    ci_lows = np.array([row["ci_low"] for row in rows])
    ci_highs = np.array([row["ci_high"] for row in rows])
    ax.barh(y, means, height=0.63, color=[row["color"] for row in rows], zorder=2)
    ax.errorbar(
        means,
        y,
        xerr=np.vstack((means - ci_lows, ci_highs - means)),
        fmt="none",
        ecolor="#253345",
        elinewidth=1.4,
        capsize=3.5,
        capthick=1.4,
        zorder=4,
    )
    for position, row in zip(y, rows):
        ax.text(
            row["ci_high"] + limit * 0.02,
            position,
            f"{row['mean']:.2f} s",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color="#253345",
            zorder=5,
        )

    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, limit)
    ax.set_xticks(ticks)
    ax.set_xlabel("Matching time (seconds)", fontsize=10.5, labelpad=8)
    ax.set_yticks(y, [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=10.5, length=0)
    ax.grid(axis="x", color="#E4EAF0", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)


def draw(rows: list[dict], output_prefix: Path) -> None:
    sns.set_theme(
        style="whitegrid", context="talk", palette="colorblind", font="DejaVu Sans"
    )
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#9AA5B1",
            "axes.labelcolor": "#283444",
            "text.color": "#1E293B",
            "xtick.color": "#465569",
            "ytick.color": "#283444",
        }
    )

    fig, (all_ax, fast_ax) = plt.subplots(
        2,
        1,
        figsize=(11.6, 8.4),
        gridspec_kw={"height_ratios": [1.15, 0.85]},
    )
    fig.subplots_adjust(left=0.28, right=0.95, top=0.79, bottom=0.13, hspace=0.55)
    fig.suptitle(
        "From the naive scan to NumPy postings",
        x=0.08,
        y=0.965,
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.882,
        "1,000 reactants × 50,000 queries  ·  serial match time  ·  lower is better",
        fontsize=11.5,
        color="#536275",
        ha="left",
    )

    bars(
        all_ax, rows, limit=85, ticks=[0, 20, 40, 60, 80], title="All measured methods"
    )
    bars(
        fast_ax,
        [rows[i] for i in FAST_METHODS],
        limit=10,
        ticks=[0, 2, 4, 6, 8, 10],
        title="Under 10 seconds · detail view",
    )
    # speedup = rows[1]["median"] / rows[-1]["median"]
    # fig.text(
    #     0.08,
    #     0.068,
    #     f"NumPy postings are {speedup:.1f}× faster than the naive benchmark.",
    #     fontsize=11,
    #     fontweight="bold",
    #     color=rows[-1]["color"],
    #     ha="left",
    # )
    fig.text(
        0.08,
        0.032,
        "Bars: mean of three runs; whiskers: 95% t confidence intervals (n=3). "
        "PostgreSQL 18 uses one connection. Index build excluded. Benchmark 04 omitted.",
        fontsize=9.2,
        color="#6B7787",
        ha="left",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=320, facecolor="white")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(svg_path, facecolor="white")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, default=HERE / "timings.csv")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "plots" / "all_methods_50k",
    )
    args = parser.parse_args()
    draw(load_timings(args.timings), args.output_prefix)
    print(
        f"Wrote {args.output_prefix.with_suffix('.png')} and "
        f"{args.output_prefix.with_suffix('.svg')}"
    )


if __name__ == "__main__":
    main()

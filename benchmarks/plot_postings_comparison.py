"""Export a blog-ready NumPy versus Roaring comparison from timings.csv.

Run from the repository root:
    uv run benchmarks/plot_postings_comparison.py
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
SAMPLES = (
    (
        "50k queries",
        "50M possible pairs",
        {
            "NumPy": "05a_numpy_postings",
            "Roaring": "05a_roaring_postings",
        },
    ),
    (
        "200k queries",
        "200M possible pairs",
        {
            "NumPy": "05a_numpy_postings_200k",
            "Roaring": "05a_roaring_postings_200k",
        },
    ),
)
METHODS = ("NumPy", "Roaring")
BAR_WIDTH = 0.28
BAR_OFFSETS = {"NumPy": -BAR_WIDTH / 2, "Roaring": BAR_WIDTH / 2}


def load_timings(path: Path) -> dict[str, dict[str, dict[str, np.ndarray | float]]]:
    frame = pd.read_csv(path)
    summary = {}
    for sample, _, labels in SAMPLES:
        summary[sample] = {}
        match_counts = {}
        for method, benchmark in labels.items():
            rows = frame[frame["benchmark"] == benchmark]
            matches = rows[rows["phase"] == "match"].sort_values("run")
            builds = rows[rows["phase"] == "index"]
            if matches["run"].tolist() != [1, 2, 3] or len(builds) != 1:
                raise ValueError(
                    f"Expected three match runs and one index build: {benchmark}"
                )
            if matches["matches"].nunique() != 1:
                raise ValueError(f"Match counts vary between runs: {benchmark}")
            match_counts[method] = int(matches["matches"].iloc[0])
            summary[sample][method] = {
                "match": matches["seconds"].to_numpy(dtype=float),
                "build": float(builds["seconds"].iloc[0]),
            }
        if len(set(match_counts.values())) != 1:
            raise ValueError(f"Match counts differ between methods: {sample}")
    return summary


def draw(summary: dict, output_prefix: Path) -> None:
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
            "ytick.color": "#465569",
            "grid.color": "#E4EAF0",
            "grid.linewidth": 0.9,
        }
    )
    colors = dict(zip(METHODS, sns.color_palette("colorblind", 2)))
    positions = np.array([0.0, 1.0])
    labels = [f"{name}\n{pairs}" for name, pairs, _ in SAMPLES]

    fig, (match_ax, build_ax) = plt.subplots(
        1,
        2,
        figsize=(12.8, 6.1),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )
    fig.subplots_adjust(left=0.075, right=0.97, top=0.77, bottom=0.25, wspace=0.31)
    fig.suptitle(
        "NumPy pulls ahead as the query set grows",
        x=0.075,
        y=0.97,
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.875,
        "Substructure search over 1,000 reactants  ·  128 selected postings  ·  lower is better",
        fontsize=11.5,
        color="#536275",
        ha="left",
    )

    for method in METHODS:
        match_runs = [summary[sample][method]["match"] for sample, _, _ in SAMPLES]
        means = np.array([runs.mean() for runs in match_runs])
        build_times = np.array(
            [summary[sample][method]["build"] for sample, _, _ in SAMPLES]
        )
        color = colors[method]
        bar_positions = positions + BAR_OFFSETS[method]

        match_bars = match_ax.bar(
            bar_positions,
            means,
            width=BAR_WIDTH,
            color=color,
            label=method,
            zorder=3,
        )
        build_bars = build_ax.bar(
            bar_positions,
            build_times,
            width=BAR_WIDTH,
            color=color,
            zorder=3,
        )

        for bar, mean in zip(match_bars, means):
            match_ax.annotate(
                f"{mean:.2f} s",
                (bar.get_x() + bar.get_width() / 2, mean),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color=color,
            )
        for bar, value in zip(build_bars, build_times):
            build_ax.annotate(
                f"{value:.2f} s",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    for ax, title, ymax, ticks in (
        (match_ax, "Matching time", 8.5, np.arange(0, 9, 2)),
        (build_ax, "Index build time", 23, np.arange(0, 25, 5)),
    ):
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(0, ymax)
        ax.set_yticks(ticks)
        ax.set_xticks(positions, labels)
        ax.set_ylabel("Seconds", fontsize=11)
        ax.tick_params(axis="both", labelsize=10.5, length=0)
        ax.grid(axis="y", zorder=0)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)

    match_ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=10.5,
        ncol=2,
        bbox_to_anchor=(-0.02, 1.02),
        handlelength=2.4,
    )
    fig.text(
        0.075,
        0.075,
        "Matching: mean of three runs. "
        "Index build: one run. Source: benchmarks/timings.csv.",
        fontsize=9.5,
        color="#6B7787",
        ha="left",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=320, facecolor="white")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(svg_path, facecolor="white")
    # Matplotlib leaves spaces at the ends of multi-line SVG path coordinates.
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
        default=HERE / "plots" / "postings_comparison",
    )
    args = parser.parse_args()
    draw(load_timings(args.timings), args.output_prefix)
    print(
        f"Wrote {args.output_prefix.with_suffix('.png')} and "
        f"{args.output_prefix.with_suffix('.svg')}"
    )


if __name__ == "__main__":
    main()

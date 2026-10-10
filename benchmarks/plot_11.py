"""Plot of benchmark 11: the index built once, saved, and loaded to search.

Reads `<run>/results.jsonl` and writes `<run>/plots/index_on_disk.{png,svg}`:
four panels, one row per variant, each on its own axis since the units differ.

    build           read queries + create index + save index (s)
    search          read index + read reactants + median matching (s)
    index on disk   the index file (MiB)
    search memory   the search process's peak RSS (MiB), split into the
                    libraries (RSS at start), the loaded index, and reactants
                    plus matching (the rest of the peak)

Styled like `plot_results.py`, whose run loading, figure and footer it reuses.

Usage:
    uv run benchmarks/plot_11.py                  # newest run
    uv run benchmarks/plot_11.py --run benchmarks/results/<run>
"""

import argparse
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.plot_results import (
    FONT,
    GRID,
    INK,
    STEPS,
    Run,
    extras,
    figure,
    footer,
    matching,
    newest_run,
)

BENCHMARK = "11_index_on_disk"

# Variant -> presentation label, top to bottom.
VARIANTS = {
    "numpy-pattern": "Python: NumPy postings, Pattern fp",
    "numpy-paths": "Python: NumPy postings, path fp",
    "rust-paths4-branches": "Rust: paths ≤ 4 + branches fp",
}

# The categorical slots of plot_results.py, in the same fixed order per panel.
BLUE, ORANGE, GREEN = (color for _, _, color in STEPS[:3])


def mib(text: str) -> float:
    return float(text.removesuffix(" MiB"))


def panels(reports: list[dict]) -> list[tuple]:
    """Per panel: title, unit, value format, and its (segment, color, values)."""
    build, search, disk, memory = [], [], [], []
    for r in reports:
        e = extras(r)
        build.append(
            (r["query_prep_seconds"], r["index_prep_seconds"], e["save index"])
        )
        search.append((e["read index"], r["reactant_prep_seconds"], matching(r)))
        disk.append((mib(e["index on disk"]),))
        start, after_index, *_ = (mib(v) for v in e["search RSS"].split(" / "))
        peak = mib(e["search peak RSS"])
        memory.append((start, after_index - start, peak - after_index))

    def segments(rows, names, colors):
        return [
            (n, c, [row[i] for row in rows])
            for i, (n, c) in enumerate(zip(names, colors))
        ]

    seconds = lambda v: f"{v:.2f} s"
    mebibytes = lambda v: f"{v:,.0f} MiB"
    return [
        (
            "Build: once",
            "time (s)",
            seconds,
            segments(
                build,
                ("read queries", "create index", "save index"),
                (BLUE, ORANGE, GREEN),
            ),
        ),
        (
            "Search: load the index and match",
            "time (s)",
            seconds,
            segments(
                search, ("read index", "read reactants", "match"), (BLUE, ORANGE, GREEN)
            ),
        ),
        (
            "Index file on disk",
            "size (MiB)",
            mebibytes,
            segments(disk, ("index file",), (BLUE,)),
        ),
        (
            "Search process memory (peak)",
            "resident memory (MiB)",
            mebibytes,
            segments(
                memory,
                ("libraries", "loaded index", "reactants + matching"),
                (BLUE, ORANGE, GREEN),
            ),
        ),
    ]


def stacked(ax, labels: list[str], title: str, unit: str, fmt, segs) -> None:
    """Horizontal stacked bars with the total at each bar's end."""
    rows = range(len(labels))
    left = [0.0] * len(labels)
    for name, color, values in segs:
        # A thin surface-colored edge is the gap between touching segments.
        ax.barh(
            rows,
            values,
            left=left,
            height=0.6,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=name,
        )
        left = [a + b for a, b in zip(left, values)]
    top = max(left)
    for y, total in enumerate(left):
        ax.text(
            total + top * 0.02, y, fmt(total), va="center", fontsize=FONT, color=INK
        )

    ax.set_yticks(list(rows), labels)
    ax.set_xlim(0, top * 1.3)
    ax.set_title(title, fontsize=8, loc="left", pad=16)
    ax.set_xlabel(unit, fontsize=FONT)
    ax.tick_params(axis="y", which="both", length=0, labelsize=FONT)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="x", top=False, labelsize=FONT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if len(segs) > 1:
        # One row between the title and the plot, clear of the bars.
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncol=len(segs),
            frameon=False,
            fontsize=6.5,
            handlelength=1.0,
            columnspacing=1.0,
            borderaxespad=0.1,
        )


def plot(run: Run) -> None:
    # Reversed: barh draws the first row at the bottom, and the shared y axis
    # cannot be inverted per panel.
    reports = run.select([(BENCHMARK, name) for name in VARIANTS])[::-1]
    labels = list(VARIANTS.values())[::-1]
    first = reports[0]
    with figure(
        run.path / "plots",
        "index_on_disk",
        nrows=2,
        ncols=2,
        figsize=(9.2, 4.9),
        sharey=True,
    ) as (fig, axes):
        for ax, (title, unit, fmt, segs) in zip(axes.flat, panels(reports)):
            stacked(ax, labels, title, unit, fmt, segs)
        fig.suptitle(
            "Index built once, saved to disk, loaded to search: Python against Rust, "
            f"one thread ({first['queries']:,} queries x {first['reactants']:,} "
            "reactants)",
            fontsize=9,
        )
        footer(
            fig,
            f"{run.machine}; all variants return {first['matches']:,} matches; "
            "build and search run in separate processes; the OS file cache is "
            "dropped before each search",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, help="default: the newest run")
    args = parser.parse_args()
    plot(Run.load(args.run or newest_run()))


if __name__ == "__main__":
    main()

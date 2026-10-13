"""Plots of a `benchmarks/run_all.py` run.

Reads `<run>/results.jsonl` and `<run>/metadata.json` once, selects each plot's
reports in memory and writes, into `<run>/plots/`:

    python_total_time.{png,svg}  the in-process Python methods (benchmarks
                                 01-03 and 05; the multi-process 04 is left out)
    rust_total_time.{png,svg}    the fastest Python method against the
                                 Rust-backed ones (benchmarks 06, 08, 10 and
                                 09's fastest configuration)
    fp_sweep.{png,svg}           benchmark 09: matching time and the pairs sent
                                 to exact matching, per fingerprint kind and size

Plots are for presentations, so titles and labels name the method, never the
benchmark number. Repeat measurements of the same code (06's NumPy variant is
05's NumPy index; 07's Rust variant is 06's) are left out, so no two bars share
a label.

The two total-time plots split each option's time into query prep, index prep,
reactant prep and median matching. Styled with SciencePlots (`science` +
`no-latex`).

Usage:
    uv run benchmarks/plot_results.py                  # newest run
    uv run benchmarks/plot_results.py --run benchmarks/results/<run>
"""

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the styles)
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
SWEEP = "09_rust_native_fp"

# A variant of FASTEST picks the benchmark's report with the lowest median
# matching time (benchmark 09's sweep).
FASTEST = "fastest"

# Each total-time plot: file stem, title, and its reports in the order the
# talk introduces them (top to bottom), each with a presentation label. "fp" is
# fingerprint; the FASTEST label gets the winning configuration appended.
TOTAL_PLOTS = [
    (
        "python_total_time",
        "Substructure search in Python, one thread",
        {
            ("01_naive", None): "RDKit match on every pair",
            ("02_pattern", None): "RDKit Pattern fp screen",
            ("03_roaring_approach", None): "Pattern fp as Roaring bitmaps",
            ("05_postings", "roaring pattern"): "Roaring posting lists, Pattern fp",
            ("05_postings", "roaring paths"): "Roaring posting lists, path fp",
            ("05_postings", "numpy pattern"): "NumPy posting lists, Pattern fp",
            ("05_postings", "numpy paths"): "NumPy posting lists, path fp",
        },
    ),
    (
        "rust_total_time",
        "Python against Rust, one thread",
        {
            ("05_postings", "numpy pattern"): "NumPy posting lists + RDKit match",
            ("06_rust_postings", "rust"): "Rust posting lists + RDKit match",
            ("10_chematic", "naive"): "chematic (Rust) match on every pair",
            ("10_chematic", "screened"): "Rust screen + chematic match",
            ("08_rust_native", None): "All in Rust, path fp",
            (SWEEP, FASTEST): "All in Rust, best fp",
        },
    ),
]

# Benchmark 09's fingerprint kinds, as the sweep names them and as plotted.
# Every path kind includes the atom labels.
FP_KINDS = {
    "atoms": "atom labels only",
    "paths2": "paths ≤ 2 bonds",
    "paths4": "paths ≤ 4 bonds",
    "paths6": "paths ≤ 6 bonds",
    "paths4-nocount": "paths ≤ 4, no counts",
    "paths4+branches": "paths ≤ 4 + branches",
    "paths4+cycles": "paths ≤ 4 + rings",
    "paths4+branches+cycles": "paths ≤ 4 + branches + rings",
    "paths4+long6+branches+cycles": "… + long paths ≤ 6",
    "paths4+long8+branches+cycles": "… + long paths ≤ 8",
}

# Stack order, left to right: the report field and the categorical slot of each
# step (validated with the dataviz skill's validate_palette.js, light mode).
STEPS = [
    ("query_prep_seconds", "query prep", "#2a78d6"),
    ("index_prep_seconds", "index prep", "#eb6834"),
    ("reactant_prep_seconds", "reactant prep", "#1baf7a"),
    ("match_median_seconds", "matching", "#eda100"),
]
INK = "#3d3d3a"
MUTED = "#73726c"
GRID = "#e2e1da"
FONT = 7

# Sequential ramps, light to dark: the dataviz palette's blue ramp (steps
# 100-700), and for a second sequential panel the next categorical hue, orange,
# stepped the same way around its slot color #eb6834.
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
ORANGE_RAMP = [
    "#fde3d6",
    "#f8bf9f",
    "#f39668",
    "#eb6834",
    "#c94f1f",
    "#9c3a14",
    "#6e280c",
]


@dataclass
class Run:
    path: Path
    reports: list[dict]
    meta: dict

    @classmethod
    def load(cls, path: Path) -> "Run":
        lines = (path / "results.jsonl").read_text().splitlines()
        meta = json.loads((path / "metadata.json").read_text())
        return cls(path, [json.loads(line) for line in lines], meta)

    @property
    def machine(self) -> str:
        """Footer text: CPU, Python, RDKit and rustc versions."""
        rustc = self.meta["rustc"].split(" (")[0]
        return (
            f"{self.meta['machine']['cpu']}, Python {self.meta['python']}, "
            f"RDKit {self.meta['packages']['rdkit']}, {rustc}"
        )

    def of(self, benchmark: str) -> list[dict]:
        return [r for r in self.reports if r["benchmark"] == benchmark]

    def select(self, keys) -> list[dict]:
        """The report for each (benchmark, variant) key, in key order."""
        by_key = {(r["benchmark"], r["variant"]): r for r in self.reports}
        for benchmark, variant in keys:
            if variant == FASTEST and (sweep := self.of(benchmark)):
                by_key[benchmark, FASTEST] = min(sweep, key=matching)
        missing = [key for key in keys if key not in by_key]
        if missing:
            raise SystemExit(f"{self.path}: no report for {missing}")
        return [by_key[key] for key in keys]


def matching(report: dict) -> float:
    return report["match_median_seconds"]


def extras(report: dict) -> dict:
    return {label: value for label, value, _ in report["extras"]}


def newest_run() -> Path:
    runs = sorted(p for p in RESULTS.iterdir() if (p / "results.jsonl").exists())
    if not runs:
        raise SystemExit(f"no runs with results.jsonl under {RESULTS}")
    return runs[-1]


@contextmanager
def figure(out: Path, stem: str, **kwargs):
    """A styled figure, saved as `<out>/<stem>.png` and `.svg` on exit."""
    with plt.style.context(["science", "no-latex"]):
        fig, axes = plt.subplots(**kwargs)
        yield fig, axes
        out.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            path = out / f"{stem}.{suffix}"
            # 600 dpi on opaque white: sharp when a slide scales the PNG up.
            fig.savefig(path, dpi=600, facecolor="white")
            print(f"wrote {path}")
        plt.close(fig)


def footer(fig, text: str) -> None:
    fig.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=5.5, color=MUTED)


def fp_config(report: dict) -> str:
    """A benchmark 09 variant, "<kind> <bits> p<postings> <order>", for a reader."""
    kind, bits, _, _ = report["variant"].split(" ")
    return f"{FP_KINDS.get(kind, kind)}, {int(bits):,} bits"


def plot_totals(run: Run, stem: str, title: str, labeled: dict) -> None:
    """Horizontal stacked bars of the prep steps and median matching."""
    # Reversed so the first report is the top bar.
    reports = run.select(list(labeled))[::-1]
    labels = [
        f"{label} ({fp_config(r)})" if variant == FASTEST else label
        for ((_, variant), label), r in zip(list(labeled.items())[::-1], reports)
    ]
    totals = [r["end_to_end_seconds"] for r in reports]
    top = max(totals)
    first = reports[0]

    with figure(run.path / "plots", stem, figsize=(6.4, 1.2 + 0.35 * len(reports))) as (
        fig,
        ax,
    ):
        left = [0.0] * len(reports)
        for field, name, color in STEPS:
            widths = [r[field] or 0.0 for r in reports]  # a step it lacks is None
            # A thin surface-colored edge is the gap between touching segments.
            ax.barh(
                labels,
                widths,
                left=left,
                height=0.62,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=name,
            )
            left = [a + b for a, b in zip(left, widths)]
        for y, total in enumerate(totals):
            ax.text(
                total + top * 0.01,
                y,
                f"{total:.2f} s",
                va="center",
                fontsize=FONT,
                color=INK,
            )

        ax.set_xlabel("total time (s): prep + median matching", fontsize=FONT)
        ax.set_xlim(0, top * 1.12)
        ax.tick_params(axis="y", which="both", length=0, labelsize=FONT)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.tick_params(axis="x", top=False, labelsize=FONT)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(STEPS),
            frameon=False,
            fontsize=FONT,
            handlelength=1.2,
            columnspacing=1.2,
        )
        ax.set_title(
            f"{title} ({first['queries']:,} queries x {first['reactants']:,} reactants)",
            fontsize=8,
            pad=18,
        )
        footer(fig, run.machine)
        fig.tight_layout()


def heatmap(fig, ax, grid, title: str, fmt, ramp, best: tuple[int, int]) -> None:
    """One log-scaled heatmap panel with labelled cells and the best cell outlined."""
    flat = [v for line in grid for v in line]
    norm = LogNorm(vmin=min(flat), vmax=max(flat))
    image = ax.imshow(
        grid,
        cmap=LinearSegmentedColormap.from_list(title, ramp),
        norm=norm,
        aspect="auto",
    )
    for i, line in enumerate(grid):
        for j, v in enumerate(line):
            # Dark ink on the light half of the ramp, white on the dark half.
            color = INK if norm(v) < 0.55 else "white"
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6, color=color)
    row, col = best
    ax.add_patch(
        Rectangle(
            (col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.4
        )
    )
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(which="both", length=0, labelsize=6.5, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    ticks = [
        t for t in (30, 50, 100, 200, 500, 1000, 3000) if norm.vmin <= t <= norm.vmax
    ]
    bar.set_ticks(ticks, labels=[f"{t:,}" for t in ticks])
    bar.ax.tick_params(labelsize=6, which="both", length=2)
    bar.ax.yaxis.set_minor_formatter(NullFormatter())
    bar.outline.set_visible(False)
    bar.set_label("log scale", fontsize=6, color=MUTED)


def plot_sweep(run: Run) -> None:
    """Benchmark 09: two heatmaps over fingerprint kind x size, matching time and
    the pairs the screen sends to exact matching, each on its own log scale."""
    sweep = run.of(SWEEP)
    if not sweep:
        raise SystemExit(f"{run.path}: no benchmark 09 reports")
    # A variant is "<kind> <bits> p<postings> <order>".
    cells = {}
    for r in sweep:
        kind, bits, _, _ = r["variant"].split(" ")
        cells[kind, int(bits)] = r
    kinds = list(dict.fromkeys(kind for kind, _ in cells))  # sweep order
    sizes = sorted({bits for _, bits in cells})
    best = min(sweep, key=matching)
    best_kind, best_bits, postings, order = best["variant"].split(" ")
    best_at = (kinds.index(best_kind), sizes.index(int(best_bits)))

    panels = [
        (
            "matching time (ms)",
            lambda r: 1000 * matching(r),
            lambda v: f"{v:.0f}",
            BLUE_RAMP,
        ),
        (
            "pairs sent to exact matching (thousands)",
            lambda r: extras(r)["exact matches run"] / 1000,
            lambda v: f"{v:,.0f}",
            ORANGE_RAMP,
        ),
    ]
    with figure(
        run.path / "plots", "fp_sweep", ncols=2, figsize=(7.2, 3.9), sharey=True
    ) as (fig, axes):
        for ax, (title, value, fmt, ramp) in zip(axes, panels):
            grid = [[value(cells[kind, bits]) for bits in sizes] for kind in kinds]
            heatmap(fig, ax, grid, title, fmt, ramp, best_at)
            ax.set_xticks(range(len(sizes)), [f"{b:,}" for b in sizes])
            ax.set_yticks(range(len(kinds)), [FP_KINDS.get(k, k) for k in kinds])
            ax.set_xlabel("fingerprint size (bits)", fontsize=FONT)
        axes[0].set_ylabel("fingerprint kind", fontsize=FONT)
        fig.suptitle(
            "All-Rust search: screening fingerprint kind and size, one thread "
            f"({postings.removeprefix('p')} posting lists); outlined: fastest",
            fontsize=8,
        )
        footer(
            fig,
            f"{run.machine}; {order} atom order; all {len(sweep)} configurations "
            f"return {sweep[0]['matches']:,} matches",
        )
        fig.tight_layout(rect=(0, 0.02, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, help="default: the newest run")
    args = parser.parse_args()

    run = Run.load(args.run or newest_run())
    for stem, title, labeled in TOTAL_PLOTS:
        plot_totals(run, stem, title, labeled)
    plot_sweep(run)


if __name__ == "__main__":
    main()

"""Diagram of benchmark 05's inverted index (posting lists), for presentations.

Writes `benchmarks/plots/inverted_index.{png,svg}`: a toy example of 8 queries and 8
fingerprint bits in three steps, and benchmark 05's real screen counts below it.

    1. query fingerprints, one row per query;
    2. the index, built once: one posting list per bit (the queries that set it),
       bits ranked by how many queries set them;
    3. one reactant: take the top-ranked bits it lacks (3 here, 128 in the
       benchmark), OR their posting lists to rule those queries out, run the full
       fingerprint check on the rest, then RDKit's exact match.

Styled like `plot_results.py`, whose colors it reuses; the PNG is 600 dpi for slides. The funnel
numbers are benchmark 05's NumPy Pattern run (50,000 queries x 1,000 reactants).

Usage:
    uv run benchmarks/plot_inverted_index.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.plot_results import GRID, INK, MUTED, STEPS

BLUE, ORANGE, GREEN, _ = (color for _, _, color in STEPS)
OUT = Path(__file__).resolve().parent / "plots"

# The toy: bits in rank order (most queries first), each query's bits, and the
# reactant's bits. Bit G is missing from the reactant but ranked below the 3
# postings used, so only the full check catches Q3.
BITS = "ABCDEFGH"
QUERIES = {
    "Q1": "AC",
    "Q2": "ABE",
    "Q3": "ACFG",
    "Q4": "BD",
    "Q5": "AD",
    "Q6": "ABF",
    "Q7": "BE",
    "Q8": "CDH",
}
REACTANT = "ACFH"
POSTINGS = 3
MATCHES = {"Q1"}  # what RDKit's exact match confirms

CELL = 0.8
# Slides scale the PNG up: 600 dpi keeps the small labels sharp on a projector.
DPI = 600
FONT = 8


def cell(ax, x, y, color, alpha=1.0, edge="white"):
    ax.add_patch(
        Rectangle(
            (x, y), CELL, CELL, facecolor=color, edgecolor=edge, lw=1.2, alpha=alpha
        )
    )


def text(ax, x, y, s, **kw):
    kw = {"fontsize": FONT, "color": INK, "ha": "center", "va": "center", **kw}
    ax.text(x, y, s, **kw)


def arrow(ax, start, end, label=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.2,
            color=MUTED,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        text(ax, mx, my + 0.45, label, fontsize=FONT - 1, color=MUTED)


def grid(ax, x0, top, rows, cols, filled, color, faded=()):
    """Cells for rows x cols, top row at `top`; `filled` holds (row, col) pairs.

    Filled cells of a `faded` row are drawn in light blue instead of `color`.
    """
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            on = (row, col) in filled
            fill = (BLUE if row in faded else color) if on else GRID
            alpha = 0.3 if on and row in faded else 1.0
            if alpha < 1.0:
                cell(ax, x0 + c * CELL, top - (r + 1) * CELL, "white")
            cell(ax, x0 + c * CELL, top - (r + 1) * CELL, fill, alpha=alpha)


def reactant_cell(ax, x, y, present):
    """A reactant bit: filled when set, an orange outline when the reactant lacks it."""
    if present:
        cell(ax, x, y, BLUE)
    else:
        ax.add_patch(
            Rectangle(
                (x + 0.06, y + 0.06),
                CELL - 0.12,
                CELL - 0.12,
                facecolor="white",
                edgecolor=ORANGE,
                lw=1.4,
            )
        )


def panel_title(ax, x, y, title, subtitle):
    text(ax, x, y, title, ha="left", fontsize=FONT + 1.5, weight="bold")
    text(ax, x, y - 0.55, subtitle, ha="left", fontsize=FONT - 0.5, color=MUTED)


def draw(ax) -> None:
    queries = list(QUERIES)
    has = {(q, b) for q, bits in QUERIES.items() for b in bits}
    top = 15.6
    missing = [b for b in BITS if b not in REACTANT]
    used = missing[:POSTINGS]
    ruled_out = {q for q in queries for b in used if b in QUERIES[q]}
    left = [q for q in queries if q not in ruled_out]
    passed = [q for q in left if set(QUERIES[q]) <= set(REACTANT)]

    # 1. Query fingerprints: rows are queries, columns bits.
    x1 = 1.2
    panel_title(ax, x1 - 0.6, top + 2.0, "1. Query fingerprints", "one row per query")
    for c, b in enumerate(BITS):
        text(ax, x1 + c * CELL + CELL / 2, top + 0.4, b)
    for r, q in enumerate(queries):
        text(ax, x1 - 0.35, top - r * CELL - CELL / 2, q)
    grid(ax, x1, top, queries, BITS, has, BLUE)
    text(
        ax,
        x1 + 4 * CELL,
        top + 0.95,
        "fingerprint bits",
        fontsize=FONT - 1,
        color=MUTED,
    )

    # The reactant's fingerprint, under the queries': same columns.
    ry = top - 8 * CELL - 2.6
    text(
        ax,
        x1 - 0.6,
        ry + 1.95,
        "Reactant fingerprint",
        ha="left",
        fontsize=FONT + 1.5,
        weight="bold",
    )
    for c, b in enumerate(BITS):
        text(ax, x1 + c * CELL + CELL / 2, ry + CELL + 0.35, b)
        reactant_cell(ax, x1 + c * CELL, ry, b in REACTANT)
    lacks = ", ".join(missing)
    text(ax, x1 - 0.6, ry - 0.45, f"lacks {lacks}", ha="left", color=ORANGE)

    # 2. The inverted index: rows are bits, columns queries.
    x2 = 11.4
    panel_title(
        ax,
        x2 - 0.9,
        top + 2.0,
        "2. Inverted index (built once)",
        "one posting list per bit",
    )
    counts = {b: sum(b in bits for bits in QUERIES.values()) for b in BITS}
    for c, q in enumerate(queries):
        text(ax, x2 + c * CELL + CELL / 2, top + 0.4, q, fontsize=FONT - 0.5)
    for r, b in enumerate(BITS):
        y = top - r * CELL - CELL / 2
        text(ax, x2 - 0.35, y, b)
        text(ax, x2 + 8 * CELL + 0.45, y, f"{counts[b]}", color=MUTED)
    text(ax, x2 + 8 * CELL + 0.45, top + 0.4, "#", color=MUTED)
    grid(ax, x2, top, BITS, queries, {(b, q) for q, b in has}, BLUE)
    text(
        ax,
        x2 + 4 * CELL,
        top - 8 * CELL - 0.45,
        "bits ranked by how many queries set them",
        fontsize=FONT - 1,
        color=MUTED,
    )
    arrow(
        ax, (x1 + 8 * CELL + 0.4, top - 4 * CELL), (x2 - 0.8, top - 4 * CELL), "invert"
    )

    # 3. One reactant against the index.
    x3 = 22.6
    panel_title(
        ax,
        x3 - 1.9,
        top + 2.0,
        "3. Screen one reactant",
        f"OR the postings of the top {POSTINGS} bits it lacks",
    )
    for c, q in enumerate(queries):
        text(ax, x3 + c * CELL + CELL / 2, top + 0.4, q, fontsize=FONT - 0.5)
    xr = x3 - 1.75  # the reactant's bit, beside each posting list
    text(ax, xr + CELL / 2, top + 0.4, "reactant", fontsize=FONT - 1.5, color=MUTED)
    ordinals = ["1st", "2nd", "3rd"]
    for r, b in enumerate(BITS):
        y = top - r * CELL - CELL / 2
        text(ax, x3 - 0.35, y, b, color=INK if b in used else MUTED)
        reactant_cell(ax, xr, top - (r + 1) * CELL, b in REACTANT)
        note_x = x3 + 8 * CELL + 0.3
        if b in used:
            label = f"{ordinals[used.index(b)]} lacking bit: OR"
            text(ax, note_x, y, label, ha="left", color=ORANGE, weight="bold")
        elif b in missing:
            label = f"lacking, not in top {POSTINGS}"
            text(ax, note_x, y, label, ha="left", fontsize=FONT - 1, color=MUTED)
        else:
            text(
                ax, note_x, y, "has it: skip", ha="left", fontsize=FONT - 1, color=MUTED
            )
    arrow(
        ax,
        (x1 + 8 * CELL + 0.3, ry + CELL / 2),
        (xr + CELL / 2, top - 8 * CELL - 0.25),
        rad=0.15,
    )
    text(
        ax,
        (x1 + 8 * CELL + xr) / 2,
        ry - 0.55,
        "the reactant's bits, in the index's rank order",
        fontsize=FONT - 1,
        color=MUTED,
    )
    faded = [b for b in BITS if b not in used]
    grid(ax, x3, top, BITS, queries, {(b, q) for q, b in has}, ORANGE, faded)

    # Result rows under the reactant panel.
    rows = [
        ("ruled out", {q: ORANGE for q in ruled_out}),
        ("left", {q: BLUE for q in left}),
        ("full fp check", {q: BLUE for q in passed}),
        ("RDKit match", {q: GREEN for q in MATCHES & set(passed)}),
    ]
    y0 = top - 8 * CELL - 0.5
    for i, (label, colors) in enumerate(rows):
        y = y0 - i * (CELL + 0.25)
        text(ax, x3 - 0.25, y - CELL / 2, label, ha="right", fontsize=FONT - 0.5)
        for c, q in enumerate(queries):
            cell(ax, x3 + c * CELL, y - CELL, colors.get(q, GRID))
    note_y = y0 - 2 * (CELL + 0.25) - CELL / 2
    text(
        ax,
        x3 + 8 * CELL + 0.3,
        note_y,
        "Q3 sets G,\nwhich the\nreactant lacks",
        ha="left",
        fontsize=FONT - 1.5,
        color=MUTED,
    )

    # Benchmark 05's real numbers: one box per stage.
    stages = [
        ("50,000,000", "query-reactant pairs", MUTED),
        ("943,741", "left after 128 checks\n(98.1% ruled out by ORs)", ORANGE),
        ("350,393", "left after the full\nfingerprint check", BLUE),
        ("27,723", "RDKit exact matches", GREEN),
    ]
    by = 0.9
    text(
        ax,
        0.6,
        by + 2.25,
        "Benchmark 05, NumPy posting lists + RDKit Pattern fp: "
        "50,000 queries x 1,000 reactants, 2,048 bits, 128 checks per reactant",
        ha="left",
        fontsize=FONT + 0.5,
        weight="bold",
    )
    width, gap = 6.2, 1.1
    for i, (number, label, color) in enumerate(stages):
        x = 0.6 + i * (width + gap)
        ax.add_patch(
            Rectangle((x, by), width, 1.7, facecolor="white", edgecolor=color, lw=1.6)
        )
        text(
            ax,
            x + width / 2,
            by + 1.2,
            number,
            fontsize=FONT + 3,
            weight="bold",
            color=color,
        )
        text(ax, x + width / 2, by + 0.5, label, fontsize=FONT - 0.5)
        if i:
            arrow(ax, (x - gap + 0.1, by + 0.85), (x - 0.1, by + 0.85))


def main() -> None:
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(10.5, 6.6))
        fig.subplots_adjust(
            0, 0, 1, 1
        )  # the drawing fills the figure; text scales with it
        ax.set_xlim(0, 32.6)
        ax.set_ylim(0, 19.0)
        ax.set_aspect("equal")
        ax.axis("off")
        draw(ax)
        OUT.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            path = OUT / f"inverted_index.{suffix}"
            # Cropped to the drawing, on an opaque white background for slides.
            fig.savefig(
                path, dpi=DPI, bbox_inches="tight", pad_inches=0.15, facecolor="white"
            )
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

"""Infographic: one 3-atom query, RDKit's Pattern fingerprint against the Rust one.

Writes `benchmarks/plots/pattern_vs_rust_query.{png,svg}` for presentations. The query
is a real reaction-template fragment from `data/substructure_sample.parquet`, an
aldehyde on an aliphatic carbon:

    C-[C&H1&D2&+0]=[O&H0&D1&+0]

Every number is computed here, over the 1,000 reactants of `data/reactant_sample.csv`:
how many pass RDKit's 2048-bit `PatternFingerprint` screen, how many pass the Rust
screen (`substructure_rs`, paths4+branches at 2048 bits, benchmark 11's
configuration), and how many really match (RDKit `HasSubstructMatch`). The reactants
only the Pattern screen passes are counted by which stated label (H count, degree,
aromaticity, charge), added back alone to the C-C=O skeleton, would rule each out. The
title gives the whole benchmark's counts, read from the newest `run_all.py` run.

Styled like `plot_results.py`, whose colors it reuses; the PNG is 600 dpi for slides.

Usage:
    uv run benchmarks/plot_pattern_vs_rust_query.py
"""

import io
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse, FancyBboxPatch
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import PatternFingerprint

import substructure_rs

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.plot_results import INK, MUTED, STEPS, newest_run

# Secondary text: darker than MUTED so it stays legible projected on a slide.
SUBTLE = "#4f4e49"
# A sans-serif with solid strokes; the science style's thin serif washes out on a projector.
FONTS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
}

RDLogger.DisableLog("rdApp.*")
BLUE, ORANGE, GREEN, YELLOW = (color for _, _, color in STEPS)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "plots"
DPI = 600

QUERY = "C-[C&H1&D2&+0]=[O&H0&D1&+0]"
FP_BITS = 2048
RUST_KIND = "paths4+branches"

# What the C-C=O of a reactant the Rust screen rejects really is; first hit wins.
# The query's stated labels, each added back alone to the C-C=O skeleton that is all
# the Pattern fingerprint sees. A label "drops" a reactant when the skeleton with just
# that label no longer matches it.
SKELETON_SMARTS = "[#6]-[#6]=[#8]"
LABELS = {
    "H count": "[#6]-[#6;H1]=[#8;H0]",
    "degree": "[#6]-[#6;D2]=[#8;D1]",
    "aromaticity": "[#6;A]-[#6;A]=[#8;A]",
    "charge": "[#6]-[#6;+0]=[#8;+0]",
}
COLLISION = "no C–C=O at all\n(Pattern bit collision)"
# One example per kind, with the labels that drop it, checked when drawn.
EXAMPLES = [
    ("CCOC(=O)C(F)(F)Oc1cccc(F)c1", {"H count", "degree"}),
    ("Cc1ccc(C(=O)O)cc1F", {"H count", "degree", "aromaticity"}),
    ("COc1ccc2ccccc2c1C=O", {"aromaticity"}),
    ("N[C@H]1C[C@H](n2c(=O)n(C3CC3)c3nc(Br)cnc32)C1", None),
]
SKELETON = Chem.MolFromSmarts(SKELETON_SMARTS)


def screen() -> pd.DataFrame:
    """Per reactant: passes the Pattern screen, the Rust screen, really matches."""
    smiles = list(pd.read_csv(ROOT / "data/reactant_sample.csv")["smiles"])
    query = Chem.MolFromSmarts(QUERY)
    query_fp = PatternFingerprint(Chem.MolFromSmarts(QUERY), FP_BITS)
    with tempfile.TemporaryDirectory() as tmp:
        one = Path(tmp) / "query.csv"
        pd.DataFrame({"substructure": [QUERY]}).to_csv(one, index=False)
        queries = substructure_rs.read_queries(one)
        index = substructure_rs.Index.build(queries, RUST_KIND, FP_BITS, "rare-root")
        rows = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            pattern = DataStructs.AllProbeBitsMatch(
                query_fp, PatternFingerprint(mol, FP_BITS)
            )
            # The Rust screen per reactant: an index of this one query, one reactant.
            path = Path(tmp) / "reactant.csv"
            pd.DataFrame({"smiles": [s]}).to_csv(path, index=False)
            reactant = substructure_rs.read_reactants(path)
            _, _, candidates = index.match_many(reactant, 128)
            rows.append((s, pattern, candidates > 0, mol.HasSubstructMatch(query)))
    df = pd.DataFrame(rows, columns=["smiles", "pattern", "rust", "match"])
    assert not (df["match"] & ~(df["pattern"] & df["rust"])).any(), (
        "a screen lost a match"
    )
    return df


def full_dataset() -> dict[str, int]:
    """The whole benchmark, from the newest `run_all.py` run: pairs sent to the exact
    match by the Pattern screen (05, NumPy) and by the Rust screen at this
    configuration (09's sweep), and the matches both find."""
    summary = pd.read_csv(newest_run() / "summary.csv")
    pattern = summary[
        (summary.benchmark == "05_postings") & (summary.variant == "numpy pattern")
    ]
    rust = summary[summary.variant == f"{RUST_KIND} {FP_BITS} p128 rare-root"]
    return {
        "queries": int(pattern["queries"].iloc[0]),
        "reactants": int(pattern["reactants"].iloc[0]),
        "pattern": int(pattern["exact matches run"].iloc[0]),
        "rust": int(rust["exact matches run"].iloc[0]),
        "matches": int(pattern["matches"].iloc[0]),
    }


def dropped_by(smiles: str) -> set[str] | None:
    """The stated labels that alone rule the reactant out; None when it has no C-C=O,
    so it passed the Pattern screen only through colliding bits."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol.HasSubstructMatch(SKELETON):
        return None
    return {
        name
        for name, smarts in LABELS.items()
        if not mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))
    }


def depiction(smiles: str, highlight: Chem.Mol, color: str) -> Image.Image:
    """A 2D drawing with the first match of `highlight` filled in `color`."""
    mol = Chem.MolFromSmiles(smiles)
    atoms = list(mol.GetSubstructMatch(highlight))
    bonds = [
        mol.GetBondBetweenAtoms(a, b).GetIdx()
        for a in atoms
        for b in atoms
        if a < b and mol.GetBondBetweenAtoms(a, b)
    ]
    rgb = tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)) + (0.45,)
    # Rendered well above its size on the slide (about 1,200 px wide at 600 dpi),
    # with bond widths and fonts scaled up to match, so it stays sharp full-screen.
    drawer = rdMolDraw2D.MolDraw2DCairo(2400, 1600)
    options = drawer.drawOptions()
    options.bondLineWidth = 12
    options.minFontSize = 100
    options.maxFontSize = 140  # RDKit caps fonts at 40 px by default
    options.scaleHighlightBondWidth = True
    options.padding = 0.05
    options.clearBackground = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms,
        highlightBonds=bonds,
        highlightAtomColors={a: rgb for a in atoms},
        highlightBondColors={b: rgb for b in bonds},
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def aspect_of(ax) -> float:
    """Height over width of `ax` on the page, to draw round shapes in axes units."""
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    return (bbox.height * fig_h) / (bbox.width * fig_w)


def text(ax, x, y, s, size=9, **kw):
    kw = {"color": INK, "ha": "center", "va": "center", **kw}
    ax.text(x, y, s, fontsize=size, transform=ax.transAxes, **kw)


def card(ax, color):
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0,rounding_size=0.04",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=color,
            lw=1.8,
        )
    )


def query_diagram(ax, y, facts, shown):
    """The three query atoms as circles, each with its stated facts below it.

    `shown[i]` is how many of atom i's facts the fingerprint uses; the rest are
    drawn crossed out.
    """
    xs = (0.22, 0.5, 0.78)
    elements = ("C", "C", "O")
    r = 0.055 * aspect_of(ax)  # the circle radius across, in axes units
    ax.plot([xs[0] + r, xs[1] - r], [y, y], color=INK, lw=2, transform=ax.transAxes)
    for dy in (-0.018, 0.018):
        ax.plot(
            [xs[1] + r, xs[2] - r],
            [y + dy, y + dy],
            color=INK,
            lw=2,
            transform=ax.transAxes,
        )
    for x, element, atom_facts, n in zip(xs, elements, facts, shown):
        ax.add_patch(
            Ellipse(
                (x, y),
                2 * r,
                0.11,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=INK,
                lw=1.6,
            )
        )
        text(ax, x, y, element, size=17, weight="bold")
        for k, fact in enumerate(atom_facts):
            used = k < n
            fy = y - 0.14 - k * 0.075
            t = ax.text(
                x,
                fy,
                fact,
                fontsize=10,
                ha="center",
                va="center",
                color=INK if used else "#8f8e88",
                transform=ax.transAxes,
            )
            if not used:
                # Cross the fact out across its rendered width.
                box = t.get_window_extent(ax.figure.canvas.get_renderer())
                x0, x1 = box.transformed(ax.transAxes.inverted()).intervalx
                ax.plot(
                    [x0, x1], [fy, fy], color=ORANGE, lw=1.4, transform=ax.transAxes
                )


FACTS = (
    ("aliphatic",),
    ("aliphatic", "H1: 1 H", "D2: 2 neighbours", "charge 0"),
    ("H0: no H", "D1: 1 neighbour", "charge 0"),
)


def draw(fig, df: pd.DataFrame, full: dict[str, int]) -> None:
    total = len(df)
    n_pattern, n_rust, n_match = (
        int(df[c].sum()) for c in ("pattern", "rust", "match")
    )
    only_pattern = df[df["pattern"] & ~df["rust"]]
    dropped = [dropped_by(smiles) for smiles in only_pattern["smiles"]]

    fig.text(
        0.02,
        0.965,
        f"RDKit's Pattern fingerprint sends {full['pattern'] / full['rust']:.1f}x more "
        "pairs to the exact match than the Rust one",
        fontsize=20,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.02,
        0.915,
        f"Whole benchmark, {full['queries']:,} queries x {full['reactants']:,} reactants, "
        f"{FP_BITS:,}-bit fingerprints: Pattern {full['pattern']:,} pairs, "
        f"Rust {full['rust']:,}; the same {full['matches']:,} matches",
        fontsize=13,
        color=SUBTLE,
        va="top",
    )

    # Row 1: the query, and what each fingerprint reads from it.
    a = fig.add_axes((0.02, 0.50, 0.30, 0.37))
    card(a, MUTED)
    text(
        a,
        0.5,
        0.91,
        "The query: an aldehyde on an aliphatic C",
        size=14,
        weight="bold",
    )
    text(a, 0.5, 0.81, QUERY, size=12, family="monospace", color=BLUE)
    query_diagram(a, 0.62, FACTS, (1, 4, 3))
    text(
        a,
        0.5,
        0.1,
        "every atom states its H count,\ndegree and charge",
        size=10.5,
        color=SUBTLE,
    )

    b = fig.add_axes((0.345, 0.50, 0.30, 0.37))
    card(b, ORANGE)
    text(b, 0.5, 0.91, "RDKit Pattern fp reads", size=14, weight="bold", color=ORANGE)
    text(b, 0.5, 0.81, "elements + bond orders only:  C – C = O", size=12)
    query_diagram(b, 0.62, FACTS, (0, 0, 0))
    text(
        b,
        0.5,
        0.1,
        "H count, degree, charge and aromaticity\nof the atoms are ignored",
        size=10.5,
        color=ORANGE,
    )

    c = fig.add_axes((0.67, 0.50, 0.31, 0.37))
    card(c, BLUE)
    text(c, 0.5, 0.91, "Rust fp reads", size=14, weight="bold", color=BLUE)
    text(c, 0.5, 0.81, "elements, bonds and every stated label:  C – CH = O", size=11)
    query_diagram(c, 0.62, FACTS, (1, 4, 3))
    text(
        c,
        0.5,
        0.1,
        "each atom's labels, and C=O as one\nfully labelled bond",
        size=10.5,
        color=BLUE,
    )

    # Row 2, left: how many reactants each step lets through.
    f = fig.add_axes((0.13, 0.04, 0.22, 0.37))
    stages = [
        ("reactants", total, MUTED),
        ("pass Pattern screen", n_pattern, ORANGE),
        ("pass Rust screen", n_rust, BLUE),
        ("really match", n_match, GREEN),
    ]
    for i, (label, n, color) in enumerate(stages):
        f.barh(i, n, color=color, height=0.62)
        f.text(
            n + total * 0.02,
            i,
            f"{n:,}",
            va="center",
            fontsize=14,
            weight="bold",
            color=color,
        )
    f.set_yticks(range(len(stages)), [s[0] for s in stages], fontsize=12)
    f.invert_yaxis()
    f.set_xlim(0, total * 1.18)
    f.set_xticks([])
    for side in ("top", "right", "bottom"):
        f.spines[side].set_visible(False)
    f.tick_params(axis="both", which="both", length=0, right=False, top=False)
    f.set_title(
        "Sent to the exact match", fontsize=14, weight="bold", loc="left", color=INK
    )

    # Row 2, middle: which stated label would drop each Pattern-only reactant.
    g = fig.add_axes((0.50, 0.04, 0.18, 0.37))
    names = [*LABELS, COLLISION]
    counts = [sum(r is not None and n in r for r in dropped) for n in LABELS]
    counts.append(sum(r is None for r in dropped))
    colors = [ORANGE] * len(LABELS) + [MUTED]
    g.barh(range(len(names)), counts, color=colors, height=0.62)
    for i, n in enumerate(counts):
        g.text(n + 6, i, f"{n}", va="center", fontsize=12, color=INK)
    g.set_yticks(range(len(names)), names, fontsize=12)
    g.invert_yaxis()
    g.set_xlim(0, max(counts) * 1.25)
    g.set_xticks([])
    for side in ("top", "right", "bottom"):
        g.spines[side].set_visible(False)
    g.tick_params(axis="both", which="both", length=0, right=False, top=False)
    g.set_title(
        f"The {len(only_pattern)} only the Pattern screen passes:\n"
        "the stated label that would drop each",
        fontsize=14,
        weight="bold",
        loc="right",
        color=INK,
    )
    g.text(
        1.0,
        -0.06,
        "a reactant can count under several labels",
        transform=g.transAxes,
        ha="right",
        fontsize=10.5,
        color=SUBTLE,
    )

    # Row 2, right: one example per kind, its C-C=O highlighted.
    cells = [(0.695, 0.215), (0.845, 0.215), (0.695, 0.02), (0.845, 0.02)]
    for (smiles, labels), (x, y) in zip(EXAMPLES, cells):
        assert dropped_by(smiles) == labels, smiles
        m = fig.add_axes((x, y, 0.14, 0.17))
        m.imshow(depiction(smiles, SKELETON, ORANGE))
        m.set_axis_off()
        caption = (
            "no C–C=O: collision"
            if labels is None
            else ", ".join(n for n in LABELS if n in labels)
        )
        m.set_title(caption, fontsize=11.5, color=INK, pad=2)
    fig.text(
        0.84,
        0.445,
        "Pattern: pass.   Rust: reject.\nWhich label drops it:",
        fontsize=12.5,
        weight="bold",
        color=INK,
        ha="center",
        va="center",
    )


def main() -> None:
    df = screen()
    with plt.style.context(["science", "no-latex"]), plt.rc_context(FONTS):
        fig = plt.figure(figsize=(13.33, 7.5))
        draw(fig, df, full_dataset())
        OUT.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            path = OUT / f"pattern_vs_rust_query.{suffix}"
            fig.savefig(path, dpi=DPI, facecolor="white")
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

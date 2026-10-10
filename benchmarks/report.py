"""One output format for every benchmark.

Each benchmark fills a `Report` and prints it with `print_report`, so every run shows the
same fields in the same order:

    == 05a_numpy_postings ==
    settings          postings 128, runs 3
    pairs             50,000 queries x 1,000 reactants = 50,000,000
    query prep            1.234 s  read + parse SMARTS
    index prep            4.797 s  Pattern fingerprints + postings
    reactant prep         0.123 s  read + parse SMILES
    matching              1.233 s  median of 3: 1.228 1.235 1.233
    end to end            7.387 s  prep + median matching
    matches              27,723    digest 72f6936d8ee5f05b
    -- benchmark specific
    dropped by postings  49,056,259  pairs ruled out by the absent-bit postings
    ...

A prep step a benchmark does not have, or does elsewhere, prints as `-` with a note.
Matching is the timed search over all pairs; the other steps are outside its timer.
With `BENCH_RESULTS=<path>` in the environment, every printed report is also appended
to that file as one JSON line; `benchmarks/run_all.py` uses this to aggregate runs.

The digest is SHA-256 over the sorted (reactant index, query index) match pairs as
little-endian u32s, the encoding `substructure_rs` uses, so every benchmark that finds
the same matches prints the same digest.

Benchmarks that compare variants also print `print_table`, one row per variant with the
same columns.
"""

import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

Extra = tuple[str, object, str]  # (label, value, note)

LABEL = 20
VALUE = 12


@dataclass
class Report:
    benchmark: str
    queries: int
    reactants: int
    match_seconds: list[float]
    matches: int
    digest: str | None = None
    variant: str | None = None
    query_prep_seconds: float | None = None
    index_prep_seconds: float | None = None
    reactant_prep_seconds: float | None = None
    settings: dict[str, object] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    extras: list[Extra] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.benchmark}: {self.variant}" if self.variant else self.benchmark

    @property
    def match_median(self) -> float:
        return statistics.median(self.match_seconds)

    @property
    def end_to_end(self) -> float:
        prep = (
            self.query_prep_seconds,
            self.index_prep_seconds,
            self.reactant_prep_seconds,
        )
        return sum(s for s in prep if s is not None) + self.match_median

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Report":
        data = dict(data)
        data["extras"] = [tuple(extra) for extra in data.get("extras", [])]
        return cls(**data)


def digest_pairs(pairs: Iterable[tuple[int, int]]) -> str:
    """SHA-256 of the sorted (reactant index, query index) pairs."""
    hasher = hashlib.sha256()
    for reactant_idx, query_idx in sorted(pairs):
        hasher.update(struct.pack("<II", reactant_idx, query_idx))
    return hasher.hexdigest()


def found_pairs(
    query_mols: Sequence, found: Iterable[Iterable]
) -> list[tuple[int, int]]:
    """(reactant index, query index) pairs from per-reactant lists of query molecules."""
    ids = {id(mol): idx for idx, mol in enumerate(query_mols)}
    return [
        (reactant_idx, ids[id(mol)])
        for reactant_idx, mols in enumerate(found)
        for mol in mols
    ]


def screen_extras(pairs: int, posted: int | None, candidates: int, matches: int):
    """The standard breakdown of where the pairs went, as report extras.

    `posted` is the pairs left after the posting filter, or None for a benchmark
    without one; `candidates` is the pairs sent to the exact match.
    """
    extras = []
    if posted is not None:
        extras.append(
            ("dropped by postings", pairs - posted, "pairs ruled out by the postings")
        )
        extras.append(
            (
                "dropped by fp check",
                posted - candidates,
                "pairs ruled out by the full fingerprint check",
            )
        )
    else:
        extras.append(
            ("dropped by fp", pairs - candidates, "pairs ruled out by the fingerprint")
        )
    extras.append(
        (
            "exact matches run",
            candidates,
            f"{100 * candidates / pairs:.2f}% of the pairs",
        )
    )
    extras.append(
        ("exact rejections", candidates - matches, "candidates that did not match")
    )
    return extras


def _value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f} s"
    return str(value)


def _line(label: str, value, note: str = "") -> str:
    return f"{label:<{LABEL}}{_value(value):>{VALUE}}  {note}".rstrip()


def _round(value, places: int = 4):
    """Floats rounded to `places`, through dicts, lists and tuples."""
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: _round(v, places) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, places) for v in value]
    return value


def record(r: Report) -> None:
    """Append `r` to the `BENCH_RESULTS` JSON-lines file, when that is set."""
    path = os.environ.get("BENCH_RESULTS")
    if path:
        row = {**r.to_dict(), "match_median_seconds": r.match_median}
        row["end_to_end_seconds"] = r.end_to_end
        with open(path, "a") as handle:
            row = _round(row)
            handle.write(json.dumps(row) + "\n")


def print_report(r: Report, record_it: bool = True) -> None:
    if record_it:
        record(r)
    pairs = r.queries * r.reactants
    runs = " ".join(f"{s:.4f}" for s in r.match_seconds)
    settings = ", ".join(f"{k} {v}" for k, v in r.settings.items())
    print(f"== {r.title} ==")
    if settings:
        print(f"{'settings':<{LABEL}}{settings}")
    print(
        f"{'pairs':<{LABEL}}{r.queries:,} queries x {r.reactants:,} reactants "
        f"= {pairs:,}"
    )
    print(_line("query prep", r.query_prep_seconds, r.notes.get("query", "")))
    print(_line("index prep", r.index_prep_seconds, r.notes.get("index", "")))
    print(_line("reactant prep", r.reactant_prep_seconds, r.notes.get("reactant", "")))
    match_note = (
        f"median of {len(r.match_seconds)}: {runs}"
        if len(r.match_seconds) > 1
        else "one run"
    )
    if r.notes.get("matching"):
        match_note += f"; {r.notes['matching']}"
    print(_line("matching", r.match_median, match_note))
    print(_line("end to end", r.end_to_end, "prep + median matching"))
    digest = f"digest {r.digest[:16]}" if r.digest else ""
    print(_line("matches", r.matches, digest))
    if r.extras:
        print("-- benchmark specific")
        for label, value, note in r.extras:
            print(_line(label, value, note))
    print()


STANDARD_COLUMNS = [
    ("query prep", lambda r: r.query_prep_seconds),
    ("index prep", lambda r: r.index_prep_seconds),
    ("reactant prep", lambda r: r.reactant_prep_seconds),
    ("matching", lambda r: r.match_median),
    ("end to end", lambda r: r.end_to_end),
    ("matches", lambda r: r.matches),
]


class Table:
    """Rows of reports, the standard columns first, then the named extras.

    Print the header with `header()` and each report with `row()` as it arrives, or a
    finished list at once with `print_table`.
    """

    def __init__(self, extra_labels: Sequence[str] = (), name_width: int = 24):
        self.extra_labels = list(extra_labels)
        self.name_width = name_width
        self.columns = [label for label, _ in STANDARD_COLUMNS] + self.extra_labels
        self.widths = [max(len(label), VALUE) for label in self.columns]

    def header(self) -> None:
        cells = "".join(f"{c:>{w + 2}}" for c, w in zip(self.columns, self.widths))
        line = f"{'variant':<{self.name_width}}{cells}  digest"
        print(line)
        print("-" * len(line))

    def row(self, r: Report) -> None:
        extras = {label: value for label, value, _ in r.extras}
        values = [get(r) for _, get in STANDARD_COLUMNS]
        values += [extras.get(label) for label in self.extra_labels]
        cells = "".join(f"{_value(v):>{w + 2}}" for v, w in zip(values, self.widths))
        name = r.variant or r.benchmark
        print(f"{name:<{self.name_width}}{cells}  {(r.digest or '-')[:16]}", flush=True)


def print_table(reports: Sequence[Report], extra_labels: Sequence[str] = ()) -> None:
    table = Table(
        extra_labels, max(24, *(len(r.variant or r.benchmark) + 2 for r in reports))
    )
    table.header()
    for r in reports:
        table.row(r)
    if len({(r.matches, r.digest) for r in reports}) == 1:
        print(f"all {len(reports)} agree: {reports[0].matches:,} matches")
    else:
        print("the variants DISAGREE on the match set")
    print()


def load_queries(path: Path) -> tuple[list, float]:
    """Query molecules parsed from the `substructure` SMARTS column, and the seconds."""
    import datamol as dm
    import pandas as pd

    start = time.perf_counter()
    smarts = pd.read_parquet(path)["substructure"]
    mols = [dm.convert.from_smarts(value) for value in smarts]
    if any(mol is None for mol in mols):
        raise ValueError("Failed to parse substructure SMARTS")
    return mols, time.perf_counter() - start


def load_reactants(path: Path) -> tuple[list, float]:
    """Reactant molecules parsed from the `smiles` column, and the seconds."""
    import datamol as dm
    import pandas as pd

    start = time.perf_counter()
    smiles = pd.read_csv(path)["smiles"]
    mols = [dm.to_mol(value) for value in smiles]
    if any(mol is None for mol in mols):
        raise ValueError("Failed to parse reactant SMILES")
    return mols, time.perf_counter() - start


def run_variants(
    script: str, option: str, variants: Sequence[str], args: Sequence[str]
):
    """Run `script --worker --<option> <variant> <args>` once per variant, each in a
    fresh process, and collect the `Report` each prints as JSON."""
    reports = []
    for variant in variants:
        print(f"Running {variant}...", flush=True)
        command = [sys.executable, script, "--worker", f"--{option}", variant, *args]
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        reports.append(Report.from_dict(json.loads(completed.stdout)))
    print()
    return reports


def print_worker(report: Report) -> None:
    """A worker's half of `run_variants`: the report as one JSON line."""
    print(json.dumps(report.to_dict()))


def mib(nbytes: int) -> str:
    return f"{nbytes / 2**20:.1f} MiB"


def from_substructure_rs(
    result: dict, benchmark: str, variant: str | None = None
) -> Report:
    """A `Report` from the dict `substructure_rs.run()` returns."""
    pairs = result["queries"] * result["reactants"]
    fp, postings, check, exact = result["phase_seconds"]
    extras = [
        *screen_extras(
            pairs, result["posted"], result["candidates"], result["matches"]
        ),
        ("reactant fps", fp, "CPU time summed over threads"),
        ("posting filter", postings, "CPU time summed over threads"),
        ("fp check", check, "CPU time summed over threads"),
        ("exact match", exact, "CPU time summed over threads"),
        ("index size", mib(result["index_bytes"]), "postings + query fingerprints"),
        (
            "bits set",
            f"{100 * result['query_density']:.1f}% / {100 * result['target_density']:.1f}%",
            "mean share of fingerprint bits set, queries / reactants",
        ),
    ]
    comparison = result.get("comparison")
    if comparison:
        extras.append(
            (
                "vs reference",
                f"{comparison['missing']} / {comparison['extra']}",
                f"missing / extra pairs against {comparison['reference_pairs']:,}",
            )
        )
    return Report(
        benchmark=benchmark,
        variant=variant,
        queries=result["queries"],
        reactants=result["reactants"],
        match_seconds=list(result["match_seconds"]),
        matches=result["matches"],
        digest=result["digest"],
        query_prep_seconds=result["query_prep_seconds"],
        index_prep_seconds=result["build_seconds"],
        reactant_prep_seconds=result["reactant_prep_seconds"],
        settings={
            "fp": f"{result['fp_kind']} {result['fp_bits']} bits",
            "postings": result["postings"],
            "order": result["order"],
            "threads": result["threads"],
            "runs": len(result["match_seconds"]),
        },
        notes={
            "query": "read + parse SMARTS (Rust)",
            "index": "query fingerprints + postings (Rust)",
            "reactant": "read + parse + sanitize SMILES (Rust)",
            "matching": "reactant fingerprints + postings + fp check + exact match",
        },
        extras=extras,
    )

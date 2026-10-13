"""Benchmark 10: substructure search with the chematic crates.

Author: Marcin Kowiel + Claude

chematic (https://github.com/kent-tokyo/chematic) is a pure-Rust
cheminformatics toolkit with its own SMILES parser, SMARTS parser and VF2
matcher. Calls `chematic_bench.run()` in one of two modes:

    naive     chematic's `has_match_with_config` on every pair, like
              benchmark 01 with RDKit
    screened  substructure_rs's fingerprint screen (benchmark 08/09) picks
              the candidates and chematic checks only those, the pairs
              benchmark 08's own matcher sees

Timed like the other benchmarks: the screen's index is built once (index
prep) and each matching run screens every reactant and runs the exact
matches. chematic's exact match alone is reported as an extra.

chematic cannot screen this workload itself: its Pattern fingerprint covers
concrete molecules only, not SMARTS queries.

uv builds the `chematic_bench` extension from `chematic_bench/`. The same
code also runs as a command, with the same options:

    uv run chematic_bench --help
"""

import argparse
import sys
from pathlib import Path

import chematic_bench

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.report import Report, print_report, screen_extras


def to_report(r: dict, args: argparse.Namespace) -> Report:
    total = r["queries"] * r["reactants"]
    screened = r["mode"] == "screened"
    extras = [
        *screen_extras(total, None, r["pairs"], r["matches"]),
        ("exact match", r["exact_median_seconds"], "chematic alone, part of matching"),
        (
            "per exact match",
            f"{1e6 * r['exact_median_seconds'] / r['pairs']:.3f} us",
            f"wall time per pair, {r['threads']} thread(s)",
        ),
    ]
    if "comparison" in r:
        c = r["comparison"]
        extras.append(
            (
                "vs reference",
                f"{c['missing']} / {c['extra']}",
                f"missing / extra pairs against {c['reference_pairs']:,}",
            )
        )
    settings = {"mode": r["mode"], "threads": r["threads"], "runs": args.runs}
    if screened:
        settings["screen"] = (
            f"{args.fp_kind} {args.fp_bits} bits, {args.postings} postings"
        )
    return Report(
        benchmark="10_chematic",
        variant=r["mode"],
        queries=r["queries"],
        reactants=r["reactants"],
        match_seconds=list(r["match_seconds"]),
        matches=r["matches"],
        digest=r["digest"],
        query_prep_seconds=r["query_prep_seconds"],
        index_prep_seconds=r["index_prep_seconds"],
        reactant_prep_seconds=r["reactant_prep_seconds"],
        settings=settings,
        notes={
            "query": "read + parse SMARTS (chematic)",
            "index": (
                "substructure_rs parses the SMARTS and builds its screen index"
                if screened
                else "none: every pair goes to chematic"
            ),
            "reactant": (
                "read + parse SMILES (chematic, and substructure_rs for the screen)"
                if screened
                else "read + parse SMILES (chematic)"
            ),
            "matching": (
                "substructure_rs screen + chematic exact match"
                if screened
                else "chematic exact match on every pair"
            ),
        },
        extras=extras,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["naive", "screened"], default="screened")
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0, help="0 = all cores")
    parser.add_argument("--fp-kind", default="paths4+branches")
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--postings", type=int, default=128)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    try:
        result = chematic_bench.run(
            mode=args.mode,
            queries_file=args.queries_file,
            reactants_file=args.reactants_file,
            runs=args.runs,
            threads=args.threads,
            fp_kind=args.fp_kind,
            fp_bits=args.fp_bits,
            postings=args.postings,
            reference=args.reference,
        )
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")
    print_report(to_report(result, args))


if __name__ == "__main__":
    main()

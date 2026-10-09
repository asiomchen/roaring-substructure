"""Benchmark 10: substructure search with the chematic crates.

Author: Marcin Kowiel + Claude

chematic (https://github.com/kent-tokyo/chematic) is a pure-Rust
cheminformatics toolkit with its own SMILES parser, SMARTS parser and VF2
matcher. Calls `chematic_bench.run()` in one of two modes:

    naive     chematic's `has_match_with_config` on every pair, like
              benchmark 01 with RDKit
    screened  substructure_rs's fingerprint screen (benchmark 08/09) picks
              the candidates outside the timer; chematic checks only those,
              so the timer measures chematic's matcher on the pairs that
              benchmark 08's own matcher sees

chematic cannot screen this workload itself: its Pattern fingerprint covers
concrete molecules only, not SMARTS queries.

uv builds the `chematic_bench` extension from `chematic_bench/`. The same
code also runs as a command, with the same options:

    uv run chematic_bench --help
"""

import argparse
from pathlib import Path

import chematic_bench


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
        chematic_bench.run(
            mode=args.mode,
            queries_file=args.queries_file,
            reactants_file=args.reactants_file,
            runs=args.runs,
            threads=args.threads,
            fp_kind=args.fp_kind,
            fp_bits=args.fp_bits,
            postings=args.postings,
            reference=args.reference,
            print_report=True,
        )
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()

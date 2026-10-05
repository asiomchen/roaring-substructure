"""Benchmark 8: the whole pipeline in Rust, without RDKit.

Author: Marcin Kowiel + Claude

Calls `substructure_rs.run()`, which parses the SMARTS queries and reactant
SMILES itself, sanitizes the reactants the way RDKit does, screens with its own
posting-list fingerprint, and runs exact matching in parallel. Its digest uses
the same encoding as benchmarks 05a-07.

uv builds the `substructure_rs` extension from `substructure_rs/` (like
`rust_index`) and rebuilds it when the Rust sources change. The same code also
runs as a command, with the same options:

    uv run substructure_rs --help

To check the result against RDKit, create a reference with
`substructure_rs/tools/rdkit_reference.py` and pass `--reference <dir>/pairs.bin`.
"""

import argparse
from pathlib import Path

import substructure_rs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument("--postings", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0, help="0 = all cores")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--dump-atoms", type=Path)
    args = parser.parse_args()

    try:
        substructure_rs.run(
            queries_file=args.queries_file,
            reactants_file=args.reactants_file,
            postings=args.postings,
            runs=args.runs,
            threads=args.threads,
            reference=args.reference,
            dump_atoms=args.dump_atoms,
            print_report=True,
        )
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()

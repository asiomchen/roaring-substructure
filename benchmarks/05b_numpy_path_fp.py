"""Benchmark 5a's NumPy postings index, screened with benchmark 08's path fingerprint.

Same corpus, same packed NumPy index, 128 absent-bit postings, full fingerprint subset
check and RDKit's exact `HasSubstructMatch`. The only change from benchmark 5a is the
screening fingerprint:

- `pattern`: RDKit's 2048-bit `PatternFingerprint` (benchmark 5a's NumPy backend);
- `paths`: benchmark 08's `paths4` fingerprint at 4096 bits, computed from the RDKit
  molecules by `benchmarks/rdkit_path_fp.py`. It hashes the H count, degree and charge
  the template SMARTS state, which the Pattern fingerprint ignores.

Both must return the same matches; the question is how many fewer pairs reach the exact
match, and whether that pays for computing the fingerprint in Python. Each variant runs
in its own process. The matching time includes each reactant's fingerprint, as in 5a;
building the query index does not.

Usage:
    uv run benchmarks/05b_numpy_path_fp.py
    uv run benchmarks/05b_numpy_path_fp.py --fingerprint paths --runs 5
"""

import argparse
import gc
import sys
import time
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.numpy_query_index import NumpySubstructureQueryIndex, pattern_bits
from benchmarks.query_index import FP_SIZE as PATTERN_FP_SIZE
from benchmarks.rdkit_path_fp import FP_SIZE as PATH_FP_SIZE
from benchmarks.rdkit_path_fp import path_fingerprint
from benchmarks.report import (
    Report,
    digest_pairs,
    found_pairs,
    load_queries,
    load_reactants,
    print_report,
    print_table,
    print_worker,
    run_variants,
    screen_extras,
)

FINGERPRINTS = {
    "pattern": (pattern_bits, PATTERN_FP_SIZE, "RDKit Pattern fingerprints"),
    "paths": (
        path_fingerprint,
        PATH_FP_SIZE,
        "benchmark 08 path fingerprints (Python)",
    ),
}


def run_worker(
    name: str, posting_limit: int, runs: int, reactants_file: Path, queries_file: Path
) -> Report:
    fingerprint, fp_size, description = FINGERPRINTS[name]
    query_mols, query_prep = load_queries(queries_file)
    reactant_mols, reactant_prep = load_reactants(reactants_file)

    start = time.perf_counter()
    index = NumpySubstructureQueryIndex.from_mols(query_mols, fingerprint, fp_size)
    index_prep = time.perf_counter() - start
    gc.collect()

    # Untimed passes: how many pairs each stage lets through, and the cost of the
    # reactant fingerprints alone (they are also inside the matching time).
    posted = candidates = 0
    for mol in reactant_mols:
        posted_ids, screened_ids = index.screen(mol, posting_limit)
        posted += len(posted_ids)
        candidates += len(screened_ids)
    start = time.perf_counter()
    for mol in reactant_mols:
        fingerprint(mol, fp_size)
    fp_seconds = time.perf_counter() - start

    times = []
    pairs = None
    for _ in range(runs):
        start = time.perf_counter()
        matches = index.match_many(reactant_mols, posting_limit)
        times.append(time.perf_counter() - start)
        if pairs is None:
            pairs = found_pairs(index.query_mols, matches)
        del matches

    total_pairs = len(query_mols) * len(reactant_mols)
    return Report(
        benchmark="05b_numpy_path_fp",
        variant=name,
        queries=len(query_mols),
        reactants=len(reactant_mols),
        match_seconds=times,
        matches=len(pairs),
        digest=digest_pairs(pairs),
        query_prep_seconds=query_prep,
        index_prep_seconds=index_prep,
        reactant_prep_seconds=reactant_prep,
        settings={"fp bits": fp_size, "postings": posting_limit, "runs": runs},
        notes={
            "query": "read + parse SMARTS",
            "index": f"{description} + NumPy postings",
            "reactant": "read + parse SMILES",
            "matching": "reactant fingerprints + postings + fp check + exact match",
        },
        extras=[
            *screen_extras(total_pairs, posted, candidates, len(pairs)),
            ("reactant fps", fp_seconds, "part of matching, timed alone"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--postings", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--fingerprint", choices=("both", *FINGERPRINTS), default="both"
    )
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.postings <= min(size for _, size, _ in FINGERPRINTS.values()):
        parser.error("--postings must be between 1 and the fingerprint size")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.worker:
        if args.fingerprint == "both":
            parser.error("--worker requires one fingerprint")
        print_worker(
            run_worker(
                args.fingerprint,
                args.postings,
                args.runs,
                args.reactants_file,
                args.queries_file,
            )
        )
        return

    reports = run_variants(
        str(Path(__file__).resolve()),
        "fingerprint",
        tuple(FINGERPRINTS) if args.fingerprint == "both" else (args.fingerprint,),
        [
            "--postings",
            str(args.postings),
            "--runs",
            str(args.runs),
            "--reactants-file",
            str(args.reactants_file),
            "--queries-file",
            str(args.queries_file),
        ],
    )
    for report in reports:
        print_report(report)
    if len(reports) > 1:
        print_table(reports, ["exact matches run", "reactant fps"])
        if len({r.digest for r in reports}) > 1:
            raise RuntimeError("The fingerprints returned different matches")


if __name__ == "__main__":
    main()

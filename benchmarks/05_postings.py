"""Substructure search with posting lists over fingerprint bits.

Each fingerprint bit maps to the IDs of queries that contain it. A query can only
match a reactant if none of its bits are absent from the reactant fingerprint. Union
postings for the most frequent absent bits (128 by default), remove those IDs, and
run the full fingerprint subset check and RDKit's exact `HasSubstructMatch` on the
remaining queries. The final check makes the answer independent of how many
postings are selected.

Two choices, any combination:

- `--index`: where the query fingerprints and postings live.
  - `roaring`: Roaring bitmaps (`benchmarks/query_index.py`);
  - `numpy`: packed NumPy `uint64` arrays (`benchmarks/numpy_query_index.py`).
- `--fingerprint`: the screening fingerprint.
  - `pattern`: RDKit's 2048-bit `PatternFingerprint`;
  - `paths`: benchmark 08's `paths4` fingerprint at 4096 bits, computed from the
    RDKit molecules by `benchmarks/rdkit_path_fp.py`. It hashes the H count, degree
    and charge the template SMARTS state, which the Pattern fingerprint ignores.

Every combination must return the same matches. Each runs in its own process, so
the resident memory measured after loading, after the index build and after
matching belongs to that combination alone. The matching time includes each
reactant's fingerprint; building the query index does not.

Usage:
    uv run benchmarks/05_postings.py                       # all four combinations
    uv run benchmarks/05_postings.py --index numpy --fingerprint paths
    uv run benchmarks/05_postings.py --index roaring numpy --fingerprint pattern
"""

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import psutil

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.numpy_query_index import NumpySubstructureQueryIndex
from benchmarks.query_index import FP_SIZE as PATTERN_FP_SIZE
from benchmarks.query_index import SubstructureQueryIndex, pattern_bits
from benchmarks.rdkit_path_fp import FP_SIZE as PATH_FP_SIZE
from benchmarks.rdkit_path_fp import path_fingerprint
from benchmarks.report import (
    Report,
    digest_pairs,
    found_pairs,
    load_queries,
    load_reactants,
    mib,
    print_report,
    print_table,
    print_worker,
    run_variants,
    screen_extras,
)

BENCHMARK = "05_postings"
INDEXES = {
    "roaring": (SubstructureQueryIndex, "Roaring"),
    "numpy": (NumpySubstructureQueryIndex, "NumPy"),
}
FINGERPRINTS = {
    "pattern": (pattern_bits, PATTERN_FP_SIZE, "RDKit Pattern fingerprints"),
    "paths": (path_fingerprint, PATH_FP_SIZE, "path fingerprints (Python)"),
}


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def run_worker(
    index_name: str,
    fp_name: str,
    posting_limit: int,
    runs: int,
    reactants_file: Path,
    queries_file: Path,
) -> Report:
    index_class, backend = INDEXES[index_name]
    fingerprint, fp_size, description = FINGERPRINTS[fp_name]
    query_mols, query_prep = load_queries(queries_file)
    reactant_mols, reactant_prep = load_reactants(reactants_file)

    gc.collect()
    rss_loaded = rss_bytes()
    start = time.perf_counter()
    index = index_class.from_mols(query_mols, fingerprint, fp_size)
    index_prep = time.perf_counter() - start
    gc.collect()
    rss_indexed = rss_bytes()

    times = []
    pairs = None
    for _ in range(runs):
        start = time.perf_counter()
        matches = index.match_many(reactant_mols, posting_limit)
        times.append(time.perf_counter() - start)
        if pairs is None:
            pairs = found_pairs(index.query_mols, matches)
        del matches
    gc.collect()
    rss_matched = rss_bytes()

    # Untimed passes: how many pairs each stage lets through, and the cost of the
    # reactant fingerprints alone (they are also inside the matching time).
    posted = candidates = 0
    for mol in reactant_mols:
        remaining, screened = index.screen(mol, posting_limit)
        posted += len(remaining)
        candidates += len(screened)
    start = time.perf_counter()
    for mol in reactant_mols:
        fingerprint(mol, fp_size)
    fp_seconds = time.perf_counter() - start

    bitmap_bytes = len(
        pickle.dumps((index.query_fps, index.postings, index.all_ids), protocol=5)
    )
    index_bytes = len(pickle.dumps(index, protocol=5))
    total_pairs = len(query_mols) * len(reactant_mols)
    return Report(
        benchmark=BENCHMARK,
        variant=f"{index_name} {fp_name}",
        queries=len(query_mols),
        reactants=len(reactant_mols),
        match_seconds=times,
        matches=len(pairs),
        digest=digest_pairs(pairs),
        query_prep_seconds=query_prep,
        index_prep_seconds=index_prep,
        reactant_prep_seconds=reactant_prep,
        settings={
            "index": index_name,
            "fingerprint": fp_name,
            "fp bits": fp_size,
            "postings": posting_limit,
            "runs": runs,
        },
        notes={
            "query": "read + parse SMARTS",
            "index": f"{description} + {backend} postings",
            "reactant": "read + parse SMILES",
            "matching": "reactant fingerprints + postings + fp check + exact match",
        },
        extras=[
            *screen_extras(total_pairs, posted, candidates, len(pairs)),
            ("reactant fps", fp_seconds, "part of matching, timed alone"),
            ("index pickle", mib(index_bytes), "whole index, pickled"),
            ("bitmap pickle", mib(bitmap_bytes), "fingerprints + postings only"),
            (
                "build RSS delta",
                mib(rss_indexed - rss_loaded),
                "resident memory after build minus after load",
            ),
            (
                "RSS",
                f"{mib(rss_loaded)} / {mib(rss_indexed)} / {mib(rss_matched)}",
                "after load / build / matching",
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n")[2:]),
    )
    parser.add_argument(
        "--index", nargs="+", choices=tuple(INDEXES), default=list(INDEXES)
    )
    parser.add_argument(
        "--fingerprint",
        nargs="+",
        choices=tuple(FINGERPRINTS),
        default=list(FINGERPRINTS),
    )
    parser.add_argument("--postings", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    max_postings = min(FINGERPRINTS[name][1] for name in args.fingerprint)
    if not 1 <= args.postings <= max_postings:
        parser.error(f"--postings must be between 1 and {max_postings}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.worker:
        if len(args.index) != 1 or len(args.fingerprint) != 1:
            parser.error("--worker requires one index and one fingerprint")
        print_worker(
            run_worker(
                args.index[0],
                args.fingerprint[0],
                args.postings,
                args.runs,
                args.reactants_file,
                args.queries_file,
            )
        )
        return

    shared = [
        "--postings",
        str(args.postings),
        "--runs",
        str(args.runs),
        "--reactants-file",
        str(args.reactants_file),
        "--queries-file",
        str(args.queries_file),
    ]
    reports = []
    for index_name in dict.fromkeys(args.index):
        reports += run_variants(
            str(Path(__file__).resolve()),
            "fingerprint",
            tuple(dict.fromkeys(args.fingerprint)),
            ["--index", index_name, *shared],
        )
    for report in reports:
        print_report(report)
    print(
        "RSS is resident memory of each worker process. OS paging can change it, so "
        "the build delta is not an index size."
    )
    if len(reports) > 1:
        print_table(reports, ["exact matches run", "reactant fps"])
        if len({r.digest for r in reports}) > 1:
            raise RuntimeError("The combinations returned different matches")


if __name__ == "__main__":
    main()

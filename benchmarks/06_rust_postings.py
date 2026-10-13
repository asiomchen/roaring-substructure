"""Compare packed NumPy uint64 postings with the Rust posting index.

Author: Marcin Kowiel + Claude

Both variants use the same corpus, 128 absent-bit postings by default, a full
fingerprint subset screen, and RDKit's exact substructure match. Each variant
runs in its own process so resident memory measurements do not overlap.
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
from benchmarks.query_index import FP_SIZE
from benchmarks.rust_query_index import RustSubstructureQueryIndex
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

BACKENDS = ("numpy", "rust")


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def run_worker(
    backend: str,
    posting_limit: int,
    runs: int,
    reactants_file: Path,
    queries_file: Path,
) -> Report:
    query_mols, query_prep = load_queries(queries_file)
    reactant_mols, reactant_prep = load_reactants(reactants_file)

    gc.collect()
    rss_loaded = rss_bytes()
    start = time.perf_counter()
    index_class = (
        NumpySubstructureQueryIndex
        if backend == "numpy"
        else RustSubstructureQueryIndex
    )
    index = index_class.from_mols(query_mols)
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

    # Untimed pass: how many pairs each stage lets through.
    if backend == "numpy":
        posted = candidates = 0
        for mol in reactant_mols:
            posted_ids, screened_ids = index.screen(mol, posting_limit)
            posted += len(posted_ids)
            candidates += len(screened_ids)
        bitmap_bytes = len(
            pickle.dumps((index.query_fps, index.postings, index.all_ids), protocol=5)
        )
        index_bytes = len(pickle.dumps(index, protocol=5))
    else:
        # The Rust screen returns only the IDs left after the full fingerprint check.
        posted = None
        candidates = sum(map(len, index.screen_many(reactant_mols, posting_limit)))
        bitmap_bytes = index.postings.nbytes()
        index_bytes = bitmap_bytes + len(pickle.dumps(index.query_mols, protocol=5))
    total_pairs = len(query_mols) * len(reactant_mols)
    return Report(
        benchmark="06_rust_postings",
        variant=backend,
        queries=len(query_mols),
        reactants=len(reactant_mols),
        match_seconds=times,
        matches=len(pairs),
        digest=digest_pairs(pairs),
        query_prep_seconds=query_prep,
        index_prep_seconds=index_prep,
        reactant_prep_seconds=reactant_prep,
        settings={"postings": posting_limit, "runs": runs},
        notes={
            "query": "read + parse SMARTS",
            "index": f"Pattern fingerprints + {backend} postings",
            "reactant": "read + parse SMILES",
            "matching": "reactant fingerprints + postings + fp check + exact match",
        },
        extras=[
            *screen_extras(total_pairs, posted, candidates, len(pairs)),
            ("index size", mib(index_bytes), "whole index incl. query molecules, pickled"),
            ("bitmap size", mib(bitmap_bytes), "fingerprints + postings only"),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postings", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--backend", choices=("both", "numpy", "rust"), default="both")
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.postings <= FP_SIZE:
        parser.error(f"--postings must be between 1 and {FP_SIZE}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.worker:
        if args.backend == "both":
            parser.error("--worker requires one backend")
        print_worker(
            run_worker(
                args.backend,
                args.postings,
                args.runs,
                args.reactants_file,
                args.queries_file,
            )
        )
        return

    reports = run_variants(
        str(Path(__file__).resolve()),
        "backend",
        BACKENDS if args.backend == "both" else (args.backend,),
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
        print_table(reports, ["exact matches run", "bitmap size"])
        if len({r.digest for r in reports}) > 1:
            raise RuntimeError("NumPy and Rust indexes returned different matches")
    print(
        "RSS values are resident memory in a fresh process per backend. OS paging can "
        "change them, so the build delta is not an index size."
    )


if __name__ == "__main__":
    main()

"""Compare the Rust posting index with and without process-parallel exact matching.

Author: Marcin Kowiel + Claude

Both variants use the same corpus, 128 absent-bit postings by default, a full
fingerprint subset screen, and RDKit's exact substructure match. Each variant
runs in its own process so resident memory measurements do not overlap.
"""

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import psutil

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.query_index import FP_SIZE
from benchmarks.rust_process_query_index import RustProcessSubstructureQueryIndex
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

BACKENDS = ("rust", "process")


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def run_worker(
    backend: str,
    posting_limit: int,
    runs: int,
    workers: int,
    reactants_file: Path,
    queries_file: Path,
) -> Report:
    query_mols, query_prep = load_queries(queries_file)
    reactant_mols, reactant_prep = load_reactants(reactants_file)

    gc.collect()
    rss_loaded = rss_bytes()
    start = time.perf_counter()
    if backend == "rust":
        index = RustSubstructureQueryIndex.from_mols(query_mols)
    else:
        index = RustProcessSubstructureQueryIndex.from_mols(query_mols, workers)
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

    # Untimed pass. The Rust screen returns only the IDs left after the full
    # fingerprint check.
    rust = index if backend == "rust" else index.rust
    posted = None
    candidates = sum(map(len, rust.screen_many(reactant_mols, posting_limit)))
    bitmap_bytes = rust.postings.nbytes()
    index_bytes = bitmap_bytes + len(pickle.dumps(index.query_mols, protocol=5))
    total_pairs = len(query_mols) * len(reactant_mols)
    return Report(
        benchmark="07_rust_postings_process",
        variant=backend,
        queries=len(query_mols),
        reactants=len(reactant_mols),
        match_seconds=times,
        matches=len(pairs),
        digest=digest_pairs(pairs),
        query_prep_seconds=query_prep,
        index_prep_seconds=index_prep,
        reactant_prep_seconds=reactant_prep,
        settings={
            "postings": posting_limit,
            "runs": runs,
            "workers": workers if backend == "process" else 1,
        },
        notes={
            "query": "read + parse SMARTS",
            "index": "Pattern fingerprints + Rust postings",
            "reactant": "read + parse SMILES",
            "matching": "reactant fingerprints + postings + fp check + exact match",
        },
        extras=[
            *screen_extras(total_pairs, posted, candidates, len(pairs)),
            ("index size", mib(index_bytes), "postings + pickled query molecules"),
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
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--backend", choices=("both", "rust", "process"), default="both"
    )
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
    if args.workers < 1:
        parser.error("--workers must be at least 1")
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
                args.workers,
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
            "--workers",
            str(args.workers),
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
            raise RuntimeError(
                "Serial and process-parallel Rust indexes returned different matches"
            )
    print(
        "RSS values are resident memory in a fresh process per backend. OS paging can "
        "change them, so the build delta is not an index size."
    )


if __name__ == "__main__":
    main()

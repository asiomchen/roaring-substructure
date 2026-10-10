"""Shared runner of benchmarks 5 (Roaring) and 5a (NumPy).

Both build a posting index over Pattern fingerprint bits and differ only in the
bitmap type, so they run the same code: `main(benchmark, index_class, backend)`.
Each report shows the timings, the screen counts, the index pickle sizes and the
process's resident memory after loading, after the build and after matching.
"""

import argparse
import gc
import pickle
import time
from pathlib import Path

import psutil

from benchmarks.query_index import FP_SIZE
from benchmarks.report import (
    Report,
    digest_pairs,
    found_pairs,
    load_queries,
    load_reactants,
    mib,
    print_report,
    screen_extras,
)


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def main(benchmark: str, index_class, backend: str, description: str) -> None:
    """Run one posting benchmark; `index_class` has `from_mols`, `match_many`,
    `screen` and the `query_fps`, `postings` and `all_ids` attributes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--postings", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    args = parser.parse_args()
    if not 1 <= args.postings <= FP_SIZE:
        parser.error(f"--postings must be between 1 and {FP_SIZE}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    query_mols, query_prep = load_queries(args.queries_file)
    reactant_mols, reactant_prep = load_reactants(args.reactants_file)

    gc.collect()
    rss_loaded = rss_bytes()
    start = time.perf_counter()
    index = index_class.from_mols(query_mols)
    index_prep = time.perf_counter() - start
    gc.collect()
    rss_indexed = rss_bytes()

    times = []
    pairs = None
    for _ in range(args.runs):
        start = time.perf_counter()
        matches = index.match_many(reactant_mols, args.postings)
        times.append(time.perf_counter() - start)
        if pairs is None:
            pairs = found_pairs(index.query_mols, matches)
        del matches
    gc.collect()
    rss_matched = rss_bytes()

    # Untimed pass: how many pairs each stage lets through.
    posted = candidates = 0
    for mol in reactant_mols:
        remaining, screened = index.screen(mol, args.postings)
        posted += len(remaining)
        candidates += len(screened)

    bitmap_bytes = len(
        pickle.dumps((index.query_fps, index.postings, index.all_ids), protocol=5)
    )
    index_bytes = len(pickle.dumps(index, protocol=5))
    total_pairs = len(query_mols) * len(reactant_mols)
    print_report(
        Report(
            benchmark=benchmark,
            queries=len(query_mols),
            reactants=len(reactant_mols),
            match_seconds=times,
            matches=len(pairs),
            digest=digest_pairs(pairs),
            query_prep_seconds=query_prep,
            index_prep_seconds=index_prep,
            reactant_prep_seconds=reactant_prep,
            settings={"postings": args.postings, "runs": args.runs},
            notes={
                "query": "read + parse SMARTS",
                "index": f"Pattern fingerprints + {backend} postings",
                "reactant": "read + parse SMILES",
                "matching": "reactant fingerprints + postings + fp check + exact match",
            },
            extras=[
                *screen_extras(total_pairs, posted, candidates, len(pairs)),
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
    )
    print(
        "RSS is resident memory of this process. OS paging can change it, so the "
        "build delta is not an index size."
    )

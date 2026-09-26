"""Compare the Rust posting index with and without process-parallel exact matching.

Both variants use the same corpus, 128 absent-bit postings by default, a full
fingerprint subset screen, and RDKit's exact substructure match. Each variant
runs in its own process so resident memory measurements do not overlap.
"""

import argparse
import gc
import hashlib
import json
import os
import pickle
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

import datamol as dm
import pandas as pd
import psutil

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.query_index import FP_SIZE
from benchmarks.rust_process_query_index import RustProcessSubstructureQueryIndex
from benchmarks.rust_query_index import RustSubstructureQueryIndex


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def run_worker(
    backend: str,
    posting_limit: int,
    runs: int,
    workers: int,
    reactants_file: Path,
    queries_file: Path,
) -> dict:
    smarts = pd.read_parquet(queries_file)["substructure"]
    query_mols = [dm.convert.from_smarts(value) for value in smarts]
    if any(mol is None for mol in query_mols):
        raise ValueError("Failed to parse substructure SMARTS")
    smiles = pd.read_csv(reactants_file)["smiles"]
    reactant_mols = [dm.to_mol(value) for value in smiles]
    if any(mol is None for mol in reactant_mols):
        raise ValueError("Failed to parse reactant SMILES")

    gc.collect()
    rss_loaded = rss_bytes()
    start = time.perf_counter()
    if backend == "rust":
        index = RustSubstructureQueryIndex.from_mols(query_mols)
    else:
        index = RustProcessSubstructureQueryIndex.from_mols(query_mols, workers)
    build_seconds = time.perf_counter() - start
    gc.collect()
    rss_indexed = rss_bytes()

    times = []
    digest = None
    match_count = None
    for _ in range(runs):
        start = time.perf_counter()
        matches = index.match_many(reactant_mols, posting_limit)
        times.append(time.perf_counter() - start)
        if digest is None:
            query_ids = {id(mol): idx for idx, mol in enumerate(index.query_mols)}
            hasher = hashlib.sha256()
            match_count = 0
            for reactant_idx, found in enumerate(matches):
                for mol in found:
                    hasher.update(struct.pack("<II", reactant_idx, query_ids[id(mol)]))
                    match_count += 1
            digest = hasher.hexdigest()
            del query_ids
        del matches
    gc.collect()
    rss_matched = rss_bytes()

    rust = index if backend == "rust" else index.rust
    bitmap_bytes = rust.postings.nbytes()
    index_bytes = bitmap_bytes + len(pickle.dumps(index.query_mols, protocol=5))
    return {
        "backend": backend,
        "queries": len(query_mols),
        "reactants": len(reactant_mols),
        "matches": match_count,
        "digest": digest,
        "build_seconds": build_seconds,
        "match_seconds": times,
        "index_bytes": index_bytes,
        "bitmap_bytes": bitmap_bytes,
        "rss_loaded_bytes": rss_loaded,
        "rss_indexed_bytes": rss_indexed,
        "rss_matched_bytes": rss_matched,
    }


def mib(value: int) -> str:
    return f"{value / 1024**2:.1f}"


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
        print(
            json.dumps(
                run_worker(
                    args.backend,
                    args.postings,
                    args.runs,
                    args.workers,
                    args.reactants_file,
                    args.queries_file,
                )
            )
        )
        return

    backends = ("rust", "process") if args.backend == "both" else (args.backend,)
    results = []
    for backend in backends:
        print(f"Running {backend}...", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--backend",
            backend,
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
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        results.append(json.loads(completed.stdout))

    if len(results) == 2 and (results[0]["matches"], results[0]["digest"]) != (
        results[1]["matches"],
        results[1]["digest"],
    ):
        raise RuntimeError(
            "Serial and process-parallel Rust indexes returned different matches"
        )

    print(
        f"{results[0]['queries']:,} queries x {results[0]['reactants']:,} reactants; "
        f"{args.postings} postings; {args.runs} matching runs per backend; "
        f"{args.workers} workers for process"
    )
    print(
        "backend   build s  match median s    index MiB  bitmaps MiB  "
        "build RSS ΔMiB  RSS load/build/match MiB"
    )
    for result in results:
        print(
            f"{result['backend']:<9} {result['build_seconds']:>7.3f}  "
            f"{statistics.median(result['match_seconds']):>14.3f}  "
            f"{mib(result['index_bytes']):>11}  "
            f"{mib(result['bitmap_bytes']):>11}  "
            f"{mib(result['rss_indexed_bytes'] - result['rss_loaded_bytes']):>14}  "
            f"{mib(result['rss_loaded_bytes'])}/"
            f"{mib(result['rss_indexed_bytes'])}/"
            f"{mib(result['rss_matched_bytes'])}"
        )
        print(
            f"  matching runs: {[round(value, 3) for value in result['match_seconds']]}; "
            f"matches: {result['matches']:,}"
        )
    print(
        "RSS values are resident memory after data load, index build, and matching, "
        "each in a fresh process; build RSS Δ is the build-minus-load change. "
        "OS paging can change these values, so the delta is not an index size."
    )


if __name__ == "__main__":
    main()

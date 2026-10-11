"""The fastest Python and Rust searches with the index built once and kept on disk.

Author: Marcin Kowiel + Claude

The other benchmarks build the query index and match in one process. A real
service builds the index once, saves it, and loads it wherever it matches
reactants. Each variant here runs that in two processes, one after the other:

    build   read queries -> create index -> save index
    search  read index -> read reactants -> match (median of --runs)

Every step is timed, and each process records its resident memory (RSS) after
each step and its peak. Each process imports its libraries before the first
step, so the steps time no imports and the start RSS includes the libraries. The search process never builds anything, so its memory
is what a service holding the index needs. The index file's size is reported.

Variants (`--variant`, any of them; all by default):

    numpy-pattern         NumPy posting lists, RDKit Pattern fp, RDKit match
    numpy-paths           NumPy posting lists, path fp in Python at 2048 bits
                          (benchmark 05 uses 4096), RDKit match
    rust-paths4-branches  all in Rust (`substructure_rs`), paths4+branches fp at
                          2048 bits, the fastest configuration of benchmark 09

What the index file holds:

- NumPy: the whole index object pickled (protocol 5): the RDKit query molecules
  in RDKit's binary format, the fingerprint and posting arrays. Loading
  unpickles it; nothing is parsed or fingerprinted again.
- Rust: the SMARTS, the fingerprint settings and the fingerprint and posting
  arrays (`substructure_rs/src/store.rs`). Loading parses the SMARTS and plans
  the exact match again; fingerprints are not recomputed.

The whole OS file cache is dropped before each search process (`purge` on
macOS, `drop_caches` on Linux), so the index file, the reactant file and the
libraries all come from disk, as on a fresh start. Dropping the cache needs sudo
without a password prompt: run `sudo -v` first, in the same terminal.

Rust runs on `--threads` rayon threads (1 by default, like the Python
variants). Every variant must return the same matches.

Usage:
    sudo -v && uv run benchmarks/11_index_on_disk.py
    sudo -v && uv run benchmarks/11_index_on_disk.py --variant numpy-pattern --runs 5
    sudo -v && uv run benchmarks/11_index_on_disk.py --index-dir data/indexes  # keep files
"""

import argparse
import json
import os
import pickle
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.report import (
    Report,
    digest_pairs,
    found_pairs,
    load_queries,
    load_reactants,
    mib,
    print_report,
    print_table,
)

BENCHMARK = "11_index_on_disk"
BUILD_STEPS = ("read queries", "create index", "save index")
SEARCH_STEPS = ("read index", "read reactants", "match")


class NumpyVariant:
    """Benchmark 05's NumPy index, pickled. RDKit is imported only here."""

    language = "Python"
    suffix = ".pkl"

    def __init__(self, fp_name: str, fp_size: int | None = None):
        self.fp_name = fp_name
        self.fp_size = fp_size  # None: the fingerprint module's FP_SIZE

    def fingerprint(self):
        if self.fp_name == "pattern":
            from benchmarks.query_index import FP_SIZE, pattern_bits

            return pattern_bits, self.fp_size or FP_SIZE, "RDKit Pattern fp"
        from benchmarks.rdkit_path_fp import FP_SIZE, path_fingerprint

        return path_fingerprint, self.fp_size or FP_SIZE, "path fp (Python)"

    def describe(self) -> str:
        _, fp_size, name = self.fingerprint()
        return f"NumPy posting lists, {name} at {fp_size} bits, RDKit match"

    def import_libraries(self) -> None:
        import datamol  # noqa: F401
        import pandas  # noqa: F401

        import benchmarks.numpy_query_index  # noqa: F401

        self.fingerprint()

    def read_queries(self, path: Path):
        return load_queries(path)[0]

    def create_index(self, queries):
        from benchmarks.numpy_query_index import NumpySubstructureQueryIndex

        fingerprint, fp_size, _ = self.fingerprint()
        return NumpySubstructureQueryIndex.from_mols(queries, fingerprint, fp_size)

    def save_index(self, index, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(index, handle, protocol=5)

    def read_index(self, path: Path):
        with path.open("rb") as handle:
            return pickle.load(handle)

    def read_reactants(self, path: Path):
        return load_reactants(path)[0]

    def match(self, index, reactants, postings: int):
        return index.match_many(reactants, postings)

    def query_count(self, index) -> int:
        return len(index.query_mols)  # len(index) counts fingerprint bits

    def pairs(self, index, found) -> list[tuple[int, int]]:
        return found_pairs(index.query_mols, found)

    def screen_counts(self, index, reactants, postings: int) -> tuple[int, int]:
        posted = candidates = 0
        for mol in reactants:
            remaining, screened = index.screen(mol, postings)
            posted += len(remaining)
            candidates += len(screened)
        return posted, candidates


class RustVariant:
    """`substructure_rs` end to end: its own parsers, screen and matcher."""

    language = "Rust"
    suffix = ".idx"

    def __init__(self, fp_kind: str, fp_bits: int):
        self.fp_kind = fp_kind
        self.fp_bits = fp_bits

    def describe(self) -> str:
        return f"all in Rust, {self.fp_kind} fp at {self.fp_bits} bits"

    def import_libraries(self) -> None:
        import substructure_rs  # noqa: F401

    def read_queries(self, path: Path):
        import substructure_rs

        return substructure_rs.read_queries(path)

    def create_index(self, queries):
        import substructure_rs

        return substructure_rs.Index.build(queries, self.fp_kind, self.fp_bits)

    def save_index(self, index, path: Path) -> None:
        index.save(path)

    def read_index(self, path: Path):
        import substructure_rs

        return substructure_rs.Index.load(path)

    def read_reactants(self, path: Path):
        import substructure_rs

        return substructure_rs.read_reactants(path)

    def match(self, index, reactants, postings: int):
        return index.match_many(reactants, postings)

    def query_count(self, index) -> int:
        return len(index)

    def pairs(self, index, found) -> list[tuple[int, int]]:
        per_reactant, _, _ = found
        return [(r, q) for r, ids in enumerate(per_reactant) for q in ids]

    def screen_counts(self, index, reactants, postings: int) -> tuple[int, int]:
        _, posted, candidates = index.match_many(reactants, postings)
        return posted, candidates


VARIANTS = {
    "numpy-pattern": NumpyVariant("pattern"),
    "numpy-paths": NumpyVariant("paths", 2048),
    "rust-paths4-branches": RustVariant("paths4+branches", 2048),
}


class Meter:
    """Times steps and records this process's RSS after each, and its peak.

    The variant's libraries are imported first, so no step pays for an import
    and the start RSS includes them.
    """

    def __init__(self, variant):
        variant.import_libraries()
        self.seconds: dict[str, float] = {}
        self.rss: dict[str, int] = {"start": rss_bytes()}

    def step(self, name: str, call, *args):
        start = time.perf_counter()
        result = call(*args)
        self.seconds[name] = time.perf_counter() - start
        self.rss[name] = rss_bytes()
        return result

    def to_dict(self) -> dict:
        return {"seconds": self.seconds, "rss": self.rss, "peak_rss": peak_rss_bytes()}


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux: KiB


def build(variant, queries_file: Path, index_file: Path) -> dict:
    meter = Meter(variant)
    queries = meter.step("read queries", variant.read_queries, queries_file)
    index = meter.step("create index", variant.create_index, queries)
    meter.step("save index", variant.save_index, index, index_file)
    return {**meter.to_dict(), "index_file_bytes": index_file.stat().st_size}


def search(
    variant, index_file: Path, reactants_file: Path, postings: int, runs: int
) -> dict:
    meter = Meter(variant)
    index = meter.step("read index", variant.read_index, index_file)
    reactants = meter.step("read reactants", variant.read_reactants, reactants_file)
    match_seconds = []
    pairs = None
    for _ in range(runs):
        start = time.perf_counter()
        found = variant.match(index, reactants, postings)
        match_seconds.append(time.perf_counter() - start)
        if pairs is None:
            pairs = variant.pairs(index, found)
        del found
    meter.rss["match"] = rss_bytes()
    result = meter.to_dict()  # before the untimed pass below can raise the peak
    posted, candidates = variant.screen_counts(index, reactants, postings)
    return {
        **result,
        "match_seconds": match_seconds,
        "queries": variant.query_count(index),
        "reactants": len(reactants),
        "matches": len(pairs),
        "digest": digest_pairs(pairs),
        "posted": posted,
        "candidates": candidates,
    }


def drop_os_cache() -> None:
    """Drop the OS file cache, so the next reads come from disk. Needs sudo
    without a password prompt."""
    if sys.platform == "darwin":
        command = ["sudo", "-n", "purge"]
    elif sys.platform.startswith("linux"):
        command = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
    else:
        raise SystemExit("dropping the OS file cache needs macOS or Linux")
    done = subprocess.run(command, text=True, capture_output=True, check=False)
    if done.returncode:
        raise SystemExit(
            f"could not drop the OS file cache with `{' '.join(command)}`: "
            f"{done.stderr.strip()}\nRun `sudo -v` first, in the same terminal."
        )


def run_worker(phase: str, name: str, args: argparse.Namespace) -> dict:
    """Run one phase of one variant in a fresh process; return its JSON result."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        phase,
        "--variant",
        name,
        "--index-file",
        str(args.index_file),
        "--queries-file",
        str(args.queries_file),
        "--reactants-file",
        str(args.reactants_file),
        "--postings",
        str(args.postings),
        "--runs",
        str(args.runs),
    ]
    env = {**os.environ, "RAYON_NUM_THREADS": str(args.threads)}
    done = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    if done.returncode:
        sys.stderr.write(done.stderr)
        raise SystemExit(f"{name} {phase} failed with exit code {done.returncode}")
    return json.loads(done.stdout)


def rss_line(rss: dict, steps: tuple[str, ...]) -> str:
    return " / ".join(mib(rss[step]) for step in ("start", *steps))


def make_report(name: str, built: dict, searched: dict, args) -> Report:
    variant = VARIANTS[name]
    b, s = built["seconds"], searched["seconds"]
    match_median = sorted(searched["match_seconds"])[
        len(searched["match_seconds"]) // 2
    ]
    pairs = searched["queries"] * searched["reactants"]
    return Report(
        benchmark=BENCHMARK,
        variant=name,
        queries=searched["queries"],
        reactants=searched["reactants"],
        match_seconds=searched["match_seconds"],
        matches=searched["matches"],
        digest=searched["digest"],
        query_prep_seconds=b["read queries"],
        index_prep_seconds=b["create index"],
        reactant_prep_seconds=s["read reactants"],
        settings={
            "language": variant.language,
            "postings": args.postings,
            "runs": args.runs,
            "threads": args.threads if variant.language == "Rust" else 1,
            "file cache": "dropped before search",
        },
        notes={
            "query": "build process: read + parse SMARTS",
            "index": f"build process: {variant.describe()}",
            "reactant": "search process: read + parse SMILES",
            "matching": "search process: reactant fps + screen + exact match",
        },
        extras=[
            ("save index", b["save index"], "build process: write the index file"),
            ("read index", s["read index"], "search process: load the index file"),
            (
                "build total",
                sum(b.values()),
                "read queries + create index + save index",
            ),
            (
                "search total",
                s["read index"] + s["read reactants"] + match_median,
                "read index + read reactants + median matching",
            ),
            ("index on disk", mib(built["index_file_bytes"]), "index file size"),
            (
                "index RSS",
                mib(searched["rss"]["read index"] - searched["rss"]["start"]),
                "search process: RSS after reading the index minus at start (libraries loaded)",
            ),
            (
                "build RSS",
                rss_line(built["rss"], BUILD_STEPS),
                "at start / after " + " / ".join(BUILD_STEPS),
            ),
            ("build peak RSS", mib(built["peak_rss"]), "build process maximum"),
            (
                "search RSS",
                rss_line(searched["rss"], SEARCH_STEPS),
                "at start / after " + " / ".join(SEARCH_STEPS),
            ),
            ("search peak RSS", mib(searched["peak_rss"]), "search process maximum"),
            (
                "dropped by postings",
                pairs - searched["posted"],
                "pairs ruled out by the postings",
            ),
            (
                "exact matches run",
                searched["candidates"],
                f"{100 * searched['candidates'] / pairs:.2f}% of the pairs",
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
        "--variant", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--postings", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1, help="Rust threads")
    parser.add_argument(
        "--index-dir", type=Path, help="keep the index files here (default: temporary)"
    )
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    parser.add_argument("--worker", choices=("build", "search"), help=argparse.SUPPRESS)
    parser.add_argument("--index-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.postings <= 2048:
        parser.error("--postings must be between 1 and 2048")
    if args.runs < 1 or args.threads < 1:
        parser.error("--runs and --threads must be at least 1")

    if args.worker:
        variant = VARIANTS[args.variant[0]]
        if args.worker == "build":
            result = build(variant, args.queries_file, args.index_file)
        else:
            result = search(
                variant, args.index_file, args.reactants_file, args.postings, args.runs
            )
        print(json.dumps(result))
        return

    drop_os_cache()  # fail now, not after the first build
    index_dir = args.index_dir or Path(tempfile.mkdtemp(prefix="substructure_index_"))
    index_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        for name in dict.fromkeys(args.variant):
            args.index_file = index_dir / f"{name}{VARIANTS[name].suffix}"
            print(f"Running {name}: build...", end=" ", flush=True)
            built = run_worker("build", name, args)
            print("drop cache...", end=" ", flush=True)
            drop_os_cache()
            print("search...", flush=True)
            searched = run_worker("search", name, args)
            reports.append(make_report(name, built, searched, args))
    finally:
        if args.index_dir is None:
            shutil.rmtree(index_dir)
    print()
    for report in reports:
        print_report(report)
    print(
        "RSS is resident memory of each process; OS paging can change it. "
        "The OS file cache was dropped before each search process."
    )
    if len(reports) > 1:
        print_table(
            reports, ["save index", "read index", "search total", "index on disk"]
        )
        if len({r.digest for r in reports}) > 1:
            raise RuntimeError("The variants returned different matches")


if __name__ == "__main__":
    main()

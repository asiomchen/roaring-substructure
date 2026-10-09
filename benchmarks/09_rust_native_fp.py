"""Benchmark 9: benchmark 08 over screening-fingerprint kinds and sizes.

Author: Marcin Kowiel + Claude

Runs `substructure_rs.run()` once per (fingerprint kind, size, posting count)
and prints one row per configuration. Every configuration must return the same
match set: the screen only decides which pairs reach the exact matcher, so a
digest that differs from the others means a fingerprint rejected a true match.

The kinds (see `substructure_rs/src/fingerprint.rs`):

    atoms                   atom labels only
    paths2 / paths4 / paths6  atoms + labelled paths up to 2, 4 or 6 bonds
    paths4-nocount          paths4 with each feature counted once
    paths4+branches         paths4 + every atom with three of its neighbours
    paths4+cycles           paths4 + labelled simple cycles of 3-8 atoms
    paths4+branches+cycles  paths4 + branches + cycles
    paths4+long6/long8+branches+cycles
                            as above, plus element/bond-class-only paths of
                            5-6 or 5-8 bonds

Sizes default to 512-8192 bits. `paths4` at 4096 bits is benchmark 08.
Candidates, the pairs that survive the screen, measure the fingerprint
independently of the machine; the matching time is what the screen is for. Use `--threads 1` for steadier timings.
The last four columns split matching into reactant fingerprints, the posting
filter, the full fingerprint check and exact matching, in milliseconds summed
over threads.

    uv run benchmarks/09_rust_native_fp.py
    uv run benchmarks/09_rust_native_fp.py --kinds paths4 paths4+cycles --bits 2048 8192
    uv run benchmarks/09_rust_native_fp.py --postings 64 128 256 --csv out.csv
"""

import argparse
import csv
from pathlib import Path

import substructure_rs

COLUMNS = [
    "fp_kind",
    "fp_bits",
    "postings",
    "query_density",
    "target_density",
    "build_seconds",
    "index_mib",
    "posted",
    "candidates",
    "matches",
    "match_median_seconds",
    "match_seconds",
    "fp_seconds",
    "postings_seconds",
    "check_seconds",
    "exact_seconds",
    "digest",
]
PHASES = ["fp_seconds", "postings_seconds", "check_seconds", "exact_seconds"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument(
        "--kinds", nargs="+", default=substructure_rs.FP_KINDS, metavar="KIND"
    )
    parser.add_argument(
        "--bits", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192], metavar="N"
    )
    parser.add_argument("--postings", nargs="+", type=int, default=[128], metavar="N")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0, help="0 = all cores")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--csv", type=Path, help="also write the rows here")
    args = parser.parse_args()

    for kind in args.kinds:
        if kind not in substructure_rs.FP_KINDS:
            parser.error(f"unknown kind {kind}; choose from {substructure_rs.FP_KINDS}")
    for bits in args.bits:
        if bits not in substructure_rs.FP_SIZES:
            parser.error(f"unsupported size {bits}; choose from {substructure_rs.FP_SIZES}")

    rows = []
    header = (
        f"{'kind':<24}{'bits':>6}{'post':>5}{'q set':>7}{'r set':>7}"
        f"{'build s':>9}{'MiB':>7}{'posted':>10}{'candidates':>12}{'matches':>9}"
        f"{'match ms':>10} |{'fp':>6}{'post':>6}{'check':>6}{'exact':>6}"
    )
    print(header)
    print("-" * len(header))
    for kind in args.kinds:
        for bits in args.bits:
            for postings in args.postings:
                try:
                    r = substructure_rs.run(
                        queries_file=args.queries_file,
                        reactants_file=args.reactants_file,
                        postings=postings,
                        runs=args.runs,
                        threads=args.threads,
                        fp_kind=kind,
                        fp_bits=bits,
                        reference=args.reference,
                    )
                except ValueError as error:
                    parser.exit(1, f"error: {error}\n")
                row = {
                    "fp_kind": kind,
                    "fp_bits": bits,
                    "postings": postings,
                    "query_density": r["query_density"],
                    "target_density": r["target_density"],
                    "build_seconds": r["build_seconds"],
                    "index_mib": r["index_bytes"] / 2**20,
                    "posted": r["posted"],
                    "candidates": r["candidates"],
                    "matches": r["matches"],
                    "match_median_seconds": r["match_median_seconds"],
                    "match_seconds": " ".join(f"{t:.4f}" for t in r["match_seconds"]),
                    **dict(zip(PHASES, r["phase_seconds"])),
                    "digest": r["digest"],
                }
                rows.append(row)
                print(
                    f"{kind:<24}{bits:>6}{postings:>5}"
                    f"{100 * row['query_density']:>6.1f}%{100 * row['target_density']:>6.1f}%"
                    f"{row['build_seconds']:>9.3f}{row['index_mib']:>7.1f}"
                    f"{row['posted']:>10}{row['candidates']:>12}{row['matches']:>9}"
                    f"{1000 * row['match_median_seconds']:>10.1f} |"
                    + "".join(f"{1000 * row[p]:>6.0f}" for p in PHASES),
                    flush=True,
                )
                comparison = r.get("comparison")
                if comparison and (comparison["missing"] or comparison["extra"]):
                    print(
                        f"  reference: missing {comparison['missing']}, "
                        f"extra {comparison['extra']}"
                    )

    digests = {row["digest"] for row in rows}
    if len(digests) == 1:
        print(f"\nall {len(rows)} configurations agree; digest {digests.pop()}")
    else:
        print("\nconfigurations DISAGREE on the match set:")
        for row in rows:
            print(f"  {row['fp_kind']:<24}{row['fp_bits']:>6}  {row['digest']}")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")

    if len(digests) > 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

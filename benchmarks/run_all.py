"""Run every benchmark on one thread and aggregate the timings.

Each benchmark runs in its single-thread configuration, one after another so
they never compete for CPU:

- the in-process Python benchmarks as they are (01-03, 05);
- 04 with one worker process, and 06/07 with `RAYON_NUM_THREADS=1`;
  07's process-parallel backend is left out, since it is multi-core by design;
- 08, 09, 10 and 11 with `--threads 1`; 11 drops the OS file cache, so run
  `sudo -v` first;
- PostgreSQL over one connection, with parallel workers off (as `compose.yaml`
  pins them): setup, load, index, match, verify. The server is stopped afterwards,
  not removed; its volume is kept.

`RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `MKL_NUM_THREADS` are set to 1 for every run. macOS
cannot pin a process to one core, so "single core" here means one thread of
work at a time, not CPU affinity.

Every report is collected through `BENCH_RESULTS` (see `benchmarks/report.py`)
and written under `benchmarks/results/<timestamp>/`:

    results.jsonl   every report, as printed
    summary.csv     one row per report, standard fields and the screen counts
    summary.md      metadata, the main table, and the benchmark 09 sweep
    metadata.json   machine, software versions, thread settings
    logs/           each benchmark's full output

Usage:
    uv run benchmarks/run_all.py
    uv run benchmarks/run_all.py --only 03 05 08
    uv run benchmarks/run_all.py --skip-postgres
"""

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV = ["uv", "run"]
PG = ["benchmarks/postgres/bench.py"]

# (key, description, commands); every command of an entry runs in order.
BENCHMARKS = [
    ("01", "naive", [["benchmarks/01_naive.py"]]),
    ("02", "RDKit Pattern fingerprint", [["benchmarks/02_pattern.py"]]),
    ("03", "Roaring fingerprint scan", [["benchmarks/03_roaring_approach.py"]]),
    (
        "04",
        "Roaring scan, one worker process",
        [["benchmarks/04_roaring_multiprocessing.py", "--processes", "1"]],
    ),
    (
        "05",
        "postings, Roaring and NumPy x Pattern and path fp",
        [["benchmarks/05_postings.py"]],
    ),
    ("06", "NumPy vs Rust posting screen", [["benchmarks/06_rust_postings.py"]]),
    (
        "07",
        "Rust posting screen (serial backend only)",
        [["benchmarks/07_rust_postings_process.py", "--backend", "rust"]],
    ),
    ("08", "native Rust", [["benchmarks/08_rust_native.py", "--threads", "1"]]),
    (
        "09",
        "native Rust fingerprint sweep",
        [["benchmarks/09_rust_native_fp.py", "--threads", "1"]],
    ),
    (
        "10",
        "chematic, naive and screened",
        [
            ["benchmarks/10_chematic.py", "--mode", "naive", "--threads", "1"],
            ["benchmarks/10_chematic.py", "--mode", "screened", "--threads", "1"],
        ],
    ),
    (
        "11",
        "index saved to disk and loaded, Python and Rust",
        [["benchmarks/11_index_on_disk.py", "--threads", "1"]],
    ),
    (
        "postgres",
        "PostgreSQL RDKit cartridge, one connection",
        [
            [*PG, "--phase", "setup"],
            [*PG, "--phase", "load"],
            [*PG, "--phase", "index"],
            [*PG, "--phase", "match", "--connections", "1"],
            [*PG, "--phase", "verify"],
        ],
    ),
]

SINGLE_THREAD = {
    name: "1"
    for name in (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
    )
}

SCREEN_COLUMNS = [
    "dropped by postings",
    "dropped by fp check",
    "dropped by fp",
    "exact matches run",
]


def run(command: list[str], **kwargs) -> str:
    return subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=True, **kwargs
    ).stdout.strip()


def sysctl(name: str) -> str:
    try:
        return run(["sysctl", "-n", name])
    except (OSError, subprocess.CalledProcessError):
        return ""


def metadata() -> dict:
    packages = {}
    for name in ("rdkit", "numpy", "pyroaring", "pandas", "datamol", "psycopg"):
        try:
            packages[name] = version(name)
        except Exception:
            packages[name] = None
    memsize = sysctl("hw.memsize")
    return {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": {
            "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "cores": os.cpu_count(),
            "memory_gib": round(int(memsize) / 2**30, 1) if memsize else None,
            "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
            "arch": platform.machine(),
        },
        "python": platform.python_version(),
        "packages": packages,
        "rustc": run(["rustc", "--version"]),
        "thread_env": SINGLE_THREAD,
        "data": {
            "queries": "data/substructure_sample.parquet",
            "reactants": "data/reactant_sample.csv",
        },
    }


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def summary_rows(reports: list[dict]) -> list[dict]:
    rows = []
    for r in reports:
        extras = {label: value for label, value, _ in r["extras"]}
        rows.append(
            {
                "benchmark": r["benchmark"],
                "variant": r["variant"] or "",
                "settings": "; ".join(f"{k} {v}" for k, v in r["settings"].items()),
                "queries": r["queries"],
                "reactants": r["reactants"],
                "query_prep_s": r["query_prep_seconds"],
                "index_prep_s": r["index_prep_seconds"],
                "reactant_prep_s": r["reactant_prep_seconds"],
                "matching_median_s": r["match_median_seconds"],
                "matching_runs_s": " ".join(f"{t:.4f}" for t in r["match_seconds"]),
                "end_to_end_s": r["end_to_end_seconds"],
                "matches": r["matches"],
                **{label: extras.get(label) for label in SCREEN_COLUMNS},
                "digest": r["digest"],
            }
        )
    return rows


def write_markdown(
    path: Path, meta: dict, rows: list[dict], failures: list[str]
) -> None:
    m = meta["machine"]
    pk = meta["packages"]
    digests = {row["digest"] for row in rows if row["digest"]}
    main = [row for row in rows if row["benchmark"] != "09_rust_native_fp"]
    sweep = sorted(
        (row for row in rows if row["benchmark"] == "09_rust_native_fp"),
        key=lambda row: row["matching_median_s"],
    )
    lines = [
        "# Single-thread benchmark run",
        "",
        f"- **Date:** {meta['date']}",
        f"- **Machine:** {m['cpu']}, {m['cores']} cores, {m['memory_gib']} GiB, "
        f"{m['os']} ({m['arch']})",
        f"- **Software:** Python {meta['python']}, RDKit {pk['rdkit']}, "
        f"NumPy {pk['numpy']}, pyroaring {pk['pyroaring']}, {meta['rustc']}",
        "- **Threads:** one thread of work per benchmark, run one after another; "
        + ", ".join(f"`{k}=1`" for k in meta["thread_env"])
        + ". No CPU pinning (macOS).",
        "- **Data:** 50,000 SMARTS queries x 1,000 reactants = 50M pairs.",
        "- **Match set:** "
        + (
            f"every report returned the same matches, digest `{digests.pop()[:16]}`."
            if len(digests) == 1
            else f"**{len(digests)} different digests** — check the logs."
        ),
        "",
        "Times are seconds. Prep steps are outside the matching timer; matching is the "
        "median of the runs; end to end is prep plus median matching. "
        '"Exact matches" counts the pairs sent to the exact matcher.',
        "",
        "| benchmark | variant | query prep | index prep | reactant prep | matching | "
        "end to end | matches | exact matches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main:
        lines.append(
            f"| {row['benchmark']} | {row['variant']} | {fmt(row['query_prep_s'])} | "
            f"{fmt(row['index_prep_s'])} | {fmt(row['reactant_prep_s'])} | "
            f"**{fmt(row['matching_median_s'])}** | {fmt(row['end_to_end_s'])} | "
            f"{fmt(row['matches'])} | {fmt(row['exact matches run'])} |"
        )
    lines += ["", "Settings per report:", ""]
    for row in main:
        name = row["benchmark"] + (f" ({row['variant']})" if row["variant"] else "")
        lines.append(f"- {name}: {row['settings'] or 'defaults'}")
    if sweep:
        lines += [
            "",
            f"## Benchmark 09 sweep ({len(sweep)} configurations, fastest first)",
            "",
            "| configuration | index prep | matching | exact matches |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in sweep:
            lines.append(
                f"| {row['variant']} | {fmt(row['index_prep_s'])} | "
                f"{fmt(row['matching_median_s'], 4)} | {fmt(row['exact matches run'])} |"
            )
    if failures:
        lines += ["", "## Failed", ""] + [f"- {f}" for f in failures]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n")[2:]),
    )
    parser.add_argument("--only", nargs="+", metavar="KEY", help="e.g. 01 05 postgres")
    parser.add_argument("--skip-postgres", action="store_true")
    parser.add_argument(
        "--out", type=Path, help="default: benchmarks/results/<timestamp>"
    )
    args = parser.parse_args()

    selected = [
        b
        for b in BENCHMARKS
        if (not args.only or b[0] in args.only)
        and not (args.skip_postgres and b[0] == "postgres")
    ]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or ROOT / "benchmarks" / "results" / f"{stamp}-single-thread"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    env = {**os.environ, **SINGLE_THREAD, "BENCH_RESULTS": str(results)}
    meta = metadata()
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    failures = []
    for key, description, commands in selected:
        start = time.perf_counter()
        print(f"[{key}] {description} ...", end=" ", flush=True)
        log = out / "logs" / f"{key}.log"
        ok = True
        with log.open("w") as handle:
            for command in commands:
                handle.write(f"$ {' '.join(command)}\n")
                handle.flush()
                done = subprocess.run(
                    [*UV, *command], cwd=ROOT, env=env, stdout=handle, stderr=handle
                )
                if done.returncode:
                    ok = False
                    failures.append(
                        f"{key}: `{' '.join(command)}` exited {done.returncode}"
                    )
                    break
        if key == "postgres":
            # Stop, do not remove: the volume and any earlier data stay.
            subprocess.run(
                ["docker", "compose", "-f", "benchmarks/postgres/compose.yaml", "stop"],
                cwd=ROOT,
                capture_output=True,
            )
        print(f"{'ok' if ok else 'FAILED'} ({time.perf_counter() - start:.0f} s)")

    reports = (
        [json.loads(line) for line in results.read_text().splitlines()]
        if results.exists()
        else []
    )
    rows = summary_rows(reports)
    if rows:
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_markdown(out / "summary.md", meta, rows, failures)
    print(f"\n{len(reports)} reports -> {out / 'summary.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

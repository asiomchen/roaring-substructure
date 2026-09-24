"""Substructure search in the RDKit PostgreSQL cartridge, one phase per run.

Runs the same task as benchmarks/01-03: for 1,000 reactant molecules, find which of
50,000 substructure SMARTS they contain. Here the substructures live in a qmol column
with a GiST index, and the cartridge screens them with the same Pattern fingerprint the
in-process benchmarks compute by hand -- gmol_compress stores
PatternFingerprintMol(mol, rdkit.sss_fp_size), and gmol_consistent sets recheck, so the
exact SubstructMatch still runs on every candidate.

The database stores the queries and nothing else: one table, substructures(idx, qmol).
The reactants are not loaded -- each query carries its own, as mol_from_smiles(), so
there is no second table to keep in sync and no JOIN.

Phases, in order (the database persists between them in the compose volume):

  setup    pull the image, start the server, create the extension, and assert the
           settings that decide whether the database can agree with Python
  load     store the 50,000 substructures as qmol
  index    CREATE INDEX ... USING gist (qmol)
  match    the search -- the number to compare with 01-03
  verify   check the match set against a screen-free Python recomputation, and prove the
           phase ran the plan it claims

Re-running load drops the table and its index, so run load before index.

The qmol traps this works around. Every one of these substructures is a query SMARTS and
100% of them fail MolFromSmiles, yet no text path yields a query molecule: text -> mol is
the documented implicit cast to SMILES, and casting text to qmol parses SMILES as well
(verified against this build -- it raises "could not create molecule from SMILES" on a
SMARTS query feature). qmol_from_smarts(cstring) is the only route, and it takes cstring
rather than the text the current RDKit sources declare. SMARTS therefore go to a staging
text column by COPY and are converted explicitly; the staging table is dropped once the
loaded graphs have been checked against the text they came from. qmol_from_smarts also
drops tetrahedral chirality markers and the brackets around atoms left with no query
features, which the check tolerates and the verify phase shows is harmless.

Usage:
  uv run benchmarks/postgres/bench.py --phase setup
  uv run benchmarks/postgres/bench.py --phase load
  uv run benchmarks/postgres/bench.py --phase load \
    --queries-file data/ord_reactant_substructures.parquet --distinct-queries
  uv run benchmarks/postgres/bench.py --phase index
  uv run benchmarks/postgres/bench.py --phase match
  uv run benchmarks/postgres/bench.py --phase match --connections 10
  uv run benchmarks/postgres/bench.py --phase match --max-parallel-workers 4 \
    --max-parallel-workers-per-gather 4
  uv run benchmarks/postgres/bench.py --phase verify     # needs no database
"""

import argparse
import csv
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datamol as dm
import pandas as pd
import psycopg
from rdkit import RDLogger
from tqdm import tqdm

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = Path(__file__).resolve().parent / "compose.yaml"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
REACTANTS_CSV = ROOT / "data" / "reactant_sample.csv"
SUBSTRUCTURES_PARQUET = ROOT / "data" / "substructure_sample.parquet"

# Must match the value compose.yaml started the server with: the fingerprint size is baked
# into the index signature, so a mismatch between what was asked for and what the server
# reports would silently measure something other than intended.
EXPECTED_FP_SIZE = int(os.environ.get("SSS_FP_SIZE", "2048"))
EXPECTED_CHIRAL = os.environ.get("DO_CHIRAL_SSS", "off")
INDEX_NAME = "substructures_qmol_idx"

# `qmol <@ mol` is the indexed direction; a qmol can never be the target side.
#
# The scalar subquery is not decoration. It makes mol_from_smiles() an InitPlan evaluated
# once per query. Written inline -- qmol <@ mol_from_smiles(%s) -- the call sits in the
# qual and is re-evaluated for every tuple the index recheck touches. As long as the
# statement is planned from the parameter values that is invisible, but once it is
# prepared and planned generically it costs 443 redundant parses per query: 23.3 ms per
# reactant against 7.2 ms, measured on the first 200 reactants.
MATCH_SQL = "SELECT idx FROM substructures WHERE qmol <@ (SELECT mol_from_smiles(%s))"

WARMUP_REACTANTS = 50
REFERENCE_REACTANTS = 100  # how many reactants `verify` recomputes in Python


def connect():
    return psycopg.connect(
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=int(os.environ.get("PGPORT", "55434")),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
        dbname=os.environ.get("PGDATABASE", "postgres"),
        autocommit=True,
    )


def compose(*args):
    subprocess.run(["docker", "compose", "-f", str(COMPOSE), *args], check=True)


def explain_text(cur, sql, params=()):
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {sql}", params)
    return "\n".join(row[0] for row in cur.fetchall())


def normalized(column):
    """A SMARTS with the two things qmol_from_smarts discards removed: tetrahedral
    chirality markers, and the brackets around an atom left with no query features."""
    return (
        f"regexp_replace(regexp_replace({column}, '@+', '', 'g'),"
        r" '\[([A-Z][a-z]?)\]', '\1', 'g')"
    )


# --------------------------------------------------------------------------- phases


def phase_setup(_args):
    print(f"compose        : {COMPOSE}")
    compose("pull")  # its own step, so pull time is visible and never counted
    try:
        compose("up", "-d", "--wait")
    except subprocess.CalledProcessError:
        raise SystemExit(
            "the server did not come up -- if postgres rejected the rdkit.* settings on\n"
            "its command line, the rdkit library is not in shared_preload_libraries"
        )

    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS rdkit")
        cur.execute("SELECT rdkit_version()")
        version = cur.fetchone()[0]
        cur.execute("SHOW server_version")
        server = cur.fetchone()[0]
        cur.execute(
            "SELECT current_setting('rdkit.sss_fp_size', true),"
            "       current_setting('rdkit.do_chiral_sss', true),"
            "       current_setting('rdkit.do_enhanced_stereo_sss', true),"
            "       current_setting('shared_buffers'),"
            "       current_setting('work_mem'),"
            "       current_setting('synchronous_commit'),"
            "       current_setting('full_page_writes')"
        )
        fp_size, chiral, enhanced, shared, work_mem, sync_commit, fpw = cur.fetchone()

    print(f"server         : PostgreSQL {server}")
    print(f"rdkit version  : {version}")
    print(f"sss_fp_size    : {fp_size}")
    print(f"do_chiral_sss  : {chiral}")
    print(f"enhanced stereo: {enhanced}")
    print(
        f"postgres conf  : shared_buffers={shared} work_mem={work_mem}"
        f" synchronous_commit={sync_commit} full_page_writes={fpw}"
    )

    # The server must have started with what compose.yaml was asked to pass it, or the
    # numbers describe something other than the configuration under test.
    problems = []
    if fp_size is None:
        problems.append("rdkit.* settings are not visible -- is the library preloaded?")
    elif int(fp_size) != EXPECTED_FP_SIZE:
        problems.append(
            f"sss_fp_size is {fp_size}, but the server was started with "
            f"SSS_FP_SIZE={EXPECTED_FP_SIZE}"
        )
    if chiral != EXPECTED_CHIRAL:
        problems.append(
            f"do_chiral_sss is {chiral}, but the server was started with "
            f"DO_CHIRAL_SSS={EXPECTED_CHIRAL}"
        )
    if enhanced != "off":
        problems.append(f"enhanced stereo is {enhanced}, expected off")
    if problems:
        for problem in problems:
            print(f"  ! {problem}")
        raise SystemExit("the server did not start with the settings asked for")

    # Deviating from Python's defaults is allowed -- it is what a tuning sweep does -- but
    # the match set is then expected to differ from the in-process reference.
    if int(fp_size) != 2048:
        print(f"  note: sss_fp_size {fp_size} is not the default 2048")
    if chiral != "off":
        print(
            f"  note: do_chiral_sss is {chiral}, so matches may differ from the "
            f"non-chiral in-process reference"
        )

    print("server ready -- run --phase load")


def phase_load(args):
    substructures = pd.read_parquet(args.queries_file)["substructure"]
    if args.distinct_queries:
        substructures = substructures.drop_duplicates(ignore_index=True)
    print(f"substructures  : {len(substructures):,}")

    start = time.time()
    with connect() as conn, conn.cursor() as cur:
        # COPY appends: without the drops, a second run silently doubles the table.
        cur.execute("DROP TABLE IF EXISTS substructures")
        cur.execute("DROP TABLE IF EXISTS sub_staging")
        cur.execute("CREATE TABLE substructures (idx int PRIMARY KEY, qmol qmol)")
        cur.execute("CREATE TABLE sub_staging (idx int, smarts text)")

        with cur.copy("COPY sub_staging (idx, smarts) FROM STDIN") as copy:
            for idx, smarts in enumerate(substructures):
                copy.write_row((idx, smarts))

        cur.execute(
            "INSERT INTO substructures (idx, qmol)"
            " SELECT idx, qmol_from_smarts(smarts::cstring) FROM sub_staging"
        )

        # qmol has no equality operator, but mol_to_smarts() reads it back, so the stored
        # graphs can be checked against the SMARTS they came from while the text is still
        # here. A NULL qmol would come back as a mismatch, so this covers that too.
        cur.execute(
            f"SELECT count(*) FROM substructures s JOIN sub_staging t USING (idx)"
            f" WHERE {normalized('t.smarts')} IS DISTINCT FROM mol_to_smarts(s.qmol)::text"
        )
        mangled = cur.fetchone()[0]
        if mangled:
            cur.execute(
                f"SELECT t.idx, {normalized('t.smarts')}, mol_to_smarts(s.qmol)::text"
                f" FROM substructures s JOIN sub_staging t USING (idx)"
                f" WHERE {normalized('t.smarts')} IS DISTINCT FROM"
                f" mol_to_smarts(s.qmol)::text LIMIT 3"
            )
            for idx, before, after in cur.fetchall():
                print(f"  ! row {idx}: {before}\n           -> {after}")

        cur.execute("SELECT count(*) FROM sub_staging WHERE smarts ~ '@'")
        chiral = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM substructures")
        loaded = cur.fetchone()[0]
        cur.execute("DROP TABLE sub_staging")
    elapsed = time.time() - start

    print(f"Load took {elapsed} seconds.")
    print(
        f"  stored       : {loaded:,} queries, {mangled} not matching the SMARTS loaded"
    )
    print(f"  chirality    : {chiral} substructures carried @ markers, dropped by qmol")

    if loaded != len(substructures) or mangled:
        raise SystemExit(
            f"load check failed: {loaded} rows, {mangled} not round-tripping"
        )


def phase_index(_args):
    start = time.time()
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        cur.execute(f"CREATE INDEX {INDEX_NAME} ON substructures USING gist (qmol)")
        cur.execute("ANALYZE substructures")
        cur.execute(
            f"SELECT pg_size_pretty(pg_relation_size('{INDEX_NAME}')),"
            "       pg_size_pretty(pg_relation_size('substructures')),"
            "       pg_size_pretty(pg_total_relation_size('substructures'))"
        )
        index_size, heap_size, total_size = cur.fetchone()
    elapsed = time.time() - start

    print(f"Index build took {elapsed} seconds.")
    print(f"  index        : {index_size} ({INDEX_NAME})")
    print(f"  heap         : {heap_size} (substructures, {total_size} with indexes)")


def set_parallel_settings(cur, args):
    """Apply the requested session-level parallel-worker limits."""
    if args.max_parallel_workers is not None:
        cur.execute(
            "SELECT set_config('max_parallel_workers', %s, false)",
            (str(args.max_parallel_workers),),
        )
    cur.execute(
        "SELECT set_config('max_parallel_workers_per_gather', %s, false)",
        (str(args.max_parallel_workers_per_gather),),
    )
    if args.min_parallel_table_scan_size is not None:
        cur.execute(
            "SELECT set_config('min_parallel_table_scan_size', %s, false)",
            (args.min_parallel_table_scan_size,),
        )


def match_concurrent(smiles_list, connections, args):
    """Run the searches over `connections` client connections, each with a slice
    of the reactants. Connections are opened before the timer starts and each gets the
    session setting the serial path uses, so the only thing being measured is the
    concurrency."""
    workers = [connect() for _ in range(connections)]
    for worker in workers:
        set_parallel_settings(worker.cursor(), args)
    chunks = [list(range(i, len(smiles_list), connections)) for i in range(connections)]

    def run_chunk(worker, chunk):
        cur = worker.cursor()
        found = []
        for idx in chunk:
            cur.execute(MATCH_SQL, (smiles_list[idx],))
            for (sidx,) in cur.fetchall():
                found.append((idx, sidx))
        return found

    try:
        start = time.time()
        with ThreadPoolExecutor(connections) as pool:
            results = list(
                tqdm(
                    pool.map(run_chunk, workers, chunks),
                    total=connections,
                    desc="chunks",
                    unit="chunk",
                )
            )
        elapsed = time.time() - start
    finally:
        for worker in workers:
            worker.close()
    return [pair for chunk in results for pair in chunk], elapsed


def phase_match(args):
    smiles_list = pd.read_csv(REACTANTS_CSV)["smiles"]
    n = len(smiles_list)
    connections = max(1, args.connections)

    with connect() as conn, conn.cursor() as cur:
        set_parallel_settings(cur, args)
        for smiles in tqdm(
            smiles_list[: min(WARMUP_REACTANTS, n)], desc="warm-up", unit="reactant"
        ):
            cur.execute(MATCH_SQL, (smiles,))
            cur.fetchall()

    if connections == 1:
        with connect() as conn, conn.cursor() as cur:
            set_parallel_settings(cur, args)
            matches = []
            start = time.time()
            for idx, smiles in enumerate(
                tqdm(smiles_list, desc="reactants", unit="reactant")
            ):
                cur.execute(MATCH_SQL, (smiles,))
                for (sidx,) in cur.fetchall():
                    matches.append((idx, sidx))
            elapsed = time.time() - start
    else:
        matches, elapsed = match_concurrent(smiles_list, connections, args)

    # One representative query, for the plan evidence verify checks: the reactant with the
    # median number of hits, which is more typical than the first one.
    counts = Counter(idx for idx, _ in matches)
    median_idx = sorted(counts, key=counts.get)[len(counts) // 2]
    with connect() as conn, conn.cursor() as cur:
        set_parallel_settings(cur, args)
        plan = explain_text(cur, MATCH_SQL, (smiles_list[median_idx],))

    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "matches.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "sidx"])
        writer.writerows(sorted(matches))
    (ARTIFACTS / "explain.txt").write_text(
        f"-- {MATCH_SQL}  -- idx = {median_idx} ({counts[median_idx]} hits)\n\n{plan}\n"
    )

    label = f" ({connections} connections)" if connections > 1 else ""
    print(f"Matching took {elapsed} seconds{label}.")
    print(f"  matches      : {len(matches):,} over {n:,} reactants")
    print("  plan         : artifacts/explain.txt")
    if INDEX_NAME not in plan:
        print(
            f"  ! the plan does not mention {INDEX_NAME} -- this phase may have "
            f"seq-scanned; check the EXPLAIN before trusting the number"
        )


def phase_verify(_args):
    failures = []

    def check(ok, message):
        print(f"  {'ok  ' if ok else 'FAIL'} {message}")
        if not ok:
            failures.append(message)

    with open(ARTIFACTS / "matches.csv") as handle:
        reader = csv.reader(handle)
        next(reader)
        matches = sorted((int(a), int(b)) for a, b in reader)
    print(f"match set      : {len(matches):,} matches")

    print("plan")
    plan = (ARTIFACTS / "explain.txt").read_text()
    check(
        INDEX_NAME in plan and "Index Scan" in plan,
        f"the match phase used {INDEX_NAME}",
    )

    # Ground truth, screen-free: 03's loop, so the reference does not itself depend on a
    # fingerprint screen being sound. Keyed by (reactant idx, substructure idx) in source
    # row order -- the same keys the database phase wrote.
    print(f"reference (03's screen-free loop, first {REFERENCE_REACTANTS} reactants)")
    reactants = pd.read_csv(REACTANTS_CSV)["smiles"]
    sub_mols = [
        dm.convert.from_smarts(smarts)
        for smarts in pd.read_parquet(SUBSTRUCTURES_PARQUET)["substructure"]
    ]
    failed = sum(mol is None for mol in sub_mols)
    if failed:
        raise SystemExit(f"failed to parse {failed} substructures")

    reference = set()
    for idx, smiles in enumerate(tqdm(reactants, total=REFERENCE_REACTANTS)):
        if idx >= REFERENCE_REACTANTS:
            break
        mol = dm.to_mol(smiles)
        for sidx, sub_mol in enumerate(sub_mols):
            if mol.HasSubstructMatch(sub_mol):
                reference.add((idx, sidx))

    window = {(idx, sidx) for idx, sidx in matches if idx < REFERENCE_REACTANTS}
    check(
        window == reference,
        f"the database agrees with the reference ({len(reference):,} matches over "
        f"{REFERENCE_REACTANTS} reactants)",
    )
    if window != reference:
        print(
            f"       reference-only {len(reference - window)}, "
            f"database-only {len(window - reference)}"
        )

    print()
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed")
    print("all checks passed")


PHASES = {
    "setup": phase_setup,
    "load": phase_load,
    "index": phase_index,
    "match": phase_match,
    "verify": phase_verify,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", required=True, choices=list(PHASES))
    parser.add_argument(
        "--connections",
        type=int,
        default=1,
        help="connections to spread the match phase over; 1 searches one "
        "reactant at a time (default 1)",
    )
    parser.add_argument(
        "--max-parallel-workers",
        type=int,
        default=None,
        help="session max_parallel_workers (default: leave the server setting unchanged)",
    )
    parser.add_argument(
        "--max-parallel-workers-per-gather",
        type=int,
        default=0,
        help="session max_parallel_workers_per_gather (default: 0)",
    )
    parser.add_argument(
        "--min-parallel-table-scan-size",
        default=None,
        help="session min_parallel_table_scan_size (default: server setting)",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=SUBSTRUCTURES_PARQUET,
        help=f"substructure parquet used by the load phase (default: {SUBSTRUCTURES_PARQUET})",
    )
    parser.add_argument(
        "--distinct-queries",
        action="store_true",
        help="drop duplicate SMARTS before loading the substructure table",
    )
    args = parser.parse_args()
    PHASES[args.phase](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Same as 03_roaring_approach.py, but the reactants are spread over worker processes
# instead of being searched one after another. This is the in-process counterpart to
# benchmarks/postgres/bench.py --phase match --connections N: both leave a single core
# idle no longer, and both report matches as (reactant index, substructure index) pairs in
# source row order, so the two can be compared.
#
# The corpus is built once at import and handed to the workers by fork, rather than
# re-parsed by each or pickled across. On a platform without fork each worker rebuilds it
# on import, which is why the work sits behind a main() guard.

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import datamol as dm
import pandas as pd
from pyroaring import BitMap
from rdkit.Chem.rdmolops import PatternFingerprint
from tqdm import auto as tqdm

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.report import Report, digest_pairs, print_report, screen_extras

START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"


def to_pattern(mol):
    fp = PatternFingerprint(mol)
    # cover the case where the molecule is None
    return BitMap(fp.GetOnBits())


def load_corpus():
    """Query molecules and fingerprints, and the seconds spent on each."""
    start = time.perf_counter()
    sub_df = pd.read_parquet("data/substructure_sample.parquet")
    sub_df["mol"] = sub_df["substructure"].apply(dm.convert.from_smarts)
    failures = sub_df["mol"].isna().sum()

    if failures > 0:
        raise ValueError(f"Failed to parse {failures} substructures.")
    parsed = time.perf_counter()

    sub_df["pattern"] = sub_df["mol"].apply(to_pattern)
    seconds = (parsed - start, time.perf_counter() - parsed)
    return list(sub_df["mol"]), list(sub_df["pattern"]), seconds


SUB_MOLS, SUB_PATS, PREP_SECONDS = load_corpus()
SUB_INDEX = range(len(SUB_MOLS))


def match_chunk(chunk):
    """One worker's share of the reactants: a list of (reactant index, smiles, mol)."""
    matches = []
    candidates = 0
    for idx, smiles, r_mol in chunk:
        r_fp = to_pattern(r_mol)
        for sidx, s_mol, s_fp in zip(SUB_INDEX, SUB_MOLS, SUB_PATS):
            if not s_fp.issubset(r_fp):
                continue
            candidates += 1
            if r_mol.HasSubstructMatch(s_mol):
                matches.append((idx, sidx))
    return matches, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count(),
        help="worker processes to spread the reactants over (default: every core)",
    )
    args = parser.parse_args()
    processes = max(1, args.processes)

    start = time.perf_counter()
    reactants_df = pd.read_csv("data/reactant_sample.csv")
    reactants_df["mol"] = reactants_df["smiles"].apply(dm.to_mol)
    reactant_prep = time.perf_counter() - start
    jobs = [
        (idx, smiles, mol)
        for idx, (smiles, mol) in enumerate(
            zip(reactants_df["smiles"], reactants_df["mol"])
        )
    ]
    chunks = [jobs[i::processes] for i in range(processes)]

    context = mp.get_context(START_METHOD)
    with context.Pool(processes) as pool:
        start_time = time.perf_counter()
        results = list(
            tqdm.tqdm(
                pool.imap(match_chunk, chunks),
                total=len(chunks),
                desc="chunks",
                unit="chunk",
            )
        )
        end_time = time.perf_counter()

    matches = [match for found, _ in results for match in found]
    candidates = sum(count for _, count in results)
    print_report(
        Report(
            benchmark="04_roaring_multiprocessing",
            queries=len(SUB_MOLS),
            reactants=len(jobs),
            match_seconds=[end_time - start_time],
            matches=len(matches),
            digest=digest_pairs(matches),
            query_prep_seconds=PREP_SECONDS[0],
            index_prep_seconds=PREP_SECONDS[1],
            reactant_prep_seconds=reactant_prep,
            settings={"processes": processes},
            notes={
                "query": "read + parse SMARTS",
                "index": "Pattern fingerprints as Roaring BitMaps",
                "reactant": "read + parse SMILES",
                "matching": f"03's loop over {processes} forked processes",
            },
            extras=screen_extras(
                len(SUB_MOLS) * len(jobs), None, candidates, len(matches)
            ),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

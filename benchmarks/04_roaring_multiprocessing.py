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

import datamol as dm
import pandas as pd
from pyroaring import BitMap
from rdkit.Chem.rdmolops import PatternFingerprint
from tqdm import auto as tqdm

START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"


def to_pattern(mol):
    fp = PatternFingerprint(mol)
    # cover the case where the molecule is None
    return BitMap(fp.GetOnBits())


def load_corpus():
    sub_df = pd.read_parquet("data/substructure_sample.parquet")
    sub_df["mol"] = sub_df["substructure"].apply(dm.convert.from_smarts)
    failures = sub_df["mol"].isna().sum()

    if failures > 0:
        raise ValueError(f"Failed to parse {failures} substructures.")

    sub_df["pattern"] = sub_df["mol"].apply(to_pattern)
    return list(sub_df["mol"]), list(sub_df["pattern"])


SUB_MOLS, SUB_PATS = load_corpus()
SUB_INDEX = range(len(SUB_MOLS))


def match_chunk(chunk):
    """One worker's share of the reactants: a list of (reactant index, smiles, mol)."""
    matches = []
    for idx, smiles, r_mol in chunk:
        r_fp = to_pattern(r_mol)
        for sidx, s_mol, s_fp in zip(SUB_INDEX, SUB_MOLS, SUB_PATS):
            if not s_fp.issubset(r_fp):
                continue
            if r_mol.HasSubstructMatch(s_mol):
                matches.append((idx, sidx))
    return matches


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

    reactants_df = pd.read_csv("data/reactant_sample.csv")
    reactants_df["mol"] = reactants_df["smiles"].apply(dm.to_mol)
    jobs = [
        (idx, smiles, mol)
        for idx, (smiles, mol) in enumerate(
            zip(reactants_df["smiles"], reactants_df["mol"])
        )
    ]
    chunks = [jobs[i::processes] for i in range(processes)]

    context = mp.get_context(START_METHOD)
    with context.Pool(processes) as pool:
        start_time = time.time()
        results = list(
            tqdm.tqdm(
                pool.imap(match_chunk, chunks),
                total=len(chunks),
                desc="chunks",
                unit="chunk",
            )
        )
        end_time = time.time()

    matches = [match for chunk in results for match in chunk]
    print(f"Matching took {end_time - start_time} seconds ({processes} processes).")
    print(f"  matches      : {len(matches):,} over {len(jobs):,} reactants")
    return 0


if __name__ == "__main__":
    sys.exit(main())

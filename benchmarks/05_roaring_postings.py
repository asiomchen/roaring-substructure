"""Substructure search with Roaring posting lists over Pattern fingerprint bits.

Each fingerprint bit maps to the IDs of queries that contain it. A query can only
match a reactant if none of its bits are absent from the reactant fingerprint. Union
postings for selected absent bits, remove those IDs, and run the full fingerprint
screen and exact RDKit match on the remaining queries. The final screen makes the
answer independent of how many postings are selected.

Index construction is outside the matching timer, as in the other benchmarks.
"""

import argparse
import sys
import time
from pathlib import Path

import datamol as dm
import pandas as pd

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.query_index import FP_SIZE, SubstructureQueryIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postings",
        type=int,
        default=128,
        help="number of absent-bit postings to union per reactant (default: 128)",
    )
    args = parser.parse_args()
    if not 1 <= args.postings <= FP_SIZE:
        parser.error(f"--postings must be between 1 and {FP_SIZE}")

    smarts = pd.read_parquet("data/substructure_sample.parquet")["substructure"]
    query_mols = [dm.convert.from_smarts(value) for value in smarts]
    if any(mol is None for mol in query_mols):
        raise ValueError("Failed to parse substructure SMARTS")
    query_mols = [mol for mol in query_mols if mol is not None]
    index = SubstructureQueryIndex.from_mols(query_mols)
    smiles = pd.read_csv("data/reactant_sample.csv")["smiles"]
    reactant_mols = [dm.to_mol(value) for value in smiles]
    orig_len = len(reactant_mols)
    reactant_mols = [mol for mol in reactant_mols if mol is not None]
    if len(reactant_mols) < orig_len:
        raise ValueError("Some reactant SMILES could not be parsed into molecules")
    if any(mol is None for mol in reactant_mols):
        raise ValueError("Failed to parse reactant SMILES")

    start = time.perf_counter()
    matches = index.match_many(reactant_mols, args.postings)
    elapsed = time.perf_counter() - start
    print(f"Matching took {elapsed:.3f} seconds ({args.postings} postings).")
    print(
        f"  matches      : {sum(map(len, matches)):,} over {len(reactant_mols):,} reactants"
    )
    return matches


if __name__ == "__main__":
    res = main()

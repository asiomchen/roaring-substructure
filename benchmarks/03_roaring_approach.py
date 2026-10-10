# Same as 02_pattern.py, but the pattern fingerprints are roaring bitmaps
# instead of rdkit bit vectors.
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

prep_start = time.perf_counter()
sub_df = pd.read_parquet("data/substructure_sample.parquet")
sub_df["mol"] = sub_df["substructure"].apply(dm.convert.from_smarts)
failures = sub_df["mol"].isna().sum()

if failures > 0:
    raise ValueError(f"Failed to parse {failures} substructures.")
query_prep = time.perf_counter() - prep_start


def to_pattern(mol):
    fp = PatternFingerprint(mol)
    return BitMap(fp.GetOnBits())


prep_start = time.perf_counter()
sub_df["pattern"] = sub_df["mol"].apply(to_pattern)

# The inner loop reads these columns 50M times, so they are materialised as lists first:
# iterating a pandas Series costs 3.7x what iterating a list does, which is about 3.3 s
# over the 50M pairs. 04_roaring_multiprocessing.py does the same, so the two line up.
sub_smiles = list(sub_df["substructure"])
sub_mols = list(sub_df["mol"])
sub_pats = list(sub_df["pattern"])
index_prep = time.perf_counter() - prep_start

prep_start = time.perf_counter()
reactants_df = pd.read_csv("data/reactant_sample.csv")
reactants_df["mol"] = reactants_df["smiles"].apply(dm.to_mol)
reactant_rows = list(zip(reactants_df["smiles"], reactants_df["mol"]))
reactant_prep = time.perf_counter() - prep_start


def match_all():
    """The timed loop. It is a function so its variables are locals: CPython resolves
    module-level names through the globals dict instead, and over these 50M pairs a bare
    loop at module scope measures 0.8 s slower than the same loop in a function."""
    matches = []
    candidates = 0
    for r_smiles, r_mol in tqdm.tqdm(reactant_rows):
        r_fp = to_pattern(r_mol)
        for s_smiles, s_mol, s_fp in zip(sub_smiles, sub_mols, sub_pats):
            if not s_fp.issubset(r_fp):
                continue
            candidates += 1
            if r_mol.HasSubstructMatch(s_mol):
                matches.append((r_smiles, s_smiles))
    return matches, candidates


start_time = time.perf_counter()
matches, candidates = match_all()
end_time = time.perf_counter()

# SMILES and SMARTS are unique in the samples, so they map back to row indices.
reactant_ids = {smiles: idx for idx, (smiles, _) in enumerate(reactant_rows)}
query_ids = {smarts: idx for idx, smarts in enumerate(sub_smiles)}
print_report(
    Report(
        benchmark="03_roaring_approach",
        queries=len(sub_mols),
        reactants=len(reactant_rows),
        match_seconds=[end_time - start_time],
        matches=len(matches),
        digest=digest_pairs((reactant_ids[r], query_ids[q]) for r, q in matches),
        query_prep_seconds=query_prep,
        index_prep_seconds=index_prep,
        reactant_prep_seconds=reactant_prep,
        notes={
            "query": "read + parse SMARTS",
            "index": "Pattern fingerprints as Roaring BitMaps",
            "reactant": "read + parse SMILES",
            "matching": "reactant fingerprints + BitMap.issubset on every pair + exact match",
        },
        extras=screen_extras(
            len(sub_mols) * len(reactant_rows), None, candidates, len(matches)
        ),
    )
)

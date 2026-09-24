# Pattern-fingerprint screening with RDKit bit vectors.
import time

import datamol as dm
import pandas as pd
from  rdkit.DataStructs.cDataStructs import AllProbeBitsMatch
from rdkit.Chem.rdmolops import PatternFingerprint
from tqdm import auto as tqdm

sub_df = pd.read_parquet("data/substructure_sample.parquet")
sub_df["mol"] = sub_df["substructure"].apply(dm.convert.from_smarts)
failures = sub_df["mol"].isna().sum()

if failures > 0:
    raise ValueError(f"Failed to parse {failures} substructures.")


def to_pattern(mol):
    fp = PatternFingerprint(mol)
    return fp


sub_df["pattern"] = sub_df["mol"].apply(to_pattern)

# The inner loop reads these columns 50M times, so they are materialised as lists first:
# iterating a pandas Series costs 3.7x what iterating a list does, which is about 3.3 s
# over the 50M pairs. 02 and 03 do the same, so the three line up.
sub_smiles = list(sub_df["substructure"])
sub_mols = list(sub_df["mol"])
sub_pats = list(sub_df["pattern"])

reactants_df = pd.read_csv("data/reactant_sample.csv")
reactants_df["mol"] = reactants_df["smiles"].apply(dm.to_mol)
reactant_rows = list(zip(reactants_df["smiles"], reactants_df["mol"]))


def match_all():
    matches = []
    for r_smiles, r_mol in tqdm.tqdm(reactant_rows):
        r_fp = to_pattern(r_mol)
        for s_smiles, s_mol, s_fp in zip(sub_smiles, sub_mols, sub_pats):
            if not AllProbeBitsMatch(s_fp, r_fp):
                continue
            if r_mol.HasSubstructMatch(s_mol):
                matches.append((r_smiles, s_smiles))
    return matches


start_time = time.time()
matches = match_all()
end_time = time.time()
print(f"Matching took {end_time - start_time} seconds.")

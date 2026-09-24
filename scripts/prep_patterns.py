import datamol as dm
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem.rdmolops import PatternFingerprint


def to_pattern(mol):
    fp = PatternFingerprint(mol)
    # cover the case where the molecule is None
    fp_base = np.array([0] * fp.GetNumBits(), dtype=np.bool)
    DataStructs.ConvertToNumpyArray(fp, fp_base)
    return fp_base


df = pd.read_parquet("data/substructure_sample.parquet")

df["mol"] = df["substructure"].apply(dm.convert.from_smarts)
failures = df["mol"].isna().sum()
if failures > 0:
    raise ValueError(f"Failed to parse {failures} substructures.")
patterns = df["mol"].apply(to_pattern)
df["pattern"] = patterns
df.drop(columns=["mol"], inplace=True)
df.to_parquet("data/substructure_patterns.parquet", index=False)

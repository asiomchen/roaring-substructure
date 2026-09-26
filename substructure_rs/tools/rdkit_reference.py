"""Write RDKit reference data for checking `substructure_rs` against RDKit.

Outputs, in OUT_DIR:
  atoms.txt  one line per reactant, in the format of `substructure_rs --dump-atoms`
  pairs.bin  (reactant index, query index) little-endian u32 pairs of RDKit matches,
             for `substructure_rs --reference`

Matches come from benchmark 07's index: RDKit's pattern-fingerprint screen, then
HasSubstructMatch. That screen can reject true matches (e.g. arynes), which then
show up as `extra` in the comparison; confirm them with HasSubstructMatch directly.

    uv run substructure_rs/tools/rdkit_reference.py OUT_DIR \
        [--reactants-file data/reactant_sample.csv] \
        [--queries-file data/substructure_sample.parquet]
"""

import argparse
import hashlib
import struct
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.rust_process_query_index import RustProcessSubstructureQueryIndex

RDLogger.DisableLog("rdApp.*")


def atom_line(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    atoms = ";".join(
        f"{a.GetAtomicNum()},{int(a.GetIsAromatic())},{a.GetTotalNumHs(True)},"
        f"{a.GetDegree()},{a.GetFormalCharge()}"
        for a in mol.GetAtoms()
    )
    bonds = ";".join(
        f"{min(b.GetBeginAtomIdx(), b.GetEndAtomIdx())}-"
        f"{max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())}-"
        f"{int(b.GetBondTypeAsDouble() * 2)}"
        for b in mol.GetBonds()
    )
    return f"{atoms}|{bonds}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--reactants-file", type=Path, default=Path("data/reactant_sample.csv")
    )
    parser.add_argument(
        "--queries-file", type=Path, default=Path("data/substructure_sample.parquet")
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    smiles = pd.read_csv(args.reactants_file)["smiles"].tolist()
    with Pool() as pool:
        lines = pool.map(atom_line, smiles, chunksize=2000)
    (args.out_dir / "atoms.txt").write_text("\n".join(lines) + "\n")

    queries = [
        Chem.MolFromSmarts(s)
        for s in pd.read_parquet(args.queries_file)["substructure"]
    ]
    reactants = [Chem.MolFromSmiles(s) for s in smiles]
    index = RustProcessSubstructureQueryIndex.from_mols(queries)
    query_ids = {id(mol): idx for idx, mol in enumerate(index.query_mols)}
    pairs = [
        (reactant_idx, query_ids[id(mol)])
        for reactant_idx, found in enumerate(index.match_many(reactants))
        for mol in found
    ]
    np.array(pairs, dtype="<u4").reshape(-1, 2).tofile(args.out_dir / "pairs.bin")
    hasher = hashlib.sha256()
    for pair in pairs:
        hasher.update(struct.pack("<II", *pair))
    print(f"{len(pairs):,} matches; digest {hasher.hexdigest()}")


if __name__ == "__main__":
    main()

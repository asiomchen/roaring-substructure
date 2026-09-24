"""Write every distinct reactant molecule of the mapped reactions to a CSV.

Reads the mapped reactions written by `scripts/fetch_ord.py` and takes their reactant-side
components - the first field of the reaction SMILES, so agents and products are left out -
writing each one as a plain SMILES with the atom mapping stripped. The result is a set of
compounds rather than reactions, so the compounds can be looked up or drawn on their own
without carrying the mapping along.

Only components carrying an atom mapping are kept. A component with no mapped atom never
enters the product - it is a reagent, solvent or counterion sitting in the reactant slot -
so it is not a reactant, and keeping it would fill the CSV with the same few common
reagents the whole set over.

Reactants are deduplicated on the unmapped canonical structure, so a building block
reached through many reactions contributes a single row.

Each compound is identified by the ORD reaction it came from plus its position in that
reaction's reactant list, e.g. `ord-56b1f4bfeebc4b8ab990b9804e798aa7_1`. Since one row
covers a whole structure, that id names only the first place the compound occurs - it is
provenance, not a structural key. A row of the mapped file can itself stand for several
ORD reactions that share a reaction SMILES, so the id uses the first of them.

Output: CSV with `id` and `smiles`, one row per distinct reactant, in mapped-file order.

Usage:
    uv run scripts/list_reactants.py
"""

import argparse
import csv
import sys
import warnings

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def unmapped(smiles):
    """Strip the atom map numbers from one reactant component.

    Returns a (smiles, reason) pair with reason None on success, so the caller can report
    why components were skipped. An empty string parses to an empty molecule rather than
    None, hence the atom check.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "unparseable"
    if not any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
        return None, "not_a_reactant"
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/ord_mapped.parquet")
    parser.add_argument("--output", default="data/unique_reactants.csv")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)

    rows = []
    seen = set()
    occurrences = 0
    reasons = {"not_a_reactant": 0, "unparseable": 0}

    for reaction_ids, mapped in tqdm(
        zip(frame["reaction_ids"], frame["mapped_reaction_smiles"]),
        total=len(frame),
        desc="reactants",
        unit="rxn",
    ):
        reactant_side = mapped.split(">")[0]
        # The index is the component's real position on the reactant side, so it does not
        # shift when a reagent is filtered out.
        for index, component in enumerate(reactant_side.split(".")):
            if not component:
                continue
            occurrences += 1
            smiles, reason = unmapped(component)
            if reason is not None:
                reasons[reason] += 1
                continue
            if smiles in seen:
                continue
            seen.add(smiles)
            # The id names the first occurrence only; later ones collapse into this row.
            rows.append((f"{reaction_ids[0]}_{index}", smiles))

    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "smiles"])
        writer.writerows(rows)

    print()
    print(f"mapped reactions read  : {len(frame):,}")
    print(f"reactant occurrences   : {occurrences:,}")
    print(f"distinct reactants     : {len(rows):,}")
    for reason, count in reasons.items():
        print(f"  {reason:<20}: {count:,}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

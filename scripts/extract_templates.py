"""Extract reactant-side reaction substructures from mapped ORD reactions (pipeline step 2).

Reads the mapped reactions written by `scripts/fetch_ord.py` and, per reaction:
    1. Template extraction via reaction-utils (`ChemicalReaction.generate_reaction_template`),
       which wraps a modified rdChiral extractor.
    2. Split the template's reactant side into its individual substructures and drop the
       atom mapping, leaving plain substructure SMARTS.

Nothing is deduplicated: every substructure occurrence becomes its own row, so a substructure
shared by two reactions appears twice and the same fragment on one reactant side appears once
per `reactant_index`. Aggregation is left to whatever reads the output, and
`scripts/list_reactants.py` is the sibling step that lists the reactant compounds themselves.

Output: parquet with `substructure`, `reaction_ids` and `reactant_index` (0-based position of
the substructure on the template's reactant side). `reaction_ids` is the collapsed list of
ORD reaction ids built by the fetch step, so a row covers every ORD reaction the input
reaction stands for.

Usage:
    uv run scripts/extract_templates.py --limit 500      # quick trial
    uv run scripts/extract_templates.py                  # full run
"""

import argparse
import os
import sys
import warnings

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rxnutils.chem.reaction import ChemicalReaction
from tqdm import tqdm

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

TEMPLATE_KWARGS = {"radius": 2, "expand_ring": True, "expand_hetero": True}


def reactant_substructures(template_smarts):
    """Split an atom-mapped template into unmapped reactant-side substructures.

    Returns (reactant_index, smarts) pairs in template order. Uses RDKit's own reaction
    parser rather than splitting on "." so that the reactant/agent/product boundaries are
    the ones RDKit actually recorded.
    """
    reaction = AllChem.ReactionFromSmarts(template_smarts)
    if reaction is None:
        return []
    reaction.Initialize()

    substructures = []
    for index in range(reaction.GetNumReactantTemplates()):
        mol = Chem.Mol(reaction.GetReactantTemplate(index))
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        smarts = Chem.MolToSmarts(mol)
        if smarts:
            substructures.append((index, smarts))
    return substructures


def extract_substructures(mapped):
    """Return the reactant substructures of one mapped reaction.

    Returns a (substructures, reason) pair where reason is None on success, so the caller
    can report why records were dropped.
    """
    try:
        chemical = ChemicalReaction(mapped)
    except Exception:
        return None, "reaction_failed"

    try:
        canonical, _retro = chemical.generate_reaction_template(**TEMPLATE_KWARGS)
    except Exception:
        return None, "template_failed"

    return reactant_substructures(canonical.smarts), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/ord_mapped.parquet")
    parser.add_argument("--output", default="data/ord_reactant_substructures.parquet")
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N reactions"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="parallel worker processes; 1 runs in-process without joblib",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    if args.limit:
        frame = frame.head(args.limit)
    reaction_ids = frame["reaction_ids"].tolist()
    mapped = frame["mapped_reaction_smiles"].tolist()
    total = len(mapped)
    print(
        f"extracting from {total:,} mapped reactions with {args.workers} workers",
        flush=True,
    )

    # One row per substructure occurrence, appended in input order so output is deterministic.
    rows = []
    reasons = {"reaction_failed": 0, "template_failed": 0}

    # `return_as="generator"` yields results as they finish but still in submission order.
    with tqdm(total=total, desc="extracting", unit="rxn", file=sys.stdout) as bar:
        results = Parallel(n_jobs=args.workers, return_as="generator")(
            delayed(extract_substructures)(smi) for smi in mapped
        )
        for done, (found, reason) in enumerate(results, start=1):
            if reason is not None:
                reasons[reason] += 1
            else:
                ids = reaction_ids[done - 1]
                for reactant_index, smarts in found:
                    rows.append((smarts, ids, reactant_index))
            bar.set_postfix_str(f"substructures={len(rows):,}")
            bar.update()

    out = pd.DataFrame(rows, columns=["substructure", "reaction_ids", "reactant_index"])
    out.to_parquet(args.output, compression="zstd", index=False)

    extracted = total - sum(reasons.values())
    print()
    print(f"reactions processed    : {total:,}")
    print(f"  templates extracted  : {extracted:,} ({extracted / total:.1%})")
    for reason, count in reasons.items():
        print(f"  {reason:<21}: {count:,}")
    print(f"substructure rows      : {len(out):,}")
    if len(out):
        print(f"  distinct substructures: {out['substructure'].nunique():,}")
        print(
            f"  ORD reactions covered : {sum(len(ids) for ids in out['reaction_ids']):,}"
        )
        print(
            f"  reactant indices      : {sorted(out['reactant_index'].unique().tolist())}"
        )
    if extracted:
        print(f"  substructures/template: {len(out) / extracted:.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())

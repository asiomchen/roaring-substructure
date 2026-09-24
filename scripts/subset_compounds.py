"""Draw a random subset of the reactant and substructure sets (pipeline side step).

Reads the two sets the pipeline derives - `data/unique_reactants.csv` from
`scripts/list_reactants.py` and `data/ord_reactant_substructures.parquet` from
`scripts/extract_templates.py` - and writes a random sample of each. Drawing is without
replacement, so X reactants give X distinct compounds.

The substructure set is deduplicated on `substructure` before drawing. That file holds one
row per occurrence, so its 1.8M rows cover far fewer distinct fragments than they suggest,
and drawing from the raw rows would repeat them. A drawn substructure keeps the provenance
of the first row it occurred in - the same convention `scripts/list_reactants.py` uses for
a compound reached through several reactions - so the id is where it was found, not a
structural key.

Each output mirrors its source: CSV with `id,smiles` for the reactants, parquet with
`substructure`, `reaction_ids`, `reactant_index` for the substructures.

Usage:
    uv run scripts/subset_compounds.py --reactants 1000 --substructures 1000
"""

import argparse
import random
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def draw(frame, n, rng):
    """`n` rows of `frame` chosen without replacement, returned in source order.

    Asking for more than there are yields the whole frame rather than an error, so the
    caller can report the shortfall.
    """
    if n >= len(frame):
        return frame.reset_index(drop=True)
    return frame.iloc[sorted(rng.sample(range(len(frame)), n))].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactants-in", default="data/unique_reactants.csv")
    parser.add_argument(
        "--substructures-in", default="data/ord_reactant_substructures.parquet"
    )
    parser.add_argument("--reactants-out", default="data/reactant_sample.csv")
    parser.add_argument(
        "--substructures-out", default="data/substructure_sample.parquet"
    )
    parser.add_argument(
        "--reactants", type=int, default=1000, help="how many reactants to draw"
    )
    parser.add_argument(
        "--substructures", type=int, default=1000, help="how many substructures to draw"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # One generator so the two draws do not share an index sequence.
    rng = random.Random(args.seed)

    reactions = pd.read_csv(args.reactants_in)
    distinct_reactants = reactions.drop_duplicates(subset="smiles")
    drawn_reactants = draw(distinct_reactants, args.reactants, rng)
    drawn_reactants.to_csv(args.reactants_out, index=False)

    substructures = pd.read_parquet(args.substructures_in)
    distinct_substructures = substructures.drop_duplicates(subset="substructure")
    drawn_substructures = draw(distinct_substructures, args.substructures, rng)
    drawn_substructures.to_parquet(
        args.substructures_out, compression="zstd", index=False
    )

    print()
    print(f"reactants available    : {len(distinct_reactants):,} distinct")
    print(f"  drawn                : {len(drawn_reactants):,}")
    print(f"substructures available: {len(distinct_substructures):,} distinct")
    print(f"  drawn                : {len(drawn_substructures):,}")
    print()
    print(f"wrote {args.reactants_out}")
    print(f"wrote {args.substructures_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch the Open Reaction Database `uspto-grants` config as atom-mapped reactions (pipeline step 1).

The `uspto-grants` config of `open-reaction-database/ord-data` holds reactions extracted
from USPTO granted patents. They arrive already atom-atom mapped, so the pipeline has no
mapping step: this file is the mapped set, and no Indigo run is needed.

The mapping is read from the record's own stored REACTION_SMILES identifier, which carries
the reactant>agent>product split as well as the map numbers. The record's components are not
used: every one of them is annotated with the REACTANT role, so rebuilding the split from
those roles would pull the whole reagent basket into the reactant slot.

A mapping is kept only if it holds together - see `mapping_is_sound`. Records that fail are
dropped and the summary reports how many.

Output: parquet with `reaction_ids` and `mapped_reaction_smiles`. Identical mapped SMILES
are collapsed into a single row that carries every ORD reaction id sharing them.

Usage:
    uv run scripts/fetch_ord.py --limit 500      # quick trial
    uv run scripts/fetch_ord.py                  # full run
"""

import argparse
import re
import sys
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from ord_schema import message_helpers
from ord_schema.proto import reaction_pb2
from rdkit import Chem, RDLogger
from tqdm import tqdm

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

REPO = "open-reaction-database/ord-data"
CONFIG = "uspto-grants"
OUT_PATH = "data/ord_mapped.parquet"

# ~2.9 KB per record, so the reaction column is never materialised in one piece.
BATCH_SIZE = 20_000

# Identifier types whose value is a reaction SMILES.
SMILES_IDENTIFIER_TYPES = (
    reaction_pb2.ReactionIdentifier.REACTION_SMILES,
    reaction_pb2.ReactionIdentifier.REACTION_CXSMILES,
)


def config_files(readme: str, config_name: str) -> list[str]:
    """Extract the data_files of one config from the dataset card frontmatter."""
    frontmatter = readme.split("---")[1]
    files: list[str] = []
    in_config = False
    for line in frontmatter.splitlines():
        if line.startswith("- config_name:"):
            in_config = line.split(":", 1)[1].strip() == config_name
        elif in_config:
            match = re.search(r"data/\S+\.parquet", line)
            if match:
                files.append(match.group(0))
    return files


def stored_reaction_smiles(message: reaction_pb2.Reaction) -> str | None:
    """The record's own reaction SMILES with any CXSMILES extension stripped, or None.

    The stored string already holds the split the submitter recorded and carries the atom
    map numbers, so it is taken as it is. Roughly half the records append a CXSMILES
    extension recording fragment grouping and atom values. The map numbers live in the
    SMILES part, so the extension carries nothing the mapping needs, and RDKit parses but
    ignores the grouping it describes; leaving it on would only make the value unparseable
    as plain SMILES. Do not try to preserve it.
    """
    for identifier in message.identifiers:
        if identifier.type in SMILES_IDENTIFIER_TYPES:
            return message_helpers.split_cxsmiles_extension(identifier.value)[0]
    return None


def mapping_is_sound(smiles: str) -> bool:
    """Whether the map numbers of a reaction SMILES describe a usable mapping.

    A sound mapping sends every product atom back to exactly one reactant atom, so both
    sides must carry the same map numbers with no repeats. Reactant atoms are allowed to be
    left unmapped - a leaving group, or a reagent that does not incorporate, has nowhere to
    go - so requiring the reactant side to be complete would reject most of the set. At
    least one reactant atom must be mapped for the reaction to be mapped at all.
    """
    fields = smiles.split(">")
    if len(fields) != 3:
        return False
    reactants, _agents, products = fields

    reactant_mol = Chem.MolFromSmiles(reactants)
    product_mol = Chem.MolFromSmiles(products)
    if reactant_mol is None or product_mol is None:
        return False

    reactant_mapped = [
        a.GetAtomMapNum() for a in reactant_mol.GetAtoms() if a.GetAtomMapNum()
    ]
    product_mapped = [
        a.GetAtomMapNum() for a in product_mol.GetAtoms() if a.GetAtomMapNum()
    ]

    # Every product atom has to come from somewhere, and no map number may name two atoms.
    if len(product_mapped) != product_mol.GetNumAtoms():
        return False
    if not reactant_mapped:
        return False
    if len(set(reactant_mapped)) != len(reactant_mapped):
        return False
    if len(set(product_mapped)) != len(product_mapped):
        return False

    # The mapped reactant atoms and the product atoms have to be the same set of atoms.
    return sorted(reactant_mapped) == sorted(product_mapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUT_PATH)
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N records"
    )
    args = parser.parse_args()

    readme = open(hf_hub_download(REPO, "README.md", repo_type="dataset")).read()
    files = config_files(readme, CONFIG)
    if not files:
        print(f"config {CONFIG!r} not found in the dataset card")
        return 1
    readers = [
        pq.ParquetFile(hf_hub_download(REPO, path, repo_type="dataset"))
        for path in files
    ]
    total = sum(reader.metadata.num_rows for reader in readers)
    if args.limit is not None:
        total = min(total, args.limit)
    print(
        f"config {CONFIG}: {len(files)} file(s), {total:,} records to read", flush=True
    )

    # Keyed by mapped SMILES: duplicates collapse into one row holding every id that shares
    # them. Dict order keeps the output deterministic (first occurrence wins).
    by_smiles: dict[str, list[str]] = {}
    read = 0
    unparseable = 0
    no_mapping = 0
    unsound = 0

    with tqdm(total=total, desc="fetching", unit="rxn", file=sys.stdout) as bar:
        for reader in readers:
            batches = reader.iter_batches(
                batch_size=BATCH_SIZE, columns=["reaction_id", "reaction"]
            )
            for batch in batches:
                ids = batch.column("reaction_id").to_pylist()
                raws = batch.column("reaction").to_pylist()
                # `total` is the ceiling, so this trims the last batch under --limit and is
                # a no-op otherwise.
                remaining = total - read
                ids, raws = ids[:remaining], raws[:remaining]

                for reaction_id, raw in zip(ids, raws):
                    read += 1
                    message = reaction_pb2.Reaction()
                    try:
                        message.ParseFromString(raw)
                    except Exception:
                        unparseable += 1
                        continue

                    smiles = stored_reaction_smiles(message)
                    if not smiles:
                        no_mapping += 1
                        continue
                    if not mapping_is_sound(smiles):
                        unsound += 1
                        continue

                    by_smiles.setdefault(smiles, []).append(reaction_id)

                bar.set_postfix_str(f"kept={len(by_smiles):,}")
                bar.update(len(ids))
                if read >= total:
                    break
            if read >= total:
                break

    out = pa.table(
        {
            "reaction_ids": pa.array(
                list(by_smiles.values()), type=pa.list_(pa.string())
            ),
            "mapped_reaction_smiles": list(by_smiles),
        }
    )
    pq.write_table(out, args.output, compression="zstd")

    kept = sum(len(ids) for ids in by_smiles.values())
    print()
    print(f"records read           : {read:,}")
    print(f"  unparseable          : {unparseable:,}")
    print(f"  no stored mapping    : {no_mapping:,}")
    print(f"  unsound mapping      : {unsound:,} ({unsound / max(read, 1):.1%})")
    print(f"records kept           : {kept:,}")
    print(f"distinct mapped SMILES : {len(by_smiles):,}")
    print(f"  collapsed duplicates : {kept - len(by_smiles):,}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

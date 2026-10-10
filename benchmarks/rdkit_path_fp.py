"""Benchmark 08's screening fingerprint, computed from RDKit molecules.

A Python port of `substructure_rs/src/fingerprint.rs` at its default kind, `paths4` at
4096 bits: hashed labelled atoms and paths of up to 4 bonds, with repeated features
counted up to 4 times. The bits differ from the Rust ones (the hashes are Python's), but
the features are the same, so it screens the same way.

The one rule that keeps it sound: a query only emits a feature when its SMARTS *states*
every label in it, and a target emits the same feature from its actual atoms and bonds.
So `query_bits <= target_bits` holds for every true match. Unlike RDKit's
`PatternFingerprint`, which hashes elements and bond orders only, this keeps the H count,
degree and charge that reaction-template SMARTS such as `[C&H2&D2&+0]` state, which is
where benchmark 08's extra selectivity comes from.

Labels per atom, `None` when a query leaves them open:
- `z`: element; `z_arom`: element and aromaticity;
- `full`: element, aromaticity, degree (`D`), total H (`H`) and charge;
- `z_degree`, `z_h`, `z_charge`: element with one of those.
Labels per bond: `class` (single and aromatic share one) and `exact` (bond type).

Features:
- every atom label on its own (views 1-6);
- every simple path of 1-4 bonds, once per labelling: elements only (10), elements and
  bond class (11), element+aromaticity and exact bond (12), and fully labelled atoms
  with exact bonds up to 2 bonds (13).

A path is keyed by the smaller of its label sequence and that sequence reversed, so it
hashes the same from either end. A query's facts are read only from the top-level AND
chain of each atom's query; anything under OR or NOT is ignored, which only drops
features and never makes the screen wrong. The match semantics are RDKit's: `H` is
`GetTotalNumHs(includeNeighbors=True)`, `D` is `GetDegree()`.

Author: Marcin Kowiel + Claude
"""

import re
from collections import Counter

from rdkit.Chem.rdchem import Atom, Bond, BondType, Mol

FP_SIZE = 4096
MAX_EDGES = 4
MAX_FULL_EDGES = 2
COUNT_CAP = 4

AROMATIC_TYPE = 1000  # RDKit's AtomType value is z + 1000 for aromatic atoms
NEGATED = re.compile(r"^not |!=| ! | not in ")

# Index of each atom label in the label tuple.
Z, Z_AROM, FULL, Z_DEGREE, Z_H, Z_CHARGE = range(6)
# Index of each bond label in the label pair.
CLASS, EXACT = range(2)
# Path labellings: (view, atom label, bond label); `None` as the bond label means the
# bond is not labelled at all (the topology-and-elements view).
PATH_VIEWS = ((10, Z, None), (11, Z, CLASS), (12, Z_AROM, EXACT), (13, FULL, EXACT))


def _atom_labels(z, aromatic, degree, h, charge):
    """The six atom labels from whichever properties are known."""

    def pair(value, scale, offset=0):
        return None if z is None or value is None else z * scale + value + offset

    full = None
    if None not in (z, aromatic, degree, h, charge):
        full = (
            (z << 16)
            | (aromatic << 15)
            | ((degree & 0xF) << 10)
            | ((h & 0xF) << 5)
            | ((charge + 8) & 0x1F)
        )
    return (
        z,
        pair(None if aromatic is None else int(aromatic), 2),
        full,
        pair(degree, 64),
        pair(h, 64),
        pair(charge, 64, 16),
    )


def _target_atom(atom: Atom):
    return _atom_labels(
        atom.GetAtomicNum(),
        atom.GetIsAromatic(),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True),
        atom.GetFormalCharge(),
    )


def _query_atom(atom: Atom):
    """Labels from what the atom's top-level AND chain states."""
    known = {}
    depth_and = 0  # deepest level reached only through non-negated ANDs
    for line in atom.DescribeQuery().splitlines():
        if not line.strip():
            continue
        depth = (len(line) - len(line.lstrip(" "))) // 2
        if depth > depth_and:
            continue  # under an OR, a NOT, or a leaf: not stated for the atom
        body = line.strip()
        depth_and = depth
        if NEGATED.search(body):
            continue
        descr, *rest = body.split(" ")
        if descr == "AtomAnd":
            depth_and = depth + 1
            continue
        value = int(rest[0]) if rest and rest[0].lstrip("-").isdigit() else None
        if descr == "AtomType":
            known["z"] = value % AROMATIC_TYPE
            known["aromatic"] = value >= AROMATIC_TYPE
        elif descr == "AtomAtomicNum":
            known["z"] = value
        elif descr == "AtomIsAromatic":
            known["aromatic"] = True
        elif descr == "AtomIsAliphatic":
            known["aromatic"] = False
        elif descr == "AtomHCount":
            known["h"] = value
        elif descr == "AtomExplicitDegree":
            known["degree"] = value
        elif descr == "AtomFormalCharge":
            known["charge"] = value
    return _atom_labels(
        known.get("z"),
        known.get("aromatic"),
        known.get("degree"),
        known.get("h"),
        known.get("charge"),
    )


def _bond_class(bond_type):
    return 1 if bond_type in (BondType.SINGLE, BondType.AROMATIC) else int(bond_type)


def _target_bond(bond: Bond):
    bond_type = bond.GetBondType()
    return _bond_class(bond_type), int(bond_type)


QUERY_BONDS = {
    "BondOrder 1 = val": (1, int(BondType.SINGLE)),
    "BondOrder 2 = val": (2, int(BondType.DOUBLE)),
    "BondOrder 3 = val": (3, int(BondType.TRIPLE)),
    "BondOrder 12 = val": (1, int(BondType.AROMATIC)),
    "SingleOrAromaticBond 1 = val": (1, None),
}


def _query_bond(bond: Bond):
    """(class, exact) as far as the bond query fixes them; anything else is open."""
    return QUERY_BONDS.get(bond.DescribeQuery().strip(), (None, None))


def _graph(mol: Mol):
    atoms = [
        _query_atom(a) if a.HasQuery() else _target_atom(a) for a in mol.GetAtoms()
    ]
    nbrs = [[] for _ in atoms]
    for bond in mol.GetBonds():
        labels = _query_bond(bond) if bond.HasQuery() else _target_bond(bond)
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        nbrs[a].append((b, labels))
        nbrs[b].append((a, labels))
    return atoms, nbrs


def features(mol: Mol, max_edges=MAX_EDGES, max_full_edges=MAX_FULL_EDGES):
    """Feature keys of `mol`, one per occurrence."""
    atoms, nbrs = _graph(mol)
    keys = [
        (view, label)
        for labels in atoms
        for view, label in enumerate(labels, 1)
        if label is not None
    ]

    def extend(path, runs):
        edges = len(path)  # bonds in the path once it grows by one atom
        if edges > max_edges:
            return
        start, last = path[0], path[-1]
        for nxt, bond in nbrs[last]:
            if nxt in path:
                continue
            atom = atoms[nxt]
            grown = []
            for (view, atom_label, bond_label), run in zip(PATH_VIEWS, runs):
                b = 0 if bond_label is None else bond[bond_label]
                a = atom[atom_label]
                too_long = view == 13 and edges > max_full_edges
                open_label = run is None or b is None or a is None
                grown.append(None if open_label or too_long else run + (b, a))
            # An open label stays open, so no longer path can emit anything either.
            if not any(grown):
                continue
            if start < nxt:
                for (view, _, _), seq in zip(PATH_VIEWS, grown):
                    if seq is not None:
                        keys.append((view, edges, min(seq, seq[::-1])))
            path.append(nxt)
            extend(path, grown)
            path.pop()

    for start, atom in enumerate(atoms):
        runs = [
            None if atom[atom_label] is None else (atom[atom_label],)
            for _, atom_label, _ in PATH_VIEWS
        ]
        extend([start], runs)
    return keys


def path_fingerprint(mol: Mol, fp_size: int = FP_SIZE) -> list[int]:
    """Sorted on-bits: the n-th copy of a feature, n below `COUNT_CAP`, sets one bit.

    Query molecules (from SMARTS) and targets (from SMILES) go through the same
    function: atoms and bonds that carry a query are labelled from what it states.
    """
    bits = set()
    for key, count in Counter(features(mol)).items():
        for copy in range(min(count, COUNT_CAP)):
            bits.add(hash((key, copy)) % fp_size)
    return sorted(bits)

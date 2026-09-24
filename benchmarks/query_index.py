"""Reusable Roaring index for substructure query molecules."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Self

from pyroaring import BitMap
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import PatternFingerprint

FP_SIZE = 2048


def pattern(mol: Mol) -> BitMap:
    return BitMap(PatternFingerprint(mol, FP_SIZE).GetOnBits())


@dataclass(frozen=True, slots=True)
class SubstructureQueryIndex(Mapping[int, BitMap]):
    """Query data and a bit-to-posting mapping.

    Iteration yields fingerprint bits by decreasing posting size. ``index[bit]``
    returns the query IDs containing that bit, and ``len(index)`` counts bits.
    """

    query_mols: tuple[Mol, ...]
    query_fps: tuple[BitMap, ...]
    postings: tuple[BitMap, ...]
    ranked_bits: tuple[int, ...]
    all_ids: BitMap

    def __iter__(self) -> Iterator[int]:
        return iter(self.ranked_bits)

    def __getitem__(self, bit: int) -> BitMap:
        if not isinstance(bit, int) or not 0 <= bit < len(self.postings):
            raise KeyError(bit)
        return self.postings[bit]

    def __len__(self) -> int:
        return len(self.postings)

    def match(self, reactant_mol: Mol, posting_limit: int = 128) -> list[Mol]:
        """Return the indexed query molecules matching one reactant."""
        if not 1 <= posting_limit <= len(self):
            raise ValueError(f"posting_limit must be between 1 and {len(self)}")
        if reactant_mol is None:
            raise ValueError("reactant molecule is None")

        reactant_fp = pattern(reactant_mol)
        excluded = BitMap()
        selected = 0
        for bit in self:
            if bit not in reactant_fp:
                excluded |= self[bit]
                selected += 1
                if selected == posting_limit:
                    break

        remaining = self.all_ids - excluded
        matches = []
        for query_idx in remaining:
            if self.query_fps[query_idx].issubset(reactant_fp):
                if reactant_mol.HasSubstructMatch(self.query_mols[query_idx]):
                    matches.append(self.query_mols[query_idx])
        return matches

    def match_many(
        self, reactant_mols: Iterable[Mol], posting_limit: int = 128
    ) -> list[list[Mol]]:
        """Return one list of matching query molecules per reactant, in input order."""
        if not 1 <= posting_limit <= len(self):
            raise ValueError(f"posting_limit must be between 1 and {len(self)}")
        return [
            self.match(reactant_mol, posting_limit) for reactant_mol in reactant_mols
        ]

    @classmethod
    def from_mols(cls, mols: Iterable[Mol]) -> Self:
        query_mols = tuple(mols)
        if any(mol is None for mol in query_mols):
            raise ValueError("Failed to parse substructure SMARTS")

        postings = [BitMap() for _ in range(FP_SIZE)]
        query_fps = []
        for idx, mol in enumerate(query_mols):
            fp = pattern(mol)
            query_fps.append(fp)
            for bit in fp:
                postings[bit].add(idx)

        # Frequent absent bits remove the most query IDs first.
        ranked_bits = sorted(
            range(FP_SIZE), key=lambda bit: len(postings[bit]), reverse=True
        )
        return cls(
            query_mols,
            tuple(query_fps),
            tuple(postings),
            tuple(ranked_bits),
            BitMap(range(len(query_mols))),
        )

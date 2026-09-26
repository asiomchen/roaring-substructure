"""Substructure query index whose posting screen runs in the Rust extension."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Self

import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import PatternFingerprint

from benchmarks.query_index import FP_SIZE
from rust_index import PostingIndex


def on_bits(mol: Mol) -> list[int]:
    return list(PatternFingerprint(mol, FP_SIZE).GetOnBits())


@dataclass(frozen=True, slots=True)
class RustSubstructureQueryIndex:
    """NumpySubstructureQueryIndex with the posting and fingerprint screen in Rust.

    The screen runs in parallel over reactants; RDKit's exact match stays in Python.
    """

    query_mols: tuple[Mol, ...]
    postings: PostingIndex

    def __len__(self) -> int:
        return len(self.postings)

    def screen_many(
        self, reactant_mols: Iterable[Mol], posting_limit: int = 128
    ) -> list[np.ndarray]:
        """Return fingerprint-screened query IDs per reactant."""
        if not 1 <= posting_limit <= len(self):
            raise ValueError(f"posting_limit must be between 1 and {len(self)}")
        bits = []
        for mol in reactant_mols:
            if mol is None:
                raise ValueError("reactant molecule is None")
            bits.append(on_bits(mol))
        offsets, ids = self.postings.screen_many(bits, posting_limit)
        offsets = np.frombuffer(offsets, dtype="<u8")
        ids = np.frombuffer(ids, dtype="<u4")
        return [ids[a:b] for a, b in pairwise(offsets)]

    def match_many(
        self, reactant_mols: Iterable[Mol], posting_limit: int = 128
    ) -> list[list[Mol]]:
        reactant_mols = list(reactant_mols)
        query_mols = self.query_mols
        return [
            [
                query_mols[query_idx]
                for query_idx in candidates.tolist()
                if reactant_mol.HasSubstructMatch(query_mols[query_idx])
            ]
            for reactant_mol, candidates in zip(
                reactant_mols, self.screen_many(reactant_mols, posting_limit)
            )
        ]

    def match(self, reactant_mol: Mol, posting_limit: int = 128) -> list[Mol]:
        return self.match_many([reactant_mol], posting_limit)[0]

    @classmethod
    def from_mols(cls, mols: Iterable[Mol]) -> Self:
        query_mols = tuple(mols)
        if any(mol is None for mol in query_mols):
            raise ValueError("Failed to parse substructure SMARTS")
        return cls(query_mols, PostingIndex([on_bits(mol) for mol in query_mols]))

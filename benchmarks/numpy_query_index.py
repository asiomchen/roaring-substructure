"""Substructure query index with packed NumPy uint64 fingerprint and posting arrays."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import PatternFingerprint

from benchmarks.query_index import FP_SIZE

WORD_BITS = 64


def pattern(mol: Mol) -> tuple[NDArray[np.bool_], NDArray[np.uint64]]:
    present = np.zeros(FP_SIZE, dtype=np.bool_)
    present[list(PatternFingerprint(mol, FP_SIZE).GetOnBits())] = True
    packed = np.packbits(present, bitorder="little").view(np.uint64)
    return present, packed


@dataclass(frozen=True, slots=True)
class NumpySubstructureQueryIndex(Mapping[int, NDArray[np.uint64]]):
    """The benchmark 5 index with packed NumPy arrays in place of Roaring bitmaps."""

    query_mols: tuple[Mol, ...]
    query_fps: NDArray[np.uint64]
    postings: NDArray[np.uint64]
    ranked_bits: tuple[int, ...]
    all_ids: NDArray[np.uint64]

    def __iter__(self) -> Iterator[int]:
        return iter(self.ranked_bits)

    def __getitem__(self, bit: int) -> NDArray[np.uint64]:
        if not isinstance(bit, int) or not 0 <= bit < len(self.postings):
            raise KeyError(bit)
        return self.postings[bit]

    def __len__(self) -> int:
        return len(self.postings)

    def match(self, reactant_mol: Mol, posting_limit: int = 128) -> list[Mol]:
        if not 1 <= posting_limit <= len(self):
            raise ValueError(f"posting_limit must be between 1 and {len(self)}")
        if reactant_mol is None:
            raise ValueError("reactant molecule is None")

        present, reactant_fp = pattern(reactant_mol)
        selected = []
        for bit in self:
            if not present[bit]:
                selected.append(bit)
                if len(selected) == posting_limit:
                    break

        excluded = (
            np.bitwise_or.reduce(self.postings[selected], axis=0)
            if selected
            else np.zeros_like(self.all_ids)
        )
        remaining = self.all_ids & ~excluded
        candidate_ids = np.flatnonzero(
            np.unpackbits(remaining.view(np.uint8), bitorder="little")
        )
        candidate_fps = self.query_fps[candidate_ids]
        screened_ids = candidate_ids[
            np.all((candidate_fps & reactant_fp) == candidate_fps, axis=1)
        ]
        return [
            self.query_mols[int(query_idx)]
            for query_idx in screened_ids
            if reactant_mol.HasSubstructMatch(self.query_mols[int(query_idx)])
        ]

    def match_many(
        self, reactant_mols: Iterable[Mol], posting_limit: int = 128
    ) -> list[list[Mol]]:
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

        word_count = (len(query_mols) + WORD_BITS - 1) // WORD_BITS
        query_fps = np.zeros((len(query_mols), FP_SIZE // WORD_BITS), dtype=np.uint64)
        postings = np.zeros((FP_SIZE, word_count), dtype=np.uint64)
        counts = np.zeros(FP_SIZE, dtype=np.int64)
        for idx, mol in enumerate(query_mols):
            bits = np.asarray(
                PatternFingerprint(mol, FP_SIZE).GetOnBits(), dtype=np.uint64
            )
            masks = np.uint64(1) << (bits % WORD_BITS)
            np.bitwise_or.at(query_fps[idx], bits // WORD_BITS, masks)
            # Each fingerprint bit occurs at most once, so these rows are distinct.
            postings[bits, idx // WORD_BITS] |= np.uint64(1) << np.uint64(
                idx % WORD_BITS
            )
            counts[bits] += 1

        ranked_bits = tuple(
            sorted(range(FP_SIZE), key=lambda bit: counts[bit], reverse=True)
        )
        all_ids = np.full(word_count, np.iinfo(np.uint64).max, dtype=np.uint64)
        if len(query_mols) % WORD_BITS:
            all_ids[-1] = (1 << (len(query_mols) % WORD_BITS)) - 1
        return cls(query_mols, query_fps, postings, ranked_bits, all_ids)

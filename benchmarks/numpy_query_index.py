"""Substructure query index with packed NumPy uint64 fingerprint and posting arrays.

The screening fingerprint is pluggable: any `fingerprint(mol, fp_size)` returning the
sorted on-bits of a query or reactant. RDKit's 2048-bit Pattern fingerprint is the
default (benchmark 5a); benchmark 5b passes benchmark 08's path fingerprint instead.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import PatternFingerprint

from benchmarks.query_index import FP_SIZE

WORD_BITS = 64

Fingerprint = Callable[[Mol, int], list[int]]


def pattern_bits(mol: Mol, fp_size: int) -> list[int]:
    return list(PatternFingerprint(mol, fp_size).GetOnBits())


def packed(
    bits: list[int], fp_size: int
) -> tuple[NDArray[np.bool_], NDArray[np.uint64]]:
    present = np.zeros(fp_size, dtype=np.bool_)
    present[bits] = True
    return present, np.packbits(present, bitorder="little").view(np.uint64)


@dataclass(frozen=True, slots=True)
class NumpySubstructureQueryIndex(Mapping[int, NDArray[np.uint64]]):
    """The benchmark 5 index with packed NumPy arrays in place of Roaring bitmaps."""

    query_mols: tuple[Mol, ...]
    query_fps: NDArray[np.uint64]
    postings: NDArray[np.uint64]
    ranked_bits: tuple[int, ...]
    all_ids: NDArray[np.uint64]
    fingerprint: Fingerprint = pattern_bits

    def __iter__(self) -> Iterator[int]:
        return iter(self.ranked_bits)

    def __getitem__(self, bit: int) -> NDArray[np.uint64]:
        if not isinstance(bit, int) or not 0 <= bit < len(self.postings):
            raise KeyError(bit)
        return self.postings[bit]

    def __len__(self) -> int:
        return len(self.postings)

    def screen(
        self, reactant_mol: Mol, posting_limit: int = 128
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """IDs left after the posting filter, and after the full fingerprint check."""
        if not 1 <= posting_limit <= len(self):
            raise ValueError(f"posting_limit must be between 1 and {len(self)}")
        if reactant_mol is None:
            raise ValueError("reactant molecule is None")

        present, reactant_fp = packed(
            self.fingerprint(reactant_mol, len(self)), len(self)
        )
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
        return candidate_ids, screened_ids

    def match(self, reactant_mol: Mol, posting_limit: int = 128) -> list[Mol]:
        _, screened_ids = self.screen(reactant_mol, posting_limit)
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
    def from_mols(
        cls,
        mols: Iterable[Mol],
        fingerprint: Fingerprint = pattern_bits,
        fp_size: int = FP_SIZE,
    ) -> Self:
        query_mols = tuple(mols)
        if any(mol is None for mol in query_mols):
            raise ValueError("Failed to parse substructure SMARTS")

        word_count = (len(query_mols) + WORD_BITS - 1) // WORD_BITS
        query_fps = np.zeros((len(query_mols), fp_size // WORD_BITS), dtype=np.uint64)
        postings = np.zeros((fp_size, word_count), dtype=np.uint64)
        counts = np.zeros(fp_size, dtype=np.int64)
        for idx, mol in enumerate(query_mols):
            bits = np.asarray(fingerprint(mol, fp_size), dtype=np.uint64)
            masks = np.uint64(1) << (bits % WORD_BITS)
            np.bitwise_or.at(query_fps[idx], bits // WORD_BITS, masks)
            # Each fingerprint bit occurs at most once, so these rows are distinct.
            postings[bits, idx // WORD_BITS] |= np.uint64(1) << np.uint64(
                idx % WORD_BITS
            )
            counts[bits] += 1

        ranked_bits = tuple(
            sorted(range(fp_size), key=lambda bit: counts[bit], reverse=True)
        )
        all_ids = np.full(word_count, np.iinfo(np.uint64).max, dtype=np.uint64)
        if len(query_mols) % WORD_BITS:
            all_ids[-1] = (1 << (len(query_mols) % WORD_BITS)) - 1
        return cls(query_mols, query_fps, postings, ranked_bits, all_ids, fingerprint)

"""Rust posting screen with RDKit's exact match spread over forked worker processes."""

import multiprocessing as mp
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from rdkit.Chem.rdchem import Mol

from benchmarks.rust_query_index import RustSubstructureQueryIndex

# Set in the parent right before forking so workers inherit them without pickling.
_query_mols: tuple[Mol, ...] = ()
_reactant_mols: list[Mol] = []
_candidates: list[np.ndarray] = []


def _match_range(bounds: tuple[int, int]) -> list[list[int]]:
    start, stop = bounds
    return [
        [
            query_idx
            for query_idx in _candidates[reactant_idx].tolist()
            if _reactant_mols[reactant_idx].HasSubstructMatch(_query_mols[query_idx])
        ]
        for reactant_idx in range(start, stop)
    ]


@dataclass(frozen=True, slots=True)
class RustProcessSubstructureQueryIndex:
    """RustSubstructureQueryIndex whose exact matches run in a fork process pool.

    HasSubstructMatch holds the GIL, so threads do not help; forked processes do.
    """

    rust: RustSubstructureQueryIndex
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    chunk_size: int = 32

    @property
    def query_mols(self) -> tuple[Mol, ...]:
        return self.rust.query_mols

    def __len__(self) -> int:
        return len(self.rust)

    def match_many(
        self, reactant_mols: Iterable[Mol], posting_limit: int = 128
    ) -> list[list[Mol]]:
        global _query_mols, _reactant_mols, _candidates
        reactant_mols = list(reactant_mols)
        candidates = self.rust.screen_many(reactant_mols, posting_limit)
        _query_mols, _reactant_mols, _candidates = (
            self.query_mols,
            reactant_mols,
            candidates,
        )
        bounds = [
            (start, min(start + self.chunk_size, len(reactant_mols)))
            for start in range(0, len(reactant_mols), self.chunk_size)
        ]
        try:
            with mp.get_context("fork").Pool(self.workers) as pool:
                chunks = pool.map(_match_range, bounds, chunksize=1)
        finally:
            _query_mols, _reactant_mols, _candidates = (), [], []
        query_mols = self.query_mols
        return [
            [query_mols[query_idx] for query_idx in found]
            for chunk in chunks
            for found in chunk
        ]

    def match(self, reactant_mol: Mol, posting_limit: int = 128) -> list[Mol]:
        return self.rust.match(reactant_mol, posting_limit)

    @classmethod
    def from_mols(cls, mols: Iterable[Mol], workers: int | None = None) -> Self:
        rust = RustSubstructureQueryIndex.from_mols(mols)
        return cls(rust) if workers is None else cls(rust, workers)

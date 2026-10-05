"""The Rust-backed indexes (benchmarks 06 and 07) return the NumPy index's matches."""

import sys
import unittest
from pathlib import Path

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.numpy_query_index import NumpySubstructureQueryIndex
from benchmarks.rust_process_query_index import RustProcessSubstructureQueryIndex
from benchmarks.rust_query_index import RustSubstructureQueryIndex

QUERIES = ["c1ccccc1", "C=O", "[#7&a]", "C#N", "[O&-]", "[C&H3&D1]-O", "c-c"]
REACTANTS = [
    "O=Cc1ccncc1",
    "CC#N",
    "C[N+](=O)[O-]",
    "COc1ccccc1",
    "c1ccccc1-c1ccccc1",
    "CCO",
]


def query_ids(index, found):
    ids = {id(mol): idx for idx, mol in enumerate(index.query_mols)}
    return [[ids[id(mol)] for mol in mols] for mols in found]


class RustIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = [Chem.MolFromSmarts(s) for s in QUERIES]
        cls.reactants = [Chem.MolFromSmiles(s) for s in REACTANTS]
        numpy_index = NumpySubstructureQueryIndex.from_mols(cls.queries)
        cls.expected = query_ids(numpy_index, numpy_index.match_many(cls.reactants))

    def test_expected_is_rdkit(self):
        for reactant, found in zip(self.reactants, self.expected):
            want = [
                i for i, q in enumerate(self.queries) if reactant.HasSubstructMatch(q)
            ]
            self.assertEqual(found, want)

    def test_rust_index(self):
        index = RustSubstructureQueryIndex.from_mols(self.queries)
        for limit in (1, 128):
            got = query_ids(index, index.match_many(self.reactants, limit))
            self.assertEqual(got, self.expected)

    def test_rust_process_index(self):
        index = RustProcessSubstructureQueryIndex.from_mols(self.queries, workers=2)
        got = query_ids(index, index.match_many(self.reactants))
        self.assertEqual(got, self.expected)

    def test_rejects_bad_posting_limit(self):
        index = RustSubstructureQueryIndex.from_mols(self.queries)
        with self.assertRaises(ValueError):
            index.match_many(self.reactants, 0)


if __name__ == "__main__":
    unittest.main()

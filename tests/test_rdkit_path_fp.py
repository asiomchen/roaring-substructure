"""Benchmark 08's fingerprint on RDKit molecules (benchmark 5b) is a lossless screen."""

import sys
import unittest
from pathlib import Path

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.numpy_query_index import NumpySubstructureQueryIndex
from benchmarks.rdkit_path_fp import path_fingerprint

QUERIES = [
    "c1ccccc1",
    "C=O",
    "[#7&a]",
    "C#N",
    "[O&-]",
    "[C&H3&D1]-O",
    "c-c",
    "[c&H0&D3&+0]:[n&H0&D2&+0]",
    "[C&H2&D2&+0]-[#8]",
    "[C,N]-O",
    "[!C]~[#6]",
    "O-C(=O)/C=C/C",
    "CC",
]
REACTANTS = [
    "O=Cc1ccncc1",
    "CC#N",
    "C[N+](=O)[O-]",
    "COc1ccccc1",
    "c1ccccc1-c1ccccc1",
    "CCO",
    "OC(=O)/C=C/C",
    "c1ccc2ncccc2c1",
]


def query_ids(index, found):
    ids = {id(mol): idx for idx, mol in enumerate(index.query_mols)}
    return [[ids[id(mol)] for mol in mols] for mols in found]


class PathFingerprintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries = [Chem.MolFromSmarts(s) for s in QUERIES]
        cls.reactants = [Chem.MolFromSmiles(s) for s in REACTANTS]

    def test_never_rejects_a_match(self):
        for r in self.reactants:
            target = set(path_fingerprint(r))
            for q in self.queries:
                if r.HasSubstructMatch(q):
                    self.assertLessEqual(set(path_fingerprint(q)), target)

    def test_stated_labels_screen_out_non_matches(self):
        # Same elements and bonds, so RDKit's Pattern fingerprint passes it; the
        # stated H count and degree do not fit methanol's carbon.
        query = Chem.MolFromSmarts("[C&H2&D2&+0]-[#8]")
        methanol = Chem.MolFromSmiles("CO")
        self.assertFalse(
            set(path_fingerprint(query)) <= set(path_fingerprint(methanol))
        )

    def test_does_not_depend_on_atom_order(self):
        for a, b in [("OCC(=O)N", "NC(=O)CO"), ("c1ccncc1C(=O)O", "OC(=O)c1cccnc1")]:
            self.assertEqual(
                path_fingerprint(Chem.MolFromSmiles(a)),
                path_fingerprint(Chem.MolFromSmiles(b)),
            )

    def test_index_matches_pattern_index(self):
        pattern = NumpySubstructureQueryIndex.from_mols(self.queries)
        paths = NumpySubstructureQueryIndex.from_mols(
            self.queries, path_fingerprint, 4096
        )
        for limit in (1, 128):
            self.assertEqual(
                query_ids(paths, paths.match_many(self.reactants, limit)),
                query_ids(pattern, pattern.match_many(self.reactants, limit)),
            )


if __name__ == "__main__":
    unittest.main()

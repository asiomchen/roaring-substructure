"""chematic_bench agrees with substructure_rs in both modes."""

import tempfile
import unittest
from pathlib import Path

import chematic_bench
import substructure_rs

QUERIES = ["c1ccccc1", "C=O", "[#7&a]", "C#N", "[O&-]"]
REACTANTS = ["O=Cc1ccncc1", "CC#N", "C[N+](=O)[O-]", "c1ccccc1", "CCO"]
# (reactant, query) pairs from RDKit's HasSubstructMatch.
EXPECTED = {(0, 1), (0, 2), (1, 3), (2, 4), (3, 0)}


class ChematicBenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.queries = tmp / "queries.csv"
        cls.queries.write_text("substructure\n" + "\n".join(QUERIES) + "\n")
        cls.reactants = tmp / "reactants.csv"
        cls.reactants.write_text("smiles\n" + "\n".join(REACTANTS) + "\n")
        cls.reference = tmp / "pairs.bin"
        cls.reference.write_bytes(
            b"".join(
                r.to_bytes(4, "little") + q.to_bytes(4, "little")
                for r, q in sorted(EXPECTED)
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_mode(self, mode):
        return chematic_bench.run(
            mode=mode,
            queries_file=self.queries,
            reactants_file=self.reactants,
            runs=1,
            reference=self.reference,
        )

    def test_modes_match_reference_and_substructure_rs(self):
        digest = substructure_rs.run(
            queries_file=self.queries, reactants_file=self.reactants, runs=1
        )["digest"]
        for mode in ("naive", "screened"):
            report = self.run_mode(mode)
            self.assertEqual(report["matches"], len(EXPECTED), mode)
            self.assertEqual(report["comparison"]["missing"], 0, mode)
            self.assertEqual(report["comparison"]["extra"], 0, mode)
            self.assertEqual(report["digest"], digest, mode)
        self.assertEqual(self.run_mode("naive")["pairs"], len(QUERIES) * len(REACTANTS))

    def test_errors_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.run_mode("fast")
        with self.assertRaises(ValueError):
            chematic_bench.run(queries_file="missing.parquet")

    def test_cli_exit_codes(self):
        self.assertEqual(chematic_bench.cli(["--help"]), 0)
        self.assertEqual(chematic_bench.cli(["--bogus"]), 1)


if __name__ == "__main__":
    unittest.main()

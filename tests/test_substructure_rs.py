"""substructure_rs works as a Python function and as a command, with the same results."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import substructure_rs

QUERIES = ["c1ccccc1", "C=O", "[#7&a]", "C#N", "[O&-]"]
REACTANTS = ["O=Cc1ccncc1", "CC#N", "C[N+](=O)[O-]", "c1ccccc1", "CCO"]
# (reactant, query) pairs from RDKit's HasSubstructMatch.
EXPECTED = {(0, 1), (0, 2), (1, 3), (2, 4), (3, 0)}


class SubstructureRsTest(unittest.TestCase):
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

    def run_fn(self, **kwargs):
        return substructure_rs.run(
            queries_file=self.queries, reactants_file=self.reactants, **kwargs
        )

    def test_run_returns_report(self):
        report = self.run_fn(runs=2, threads=2, reference=self.reference)
        self.assertEqual(report["queries"], len(QUERIES))
        self.assertEqual(report["reactants"], len(REACTANTS))
        self.assertEqual(report["matches"], len(EXPECTED))
        self.assertEqual(report["threads"], 2)
        self.assertEqual(len(report["match_seconds"]), 2)
        self.assertEqual(report["comparison"]["missing"], 0)
        self.assertEqual(report["comparison"]["extra"], 0)

    def test_repeated_calls_are_deterministic(self):
        self.assertEqual(
            self.run_fn(runs=1, threads=1)["digest"],
            self.run_fn(runs=1, threads=3)["digest"],
        )

    def test_errors_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.run_fn(postings=0)
        with self.assertRaises(ValueError):
            substructure_rs.run(queries_file="missing.parquet")

    def test_command_matches_function(self):
        digest = self.run_fn(runs=1)["digest"]
        command = Path(sys.executable).parent / "substructure_rs"
        out = subprocess.run(
            [
                command,
                "--queries-file",
                self.queries,
                "--reactants-file",
                self.reactants,
                "--runs",
                "1",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertIn(f"digest          {digest}", out)

    def test_cli_function_exit_codes(self):
        self.assertEqual(substructure_rs.cli(["--help"]), 0)
        self.assertEqual(substructure_rs.cli(["--bogus"]), 1)


if __name__ == "__main__":
    unittest.main()

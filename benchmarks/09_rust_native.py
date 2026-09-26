"""Benchmark 9: the whole pipeline in Rust, without RDKit.

Builds and runs the `substructure_rs` binary, which parses the SMARTS queries
and reactant SMILES itself, sanitizes the reactants the way RDKit does, screens
with its own posting-list fingerprint, and runs exact matching in parallel.
Its digest uses the same encoding as benchmarks 05a-07.

Any extra arguments go to the binary; see `--help`. To check the result
against RDKit, create a reference with `substructure_rs/tools/rdkit_reference.py`
and pass `--reference <dir>/pairs.bin`.
"""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = root / "substructure_rs" / "Cargo.toml"
binary = root / "substructure_rs" / "target" / "release" / "substructure_rs"


def main() -> None:
    args = sys.argv[1:]
    if "--no-build" in args:
        args.remove("--no-build")
    else:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--quiet",
                "--manifest-path",
                str(manifest),
            ],
            check=True,
        )
    sys.exit(subprocess.run([str(binary), *args], cwd=root, check=False).returncode)


if __name__ == "__main__":
    main()

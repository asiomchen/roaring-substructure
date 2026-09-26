"""Benchmark 9: the whole pipeline in Rust, without RDKit.

Runs the `substructure_rs` binary, which parses the SMARTS queries and reactant
SMILES itself, sanitizes the reactants the way RDKit does, screens with its own
posting-list fingerprint, and runs exact matching in parallel. Its digest uses
the same encoding as benchmarks 05a-07.

uv builds and installs the binary from `substructure_rs/` (a maturin `bin`
package), like `rust_index`, and rebuilds it when the Rust sources change.

Any arguments go to the binary; see `--help`. To check the result against
RDKit, create a reference with `substructure_rs/tools/rdkit_reference.py` and
pass `--reference <dir>/pairs.bin`.
"""

import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def main() -> None:
    binary = Path(sys.executable).parent / "substructure_rs"
    if not binary.exists():
        found = shutil.which("substructure_rs")
        if found is None:
            sys.exit("substructure_rs is not installed; run `uv sync`")
        binary = Path(found)
    sys.exit(
        subprocess.run([str(binary), *sys.argv[1:]], cwd=root, check=False).returncode
    )


if __name__ == "__main__":
    main()

"""Substructure search with Roaring posting lists over Pattern fingerprint bits.

Each fingerprint bit maps to the IDs of queries that contain it. A query can only
match a reactant if none of its bits are absent from the reactant fingerprint. Union
postings for selected absent bits, remove those IDs, and run the full fingerprint
screen and exact RDKit match on the remaining queries. The final screen makes the
answer independent of how many postings are selected.

Index construction is outside the matching timer, as in the other benchmarks.
Benchmark 5a runs the same code (`benchmarks/postings_bench.py`) with packed NumPy
arrays, so the two reports compare field by field. Resident memory is measured
after loading, after the index build and after matching; run the script on its own
so nothing else shares the process.
"""

import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.postings_bench import main
from benchmarks.query_index import SubstructureQueryIndex

if __name__ == "__main__":
    main("05_roaring_postings", SubstructureQueryIndex, "Roaring", __doc__)

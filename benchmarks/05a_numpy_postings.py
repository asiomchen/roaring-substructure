"""Benchmark 5 with packed NumPy uint64 arrays in place of Roaring bitmaps.

The same algorithm and measurements as benchmark 5 (`05_roaring_postings.py`):
Pattern fingerprints, 128 absent-bit postings by default, a full fingerprint
subset screen and RDKit's exact substructure match, with the query fingerprints
and posting lists stored as packed NumPy `uint64` arrays. Both run
`benchmarks/postings_bench.py`, so their reports compare field by field.

Resident memory is measured after loading, after the index build and after
matching. Run the script on its own so nothing else shares the process.
"""

import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
from benchmarks.numpy_query_index import NumpySubstructureQueryIndex
from benchmarks.postings_bench import main

if __name__ == "__main__":
    main("05a_numpy_postings", NumpySubstructureQueryIndex, "NumPy", __doc__)

# New benchmarks

Two benchmarks were added after PR #1. Both build on benchmark 08, the all-Rust
`substructure_rs` pipeline:

- **Benchmark 09** tunes benchmark 08’s screening fingerprint.
- **Benchmark 10** swaps in a third-party Rust matcher, both to compare speed and to
  cross-check correctness.

## Where they fit

| # | Script | What it measures |
| --- | --- | --- |
| 01 | `01_naive.py` | RDKit `HasSubstructMatch` on all 50M pairs (1k reactants × 50k SMARTS). Baseline, about 76 s. |
| 02 | `02_pattern.py` | RDKit 2048-bit `PatternFingerprint` + `AllProbeBitsMatch` screen, then exact match. About 23 s. |
| 03 | `03_roaring_approach.py` | The same screen with pyroaring `BitMap.issubset`. About 5.9 s. |
| 04 | `04_roaring_multiprocessing.py` | The Roaring loop spread over worker processes. 1.30 s with 10 workers. |
| 05 | `05_roaring_postings.py` | Inverted index: Roaring postings for 128 frequent absent bits, then the full screen and exact match. About 1.6 s. |
| 05a | `05a_numpy_postings.py` | The same algorithm with packed NumPy `uint64`. Compares build, match, pickle size and RSS. About 1.25 s. |
| pg | `postgres/bench.py` | RDKit PostgreSQL cartridge with a GiST index. 7.9 s on 1 connection, 1.6 s on 10. |
| 06 | `06_rust_postings.py` | Rust posting screen (`rust_index/`). Exact match stays in RDKit. |
| 07 | `07_rust_postings_process.py` | Rust postings, with and without process-parallel exact matching. |
| 08 | `08_rust_native.py` | The whole pipeline in Rust (`substructure_rs`): parsing, fingerprint, postings and matcher. About 0.1 s on 1 thread. |
| **09** | `09_rust_native_fp.py` | **New.** Sweeps benchmark 08 over fingerprint kinds × sizes × posting counts. |
| **10** | `10_chematic.py` | **New.** The chematic crates: brute force, and exact matching behind the 08/09 screen. |

## Benchmark 09: fingerprint kinds and sizes

Landed in `9b259cf` (originally PR #2) and extended in PRs #3 and #4.

### What it brings

Benchmark 08 hard-coded its screening fingerprint as compile-time constants: labelled
paths of up to 4 bonds, 4096 bits, counted features.
Benchmark 09 makes the feature set and the size choosable at run time, so it can answer
two questions:

1. How much do richer features or more bits cut the candidates that reach exact
   matching, and what do they cost to compute?
2. Is every fingerprint a lossless screen?
   The fingerprint only decides which pairs reach the exact matcher, so every
   configuration must give the same match digest.
   If any digest differs, the script exits with status 1.

### Fingerprint kinds (`substructure_rs.FP_KINDS`)

| Kind | Features |
| --- | --- |
| `atoms` | Atom labels only. |
| `paths2` / `paths4` / `paths6` | Atoms plus labelled paths of up to 2, 4 or 6 bonds. |
| `paths4-nocount` | `paths4` with each feature counted once (count cap 1 instead of 4). |
| `paths4+branches` | Adds each atom together with every 3 of its neighbours. |
| `paths4+cycles` | Adds labelled simple cycles of 3–8 atoms. |
| `paths4+branches+cycles` | Both of the above. |
| `paths4+long6+…`, `paths4+long8+…` | Added in `ff1a34d`: longer paths labelled by element and bond class only. |

Sizes are 512, 1024, 2048, 4096, 8192 and 16384 bits.
The default is `paths4` @ 4096, which reproduces benchmark 08.

### Running it

```bash
uv run benchmarks/09_rust_native_fp.py --threads 1 --csv benchmarks/09_rust_native_fp_1thread.csv
uv run benchmarks/09_rust_native_fp.py --kinds paths4 paths4+cycles --bits 2048 8192
```

Flags:

- `--kinds`, `--bits`, `--postings`: the sweep axes.
  Postings default to 128.
- `--orders`: the exact-match atom orders to sweep.
  Added in #3.
- `--runs`: number of runs; the result is their median.
  Default 5.
- `--threads`, `--queries-file`, `--reactants-file`
- `--reference`: an RDKit `pairs.bin` to compare against.
- `--csv`: write the results to a file.

**CSV columns.** `benchmarks/09_rust_native_fp_1thread.csv` has 50 rows (10 kinds × 5
sizes) with these columns:

- `fp_kind, fp_bits, postings, order`
- `query_density, target_density`: the mean fraction of fingerprint bits set.
- `build_seconds, index_mib`
- `posted, candidates, matches`: pairs left after the posting filter, pairs left after
  the full fingerprint check, and matches.
- `match_median_seconds, match_seconds`
- `fp_seconds, postings_seconds, check_seconds, exact_seconds`: time per phase.
- `digest`

The CSV started as `fp_sweep_1thread.csv` and was renamed and widened in `ff1a34d`.

### Findings

From the first sweep (one thread, 128 postings, 4096 bits):

| Kind | Candidates | Match time |
| --- | ---: | ---: |
| `atoms` | 3,031,375 | 2013 ms |
| `paths2` | 246,688 | 226 ms |
| `paths4` (benchmark 08) | 88,651 | 115 ms |
| `paths4-nocount` | 160,204 | — |
| `paths6` | 70,834 | — |
| `paths4+branches+cycles` | 66,453 | — |

- **Counting repeated features** gives about 1.7–1.8× fewer candidates.
  It is the cheapest gain.
- **Branches and cycles** cut candidates by about 25% for about 14% more index build
  time.
- **`paths6`** cuts a similar share of candidates, but its paths cost more to compute
  per reactant, so overall it is slower.
- **Size:** doubling the bits removes at most about 12% of candidates but doubles the
  bytes compared per candidate.
  1024–2048 bits is fastest for every kind except `atoms`, the same trade-off as
  PostgreSQL’s `sss_fp_size`. Below 1024 bits the screen saturates: at 512 bits,
  `paths6` passes 6× more pairs.
  Index size grows from 24.4 MiB at 2048 bits to 97.7 MiB at 8192.
- **The posting count is not a lever:** 128 is best or tied.
- **Folding noise:** which features happen to share bits after folding moves candidate
  counts by a few percent.
  Re-salting the hash gave `paths4` @ 2048 anywhere from 95,748 to 101,792 candidates.
  Small differences between kinds are not signal (#4).
- **Best result:** `paths4+branches` / `paths4+branches+cycles` at 2048 bits, about 50
  ms after all the speedups in [performance.md](performance.md).

## Benchmark 10: the chematic crates

Landed in PR #5 (`eef5629`).

### What it brings

[chematic](https://github.com/kent-tokyo/chematic) is a general-purpose pure-Rust
cheminformatics toolkit with its own SMILES parser, SMARTS parser and VF2 matcher.
This benchmark uses `chematic-core`, `chematic-smiles` and `chematic-smarts`, pinned at
1.0.37. It answers two questions:

1. **Speed.** How fast is a general Rust matcher, both brute force and behind a good
   screen, compared with RDKit and with substructure_rs’s specialised matcher?
2. **Correctness.** Does a third, independent implementation agree with RDKit and
   substructure_rs on every pair?

chematic cannot screen this workload itself.
Its Pattern fingerprint is a port of RDKit’s and covers concrete molecules only, not
SMARTS queries. So the benchmark borrows substructure_rs’s screen.

### The two modes

- **`--mode naive`** has chematic parse everything and check all 50M (reactant, query)
  pairs. It mirrors benchmark 01.
- **`--mode screened`** uses substructure_rs’s fingerprint and postings screen
  (`substructure_rs::Screen`) to pick the pairs, and chematic checks only those, the
  same pairs benchmark 08’s matcher sees.
  As first merged in #5, the screen ran outside the timer, so the timer measured
  chematic’s exact matcher alone.
  It is now timed like the other benchmarks: the screen index is built once as index
  prep, and each matching run includes the screen; the exact match alone is reported
  separately.

```bash
uv run benchmarks/10_chematic.py --mode naive
uv run benchmarks/10_chematic.py --mode screened --threads 1
```

Other flags: `--fp-kind` (default `paths4+branches`), `--fp-bits` (default 2048),
`--postings`, `--runs`, `--threads` and `--reference`. The output includes:

- parse times;
- the screen time (reported, not counted);
- the median match time and ns per pair;
- the pair and match counts;
- a SHA-256 digest in the same encoding as benchmarks 05a–09;
- missing and extra pairs against an RDKit reference.

### Results

Both modes return the **same 27,723 matches and digest as RDKit and benchmark 08**.

|  | 1 thread | 14 threads |
| --- | ---: | ---: |
| chematic `naive` (50M pairs) | 22.0 s | 2.96 s |
| chematic `screened` (82,461 pairs, exact match only) | 0.15 s | 18 ms |
| benchmark 08 matcher, same pairs | 0.019 s | — |
| benchmark 08, whole match step | 0.050 s | 5 ms |

- **Brute force:** 3.4× faster than RDKit’s 75.5 s. The README credits an element-count
  check chematic runs before each search.
- **On the screened pairs:** about 1.8 µs per pair against about 0.2 µs for
  substructure_rs. These pairs are the hard ones: about a third of them match, and most
  of the rest nearly do.
  As a drop-in exact matcher, chematic would take matching from about 50 ms to about 180
  ms.
- **Its main value** is as an independent, RDKit-equivalent check of substructure_rs’s
  parsing and matching, which it confirms on all 50M pairs.

No results CSV is committed for benchmark 10. The numbers were only in the README, and
the `97a70fa` rewrite removed them.

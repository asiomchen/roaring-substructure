# Changes since PR #1

This folder describes everything that landed after PR #1 (`18ba69b`, “rust benchmark”)
up to `97a70fa` on the `explain` branch, plus the uncommitted work in the tree.
All new Rust code is in `substructure_rs` (0.1.0 → 0.4.0) and the new `chematic_bench`
crate.

## Timeline

| PR / commit | Branch | What it does |
| --- | --- | --- |
| #2 `c01bc56` | — | Benchmark 09 and a faster Rust pipeline, squashed into one commit. |
| #6 `9a26d9e` | — | Reverts #2. The commit message gives no reason. |
| #3 merge `3b82fce` | `10_search_order` | Brings #2 back as its original commits (`9b259cf`, `ff1a34d`), then adds rare-root search order (`812828f`) and version bumps to 0.2.0 and 0.3.0. |
| #4 merge `9e941c0` | `11_fp_buffers` | Incremental path hashing for fingerprints (`b5a7456`), bump to 0.4.0. |
| #5 merge `5d84e4e` | `13_chematic` | Benchmark 10: the chematic crates (`eef5629`). |
| `97a70fa` | `explain` | Adds the flowmark skill and `AGENTS.md`, and rewrites `README.md`. **This rewrite drops the documentation for 09 and 10** (see [docs-and-wip.md](docs-and-wip.md)). |
| uncommitted | `explain` | `ARTICLE_PART_2.md`, `FINGERPRINT_SELECTIVITY_EXPLAINED.md`, `examples/known_vs_observed.py`. |

The code #3 brought back is the same as #2 except for version strings
(`git diff c01bc56 ff1a34d`).

## What each benchmark brings

| Benchmark | Question it answers | Main result |
| --- | --- | --- |
| **09** `benchmarks/09_rust_native_fp.py` | Which fingerprint features and size make the best screen for the benchmark 08 Rust pipeline? Is every fingerprint lossless? | All 10 kinds × 512–8192 bits return the same 27,723 matches and digest. The best are `paths4+branches(+cycles)` at 1024–2048 bits: about 98 ms at first, about 50 ms after #3 and #4, against 115 ms for the benchmark 08 default (`paths4` @ 4096). |
| **10** `benchmarks/10_chematic.py` | How does a general-purpose pure-Rust toolkit (chematic) compare with substructure_rs’s specialised matcher? Does an independent implementation agree with RDKit? | Brute force over all 50M pairs takes 22 s, 3.4× faster than RDKit’s 75.5 s. On the 82k screened pairs, chematic needs about 1.8 µs per pair against about 0.2 µs for substructure_rs. Results are identical to RDKit, so it works as a correctness cross-check. |

PRs #3 and #4 add no new benchmark scripts.
They speed up the pipeline that benchmarks 08, 09 and 10 run, and they extend benchmark
09’s output: per-phase columns, the `--orders` flag and a regenerated CSV.

## Pages, by category

- [benchmarks.md](benchmarks.md): benchmark 09 and benchmark 10 in detail, and where
  they sit among benchmarks 01–10.
- [performance.md](performance.md): the speedups in fingerprint construction, exact
  matching and search order (PRs #3 and #4), with before/after numbers.
- [analysis-tooling.md](analysis-tooling.md): per-phase timing, the false-positive
  analysis and search statistics.
- [api-build-tests.md](api-build-tests.md): the Python and CLI API, build setup, version
  bumps and tests.
- [docs-and-wip.md](docs-and-wip.md): the README rewrite and what it lost, the flowmark
  skill, and the uncommitted article and example files.

## How the numbers moved

One thread, best fingerprint configuration, full match step (fingerprint + postings +
check + exact):

| State | Configuration | Time |
| --- | --- | ---: |
| Benchmark 08 default | `paths4` @ 4096 | 115 ms |
| Benchmark 09 first sweep (`9b259cf`) | `paths4+branches+cycles` @ 2048 | 98 ms |
| After `ff1a34d` (packed atom tests, faster fingerprint folding) | `paths4+branches` @ 2048 | 63 ms |
| After `812828f` (rare-root order) | `paths4+branches` @ 2048 | 52.5 ms |
| After #4 (incremental path hashing) | `paths4+branches(+cycles)` @ 2048 | 50 ms |

Every row returns 27,723 matches with digest `72f6936d…7310b`.

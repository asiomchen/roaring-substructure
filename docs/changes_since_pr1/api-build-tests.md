# API, build and tests

## `substructure_rs` Python API

| Added in | Change |
| --- | --- |
| `9b259cf` | `run(fp_kind="paths4", fp_bits=4096)`. Report gains `fp_kind`, `fp_bits`, `query_density`, `target_density`. Module exports `FP_KINDS` and `FP_SIZES`. |
| `ff1a34d` | Report gains `phase_seconds` (fp / postings / check / exact) and `posted`. Two more fingerprint kinds (`long6`, `long8`). |
| `812828f` | `run(order="rare-root")`. Report gains `order`. Module exports `ORDERS = ["rare-root", "score", "rare"]`. |

Invalid kinds, sizes, orders, or postings greater than `fp_bits` raise `ValueError` via
`Options::validate()`.

## `substructure_rs` CLI

- `--fp-kind`, `--fp-bits`, `--order`
- `--analyze-false-positives`, `--search-stats`: CLI only.

## `substructure_rs` Rust API

- `Index<const W>` and `Fp<const W> = [u64; W]` are generic over the fingerprint width.
  `run_in_pool` dispatches 512–16384 bits to W = 8–256.
- `FpKind` describes a fingerprint kind with these fields: `max_edges`, `long_edges`,
  `max_full_edges`, `count_cap`, `branches`, `max_cycle`.
- `Plan::with_order(q, order, &freq)` and `Order { Score, RareRoot, Rare }`.
- Added for benchmark 10 (#5):
  - `pub fn read_column(path, column)`
  - `pub struct Candidates`
  - `pub fn candidates(&Options)`: returns the screened query IDs per reactant.
  - Since replaced by `pub struct Screen` (`Screen::build`, `Screen::targets`,
    `Screen::screen`), so benchmark 10 can build the screen once as index prep and
    screen inside each timed matching run.
    The `run()` report now has `query_prep_seconds` and `reactant_prep_seconds` (each
    reads and parses its own file) in place of `read_seconds`, `parse_queries_seconds`
    and `parse_reactants_seconds`.

## Versions

`substructure_rs` went from 0.1.0 to 0.2.0 (`7652d66`), then 0.3.0 (`89c80f5`), then
0.4.0 (`f9d139c`). Each bump touches `Cargo.toml`, `Cargo.lock`, `pyproject.toml` and
`uv.lock`.

## New crate `chematic_bench/` (#5)

- **Crate type:** `cdylib` + `rlib` + a binary.
  The optional `python` feature adds pyo3 0.26.
- **Dependencies:** rayon, sha2, `substructure_rs` (by path), and the chematic crates
  pinned at 1.0.37.
- **Release profile:** fat LTO, `codegen-units = 1`.
- **Python module:**
  - `chematic_bench.run(...)` releases the GIL.
  - `cli(args)`, `main()`, `USAGE`.
  - Console script `uv run chematic_bench`.
- **Wiring:** the root `pyproject.toml` depends on it through `[tool.uv.sources]`. The
  source is not editable.
  `cache-keys` include `../substructure_rs/src/**/*.rs`, so a change to the screen
  triggers a rebuild.
- **Build quirks.** `.cargo/config.toml` matches `substructure_rs`:
  - `target-cpu=native`;
  - `RUSTC_STAGE=0`, which skips proc-macro2’s compiler probe;
  - `PYO3_CONFIG_FILE=pyo3-config.txt`, which hardcodes CPython 3.13. A different Python
    version needs that file edited.

## Tests

`substructure_rs/src/tests.rs`:

- `fingerprint_never_rejects_a_match`: now loops over every `FP_KINDS` (`9b259cf`).
- `cycles_and_branches_screen_out_non_matches` (`9b259cf`).
- `compiled_atom_tests_agree_with_expressions`: `compile_atom` vs `Expr::matches` on 15
  SMARTS (`ff1a34d`).
- `every_atom_order_finds_the_same_matches`: every order matches `Plan::new`
  (`812828f`).
- `fingerprints_do_not_depend_on_atom_order`: each molecule written from two different
  starting atoms gives the same fingerprint, for every kind (#4).

`tests/test_substructure_rs.py`:

- `test_fingerprint_kinds_and_sizes_agree`: every kind × 512–8192 bits gives the same
  digest.
- `test_bad_fingerprint_raises_value_error`
- `test_atom_orders_agree`: also checks that `order="random"` raises.

`tests/test_chematic_bench.py`:

- `test_modes_match_reference_and_substructure_rs`: both modes on 5 queries × 5
  reactants against an RDKit `pairs.bin`. The digest must equal substructure_rs’s.
- `test_errors_raise_value_error`
- `test_cli_exit_codes`

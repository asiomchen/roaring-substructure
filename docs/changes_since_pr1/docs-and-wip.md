# Docs, tooling and uncommitted work

## Commit `97a70fa` “add skill”

**Tooling.**

- Adds `.claude/skills/flowmark/` (`SKILL.md`, `references/project-setup.md`).
- Adds `AGENTS.md`, which says to run `flowmark --auto` on Markdown, using pinned `uvx`
  runners: `flowmark-rs==0.4.0` or `flowmark==0.8.0`.
- `ARTICLE.md` gets 9 blank lines inside its Python code samples.
  This looks like formatter output; there is no new prose.

**README rewrite: content regression.**

The new `README.md` is article-oriented: 215 lines instead of 449. It adds:

- a table of contents;
- a subsection for each approach;
- a reusable index snippet;
- a reference table for Rust benchmarks 06–08;
- instructions for rebuilding the dataset with `run_pipeline.sh`.

But it is the stashed draft `stash@{0}` ("WIP on main: 18ba69b"), written before PRs
#2–#5. `git diff 25126d7 97a70fa -- README.md` is empty.
Committing it overwrote the merged README and removed:

- **Benchmark 09:** the kind × size table, the time breakdown, the search-order and
  false-positive results, and the notes on incremental path hashing.
- **Benchmark 10:** the chematic results.
- **PostgreSQL:** the per-version table and the whole Tuning section (`sss_fp_size`,
  `shared_buffers`, chirality, connection scaling, the LATERAL batching pitfall).
- **Other sections:** Method, “Things worth knowing”, the 5a memory and RSS numbers, and
  the commands for 06–10.

The README at HEAD does not mention benchmarks 09 or 10 at all.
The removed text is still at `git show 5d84e4e:README.md`. The pages in this folder
cover the 09 and 10 material, but the README should probably be merged with the
`5d84e4e` version.

## Uncommitted files

- **`ARTICLE_PART_2.md`** (259 lines).
  Draft of article part 2, “Moving the substructure search into Rust”.
  - Covers: a Rust/PyO3/maturin primer, the one-core timing policy, benchmark 06 (Rust
    screen, about 15% saving) and benchmark 08 (full Rust, about 9× faster).
  - Main point: the gain comes mostly from the more selective fingerprint, which passes
    88,651 candidates against RDKit’s 350,393.
  - Does not yet cover 09 or 10.
  - Rough edges:
    - the text says “two experiments” but the heading says “three”;
    - one heading is numbered “### 3.”;
    - the rdkit#151 and rdkit#1496 references are not Markdown links.
- **`FINGERPRINT_SELECTIVITY_EXPLAINED.md`** (188 lines).
  Explains why benchmark 08’s fingerprint rejects more than RDKit’s
  `PatternFingerprint`.
  - The rule: a query may set bits only for properties it guarantees.
  - The templates state H, degree and charge.
    RDKit ignores them; 08 hashes them.
  - Includes an ablation table and two proposals for bringing this to RDKit: a
    `rdFingerprintGenerator`-based “06b”, or an opt-in RDKit option.
- **`examples/known_vs_observed.py`** (201 lines).
  An RDKit demo for the explainer above:
  `uv run examples/known_vs_observed.py [--smarts … --smiles …]`.
  - Separates query-atom properties a SMARTS *states* from those it only *observes* in
    the query graph.
  - Proves each “unsafe” verdict with a counterexample molecule.
  - `examples/__pycache__/` should be gitignored.

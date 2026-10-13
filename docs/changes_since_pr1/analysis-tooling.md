# Analysis tooling

PR #3 adds instrumentation to `substructure_rs` that explains where time goes and why
the screen lets pairs through.
It is CLI-only: the flags are hard-coded off in the Python binding.

## Per-phase timing (`ff1a34d`)

`Index::match_one` now returns a `Matched` struct instead of a tuple:

- `posted`: pairs left after the posting filter.
- `candidates`: pairs left after the full fingerprint check.
- `found`: the matches.
- `seconds: [f64; 3]`: time in each step.

The old `screen()` was split into `posting_filter()` and `subset_filter()`.
`Report.phase_seconds` gives the fp / postings / check / exact split, summed over
threads. This is what benchmark 09’s `*_seconds` CSV columns come from.

## False-positive analysis (`--analyze-false-positives`, `analysis.rs`)

A false positive is a pair that passes the fingerprint screen but fails the exact match.
For each one, `false_positives()` does three things:

- **Relaxed re-matching.** It re-matches against 8 relaxed versions of the query.
  Six drop one constraint each: H count, degree, charge, aromaticity, element or bond
  order. Two drop groups: “all atom labels” and “all labels”.
- **Folding collisions.** It counts pairs that pass only because folding mapped
  different features onto the same bit, so the unfolded feature sets are not subsets but
  the folded bits are.
- **Breakdowns.** It splits false positives by query diameter, rings, open-element
  queries, and how concentrated they are in the top 10, 100 and 1000 queries.

**Findings** for `paths4+branches+cycles` @ 2048 bits, 44,754 false positives, measured
before #4:

- 21% are folding collisions (4% at 8192 bits).
- Dropping H, degree or charge fixes under 0.5% between them.
  Aromaticity fixes about 2%, bond order about 2% and element 23%.
- 74% need more than one constraint dropped.
  Two-thirds of those match the bare query graph: the shape and each label are present,
  but not on the same atoms.
- Queries spanning at most 4 bonds are 13% of queries but 44% of false positives.

**Conclusion.** The screen misses combinations of labels spread over more atoms than its
features cover, not any one kind of label.
Longer paths would help, but they cost more per reactant than a false positive does, now
that rejecting one takes about 0.3 µs.

## Search statistics (`--search-stats`, `812828f`)

`search_stats()` reports, split into matches and non-matches:

- atoms offered, root offers and atoms placed (median, p99, max);
- the depth at which non-matches fail;
- a one-thread timing of the same pairs, run reactant by reactant and then query by
  query.

**Findings:**

- The search was already close to minimal: a matching pair places about 10 atoms for a
  7-atom query, and a non-match 12–14.
- Running query by query, with each plan’s data hot in cache, saved only 4%.
- So the cost was the search work itself, mainly the root scan.
  That is why the search-order work in [performance.md](performance.md) targets the
  root.

The `Counter` trait threads these counts through `has_match_counted`. Its `()`
implementation compiles to nothing, so normal runs pay no overhead.

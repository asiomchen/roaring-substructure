# Performance work in the Rust pipeline

PRs #3 and #4 make `substructure_rs` faster without changing its results.
Every change below keeps the same 27,723 matches and digest.
The pipeline has four phases:

1. **fp:** build the reactant’s fingerprint.
2. **postings:** filter the queries with the posting lists.
3. **check:** run the full fingerprint subset check.
4. **exact:** run the exact match.

Time per phase for `paths4+branches+cycles` @ 2048 bits, one thread:

| State | fp | postings | check | exact | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before `ff1a34d` | 33 | 15 | 7 | 45 | ~98 ms |
| After `ff1a34d` | 19 | 15 | 6 | 24 | ~64 ms |
| After #4 (incl. `812828f`) | 13 | 15 | 6 | 16 | ~50 ms |

## Fingerprint construction

### Counting by hash table instead of sort (`ff1a34d`, `fingerprint.rs`)

- **Folding.** `fold<W>` counts copies of each feature key in a small open-addressing
  table instead of sorting the keys.
  The n-th copy of a key sets bit `mix(key, n)` while n < `count_cap`. That is the same
  bits as before, without the O(n log n) sort.
- **Path walk.** The walk reuses one pre-sized `path`/`bonds` buffer across start atoms,
  and `emit_path` uses `[u32; 16]` stack arrays instead of heap Vecs.
- **Result:** about 40% less fingerprint time.

### Incremental path hashing (#4, `b5a7456`)

**Before.** For each path, `emit_path` copied the labels for each labelling (view) and
compared the forward and reversed sequences to pick a direction.
It then hashed the whole sequence.
Each path cost O(length) per view, and nothing was reused between a path and its prefix.

**After.** `extend` carries, for each view, a forward and a backward polynomial hash
(`K = 0x9e3779b97f4a7c15`):

- forward: `f = (f·K + bond)·K + atom`
- backward: `r += bond·pow + atom·pow·K`, where `pow = K^len`

Reversing a path swaps the two hashes, so `min(f, r)` is a direction-independent key
with no comparison. Each extension costs a few multiply-adds per view.

There is also a new pruning rule: when no view can still emit, because every view has
hit an open query label, the walk stops going deeper.

**Not bit-identical.** The hash function changed, so densities and candidate counts move
by a few percent: `paths4` @ 2048 went from 95,748 to 98,665 candidates.
Results do not change.
The new test `fingerprints_do_not_depend_on_atom_order` guards the canonicalisation.

**A negative result.** Per-thread `Scratch` buffers (`thread_local!`), a flat `Nbrs`
neighbour list, a reusable fold table and stack arrays in `emit_cycle` were tried first.
They saved nothing measurable: the cost was hashing, not allocation.
The branch name `11_fp_buffers` comes from this part.

Fingerprint time before → after #4:

| kind | fp ms |
| --- | --- |
| `atoms` | 2.4 → 1.1 |
| `paths2` | 6.1 → 4.3 |
| `paths4` | 14.5 → 9.3 |
| `paths6` | 31.5 → 17.4 |
| `paths4+branches+cycles` | 18.6 → 13.3 |
| `paths4+long8+branches+cycles` | 37.4 → 24.3 |

The CSV refreshed in #4 also picks up the exact-match gains from #3. Only `fp_seconds`
isolates #4’s effect.

## Exact matching

### Packed atom tests (`ff1a34d`, `smarts.rs`, `matcher.rs`)

- `TargetAtom::packed()` puts element, aromaticity, total H, degree and charge into one
  `u64`.
- `compile_atom()` turns a SMARTS atom expression into a small disjunction of
  `packed & mask == value` terms:
  - AND becomes a cross product, with contradictory terms dropped;
  - OR is capped at 8 terms;
  - NOT is supported for `!aromatic` only.
  - Anything else falls back to `AtomTest::Expr`.
- `has_match` keeps the mapping and a `u64` used-bitmap on the stack for queries of up
  to 64 atoms. No allocation per pair.
- **Result:** exact matching roughly halved, from 45 to 24 ms.
  Benchmark 08 went from 12 to 8 ms on all cores.

### Search order (`812828f`)

These are three changes to the backtracking (VF2-style) search, with time per pair after
each:

| Change | ns per pair |
| --- | ---: |
| Baseline | 282 |
| 1. Tight root scan | 248 |
| 2. One `Step` struct per step | 223 |
| 3. Rare-root ordering | about 190 |

1. **Tight root scan.** The first query atom has no anchor, so all of the target’s atoms
   (about 30 per reactant) are candidates.
   When the root test is one packed term, a branch-light loop over `t.packed` filters
   them before the full `try_atom`.
2. **One `Step` struct per step.** The parallel vectors (`back`, `anchored`, `tests`)
   are replaced by one `Step { test, anchor, closures }`, so everything a step needs
   sits together in memory.
3. **Rare-root ordering** (`--order rare-root`, now the default).
   - `LabelFreq` builds a histogram of atom labels from the queries.
     Rarity is `-ln(share)`.
   - Each connected component of the query starts at its rarest atom (for example N+ or
     Cl rather than aromatic C), then grows by the old rule: most bonds to placed atoms.
   - Most non-matches now fail at the root.
   - The `rare` order, which also uses rarity to break ties while growing, was no
     better.

| kind @ 2048 | exact `score` | exact `rare-root` | total `rare-root` |
| --- | ---: | ---: | ---: |
| `paths4` | 25.0 | 22.0 | 56.9 |
| `paths4+branches` | 19.7 | 17.0 | **52.5** |
| `paths4+branches+cycles` | 18.0 | 15.0 | 54.0 |

The search statistics in [analysis-tooling.md](analysis-tooling.md) are what led to this
work. They showed the search already placed close to the minimum number of atoms, and
that the root scan dominated.

## Overall effect

| kind @ 2048, 1 thread | Before #3 | After #3 | After #4 |
| --- | ---: | ---: | ---: |
| `paths4` | 113 ms | 68 ms | 54 ms |
| `paths4+branches` | 100 ms | 63 ms | 50 ms |
| `paths4+branches+cycles` | 98 ms | 64 ms | 50 ms |
| `atoms` | 2161 ms | 811 ms | 490 ms |

After #4, the postings filter (about 15 ms) costs about as much as fingerprinting or
exact matching, so it is the next target.

//! Why do screened pairs fail to match? (`--analyze-false-positives`)
//! How much searching does exact matching do? (`--search-stats`)
//!
//! For every pair that passes the fingerprint screen but not the exact match,
//! loosen one kind of query constraint at a time and match again. A pair that
//! matches once, say, H counts are dropped failed only on H counts, so a
//! fingerprint that encoded them better could have screened it out.
//!
//! Author: Marcin Kowiel + Claude

use crate::fingerprint::{query_features, target_features, FpKind};
use crate::index::Index;
use crate::matcher::{has_match, has_match_counted, Plan, SearchCounts, Target};
use crate::smarts::{BondQuery, Expr, Prim, Query};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write;

/// Constraint kinds, each loosened on its own; then two that loosen whole groups.
const RELAXATIONS: [&str; 8] =
    ["H count", "degree", "charge", "aromaticity", "element", "bond order", "all atom labels", "all labels"];
const SINGLE: usize = 6;

fn touches(e: &Expr, drop: &dyn Fn(&Prim) -> Option<Expr>) -> bool {
    match e {
        Expr::Prim(p) => drop(p).is_some(),
        Expr::Not(inner) => touches(inner, drop),
        Expr::And(es) | Expr::Or(es) => es.iter().any(|x| touches(x, drop)),
    }
}

fn relax_expr(e: &Expr, drop: &dyn Fn(&Prim) -> Option<Expr>) -> Expr {
    match e {
        Expr::Prim(p) => drop(p).unwrap_or_else(|| e.clone()),
        // Loosening under a negation would tighten it; drop the whole negation instead.
        Expr::Not(inner) if touches(inner, drop) => Expr::Prim(Prim::True),
        Expr::Not(_) => e.clone(),
        Expr::And(es) => Expr::And(es.iter().map(|x| relax_expr(x, drop)).collect()),
        Expr::Or(es) => Expr::Or(es.iter().map(|x| relax_expr(x, drop)).collect()),
    }
}

fn relax(q: &Query, kind: usize) -> Query {
    let mut q = q.clone();
    let drop: &dyn Fn(&Prim) -> Option<Expr> = match kind {
        0 => &|p| matches!(p, Prim::TotalH(_)).then_some(Expr::Prim(Prim::True)),
        1 => &|p| matches!(p, Prim::Degree(_)).then_some(Expr::Prim(Prim::True)),
        2 => &|p| matches!(p, Prim::Charge(_)).then_some(Expr::Prim(Prim::True)),
        3 => &|p| match *p {
            Prim::Aromatic(_) => Some(Expr::Prim(Prim::True)),
            Prim::Element { z, aromatic: Some(_) } => Some(Expr::Prim(Prim::Element { z, aromatic: None })),
            _ => None,
        },
        4 => &|p| match *p {
            Prim::Element { aromatic: Some(a), .. } => Some(Expr::Prim(Prim::Aromatic(a))),
            Prim::Element { aromatic: None, .. } => Some(Expr::Prim(Prim::True)),
            _ => None,
        },
        5 => &|_| None,
        _ => &|_| Some(Expr::Prim(Prim::True)),
    };
    for a in &mut q.atoms {
        a.expr = relax_expr(&a.expr, drop);
    }
    if kind == 5 || kind == 7 {
        for b in &mut q.bonds {
            b.query = BondQuery::Any;
        }
    }
    q
}

/// `candidates[r]` are the screened query IDs for reactant `r`, `found[r]` the matches.
pub fn false_positives(
    kind: &FpKind,
    queries: &[Query],
    targets: &[Target],
    candidates: &[Vec<u32>],
    found: &[Vec<u32>],
) -> String {
    let pairs: Vec<(usize, u32)> = candidates
        .iter()
        .zip(found)
        .enumerate()
        .flat_map(|(r, (c, f))| c.iter().filter(|q| f.binary_search(q).is_err()).map(move |&q| (r, q)))
        .collect();
    let relaxed: Vec<Vec<(Query, Plan)>> = queries
        .par_iter()
        .map(|q| {
            (0..RELAXATIONS.len())
                .map(|k| {
                    let r = relax(q, k);
                    let p = Plan::new(&r);
                    (r, p)
                })
                .collect()
        })
        .collect();
    // Bit k set: loosening constraint kind k alone makes the pair match.
    let fixes: Vec<u32> = pairs
        .par_iter()
        .map(|&(r, q)| {
            let mut m = 0;
            for (k, (rq, plan)) in relaxed[q as usize].iter().enumerate() {
                if has_match(rq, plan, &targets[r]) {
                    m |= 1 << k;
                }
            }
            m
        })
        .collect();

    // Pairs that pass only because folding made distinct features share a bit.
    let tf: Vec<Vec<u64>> = targets.par_iter().map(|t| target_features(kind, t)).collect();
    let collisions = pairs
        .par_iter()
        .filter(|&&(r, q)| {
            let t = &tf[r];
            !query_features(kind, &queries[q as usize]).iter().all(|f| t.binary_search(f).is_ok())
        })
        .count();

    let n = pairs.len().max(1) as f64;
    let mut out = String::new();
    let _ = writeln!(out, "false positives {} (screened pairs that do not match)", pairs.len());
    let _ = writeln!(
        out,
        "  {collisions} ({:.1}%) pass only through folding collisions; unfolded features would reject them",
        100.0 * collisions as f64 / n
    );
    let _ = writeln!(out, "  matches once this constraint alone is dropped:");
    let single = (1u32 << SINGLE) - 1;
    for (k, name) in RELAXATIONS.iter().enumerate().take(SINGLE) {
        let c = fixes.iter().filter(|&&m| m >> k & 1 == 1).count();
        let only = fixes.iter().filter(|&&m| m & single == 1 << k).count();
        let _ = writeln!(out, "    {name:<16} {c:>7} ({:>5.1}%), only this one: {only:>7}", 100.0 * c as f64 / n);
    }
    let none = fixes.iter().filter(|&&m| m & single == 0).count();
    let _ = writeln!(out, "    {:<16} {none:>7} ({:>5.1}%)", "none of these", 100.0 * none as f64 / n);
    let _ = writeln!(out, "  of those, matching once these groups are dropped:");
    for (k, name) in RELAXATIONS.iter().enumerate().skip(SINGLE) {
        let c = fixes.iter().filter(|&&m| m & single == 0 && m >> k & 1 == 1).count();
        let _ = writeln!(out, "    {name:<16} {c:>7} ({:>5.1}%)", 100.0 * c as f64 / n);
    }

    // Query diameter (longest shortest path, in bonds) and ring closures.
    let diameter = |q: &Query| {
        let n = q.atoms.len();
        let mut best = 0;
        for s in 0..n {
            let mut dist = vec![usize::MAX; n];
            dist[s] = 0;
            let mut queue = std::collections::VecDeque::from([s]);
            while let Some(a) = queue.pop_front() {
                for &(b, _) in &q.nbrs[a] {
                    if dist[b] == usize::MAX {
                        dist[b] = dist[a] + 1;
                        best = best.max(dist[b]);
                        queue.push_back(b);
                    }
                }
            }
        }
        best
    };
    let diam: Vec<usize> = queries.iter().map(diameter).collect();
    let _ = writeln!(out, "  query diameter in bonds: share of false positives / of queries");
    for (lo, hi) in [(0, 2), (3, 4), (5, 6), (7, 8), (9, usize::MAX)] {
        let inr = |d: usize| (lo..=hi).contains(&d);
        let f = pairs.iter().filter(|&&(_, q)| inr(diam[q as usize])).count();
        let a = diam.iter().filter(|&&d| inr(d)).count();
        let label = if hi == usize::MAX { format!("{lo}+") } else { format!("{lo}-{hi}") };
        let _ = writeln!(
            out,
            "    {label:<6} {:>5.1}% / {:>5.1}%",
            100.0 * f as f64 / n,
            100.0 * a as f64 / queries.len().max(1) as f64
        );
    }
    let ringed = |q: &Query| q.bonds.len() >= q.atoms.len();
    let f = pairs.iter().filter(|&&(_, q)| ringed(&queries[q as usize])).count();
    let a = queries.iter().filter(|q| ringed(q)).count();
    let _ = writeln!(
        out,
        "  queries with a ring: {:.1}% of false positives, {:.1}% of queries",
        100.0 * f as f64 / n,
        100.0 * a as f64 / queries.len().max(1) as f64
    );

    let wildcard = |q: &Query| q.atoms.iter().any(|a| a.known.z.is_none());
    let fp_wild = pairs.iter().filter(|&&(_, q)| wildcard(&queries[q as usize])).count();
    let all_wild = queries.iter().filter(|q| wildcard(q)).count();
    let _ = writeln!(
        out,
        "  queries with an open element: {:.1}% of false positives, {:.1}% of queries",
        100.0 * fp_wild as f64 / n,
        100.0 * all_wild as f64 / queries.len().max(1) as f64
    );

    let mut per_query: HashMap<u32, usize> = HashMap::new();
    for &(_, q) in &pairs {
        *per_query.entry(q).or_default() += 1;
    }
    let mut counts: Vec<usize> = per_query.values().copied().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    let top = |k: usize| 100.0 * counts.iter().take(k).sum::<usize>() as f64 / n;
    let _ = writeln!(
        out,
        "  from {} distinct queries; the top 10 / 100 / 1000 give {:.1}% / {:.1}% / {:.1}%",
        counts.len(),
        top(10),
        top(100),
        top(1000)
    );
    out
}

/// Search work over every screened pair, split by whether the pair matches.
pub fn search_stats<const W: usize>(
    index: &Index<W>,
    targets: &[Target],
    candidates: &[Vec<u32>],
    found: &[Vec<u32>],
) -> String {
    // (matched, query size, counts) per screened pair.
    let pairs: Vec<(bool, usize, SearchCounts)> = candidates
        .par_iter()
        .zip(found)
        .zip(targets)
        .flat_map_iter(|((cands, found), t)| {
            cands.iter().map(move |&q| {
                let mut c = SearchCounts::default();
                let plan = index.plan(q);
                let hit = has_match_counted(&index.queries[q as usize], plan, t, &mut c);
                debug_assert_eq!(hit, found.binary_search(&q).is_ok());
                (hit, plan.len(), c)
            })
        })
        .collect();

    let mut out = String::new();
    let _ = writeln!(out, "search stats    per screened pair: target atoms offered (root scan), atoms placed");
    for (label, hit) in [("matches", true), ("non-matches", false)] {
        let sel: Vec<&(bool, usize, SearchCounts)> = pairs.iter().filter(|p| p.0 == hit).collect();
        let n = sel.len().max(1) as f64;
        let mean = |f: &dyn Fn(&SearchCounts) -> u64| sel.iter().map(|p| f(&p.2)).sum::<u64>() as f64 / n;
        let mut placed: Vec<u64> = sel.iter().map(|p| p.2.placed).collect();
        placed.sort_unstable();
        let pct = |q: f64| placed.get(((placed.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(0);
        let size = sel.iter().map(|p| p.1).sum::<usize>() as f64 / n;
        let _ = writeln!(
            out,
            "  {label:<12} {:>7} pairs, {size:>4.1} query atoms: offered {:>6.1} ({:>5.1} root), \
placed {:>6.1} (median {}, p99 {}, max {})",
            sel.len(),
            mean(&|c| c.offered),
            mean(&|c| c.root_offered),
            mean(&|c| c.placed),
            pct(0.5),
            pct(0.99),
            placed.last().copied().unwrap_or(0)
        );
    }
    // The same pairs, timed on one thread in two orders: reactant by reactant
    // (as the benchmark runs) and query by query (each plan read once, hot).
    let mut by_reactant: Vec<(u32, u32)> = candidates
        .iter()
        .enumerate()
        .flat_map(|(r, c)| c.iter().map(move |&q| (r as u32, q)))
        .collect();
    let time = |pairs: &[(u32, u32)]| {
        let start = std::time::Instant::now();
        let hits = pairs
            .iter()
            .filter(|&&(r, q)| has_match(&index.queries[q as usize], index.plan(q), &targets[r as usize]))
            .count();
        (start.elapsed().as_secs_f64(), hits)
    };
    let (t_reactant, h1) = time(&by_reactant);
    by_reactant.sort_unstable_by_key(|&(r, q)| (q, r));
    let (t_query, h2) = time(&by_reactant);
    assert_eq!(h1, h2);
    let _ = writeln!(
        out,
        "  one thread, {} pairs: reactant by reactant {:.1} ms ({:.0} ns/pair), query by query {:.1} ms ({:.0} ns/pair)",
        by_reactant.len(),
        1000.0 * t_reactant,
        1e9 * t_reactant / by_reactant.len().max(1) as f64,
        1000.0 * t_query,
        1e9 * t_query / by_reactant.len().max(1) as f64
    );
    let misses: Vec<&(bool, usize, SearchCounts)> = pairs.iter().filter(|p| !p.0).collect();
    let n = misses.len().max(1) as f64;
    let share = |f: &dyn Fn(usize, usize) -> bool| {
        100.0 * misses.iter().filter(|p| f(p.2.depth, p.1)).count() as f64 / n
    };
    let _ = writeln!(
        out,
        "  non-matches fail with the root unplaced {:.1}%, under half placed {:.1}%, half or more {:.1}%",
        share(&|d, _| d == 0),
        share(&|d, s| d > 0 && 2 * d < s),
        share(&|d, s| d > 0 && 2 * d >= s)
    );
    out
}

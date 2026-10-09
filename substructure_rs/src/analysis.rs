//! Why do screened pairs fail to match? (`--analyze-false-positives`)
//!
//! For every pair that passes the fingerprint screen but not the exact match,
//! loosen one kind of query constraint at a time and match again. A pair that
//! matches once, say, H counts are dropped failed only on H counts, so a
//! fingerprint that encoded them better could have screened it out.
//!
//! Author: Marcin Kowiel + Claude

use crate::fingerprint::{query_features, target_features, FpKind};
use crate::matcher::{has_match, Plan, Target};
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

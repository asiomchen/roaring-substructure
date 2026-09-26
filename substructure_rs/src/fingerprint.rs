//! Screening fingerprint: hashed labelled paths, with counts.
//!
//! Every feature emitted for a query is also emitted for any target it
//! matches, so `query_fp & !target_fp == 0` never rejects a true match. A query
//! only emits a feature when its atoms and bonds pin down every label in it.
//!
//! Author: Marcin Kowiel + Claude

use crate::matcher::Target;
use crate::mol::{BondType, TargetAtom};
use crate::smarts::{BondQuery, Known, Query};

pub const FP_BITS: usize = 4096;
pub const FP_WORDS: usize = FP_BITS / 64;
pub type Fp = [u64; FP_WORDS];

/// Longest path, in bonds.
const MAX_EDGES: usize = 4;
/// Longest path using fully specified atoms, in bonds.
const MAX_FULL_EDGES: usize = 2;
/// How many copies of a repeated feature are counted.
const COUNT_CAP: usize = 4;

/// Atom and bond labels at each level of detail; `None` = the query leaves it open.
#[derive(Clone, Copy)]
struct AtomLabels {
    z: Option<u32>,
    z_arom: Option<u32>,
    full: Option<u32>,
    z_degree: Option<u32>,
    z_h: Option<u32>,
    z_charge: Option<u32>,
}

#[derive(Clone, Copy)]
struct BondLabels {
    class: Option<u32>,
    exact: Option<u32>,
}

fn full_label(z: u8, arom: bool, degree: u8, h: u8, charge: i8) -> u32 {
    ((z as u32) << 16) | ((arom as u32) << 15) | ((degree as u32 & 0xf) << 10) | ((h as u32 & 0xf) << 5) | ((charge as i32 + 8) as u32 & 0x1f)
}

impl AtomLabels {
    fn target(a: &TargetAtom) -> Self {
        let z = a.z as u32;
        AtomLabels {
            z: Some(z),
            z_arom: Some(z * 2 + a.aromatic as u32),
            full: Some(full_label(a.z, a.aromatic, a.degree, a.total_h, a.charge)),
            z_degree: Some(z * 64 + a.degree as u32),
            z_h: Some(z * 64 + a.total_h as u32),
            z_charge: Some(z * 64 + (a.charge as i32 + 16) as u32),
        }
    }

    fn query(k: &Known) -> Self {
        let z = k.z.map(|z| z as u32);
        AtomLabels {
            z,
            z_arom: z.zip(k.aromatic).map(|(z, a)| z * 2 + a as u32),
            full: match (k.z, k.aromatic, k.degree, k.total_h, k.charge) {
                (Some(z), Some(a), Some(d), Some(h), Some(q)) => Some(full_label(z, a, d, h, q)),
                _ => None,
            },
            z_degree: z.zip(k.degree).map(|(z, d)| z * 64 + d as u32),
            z_h: z.zip(k.total_h).map(|(z, h)| z * 64 + h as u32),
            z_charge: z.zip(k.charge).map(|(z, q)| z * 64 + (q as i32 + 16) as u32),
        }
    }
}

fn bond_code(kind: BondType) -> u32 {
    match kind {
        BondType::Single => 1,
        BondType::Double => 2,
        BondType::Triple => 3,
        BondType::Quadruple => 4,
        BondType::Aromatic => 5,
        BondType::Dative => 6,
    }
}

impl BondLabels {
    fn target(kind: BondType) -> Self {
        let class = match kind {
            BondType::Single | BondType::Aromatic => 1,
            other => bond_code(other),
        };
        BondLabels { class: Some(class), exact: Some(bond_code(kind)) }
    }

    fn query(q: BondQuery) -> Self {
        let (class, exact) = match q {
            BondQuery::Single => (Some(1), Some(bond_code(BondType::Single))),
            BondQuery::Aromatic => (Some(1), Some(bond_code(BondType::Aromatic))),
            BondQuery::SingleOrAromatic => (Some(1), None),
            BondQuery::Double => (Some(2), Some(2)),
            BondQuery::Triple => (Some(3), Some(3)),
            BondQuery::Any => (None, None),
        };
        BondLabels { class, exact }
    }
}

#[inline]
fn mix(mut h: u64, v: u64) -> u64 {
    h ^= v.wrapping_add(0x9e37_79b9_7f4a_7c15).wrapping_add(h << 6).wrapping_add(h >> 2);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb)
}

/// Hash a label sequence independent of direction.
fn path_key(view: u64, atoms: &[u32], bonds: &[u32]) -> u64 {
    let fwd = atoms.iter().copied().zip(bonds.iter().copied().chain([0]));
    let rev = atoms.iter().rev().copied().zip(bonds.iter().rev().copied().chain([0]));
    let forward_smaller = fwd.clone().cmp(rev.clone()) != std::cmp::Ordering::Greater;
    let mut h = mix(view, atoms.len() as u64);
    if forward_smaller {
        for (a, b) in fwd {
            h = mix(mix(h, a as u64), b as u64);
        }
    } else {
        for (a, b) in rev {
            h = mix(mix(h, a as u64), b as u64);
        }
    }
    h
}

/// Collect path features from a graph given as labels and adjacency.
fn collect(atoms: &[AtomLabels], nbrs: &[Vec<(usize, BondLabels)>], keys: &mut Vec<u64>) {
    for (i, a) in atoms.iter().enumerate() {
        for (view, label) in [(1, a.z), (2, a.z_arom), (3, a.full), (4, a.z_degree), (5, a.z_h), (6, a.z_charge)] {
            if let Some(l) = label {
                keys.push(mix(view, l as u64));
            }
        }
        let mut path = vec![i];
        let mut bonds = Vec::new();
        extend(atoms, nbrs, &mut path, &mut bonds, keys);
    }
}

fn extend(
    atoms: &[AtomLabels],
    nbrs: &[Vec<(usize, BondLabels)>],
    path: &mut Vec<usize>,
    bonds: &mut Vec<BondLabels>,
    keys: &mut Vec<u64>,
) {
    let last = *path.last().unwrap();
    if bonds.len() == MAX_EDGES {
        return;
    }
    for &(next, bl) in &nbrs[last] {
        if path.contains(&next) {
            continue;
        }
        path.push(next);
        bonds.push(bl);
        if path[0] < next {
            emit_path(atoms, path, bonds, keys);
        }
        extend(atoms, nbrs, path, bonds, keys);
        path.pop();
        bonds.pop();
    }
}

fn emit_path(atoms: &[AtomLabels], path: &[usize], bonds: &[BondLabels], keys: &mut Vec<u64>) {
    let views: [(u64, fn(&AtomLabels) -> Option<u32>, fn(&BondLabels) -> Option<u32>); 3] = [
        (10, |a| a.z, |_| Some(0)),
        (11, |a| a.z, |b| b.class),
        (12, |a| a.z_arom, |b| b.exact),
    ];
    let mut al = Vec::with_capacity(path.len());
    let mut bl = Vec::with_capacity(bonds.len());
    let mut run = |view: u64, af: fn(&AtomLabels) -> Option<u32>, bf: fn(&BondLabels) -> Option<u32>| {
        al.clear();
        bl.clear();
        for &i in path {
            match af(&atoms[i]) {
                Some(l) => al.push(l),
                None => return,
            }
        }
        for b in bonds {
            match bf(b) {
                Some(l) => bl.push(l),
                None => return,
            }
        }
        keys.push(path_key(view, &al, &bl));
    };
    for (view, af, bf) in views {
        run(view, af, bf);
    }
    if bonds.len() <= MAX_FULL_EDGES {
        run(13, |a| a.full, |b| b.exact);
    }
}

fn finish(mut keys: Vec<u64>) -> Fp {
    keys.sort_unstable();
    let mut fp = [0u64; FP_WORDS];
    let mut k = 0;
    while k < keys.len() {
        let end = keys[k..].iter().position(|&x| x != keys[k]).map_or(keys.len(), |p| k + p);
        for copy in 0..(end - k).min(COUNT_CAP) {
            let bit = (mix(keys[k], copy as u64) % FP_BITS as u64) as usize;
            fp[bit / 64] |= 1 << (bit % 64);
        }
        k = end;
    }
    fp
}

pub fn target_fp(t: &Target) -> Fp {
    let atoms: Vec<AtomLabels> = t.atoms.iter().map(AtomLabels::target).collect();
    let nbrs: Vec<Vec<(usize, BondLabels)>> = t
        .nbrs
        .iter()
        .map(|ns| ns.iter().map(|&(n, k)| (n as usize, BondLabels::target(k))).collect())
        .collect();
    let mut keys = Vec::new();
    collect(&atoms, &nbrs, &mut keys);
    finish(keys)
}

pub fn query_fp(q: &Query) -> Fp {
    let atoms: Vec<AtomLabels> = q.atoms.iter().map(|a| AtomLabels::query(&a.known)).collect();
    let nbrs: Vec<Vec<(usize, BondLabels)>> = q
        .nbrs
        .iter()
        .map(|ns| ns.iter().map(|&(n, b)| (n, BondLabels::query(q.bonds[b].query))).collect())
        .collect();
    let mut keys = Vec::new();
    collect(&atoms, &nbrs, &mut keys);
    finish(keys)
}

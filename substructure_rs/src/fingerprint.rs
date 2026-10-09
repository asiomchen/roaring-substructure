//! Screening fingerprint: hashed labelled features, with counts.
//!
//! Every feature emitted for a query is also emitted for any target it
//! matches, so `query_fp & !target_fp == 0` never rejects a true match. A query
//! only emits a feature when its atoms and bonds pin down every label in it.
//! Counts stay sound because a match maps distinct query features to distinct
//! target features with the same labels.
//!
//! The feature families (atoms, paths, branches, cycles) and the folded size
//! are chosen at run time; see `FP_KINDS` and `FP_SIZES`. The default,
//! `paths4` at 4096 bits, is the fingerprint of benchmark 08.
//!
//! Author: Marcin Kowiel + Claude

use crate::matcher::Target;
use crate::mol::{BondType, TargetAtom};
use crate::smarts::{BondQuery, Known, Query};

/// Supported fingerprint sizes, in bits.
pub const FP_SIZES: [usize; 6] = [512, 1024, 2048, 4096, 8192, 16384];
pub const DEFAULT_FP_BITS: usize = 4096;
pub const DEFAULT_FP_KIND: &str = "paths4";

/// A fingerprint of `W` 64-bit words.
pub type Fp<const W: usize> = [u64; W];

/// Which features a fingerprint hashes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FpKind {
    pub name: &'static str,
    /// Longest path, in bonds; 0 = atoms only.
    pub max_edges: usize,
    /// Longest path using fully specified atoms, in bonds.
    pub max_full_edges: usize,
    /// Longest path, in bonds, labelled by element and bond class only;
    /// paths longer than `max_edges` get just this one label.
    pub long_edges: usize,
    /// How many copies of a repeated feature are counted.
    pub count_cap: usize,
    /// Atom with three of its neighbours (a tree, not a path).
    pub branches: bool,
    /// Simple cycles up to this many atoms; 0 = none.
    pub max_cycle: usize,
}

const fn kind(name: &'static str, max_edges: usize, count_cap: usize, branches: bool, max_cycle: usize) -> FpKind {
    let max_full_edges = if max_edges < 3 { max_edges } else { max_edges / 2 };
    FpKind { name, max_edges, max_full_edges, long_edges: max_edges, count_cap, branches, max_cycle }
}

const fn long(kind: FpKind, name: &'static str, long_edges: usize) -> FpKind {
    FpKind { name, long_edges, ..kind }
}

pub const FP_KINDS: [FpKind; 10] = [
    kind("atoms", 0, 4, false, 0),
    kind("paths2", 2, 4, false, 0),
    kind("paths4", 4, 4, false, 0),
    kind("paths4-nocount", 4, 1, false, 0),
    kind("paths6", 6, 4, false, 0),
    kind("paths4+branches", 4, 4, true, 0),
    kind("paths4+cycles", 4, 4, false, 8),
    kind("paths4+branches+cycles", 4, 4, true, 8),
    long(kind("", 4, 4, true, 8), "paths4+long6+branches+cycles", 6),
    long(kind("", 4, 4, true, 8), "paths4+long8+branches+cycles", 8),
];

pub fn fp_kind(name: &str) -> Result<FpKind, String> {
    FP_KINDS.iter().copied().find(|k| k.name == name).ok_or_else(|| {
        let names: Vec<&str> = FP_KINDS.iter().map(|k| k.name).collect();
        format!("unknown fingerprint kind {name:?}; choose one of {}", names.join(", "))
    })
}

pub fn check_fp_bits(bits: usize) -> Result<(), String> {
    if FP_SIZES.contains(&bits) {
        Ok(())
    } else {
        Err(format!("fingerprint size {bits} not supported; choose one of {FP_SIZES:?}"))
    }
}

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

/// Collect features from a graph given as labels and adjacency.
fn collect(kind: &FpKind, atoms: &[AtomLabels], nbrs: &[Vec<(usize, BondLabels)>], keys: &mut Vec<u64>) {
    for a in atoms {
        for (view, label) in [(1, a.z), (2, a.z_arom), (3, a.full), (4, a.z_degree), (5, a.z_h), (6, a.z_charge)] {
            if let Some(l) = label {
                keys.push(mix(view, l as u64));
            }
        }
    }
    let mut path = Vec::with_capacity(kind.max_edges.max(kind.long_edges) + 1);
    let mut bonds = Vec::with_capacity(path.capacity());
    for i in 0..atoms.len() {
        path.push(i);
        extend(kind, atoms, nbrs, &mut path, &mut bonds, keys);
        path.pop();
    }
    if kind.branches {
        branches(atoms, nbrs, keys);
    }
    if kind.max_cycle >= 3 {
        cycles(kind.max_cycle, atoms, nbrs, keys);
    }
}

fn extend(
    kind: &FpKind,
    atoms: &[AtomLabels],
    nbrs: &[Vec<(usize, BondLabels)>],
    path: &mut Vec<usize>,
    bonds: &mut Vec<BondLabels>,
    keys: &mut Vec<u64>,
) {
    let last = *path.last().unwrap();
    if bonds.len() >= kind.max_edges.max(kind.long_edges) {
        return;
    }
    for &(next, bl) in &nbrs[last] {
        if path.contains(&next) {
            continue;
        }
        path.push(next);
        bonds.push(bl);
        if path[0] < next {
            emit_path(kind, atoms, path, bonds, keys);
        }
        extend(kind, atoms, nbrs, path, bonds, keys);
        path.pop();
        bonds.pop();
    }
}

fn emit_path(kind: &FpKind, atoms: &[AtomLabels], path: &[usize], bonds: &[BondLabels], keys: &mut Vec<u64>) {
    let views: [(u64, AtomView, BondView); 3] = [
        (10, |a| a.z, |_| Some(0)),
        (11, |a| a.z, |b| b.class),
        (12, |a| a.z_arom, |b| b.exact),
    ];
    let views = if bonds.len() > kind.max_edges { &views[1..2] } else { &views[..] };
    let mut al = [0u32; MAX_PATH_ATOMS];
    let mut bl = [0u32; MAX_PATH_ATOMS];
    let mut run = |view: u64, af: AtomView, bf: BondView| {
        for (slot, &i) in al.iter_mut().zip(path) {
            match af(&atoms[i]) {
                Some(l) => *slot = l,
                None => return,
            }
        }
        for (slot, b) in bl.iter_mut().zip(bonds) {
            match bf(b) {
                Some(l) => *slot = l,
                None => return,
            }
        }
        keys.push(path_key(view, &al[..path.len()], &bl[..bonds.len()]));
    };
    for &(view, af, bf) in views {
        run(view, af, bf);
    }
    if bonds.len() <= kind.max_full_edges {
        run(13, |a| a.full, |b| b.exact);
    }
}

/// Paths are at most `MAX_PATH_ATOMS - 1` bonds long.
const MAX_PATH_ATOMS: usize = 16;

type AtomView = fn(&AtomLabels) -> Option<u32>;
type BondView = fn(&BondLabels) -> Option<u32>;

/// Each atom with every three of its neighbours: the centre label and the
/// sorted (bond, atom) labels of the arms.
fn branches(atoms: &[AtomLabels], nbrs: &[Vec<(usize, BondLabels)>], keys: &mut Vec<u64>) {
    let views: [(u64, AtomView, BondView); 3] = [
        (20, |a| a.z, |b| b.class),
        (21, |a| a.z_arom, |b| b.exact),
        (22, |a| a.full, |b| b.exact),
    ];
    for (c, ns) in nbrs.iter().enumerate() {
        for i in 0..ns.len() {
            for j in i + 1..ns.len() {
                for k in j + 1..ns.len() {
                    'view: for (view, af, bf) in views {
                        let Some(centre) = af(&atoms[c]) else { continue };
                        let mut arms = [(0, 0); 3];
                        for (arm, &(n, bl)) in arms.iter_mut().zip([&ns[i], &ns[j], &ns[k]]) {
                            match (bf(&bl), af(&atoms[n])) {
                                (Some(b), Some(a)) => *arm = (b, a),
                                _ => continue 'view,
                            }
                        }
                        arms.sort_unstable();
                        let mut h = mix(view, centre as u64);
                        for (b, a) in arms {
                            h = mix(mix(h, b as u64), a as u64);
                        }
                        keys.push(h);
                    }
                }
            }
        }
    }
}

/// Simple cycles of 3..=`max_len` atoms, each found once: from its lowest
/// atom, in the direction whose second atom is lower than its last.
fn cycles(max_len: usize, atoms: &[AtomLabels], nbrs: &[Vec<(usize, BondLabels)>], keys: &mut Vec<u64>) {
    fn walk(
        max_len: usize,
        atoms: &[AtomLabels],
        nbrs: &[Vec<(usize, BondLabels)>],
        path: &mut Vec<usize>,
        bonds: &mut Vec<BondLabels>,
        keys: &mut Vec<u64>,
    ) {
        let (start, last) = (path[0], *path.last().unwrap());
        for &(next, bl) in &nbrs[last] {
            if next == start {
                if path.len() >= 3 && path[1] < last {
                    bonds.push(bl);
                    emit_cycle(atoms, path, bonds, keys);
                    bonds.pop();
                }
            } else if next > start && path.len() < max_len && !path.contains(&next) {
                path.push(next);
                bonds.push(bl);
                walk(max_len, atoms, nbrs, path, bonds, keys);
                path.pop();
                bonds.pop();
            }
        }
    }
    let mut path = Vec::with_capacity(max_len);
    let mut bonds = Vec::with_capacity(max_len);
    for s in 0..atoms.len() {
        path.push(s);
        walk(max_len, atoms, nbrs, &mut path, &mut bonds, keys);
        path.pop();
    }
}

/// `bonds[i]` joins `path[i]` and `path[i + 1]` (cyclically).
fn emit_cycle(atoms: &[AtomLabels], path: &[usize], bonds: &[BondLabels], keys: &mut Vec<u64>) {
    let views: [(u64, AtomView, BondView); 3] = [
        (30, |_| Some(0), |_| Some(0)),
        (31, |a| a.z, |b| b.class),
        (32, |a| a.z_arom, |b| b.exact),
    ];
    let n = path.len();
    let mut seq = Vec::with_capacity(n);
    'view: for (view, af, bf) in views {
        seq.clear();
        for (&i, b) in path.iter().zip(bonds) {
            match (af(&atoms[i]), bf(b)) {
                (Some(a), Some(b)) => seq.push((a, b)),
                _ => continue 'view,
            }
        }
        // Smallest of the n rotations in each direction. Reversed, atom i is
        // followed by the bond that preceded it.
        let forward = |r: usize, i: usize| seq[(r + i) % n];
        let backward = |r: usize, i: usize| {
            let a = (r + n - i) % n;
            (seq[a].0, seq[(a + n - 1) % n].1)
        };
        let mut best: Vec<(u32, u32)> = (0..n).map(|i| forward(0, i)).collect();
        let mut cand = Vec::with_capacity(n);
        for r in 0..n {
            for dir in [0, 1] {
                cand.clear();
                cand.extend((0..n).map(|i| if dir == 0 { forward(r, i) } else { backward(r, i) }));
                if cand < best {
                    std::mem::swap(&mut cand, &mut best);
                }
            }
        }
        let mut h = mix(view, n as u64);
        for (a, b) in best {
            h = mix(mix(h, a as u64), b as u64);
        }
        keys.push(h);
    }
}

/// Feature IDs before folding: one per (feature, copy up to `count_cap`), sorted.
/// Folding them gives the same bits as `fold`.
fn feature_ids(mut keys: Vec<u64>, count_cap: usize) -> Vec<u64> {
    keys.sort_unstable();
    let mut ids = Vec::with_capacity(keys.len());
    let mut k = 0;
    while k < keys.len() {
        let end = keys[k..].iter().position(|&x| x != keys[k]).map_or(keys.len(), |p| k + p);
        ids.extend((0..(end - k).min(count_cap)).map(|copy| mix(keys[k], copy as u64)));
        k = end;
    }
    ids.sort_unstable();
    ids
}

/// Fold features into `W` words: the n-th copy of a feature, for n below
/// `count_cap`, sets bit `mix(key, n)`. Copies are counted in a small
/// open-addressing table rather than by sorting.
fn fold<const W: usize>(keys: &[u64], count_cap: usize) -> Fp<W> {
    let bits = (W * 64) as u64;
    let mut fp = [0u64; W];
    let mut set = |id: u64| {
        let bit = (id % bits) as usize;
        fp[bit / 64] |= 1 << (bit % 64);
    };
    let size = (2 * keys.len()).next_power_of_two().max(16);
    let mask = size - 1;
    // (key, copies seen); keys are hashes, so their low bits index the table.
    let mut table = vec![(0u64, 0u32); size];
    for &k in keys {
        let mut i = k as usize & mask;
        loop {
            let (tk, tc) = &mut table[i];
            if *tc == 0 {
                *tk = k;
                *tc = 1;
                set(mix(k, 0));
                break;
            }
            if *tk == k {
                if (*tc as usize) < count_cap {
                    set(mix(k, *tc as u64));
                    *tc += 1;
                }
                break;
            }
            i = (i + 1) & mask;
        }
    }
    fp
}

fn target_keys(kind: &FpKind, t: &Target) -> Vec<u64> {
    let atoms: Vec<AtomLabels> = t.atoms.iter().map(AtomLabels::target).collect();
    let nbrs: Vec<Vec<(usize, BondLabels)>> = t
        .nbrs
        .iter()
        .map(|ns| ns.iter().map(|&(n, k)| (n as usize, BondLabels::target(k))).collect())
        .collect();
    let mut keys = Vec::new();
    collect(kind, &atoms, &nbrs, &mut keys);
    keys
}

fn query_keys(kind: &FpKind, q: &Query) -> Vec<u64> {
    let atoms: Vec<AtomLabels> = q.atoms.iter().map(|a| AtomLabels::query(&a.known)).collect();
    let nbrs: Vec<Vec<(usize, BondLabels)>> = q
        .nbrs
        .iter()
        .map(|ns| ns.iter().map(|&(n, b)| (n, BondLabels::query(q.bonds[b].query))).collect())
        .collect();
    let mut keys = Vec::new();
    collect(kind, &atoms, &nbrs, &mut keys);
    keys
}

pub fn target_features(kind: &FpKind, t: &Target) -> Vec<u64> {
    feature_ids(target_keys(kind, t), kind.count_cap)
}

pub fn query_features(kind: &FpKind, q: &Query) -> Vec<u64> {
    feature_ids(query_keys(kind, q), kind.count_cap)
}

pub fn target_fp<const W: usize>(kind: &FpKind, t: &Target) -> Fp<W> {
    fold(&target_keys(kind, t), kind.count_cap)
}

pub fn query_fp<const W: usize>(kind: &FpKind, q: &Query) -> Fp<W> {
    fold(&query_keys(kind, q), kind.count_cap)
}

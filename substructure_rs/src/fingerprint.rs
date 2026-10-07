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

/// Adjacency in one flat list: atom `i`'s neighbours are `edges[start[i]..start[i + 1]]`.
#[derive(Default)]
struct Nbrs {
    start: Vec<usize>,
    edges: Vec<(usize, BondLabels)>,
}

impl Nbrs {
    /// Refill from per-atom neighbour lists.
    fn fill<'a, I, J>(&mut self, lists: I)
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = (usize, BondLabels)>,
    {
        self.start.clear();
        self.edges.clear();
        self.start.push(0);
        for ns in lists {
            self.edges.extend(ns);
            self.start.push(self.edges.len());
        }
    }

    fn len(&self) -> usize {
        self.start.len() - 1
    }
}

impl std::ops::Index<usize> for Nbrs {
    type Output = [(usize, BondLabels)];

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.edges[self.start[i]..self.start[i + 1]]
    }
}

/// Buffers reused from one molecule to the next on each thread.
#[derive(Default)]
struct Scratch {
    atoms: Vec<AtomLabels>,
    nbrs: Nbrs,
    keys: Vec<u64>,
    table: Vec<(u64, u32)>,
    path: Vec<usize>,
    bonds: Vec<BondLabels>,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// Collect features from a graph given as labels and adjacency.
fn collect(
    kind: &FpKind,
    atoms: &[AtomLabels],
    nbrs: &Nbrs,
    keys: &mut Vec<u64>,
    path: &mut Vec<usize>,
    bonds: &mut Vec<BondLabels>,
) {
    for a in atoms {
        for (view, label) in [(1, a.z), (2, a.z_arom), (3, a.full), (4, a.z_degree), (5, a.z_h), (6, a.z_charge)] {
            if let Some(l) = label {
                keys.push(mix(view, l as u64));
            }
        }
    }
    path.clear();
    for i in 0..atoms.len() {
        paths_from(kind, atoms, nbrs, i, path, keys);
    }
    if kind.branches {
        branches(atoms, nbrs, keys);
    }
    if kind.max_cycle >= 3 {
        cycles(kind.max_cycle, atoms, nbrs, keys, path, bonds);
    }
}

/// Path labellings: (view, atom label, bond label). Paths longer than
/// `max_edges` use only the second; the last only up to `max_full_edges`.
const PATH_VIEWS: [(u64, AtomView, BondView); 4] = [
    (10, |a| a.z, |_| Some(0)),
    (11, |a| a.z, |b| b.class),
    (12, |a| a.z_arom, |b| b.exact),
    (13, |a| a.full, |b| b.exact),
];

/// Odd multiplier of the polynomial path hashes.
const K: u64 = 0x9e37_79b9_7f4a_7c15;

#[inline]
fn atom_token(l: u32) -> u64 {
    mix(0xa70, l as u64)
}

#[inline]
fn bond_token(l: u32) -> u64 {
    mix(0xb0d, l as u64)
}

/// Per path labelling, polynomial hashes of the label sequence read forward
/// and backward, or `None` once a label is open. A path and its reverse swap
/// the two, so the smaller one keys the path in either direction; both
/// extend in constant time as the path grows.
type Runs = [Option<(u64, u64)>; 4];

fn paths_from(kind: &FpKind, atoms: &[AtomLabels], nbrs: &Nbrs, start: usize, path: &mut Vec<usize>, keys: &mut Vec<u64>) {
    let mut runs: Runs = [None; 4];
    for (run, &(_, af, _)) in runs.iter_mut().zip(&PATH_VIEWS) {
        *run = af(&atoms[start]).map(|l| (atom_token(l), atom_token(l)));
    }
    path.push(start);
    // `pow` = K^(labels so far): one atom.
    extend(kind, atoms, nbrs, path, runs, K, keys);
    path.pop();
}

fn extend(kind: &FpKind, atoms: &[AtomLabels], nbrs: &Nbrs, path: &mut Vec<usize>, runs: Runs, pow: u64, keys: &mut Vec<u64>) {
    // Bonds in the extended path.
    let edges = path.len();
    if edges > kind.max_edges.max(kind.long_edges) {
        return;
    }
    let wanted = [edges <= kind.max_edges, true, edges <= kind.max_edges, edges <= kind.max_full_edges];
    let last = *path.last().unwrap();
    for &(next, bl) in &nbrs[last] {
        if path.contains(&next) {
            continue;
        }
        let mut next_runs: Runs = [None; 4];
        let mut any = false;
        for v in 0..PATH_VIEWS.len() {
            let (Some((f, r)), true) = (runs[v], wanted[v]) else { continue };
            let (_, af, bf) = PATH_VIEWS[v];
            if let (Some(b), Some(a)) = (bf(&bl), af(&atoms[next])) {
                let (tb, ta) = (bond_token(b), atom_token(a));
                let f = f.wrapping_mul(K).wrapping_add(tb).wrapping_mul(K).wrapping_add(ta);
                let r = r.wrapping_add(tb.wrapping_mul(pow)).wrapping_add(ta.wrapping_mul(pow.wrapping_mul(K)));
                next_runs[v] = Some((f, r));
                any = true;
            }
        }
        // Open labels stay open, so no longer path can emit anything either.
        if !any {
            continue;
        }
        if path[0] < next {
            for (v, &run) in next_runs.iter().enumerate() {
                if let Some((f, r)) = run {
                    keys.push(mix(mix(PATH_VIEWS[v].0, edges as u64), f.min(r)));
                }
            }
        }
        path.push(next);
        extend(kind, atoms, nbrs, path, next_runs, pow.wrapping_mul(K).wrapping_mul(K), keys);
        path.pop();
    }
}

/// Paths are at most `MAX_PATH_ATOMS - 1` bonds long.
const MAX_PATH_ATOMS: usize = 16;

type AtomView = fn(&AtomLabels) -> Option<u32>;
type BondView = fn(&BondLabels) -> Option<u32>;

/// Each atom with every three of its neighbours: the centre label and the
/// sorted (bond, atom) labels of the arms.
fn branches(atoms: &[AtomLabels], nbrs: &Nbrs, keys: &mut Vec<u64>) {
    let views: [(u64, AtomView, BondView); 3] = [
        (20, |a| a.z, |b| b.class),
        (21, |a| a.z_arom, |b| b.exact),
        (22, |a| a.full, |b| b.exact),
    ];
    for c in 0..nbrs.len() {
        let ns = &nbrs[c];
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
fn cycles(
    max_len: usize,
    atoms: &[AtomLabels],
    nbrs: &Nbrs,
    keys: &mut Vec<u64>,
    path: &mut Vec<usize>,
    bonds: &mut Vec<BondLabels>,
) {
    fn walk(
        max_len: usize,
        atoms: &[AtomLabels],
        nbrs: &Nbrs,
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
    path.clear();
    bonds.clear();
    for s in 0..atoms.len() {
        path.push(s);
        walk(max_len, atoms, nbrs, path, bonds, keys);
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
    let mut seq = [(0u32, 0u32); MAX_PATH_ATOMS];
    'view: for (view, af, bf) in views {
        for ((slot, &i), b) in seq.iter_mut().zip(path).zip(bonds) {
            match (af(&atoms[i]), bf(b)) {
                (Some(a), Some(b)) => *slot = (a, b),
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
        let mut best = [(0u32, 0u32); MAX_PATH_ATOMS];
        let mut cand = [(0u32, 0u32); MAX_PATH_ATOMS];
        for (i, slot) in best[..n].iter_mut().enumerate() {
            *slot = forward(0, i);
        }
        for r in 0..n {
            for dir in [0, 1] {
                for (i, slot) in cand[..n].iter_mut().enumerate() {
                    *slot = if dir == 0 { forward(r, i) } else { backward(r, i) };
                }
                if cand[..n] < best[..n] {
                    best = cand;
                }
            }
        }
        let mut h = mix(view, n as u64);
        for &(a, b) in &best[..n] {
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
fn fold<const W: usize>(keys: &[u64], count_cap: usize, table: &mut Vec<(u64, u32)>) -> Fp<W> {
    let bits = (W * 64) as u64;
    let mut fp = [0u64; W];
    let mut set = |id: u64| {
        let bit = (id % bits) as usize;
        fp[bit / 64] |= 1 << (bit % 64);
    };
    let size = (2 * keys.len()).next_power_of_two().max(16);
    let mask = size - 1;
    // (key, copies seen); keys are hashes, so their low bits index the table.
    table.clear();
    table.resize(size, (0, 0));
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

impl Scratch {
    fn load_target(&mut self, t: &Target) {
        self.atoms.clear();
        self.atoms.extend(t.atoms.iter().map(AtomLabels::target));
        self.nbrs.fill(t.nbrs.iter().map(|ns| ns.iter().map(|&(n, k)| (n as usize, BondLabels::target(k)))));
    }

    fn load_query(&mut self, q: &Query) {
        self.atoms.clear();
        self.atoms.extend(q.atoms.iter().map(|a| AtomLabels::query(&a.known)));
        self.nbrs.fill(q.nbrs.iter().map(|ns| ns.iter().map(|&(n, b)| (n, BondLabels::query(q.bonds[b].query)))));
    }

    /// Features of the loaded molecule into `self.keys`.
    fn collect(&mut self, kind: &FpKind) {
        self.keys.clear();
        let Scratch { atoms, nbrs, keys, path, bonds, .. } = self;
        collect(kind, atoms, nbrs, keys, path, bonds);
    }

    fn fold<const W: usize>(&mut self, kind: &FpKind) -> Fp<W> {
        fold(&self.keys, kind.count_cap, &mut self.table)
    }
}

/// Run `f` with this thread's scratch buffers.
fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    SCRATCH.with_borrow_mut(f)
}

fn target_keys(kind: &FpKind, t: &Target) -> Vec<u64> {
    with_scratch(|s| {
        s.load_target(t);
        s.collect(kind);
        s.keys.clone()
    })
}

fn query_keys(kind: &FpKind, q: &Query) -> Vec<u64> {
    with_scratch(|s| {
        s.load_query(q);
        s.collect(kind);
        s.keys.clone()
    })
}

pub fn target_features(kind: &FpKind, t: &Target) -> Vec<u64> {
    feature_ids(target_keys(kind, t), kind.count_cap)
}

pub fn query_features(kind: &FpKind, q: &Query) -> Vec<u64> {
    feature_ids(query_keys(kind, q), kind.count_cap)
}

pub fn target_fp<const W: usize>(kind: &FpKind, t: &Target) -> Fp<W> {
    with_scratch(|s| {
        s.load_target(t);
        s.collect(kind);
        s.fold(kind)
    })
}

pub fn query_fp<const W: usize>(kind: &FpKind, q: &Query) -> Fp<W> {
    with_scratch(|s| {
        s.load_query(q);
        s.collect(kind);
        s.fold(kind)
    })
}

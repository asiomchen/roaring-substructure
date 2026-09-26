//! Port of RDKit's `MolOps::findSSSR` and `MolOps::symmetrizeSSSR`
//! (FindRings.cpp, RDKit 2026.03).
//!
//! Aromaticity perception depends on which rings are found and in what order
//! (an atom's electron-donor type is fixed by the first ring that contains it),
//! so this follows RDKit's traversal order step by step rather than using a
//! textbook minimum cycle basis.
//!
//! Author: Marcin Kowiel + Claude

use crate::mol::{BondType, Mol};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

pub struct Rings {
    /// Atom indices in ring order, in RDKit's `RingInfo::atomRings()` order.
    pub atom_rings: Vec<Vec<usize>>,
    pub bond_in_ring: Vec<bool>,
}

/// `computeRingInvariant`: the ring's atom set as a bitset.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Inv(Vec<u64>);

impl Inv {
    fn new(ring: &[usize], n: usize) -> Self {
        let mut v = vec![0u64; n.div_ceil(64).max(1)];
        for &a in ring {
            v[a / 64] |= 1 << (a % 64);
        }
        Inv(v)
    }
}

/// boost::dynamic_bitset `operator<`: compare as numbers, most significant block first.
impl Ord for Inv {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}

impl PartialOrd for Inv {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct Finder<'a> {
    mol: &'a Mol,
    active: Vec<bool>,
    degrees: Vec<i32>,
    invars: HashSet<Inv>,
    ring_atoms: Vec<bool>,
    ring_bonds: Vec<bool>,
    /// Rings removed by `removeExtraRings`, used by `symmetrizeSSSR`.
    extras: Vec<Vec<usize>>,
}

impl Finder<'_> {
    fn n(&self) -> usize {
        self.mol.atoms.len()
    }

    fn bond_between(&self, a: usize, b: usize) -> usize {
        self.mol.bond_between(a, b).expect("ring bond")
    }

    fn trim_bonds(&mut self, cand: usize, changed: &mut VecDeque<usize>) {
        trim_bonds(self.mol, cand, changed, &mut self.degrees, &mut self.active);
    }

    fn add_ring(&mut self, res: &mut Vec<Vec<usize>>, ring: Vec<usize>, mark: bool) -> bool {
        let inv = Inv::new(&ring, self.n());
        if !self.invars.insert(inv) {
            return false;
        }
        if mark {
            for i in 0..ring.len() - 1 {
                let b = self.bond_between(ring[i], ring[i + 1]);
                self.ring_bonds[b] = true;
                self.ring_atoms[ring[i]] = true;
            }
            let b = self.bond_between(ring[0], ring[ring.len() - 1]);
            self.ring_bonds[b] = true;
            self.ring_atoms[ring[ring.len() - 1]] = true;
        }
        res.push(ring);
        true
    }

    fn pick_d2_nodes(&self, frag: &[usize]) -> Vec<usize> {
        let mut d2 = Vec::new();
        let mut forb = vec![false; self.n()];
        loop {
            let Some(&root) = frag.iter().find(|&&a| self.degrees[a] == 2 && !forb[a]) else {
                break;
            };
            d2.push(root);
            forb[root] = true;
            self.mark_useless_d2s(root, &mut forb);
        }
        d2
    }

    fn mark_useless_d2s(&self, root: usize, forb: &mut [bool]) {
        for &(o, b) in &self.mol.nbrs[root] {
            if !self.active[b] {
                continue;
            }
            if !forb[o] && self.degrees[o] == 2 {
                forb[o] = true;
                self.mark_useless_d2s(o, forb);
            }
        }
    }

    fn find_rings_d2_nodes(&mut self, res: &mut Vec<Vec<usize>>, d2nodes: &[usize]) {
        let mut dup_d2_cands: BTreeMap<Inv, Vec<usize>> = BTreeMap::new();
        let mut dup_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for &cand in d2nodes {
            let mut srings = Vec::new();
            smallest_rings_bfs(self.mol, cand, &mut srings, &self.active, &[]);
            for nring in &srings {
                let inv = Inv::new(nring, self.n());
                let known = self.invars.contains(&inv);
                let dups = dup_d2_cands.entry(inv).or_default();
                if !known {
                    self.add_ring(res, nring.clone(), true);
                } else {
                    for &other in dups.iter() {
                        dup_map.entry(cand).or_default().push(other);
                        dup_map.entry(other).or_default().push(cand);
                    }
                }
                dup_d2_cands.get_mut(&Inv::new(nring, self.n())).unwrap().push(cand);
            }
            if srings.is_empty() {
                let mut changed = VecDeque::from([cand]);
                while let Some(local) = changed.pop_front() {
                    self.trim_bonds(local, &mut changed);
                }
            }
        }
        // findSSSRforDupCands
        for dup_cands in dup_d2_cands.values() {
            if dup_cands.len() <= 1 {
                continue;
            }
            let mut nrings: Vec<Vec<usize>> = Vec::new();
            let mut min_size = usize::MAX;
            for &dup_cand in dup_cands {
                let mut degrees = self.degrees.clone();
                let mut active = self.active.clone();
                let mut changed = VecDeque::new();
                for &dn in dup_map.get(&dup_cand).expect("duplicate could not be found") {
                    trim_bonds(self.mol, dn, &mut changed, &mut degrees, &mut active);
                }
                let mut srings = Vec::new();
                smallest_rings_bfs(self.mol, dup_cand, &mut srings, &active, &[]);
                for r in srings {
                    min_size = min_size.min(r.len());
                    nrings.push(r);
                }
            }
            for r in nrings {
                if r.len() == min_size {
                    self.add_ring(res, r, false);
                }
            }
        }
    }

    fn find_rings_d3_node(&mut self, res: &mut Vec<Vec<usize>>, cand: usize) {
        let mut srings = Vec::new();
        let nsmall = smallest_rings_bfs(self.mol, cand, &mut srings, &self.active, &[]);
        for r in &srings {
            self.add_ring(res, r.clone(), false);
        }
        if nsmall >= 3 {
            return;
        }
        let nbr: Vec<usize> = self.mol.nbrs[cand]
            .iter()
            .filter(|&&(_, b)| self.active[b])
            .map(|&(o, _)| o)
            .take(3)
            .collect();
        assert!(nbr.len() == 3, "neighbor not found");
        let (n1, n2, n3) = (nbr[0], nbr[1], nbr[2]);
        let mut forbidden_runs: Vec<usize> = Vec::new();
        if nsmall == 2 {
            let both = |x: usize| srings[0].contains(&x) && srings[1].contains(&x);
            let f = if both(n1) {
                n1
            } else if both(n2) {
                n2
            } else if both(n3) {
                n3
            } else {
                panic!("third ring not found");
            };
            forbidden_runs.push(f);
        } else if nsmall == 1 {
            let (f1, f2) = if !srings[0].contains(&n1) {
                (n2, n3)
            } else if !srings[0].contains(&n2) {
                (n1, n3)
            } else if !srings[0].contains(&n3) {
                (n1, n2)
            } else {
                panic!("rings not found");
            };
            forbidden_runs.push(f2);
            forbidden_runs.push(f1);
        }
        for f in forbidden_runs {
            let mut trings = Vec::new();
            smallest_rings_bfs(self.mol, cand, &mut trings, &self.active, &[f]);
            for r in trings {
                self.add_ring(res, r, false);
            }
        }
    }

    fn atom_search_bfs(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        let mut q: VecDeque<Vec<usize>> = VecDeque::from([vec![start]]);
        while let Some(tv) = q.pop_front() {
            let curr = *tv.last().unwrap();
            for &(nbr, _) in &self.mol.nbrs[curr] {
                if nbr == end {
                    if curr != start {
                        let mut nv = tv.clone();
                        nv.push(nbr);
                        if !self.invars.contains(&Inv::new(&nv, self.n())) {
                            return Some(nv);
                        }
                    }
                } else if self.ring_atoms[nbr] && !tv.contains(&nbr) {
                    let mut nv = tv.clone();
                    nv.push(nbr);
                    q.push_back(nv);
                }
            }
        }
        None
    }

    /// `removeExtraRings`: keep a minimal set, remember the rest as extras.
    fn remove_extra_rings(&mut self, res: &mut Vec<Vec<usize>>) {
        res.sort_by_key(|r| r.len()); // small inputs: std::sort behaves stably
        let nb = self.mol.bonds.len();
        let brings: Vec<Vec<bool>> = res
            .iter()
            .map(|r| {
                let mut v = vec![false; nb];
                for k in 0..r.len() {
                    v[self.bond_between(r[k], r[(k + 1) % r.len()])] = true;
                }
                v
            })
            .collect();
        let sizes: Vec<usize> = res.iter().map(|r| r.len()).collect();
        let subset = |a: &[bool], u: &[bool]| a.iter().zip(u).all(|(&x, &y)| !x || y);
        let mut avail = vec![true; res.len()];
        let mut keep = vec![false; res.len()];
        let mut munion = vec![false; nb];
        for i in 0..res.len() {
            if subset(&brings[i], &munion) {
                avail[i] = false;
            }
            if !avail[i] {
                continue;
            }
            for (m, &b) in munion.iter_mut().zip(&brings[i]) {
                *m |= b;
            }
            keep[i] = true;
            let mut consider: Vec<bool> = (0..res.len()).map(|j| j > i && avail[j] && sizes[j] == sizes[i]).collect();
            while consider.iter().any(|&c| c) {
                let mut best_j = i + 1;
                let mut best_overlap: i64 = -1;
                let mut j = i + 1;
                while j < res.len() && sizes[j] == sizes[i] {
                    if consider[j] && avail[j] {
                        let overlap = brings[j].iter().zip(&munion).filter(|(&x, &y)| x && y).count() as i64;
                        if overlap > best_overlap {
                            best_overlap = overlap;
                            best_j = j;
                        }
                    }
                    j += 1;
                }
                consider[best_j] = false;
                if subset(&brings[best_j], &munion) {
                    avail[best_j] = false;
                } else {
                    keep[best_j] = true;
                    avail[best_j] = false;
                    for (m, &b) in munion.iter_mut().zip(&brings[best_j]) {
                        *m |= b;
                    }
                }
            }
        }
        let all = std::mem::take(res);
        for (i, r) in all.into_iter().enumerate() {
            if keep[i] {
                res.push(r);
            } else {
                self.extras.push(r);
            }
        }
    }
}

fn trim_bonds(mol: &Mol, cand: usize, changed: &mut VecDeque<usize>, degrees: &mut [i32], active: &mut [bool]) {
    for &(o, b) in &mol.nbrs[cand] {
        if !active[b] {
            continue;
        }
        if degrees[o] <= 2 {
            changed.push_back(o);
        }
        active[b] = false;
        degrees[o] -= 1;
        degrees[cand] -= 1;
    }
}

/// `smallestRingsBfs`: all smallest rings through `root`.
fn smallest_rings_bfs(mol: &Mol, root: usize, rings: &mut Vec<Vec<usize>>, active: &[bool], forbidden: &[usize]) -> usize {
    const WHITE: u8 = 0;
    const GRAY: u8 = 1;
    const BLACK: u8 = 2;
    let n = mol.atoms.len();
    let mut done = vec![WHITE; n];
    for &f in forbidden {
        done[f] = BLACK;
    }
    let mut parents: Vec<isize> = vec![-1; n];
    let mut depths = vec![0usize; n];
    let mut q = VecDeque::from([root]);
    let mut cur_size = usize::MAX;
    while let Some(curr) = q.pop_front() {
        done[curr] = BLACK;
        let depth = depths[curr] + 1;
        if depth > cur_size {
            break;
        }
        for &(nbr, b) in &mol.nbrs[curr] {
            if !active[b] {
                continue;
            }
            if done[nbr] == BLACK || parents[curr] == nbr as isize {
                continue;
            }
            if done[nbr] == WHITE {
                parents[nbr] = curr as isize;
                done[nbr] = GRAY;
                depths[nbr] = depth;
                q.push_back(nbr);
            } else {
                let mut ring = vec![nbr];
                let mut parent = parents[nbr];
                while parent != -1 && parent != root as isize {
                    ring.push(parent as usize);
                    parent = parents[parent as usize];
                }
                ring.insert(0, curr);
                parent = parents[curr];
                while parent != -1 {
                    if ring.contains(&(parent as usize)) {
                        ring.clear();
                        break;
                    }
                    ring.insert(0, parent as usize);
                    parent = parents[parent as usize];
                }
                if ring.len() > 1 {
                    if ring.len() <= cur_size {
                        cur_size = ring.len();
                        rings.push(ring);
                    } else {
                        return rings.len();
                    }
                }
            }
        }
    }
    rings.len()
}

/// `getMolFrags`: connected components, atoms ascending, ordered by lowest atom.
fn fragments(mol: &Mol) -> Vec<Vec<usize>> {
    let n = mol.atoms.len();
    let mut comp = vec![usize::MAX; n];
    let mut frags: Vec<Vec<usize>> = Vec::new();
    for s in 0..n {
        if comp[s] != usize::MAX {
            continue;
        }
        let id = frags.len();
        let mut stack = vec![s];
        comp[s] = id;
        let mut atoms = Vec::new();
        while let Some(u) = stack.pop() {
            atoms.push(u);
            for &(v, _) in &mol.nbrs[u] {
                if comp[v] == usize::MAX {
                    comp[v] = id;
                    stack.push(v);
                }
            }
        }
        atoms.sort_unstable();
        frags.push(atoms);
    }
    frags
}

/// `MolOps::findSSSR`; returns the rings and the extras for symmetrization.
fn find_sssr(mol: &Mol) -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let n = mol.atoms.len();
    let nb_total = mol.bonds.len();
    let active: Vec<bool> = mol.bonds.iter().map(|b| b.kind != BondType::Dative).collect();
    let degrees_with_zero: Vec<i32> = (0..n).map(|i| mol.nbrs[i].len() as i32).collect();
    let degrees: Vec<i32> = (0..n)
        .map(|i| mol.nbrs[i].iter().filter(|&&(_, b)| active[b]).count() as i32)
        .collect();
    let mut f = Finder {
        mol,
        active,
        degrees,
        invars: HashSet::new(),
        ring_atoms: vec![false; n],
        ring_bonds: vec![false; nb_total],
        extras: Vec::new(),
    };
    let mut res = Vec::new();
    for frag in fragments(mol) {
        if frag.len() < 3 {
            continue;
        }
        let mut changed = VecDeque::new();
        let mut bonds_with_zero = 0;
        let mut nbnds = 0i32;
        for &a in &frag {
            bonds_with_zero += degrees_with_zero[a];
            let deg = f.degrees[a];
            nbnds += deg;
            if deg < 2 {
                changed.push_back(a);
            }
        }
        let bonds_with_zero = bonds_with_zero / 2;
        if bonds_with_zero - frag.len() as i32 + 1 < 1 {
            continue;
        }
        let nbnds = nbnds / 2;
        let mut done = vec![false; n];
        let mut n_done = 0usize;
        let mut frag_res: Vec<Vec<usize>> = Vec::new();
        while n_done <= frag.len() - 3 {
            while let Some(cand) = changed.pop_front() {
                if !done[cand] {
                    done[cand] = true;
                    n_done += 1;
                    f.trim_bonds(cand, &mut changed);
                }
            }
            let d2 = f.pick_d2_nodes(&frag);
            if !d2.is_empty() {
                f.find_rings_d2_nodes(&mut frag_res, &d2);
                for &d in &d2 {
                    done[d] = true;
                    n_done += 1;
                    f.trim_bonds(d, &mut changed);
                }
            } else if n_done <= frag.len() - 3 {
                let Some(&cand) = frag.iter().find(|&&a| f.degrees[a] == 3) else { break };
                f.find_rings_d3_node(&mut frag_res, cand);
                done[cand] = true;
                n_done += 1;
                f.trim_bonds(cand, &mut changed);
            }
        }
        let nexpt = nbnds - frag.len() as i32 + 1;
        if (frag_res.len() as i32) < nexpt {
            // RDKit scans bond indices below the fragment's bond count.
            let pick = |f: &Finder, dead: &[bool]| {
                (0..nbnds as usize).find(|&i| {
                    i < nb_total
                        && !f.ring_bonds[i]
                        && !dead[i]
                        && f.ring_atoms[mol.bonds[i].a]
                        && f.ring_atoms[mol.bonds[i].b]
                })
            };
            let mut dead = vec![false; nb_total];
            let mut possible = pick(&f, &dead);
            while let Some(b) = possible {
                let (s, e) = (mol.bonds[b].a, mol.bonds[b].b);
                match f.atom_search_bfs(s, e) {
                    Some(r) => {
                        f.add_ring(&mut frag_res, r, true);
                    }
                    None => dead[b] = true,
                }
                possible = pick(&f, &dead);
            }
            if (frag_res.len() as i32) < nexpt {
                return None; // RDKit falls back to fastFindRings here
            }
        }
        if frag_res.len() as i32 > nexpt {
            f.remove_extra_rings(&mut frag_res);
        }
        res.extend(frag_res);
    }
    Some((res, f.extras))
}

/// `MolOps::symmetrizeSSSR`.
pub fn symmetrized_sssr(mol: &Mol) -> Rings {
    let nb = mol.bonds.len();
    let ring_bonds = |r: &[usize]| -> Vec<usize> {
        (0..r.len()).map(|k| mol.bond_between(r[k], r[(k + 1) % r.len()]).expect("ring bond")).collect()
    };
    let (sssr, extras) = find_sssr(mol).unwrap_or_else(|| (fallback_rings(mol), Vec::new()));
    let mut res = sssr.clone();
    let bond_sssrs: Vec<Vec<usize>> = sssr.iter().map(|r| ring_bonds(r)).collect();
    let mut bond_counts = vec![0; nb];
    for r in &bond_sssrs {
        for &b in r {
            bond_counts[b] += 1;
        }
    }
    for extra in &extras {
        let extra_bonds = ring_bonds(extra);
        for ring in &bond_sssrs {
            if ring.len() != extra_bonds.len() {
                continue;
            }
            let mut share = false;
            let mut replaces_all_unique = true;
            for &b in ring {
                if bond_counts[b] == 1 || !share {
                    if extra_bonds.contains(&b) {
                        share = true;
                    } else if bond_counts[b] == 1 {
                        replaces_all_unique = false;
                    }
                }
            }
            if share && replaces_all_unique {
                res.push(extra.clone());
                break;
            }
        }
    }
    let mut bond_in_ring = vec![false; nb];
    for r in &res {
        for b in ring_bonds(r) {
            bond_in_ring[b] = true;
        }
    }
    Rings { atom_rings: res, bond_in_ring }
}

/// Simple cycle basis for the rare molecules where RDKit gives up on SSSR.
fn fallback_rings(mol: &Mol) -> Vec<Vec<usize>> {
    let n = mol.atoms.len();
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut depth = vec![usize::MAX; n];
    let mut rings = Vec::new();
    let mut seen_bond = vec![false; mol.bonds.len()];
    for s in 0..n {
        if depth[s] != usize::MAX {
            continue;
        }
        depth[s] = 0;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for &(v, b) in &mol.nbrs[u] {
                if seen_bond[b] || mol.bonds[b].kind == BondType::Dative {
                    continue;
                }
                seen_bond[b] = true;
                if depth[v] == usize::MAX {
                    depth[v] = depth[u] + 1;
                    parent[v] = Some(u);
                    stack.push(v);
                } else {
                    let (mut a, mut c) = (u, v);
                    let (mut left, mut right) = (vec![a], vec![c]);
                    while a != c {
                        if depth[a] >= depth[c] {
                            a = parent[a].unwrap();
                            left.push(a);
                        } else {
                            c = parent[c].unwrap();
                            right.push(c);
                        }
                    }
                    right.pop();
                    left.extend(right.into_iter().rev());
                    rings.push(left);
                }
            }
        }
    }
    rings
}

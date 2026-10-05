//! Reactant molecules: SMILES parsing and a port of the parts of RDKit's
//! sanitization that decide aromaticity and hydrogen counts.
//!
//! Each step follows the RDKit 2026.03 C++ source it is named after, so that
//! matches agree with `Chem.MolFromSmiles(smiles).HasSubstructMatch(query)`.
//!
//! Author: Marcin Kowiel + Claude

use crate::periodic::{more_electronegative, TABLE};
use crate::rings::{symmetrized_sssr, Rings};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Quadruple,
    Aromatic,
    /// `a->b`: stored with `a` as the donor (bond.a) and `b` as the acceptor (bond.b).
    Dative,
}

impl BondType {
    /// Doubled bond order, so aromatic bonds stay integral.
    fn order2(self) -> i32 {
        match self {
            BondType::Single => 2,
            BondType::Double => 4,
            BondType::Triple => 6,
            BondType::Quadruple => 8,
            BondType::Aromatic => 3,
            BondType::Dative => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Atom {
    pub z: u8,
    pub charge: i8,
    pub isotope: u16,
    pub aromatic: bool,
    pub explicit_h: u8,
    pub no_implicit: bool,
    pub implicit_h: u8,
    pub radicals: u8,
    explicit_valence: i32,
}

#[derive(Clone, Debug)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
    pub kind: BondType,
    pub aromatic: bool,
    /// Written with `/` or `\\` in the SMILES.
    pub directional: bool,
}

/// A sanitized molecule, ready for matching.
#[derive(Clone, Debug, Default)]
pub struct Mol {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// Per atom: (neighbor atom, bond index).
    pub nbrs: Vec<Vec<(usize, usize)>>,
}

/// Matching view of one atom after sanitization and hydrogen removal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetAtom {
    pub z: u8,
    pub aromatic: bool,
    pub total_h: u8,
    pub degree: u8,
    pub charge: i8,
}

impl Mol {
    fn add_bond(&mut self, a: usize, b: usize, kind: BondType, aromatic: bool, directional: bool) {
        let idx = self.bonds.len();
        self.bonds.push(Bond { a, b, kind, aromatic, directional });
        self.nbrs[a].push((b, idx));
        self.nbrs[b].push((a, idx));
    }

    /// RDKit's `Bond::getValenceContrib(atom)`, doubled: a dative bond counts
    /// only towards its acceptor.
    fn contrib2(&self, b: usize, atom: usize) -> i32 {
        let bond = &self.bonds[b];
        if bond.kind == BondType::Dative && bond.b != atom {
            0
        } else {
            bond.kind.order2()
        }
    }

    pub fn degree(&self, atom: usize) -> usize {
        self.nbrs[atom].len()
    }

    pub fn bond_between(&self, a: usize, b: usize) -> Option<usize> {
        self.nbrs[a].iter().find(|&&(n, _)| n == b).map(|&(_, bi)| bi)
    }

    pub fn target_atoms(&self) -> Vec<TargetAtom> {
        (0..self.atoms.len())
            .map(|i| {
                let a = &self.atoms[i];
                let h_nbrs = self.nbrs[i].iter().filter(|&&(n, _)| self.atoms[n].z == 1).count();
                TargetAtom {
                    z: a.z,
                    aromatic: a.aromatic,
                    total_h: a.explicit_h + a.implicit_h + h_nbrs as u8,
                    degree: self.degree(i) as u8,
                    charge: a.charge,
                }
            })
            .collect()
    }

    // ---------------------------------------------------------------- valence

    fn is_aromatic_atom(&self, i: usize) -> bool {
        self.atoms[i].aromatic
            || self.nbrs[i].iter().any(|&(_, b)| {
                self.bonds[b].aromatic || self.bonds[b].kind == BondType::Aromatic
            })
    }

    fn effective_z(&self, i: usize) -> usize {
        let a = &self.atoms[i];
        let ovalens = TABLE[a.z as usize].valences;
        if ovalens.len() > 1 || ovalens[0] != -1 {
            (a.z as i32 - a.charge as i32).clamp(0, 118) as usize
        } else {
            a.z as usize
        }
    }

    /// `calculateExplicitValence(atom, strict=false)`.
    fn calc_explicit_valence(&self, i: usize) -> i32 {
        let a = &self.atoms[i];
        let mut accum: f64 = self.nbrs[i]
            .iter()
            .map(|&(_, b)| self.contrib2(b, i) as f64 / 2.0)
            .sum::<f64>()
            + a.explicit_h as f64;
        let effz = self.effective_z(i);
        let dv = TABLE[effz].default_valence;
        if accum > dv as f64 && self.is_aromatic_atom(i) {
            let mut pval = dv;
            for &val in TABLE[effz].valences {
                if val == -1 || val as f64 > accum {
                    break;
                }
                pval = val;
            }
            if accum - pval as f64 <= 1.5 {
                accum = pval as f64;
            }
        }
        (accum + 0.1).round() as i32
    }

    /// `calculateImplicitValence(atom, strict=false)`.
    fn calc_implicit_h(&self, i: usize) -> i32 {
        let a = &self.atoms[i];
        if a.no_implicit || a.z == 0 {
            return 0;
        }
        let ev = a.explicit_valence;
        if ev == 0 && a.radicals == 0 && a.z == 1 {
            return if a.charge == 0 { 1 } else { 0 };
        }
        let mut explicit_plus_rad = ev + a.radicals as i32;
        let mut effz = self.effective_z(i);
        if effz == 0 {
            return 0;
        }
        let dv = TABLE[effz].default_valence;
        if dv == -1 {
            return 0;
        }
        let z = a.z as usize;
        if (effz > 16 && (z == 15 || z == 16)) || (effz > 34 && (z == 33 || z == 34)) {
            effz = z;
            explicit_plus_rad -= a.charge as i32;
        }
        let valens = TABLE[effz].valences;
        if self.is_aromatic_atom(i) {
            if explicit_plus_rad <= dv {
                dv - explicit_plus_rad
            } else {
                0
            }
        } else {
            valens
                .iter()
                .take_while(|&&v| v >= 0)
                .find(|&&v| explicit_plus_rad <= v)
                .map_or(0, |&v| v - explicit_plus_rad)
        }
    }

    fn update_property_cache(&mut self) {
        for i in 0..self.atoms.len() {
            self.atoms[i].explicit_valence = self.calc_explicit_valence(i);
            self.atoms[i].implicit_h = self.calc_implicit_h(i).max(0) as u8;
        }
    }

    fn total_h(&self, i: usize) -> i32 {
        (self.atoms[i].explicit_h + self.atoms[i].implicit_h) as i32
    }

    fn total_valence(&self, i: usize) -> i32 {
        self.atoms[i].explicit_valence + self.atoms[i].implicit_h as i32
    }

    // ---------------------------------------------------------------- cleanUp

    /// `MolOps::cleanUp`: nitro/azide nitrogens, phosphorus, and halogen oxides.
    fn clean_up(&mut self) {
        let mut considered = Vec::new();
        for i in 0..self.atoms.len() {
            if self.atoms[i].z != 7 || self.atoms[i].charge != 0 {
                continue;
            }
            if self.calc_explicit_valence(i) != 5 {
                continue;
            }
            considered.push(i);
            let hold = self.atoms[i].aromatic;
            self.atoms[i].aromatic = false;
            for k in 0..self.nbrs[i].len() {
                let (n, b) = self.nbrs[i][k];
                if self.atoms[n].z == 8 && self.atoms[n].charge == 0 && self.bonds[b].kind == BondType::Double {
                    self.bonds[b].kind = BondType::Single;
                    self.atoms[i].charge = 1;
                    self.atoms[n].charge = -1;
                    break;
                }
            }
            self.atoms[i].aromatic = hold;
        }
        for &i in &considered {
            let hold = self.atoms[i].aromatic;
            self.atoms[i].aromatic = false;
            for k in 0..self.nbrs[i].len() {
                let (n, b) = self.nbrs[i][k];
                if self.atoms[n].z == 7 && self.atoms[n].charge == 0 && self.bonds[b].kind == BondType::Triple {
                    self.bonds[b].kind = BondType::Double;
                    self.atoms[i].charge = 1;
                    self.atoms[n].charge = -1;
                    break;
                }
            }
            self.atoms[i].aromatic = hold;
        }
        for i in 0..self.atoms.len() {
            match self.atoms[i].z {
                15 => self.phosphorus_cleanup(i),
                17 | 35 | 53 => self.halogen_cleanup(i),
                _ => {}
            }
        }
    }

    fn phosphorus_cleanup(&mut self, i: usize) {
        if self.atoms[i].charge != 0 || self.calc_explicit_valence(i) != 5 || self.degree(i) != 3 {
            return;
        }
        let mut dbl_to_o = None;
        let mut has_double_to_c_or_n = false;
        for &(n, b) in &self.nbrs[i] {
            let nz = self.atoms[n].z;
            if nz == 8 && self.atoms[n].charge == 0 && self.bonds[b].kind == BondType::Double {
                dbl_to_o = Some((n, b));
            } else if (nz == 6 || nz == 7) && self.degree(n) >= 2 && self.bonds[b].kind == BondType::Double {
                has_double_to_c_or_n = true;
            }
        }
        if let (true, Some((o, b))) = (has_double_to_c_or_n, dbl_to_o) {
            self.atoms[o].charge = -1;
            self.bonds[b].kind = BondType::Single;
            self.atoms[i].charge = 1;
        }
    }

    fn halogen_cleanup(&mut self, i: usize) {
        let ev = self.calc_explicit_valence(i);
        if self.atoms[i].charge != 0 || !(ev == 7 || ev == 5 || ev == 3) {
            return;
        }
        if !self.nbrs[i].iter().all(|&(n, _)| self.atoms[n].z == 8) {
            return;
        }
        let mut charge = 0;
        for k in 0..self.nbrs[i].len() {
            let (n, b) = self.nbrs[i][k];
            if self.bonds[b].kind == BondType::Double {
                self.bonds[b].kind = BondType::Single;
                charge += 1;
                self.atoms[n].charge = -1;
            }
        }
        self.atoms[i].charge = charge;
    }

    // ---------------------------------------------------------------- kekulize

    /// `markDbondCands` for one non-dummy aromatic atom.
    fn needs_double_bond(&self, i: usize) -> bool {
        let a = &self.atoms[i];
        let mut sbo = 0;
        let mut n_to_ignore = 0;
        for &(_, b) in &self.nbrs[i] {
            let bond = &self.bonds[b];
            if bond.aromatic
                && matches!(bond.kind, BondType::Single | BondType::Double | BondType::Aromatic)
            {
                sbo += 1;
            } else {
                let contrib = (self.contrib2(b, i) as f64 / 2.0).round() as i32;
                sbo += contrib;
                if contrib == 0 {
                    n_to_ignore += 1;
                }
            }
        }
        sbo += self.total_h(i);
        let z = a.z as usize;
        let mut dv = TABLE[z].default_valence;
        let mut chrg = a.charge as i32;
        if is_early_atom(a.z) {
            chrg = -chrg;
        }
        if a.z == 6 && chrg > 0 {
            chrg = -chrg;
        }
        dv += chrg;
        let tbo = self.total_valence(i);
        let n_radicals = a.radicals as i32;
        let total_degree = self.degree(i) as i32 + a.implicit_h as i32 - n_to_ignore;
        let val_list = TABLE[z].valences;
        let mut vi = 1;
        while tbo > dv && vi < val_list.len() && val_list[vi] > 0 {
            dv = val_list[vi] + chrg;
            vi += 1;
        }
        if tbo == 5
            && sbo == 4
            && dv == 3
            && total_degree == 3
            && n_radicals == 0
            && chrg == 0
            && self.total_h(i) == 0
            && matches!(a.z, 7 | 15 | 33)
        {
            dv = 5;
        }
        if total_degree + n_radicals >= dv {
            return false;
        }
        dv == sbo + 1 + n_radicals || (n_radicals == 0 && a.no_implicit && dv == sbo + 2)
    }

    /// Replace aromatic bonds with a Kekulé structure and clear aromatic flags.
    fn kekulize(&mut self) -> Result<(), String> {
        let n = self.atoms.len();
        let in_arom: Vec<bool> = (0..n).map(|i| self.is_aromatic_atom(i)).collect();
        if !in_arom.iter().any(|&x| x) {
            return Ok(());
        }
        let cands: Vec<bool> = (0..n).map(|i| in_arom[i] && self.atoms[i].z != 0 && self.needs_double_bond(i)).collect();
        // Aromatic bonds between two candidates can take a double bond.
        let edges: Vec<Vec<(usize, usize)>> = (0..n)
            .map(|i| {
                if !cands[i] {
                    return Vec::new();
                }
                self.nbrs[i]
                    .iter()
                    .copied()
                    .filter(|&(m, b)| cands[m] && (self.bonds[b].aromatic || self.bonds[b].kind == BondType::Aromatic))
                    .collect()
            })
            .collect();
        let mut mate: Vec<Option<usize>> = vec![None; n];
        let mut budget = 2_000_000u64;
        if !perfect_matching(&cands, &edges, &mut mate, &mut budget) {
            return Err("can't kekulize".into());
        }
        for b in 0..self.bonds.len() {
            let bond = &mut self.bonds[b];
            if bond.aromatic || bond.kind == BondType::Aromatic {
                bond.kind = if mate[bond.a] == Some(b) { BondType::Double } else { BondType::Single };
                bond.aromatic = false;
            }
        }
        for a in &mut self.atoms {
            a.aromatic = false;
        }
        for i in 0..n {
            self.atoms[i].explicit_valence = self.calc_explicit_valence(i);
        }
        Ok(())
    }

    // ---------------------------------------------------------------- radicals

    /// `MolOps::assignRadicals`.
    fn assign_radicals(&mut self) {
        for i in 0..self.atoms.len() {
            let a = &self.atoms[i];
            if !a.no_implicit || a.z == 0 {
                continue;
            }
            let valens = TABLE[a.z as usize].valences;
            let chg = a.charge as i32;
            let n_outer = TABLE[a.z as usize].outer;
            let radicals = if valens.len() != 1 || valens[0] != -1 {
                let accum: f64 = self.nbrs[i]
                    .iter()
                    .map(|&(_, b)| self.contrib2(b, i) as f64 / 2.0)
                    .sum::<f64>()
                    + a.explicit_h as f64;
                let total_valence = (accum + 0.1) as i32;
                let base_count = if a.z == 1 || a.z == 2 { 2 } else { 8 };
                let mut num = base_count - n_outer - total_valence + chg;
                if num < 0 {
                    num = 0;
                    if valens.len() > 1 {
                        if let Some(&v) = valens.iter().find(|&&v| v - total_valence + chg >= 0) {
                            num = v - total_valence + chg;
                        }
                    }
                }
                let num2 = n_outer - total_valence - chg;
                if num2 >= 0 {
                    num = num.min(num2);
                }
                num
            } else if self.degree(i) > 0 {
                0
            } else {
                let n_valence = n_outer - chg;
                if n_valence < 0 {
                    0
                } else {
                    n_valence % 2
                }
            };
            self.atoms[i].radicals = radicals.max(0) as u8;
        }
    }

    // ---------------------------------------------------------------- aromaticity

    fn in_ring_bond(&self, rings: &Rings, b: usize) -> bool {
        rings.bond_in_ring[b]
    }

    fn incident_noncyclic_multiple_bond(&self, rings: &Rings, i: usize) -> Option<usize> {
        self.nbrs[i]
            .iter()
            .find(|&&(_, b)| !self.in_ring_bond(rings, b) && self.contrib2(b, i) >= 4)
            .map(|&(n, _)| n)
    }

    fn incident_cyclic_multiple_bond(&self, rings: &Rings, i: usize) -> bool {
        self.nbrs[i]
            .iter()
            .any(|&(_, b)| self.in_ring_bond(rings, b) && self.contrib2(b, i) >= 4)
    }

    fn incident_multiple_bond(&self, i: usize) -> bool {
        let zero = self.nbrs[i].iter().filter(|&&(_, b)| (self.contrib2(b, i) as f64 / 2.0).round() == 0.0).count();
        let deg = (self.degree(i) - zero) as i32 + self.atoms[i].explicit_h as i32;
        self.atoms[i].explicit_valence != deg
    }

    /// `MolOps::countAtomElec`.
    fn count_atom_elec(&self, i: usize) -> i32 {
        let a = &self.atoms[i];
        let dv = TABLE[a.z as usize].default_valence;
        if dv <= 1 {
            return -1;
        }
        let zero = self.nbrs[i].iter().filter(|&&(_, b)| self.contrib2(b, i) == 0).count() as i32;
        let degree = self.degree(i) as i32 + self.total_h(i) - zero;
        if degree > 3 {
            return -1;
        }
        let nlp = (TABLE[a.z as usize].outer - dv - a.charge as i32).max(0);
        let mut res = (dv - degree) + nlp - a.radicals as i32;
        if res > 1 && a.explicit_valence - self.degree(i) as i32 > 1 {
            res = 1;
        }
        res
    }

    /// `getAtomDonorTypeArom(at, exocyclicBondsStealElectrons=true)`.
    fn donor_type(&self, rings: &Rings, i: usize) -> Donor {
        let a = &self.atoms[i];
        if a.z == 0 {
            return if self.incident_cyclic_multiple_bond(rings, i) { Donor::One } else { Donor::Any };
        }
        let mut nelec = self.count_atom_elec(i);
        if nelec < 0 {
            Donor::No
        } else if nelec == 0 {
            if self.incident_noncyclic_multiple_bond(rings, i).is_some() {
                Donor::Vacant
            } else if self.incident_cyclic_multiple_bond(rings, i) {
                Donor::One
            } else {
                Donor::No
            }
        } else if nelec == 1 {
            if let Some(who) = self.incident_noncyclic_multiple_bond(rings, i) {
                if more_electronegative(self.atoms[who].z, a.z) {
                    Donor::Vacant
                } else {
                    Donor::One
                }
            } else if self.incident_multiple_bond(i) {
                Donor::One
            } else if a.charge == 1 {
                Donor::Vacant
            } else {
                Donor::No
            }
        } else {
            if let Some(who) = self.incident_noncyclic_multiple_bond(rings, i) {
                if more_electronegative(self.atoms[who].z, a.z) {
                    nelec -= 1;
                }
            }
            if nelec % 2 == 1 {
                Donor::One
            } else {
                Donor::Two
            }
        }
    }

    /// `isAtomCandForArom` with RDKit's default arguments.
    fn is_arom_candidate(&self, i: usize, donor: Donor) -> bool {
        let a = &self.atoms[i];
        if a.z > 18 && a.z != 34 && a.z != 52 {
            return false;
        }
        if donor == Donor::No {
            return false;
        }
        let def_val = TABLE[a.z as usize].default_valence;
        let shifted = (a.z as i32 - a.charge as i32).clamp(0, 118) as usize;
        if def_val > 0 && self.total_valence(i) > TABLE[shifted].default_valence {
            return false;
        }
        if a.radicals > 0 && (a.z != 6 || a.charge != 0) {
            return false;
        }
        if a.explicit_valence - self.degree(i) as i32 > 1 {
            let n_mult = self.nbrs[i]
                .iter()
                .filter(|&&(_, b)| matches!(self.bonds[b].kind, BondType::Double | BondType::Triple))
                .count();
            if n_mult > 1 {
                return false;
            }
        }
        true
    }

    /// `aromaticityHelper(mol, srings, 0, 0, includeFused=true)`.
    fn set_aromaticity(&mut self, rings: &Rings) {
        let n = self.atoms.len();
        let mut seen = vec![false; n];
        let mut cand = vec![false; n];
        let mut edon = vec![Donor::No; n];
        let mut c_rings: Vec<&Vec<usize>> = Vec::new();
        for ring in &rings.atom_rings {
            let size = ring.len();
            let mut all_arom = true;
            let mut all_dummy = true;
            for &i in ring {
                if self.atoms[i].z != 0 {
                    all_dummy = false;
                }
                if seen[i] {
                    if !cand[i] {
                        all_arom = false;
                    }
                    continue;
                }
                seen[i] = true;
                let mut donor = self.donor_type(rings, i);
                if donor == Donor::Two && size >= 9 {
                    let a = &self.atoms[i];
                    if (a.z == 8 || a.z == 16)
                        && self.degree(i) == 2
                        && a.charge == 0
                        && !self.nbrs[i]
                            .iter()
                            .any(|&(_, b)| matches!(self.bonds[b].kind, BondType::Double | BondType::Triple))
                    {
                        donor = Donor::No;
                    }
                }
                edon[i] = donor;
                cand[i] = self.is_arom_candidate(i, donor);
                if !cand[i] {
                    all_arom = false;
                }
            }
            if all_arom && !all_dummy {
                c_rings.push(ring);
            }
        }
        if c_rings.is_empty() {
            return;
        }
        let b_rings: Vec<Vec<usize>> = c_rings.iter().map(|r| self.ring_bonds(r)).collect();
        // makeRingNeighborMap(maxSize=24, maxOverlapSize=1)
        let nr = c_rings.len();
        let mut neigh: Vec<Vec<usize>> = vec![Vec::new(); nr];
        for i in 0..nr {
            if b_rings[i].len() > 24 {
                continue;
            }
            for j in i + 1..nr {
                if b_rings[j].len() > 24 {
                    continue;
                }
                let inter = b_rings[i].iter().filter(|b| b_rings[j].contains(b)).count();
                if inter == 1 {
                    neigh[i].push(j);
                    neigh[j].push(i);
                }
            }
        }
        let mut done = vec![false; nr];
        for start in 0..nr {
            if done[start] {
                continue;
            }
            let mut fused = Vec::new();
            pick_fused(start, &neigh, &mut fused, &mut done);
            self.huckel_fused(&c_rings, &b_rings, &fused, &edon, &neigh);
        }
    }

    fn ring_bonds(&self, ring: &[usize]) -> Vec<usize> {
        (0..ring.len())
            .map(|k| self.bond_between(ring[k], ring[(k + 1) % ring.len()]).expect("ring bond"))
            .collect()
    }

    fn huckel_fused(
        &mut self,
        a_rings: &[&Vec<usize>],
        b_rings: &[Vec<usize>],
        fused: &[usize],
        edon: &[Donor],
        neigh: &[Vec<usize>],
    ) {
        let nrings = fused.len();
        let mut fused_bonds: Vec<usize> = fused.iter().flat_map(|&r| b_rings[r].iter().copied()).collect();
        fused_bonds.sort_unstable();
        fused_bonds.dedup();
        let n_ring_bonds = fused_bonds.len();
        let mut done_bonds: Vec<bool> = vec![false; self.bonds.len()];
        let mut n_done = 0;
        let mut counts = vec![0u8; self.atoms.len()];
        for cur_size in 1..=nrings.min(6) {
            if n_done >= n_ring_bonds {
                break;
            }
            let mut comb: Vec<usize> = (0..cur_size).collect();
            loop {
                let cur: Vec<usize> = comb.iter().map(|&k| fused[k]).collect();
                if check_fused(&cur, neigh) {
                    counts.iter_mut().for_each(|c| *c = 0);
                    for &r in &cur {
                        for &a in a_rings[r] {
                            counts[a] += 1;
                        }
                    }
                    let unon: Vec<usize> = (0..counts.len()).filter(|&a| counts[a] == 1 || counts[a] == 2).collect();
                    if apply_huckel(&unon, edon) {
                        n_done += self.mark_arom(b_rings, &cur, &mut done_bonds);
                    }
                }
                if !next_combination(&mut comb, nrings) {
                    break;
                }
            }
        }
    }

    /// `markAtomsBondsArom`; returns the number of newly done bonds.
    fn mark_arom(&mut self, b_rings: &[Vec<usize>], ring_ids: &[usize], done: &mut [bool]) -> usize {
        let mut cnt: Vec<(usize, u32)> = Vec::new();
        for &r in ring_ids {
            for &b in &b_rings[r] {
                match cnt.iter_mut().find(|(x, _)| *x == b) {
                    Some(e) => e.1 += 1,
                    None => cnt.push((b, 1)),
                }
            }
        }
        let mut newly = 0;
        for (b, c) in cnt {
            if c != 1 {
                continue;
            }
            let bond = &mut self.bonds[b];
            bond.aromatic = true;
            if matches!(bond.kind, BondType::Single | BondType::Double) {
                bond.kind = BondType::Aromatic;
                let (x, y) = (bond.a, bond.b);
                self.atoms[x].aromatic = true;
                self.atoms[y].aromatic = true;
            }
            if !done[b] {
                done[b] = true;
                newly += 1;
            }
        }
        newly
    }

    // ---------------------------------------------------------------- removeHs

    /// Is the H on this directional bond the reference for double-bond stereo?
    fn defines_bond_stereo(&self, h: usize) -> bool {
        let (heavy, hb) = self.nbrs[h][0];
        self.bonds[hb].directional
            && self.nbrs[heavy].iter().any(|&(x, db)| {
                self.bonds[db].kind == BondType::Double
                    && self.nbrs[x].iter().any(|&(y, b)| y != heavy && self.bonds[b].directional)
            })
    }

    /// Drop plain `[H]` atoms bonded to one heavy atom, as `removeHs` does by default.
    fn remove_hs(&mut self) {
        let removable: Vec<bool> = (0..self.atoms.len())
            .map(|i| {
                let a = &self.atoms[i];
                a.z == 1
                    && a.isotope == 0
                    && a.charge == 0
                    && self.nbrs[i].len() == 1
                    && self.atoms[self.nbrs[i][0].0].z != 1
                    && self.bonds[self.nbrs[i][0].1].kind == BondType::Single
                    && !self.defines_bond_stereo(i)
            })
            .collect();
        if !removable.iter().any(|&r| r) {
            return;
        }
        let mut new_idx = vec![usize::MAX; self.atoms.len()];
        let mut atoms = Vec::new();
        for (i, a) in self.atoms.iter().enumerate() {
            if !removable[i] {
                new_idx[i] = atoms.len();
                atoms.push(a.clone());
            }
        }
        for (i, &r) in removable.iter().enumerate() {
            if r {
                let heavy = new_idx[self.nbrs[i][0].0];
                atoms[heavy].explicit_h += 1;
            }
        }
        let old_bonds = std::mem::take(&mut self.bonds);
        self.atoms = atoms;
        self.nbrs = vec![Vec::new(); self.atoms.len()];
        for b in old_bonds {
            if new_idx[b.a] != usize::MAX && new_idx[b.b] != usize::MAX {
                self.add_bond(new_idx[b.a], new_idx[b.b], b.kind, b.aromatic, b.directional);
            }
        }
    }

    /// Sanitize a freshly parsed molecule the way `MolFromSmiles` does.
    pub fn sanitize(&mut self) -> Result<(), String> {
        self.clean_up();
        self.update_property_cache();
        let rings = symmetrized_sssr(self);
        self.kekulize()?;
        self.assign_radicals();
        self.set_aromaticity(&rings);
        self.remove_hs();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Donor {
    Vacant,
    One,
    Two,
    Any,
    No,
}

fn electron_range(d: Donor) -> (i32, i32) {
    match d {
        Donor::Any => (1, 2),
        Donor::One => (1, 1),
        Donor::Two => (2, 2),
        Donor::Vacant | Donor::No => (0, 0),
    }
}

fn apply_huckel(ring: &[usize], edon: &[Donor]) -> bool {
    let (mut lo, mut hi) = (0, 0);
    let mut n_any = 0;
    for &i in ring {
        if edon[i] == Donor::Any {
            n_any += 1;
            if n_any > 1 {
                return false;
            }
        }
        let (l, h) = electron_range(edon[i]);
        lo += l;
        hi += h;
    }
    if hi >= 6 {
        (lo..=hi).any(|e| (e - 2) % 4 == 0)
    } else {
        hi == 2
    }
}

fn pick_fused(cur: usize, neigh: &[Vec<usize>], res: &mut Vec<usize>, done: &mut [bool]) {
    done[cur] = true;
    res.push(cur);
    for &n in &neigh[cur] {
        if !done[n] {
            pick_fused(n, neigh, res, done);
        }
    }
}

/// `RingUtils::checkFused`: are the rings connected through the neighbor map?
fn check_fused(rids: &[usize], neigh: &[Vec<usize>]) -> bool {
    if rids.len() == 1 {
        return true;
    }
    let mut seen = vec![rids[0]];
    let mut stack = vec![rids[0]];
    while let Some(r) = stack.pop() {
        for &n in &neigh[r] {
            if rids.contains(&n) && !seen.contains(&n) {
                seen.push(n);
                stack.push(n);
            }
        }
    }
    seen.len() == rids.len()
}

fn next_combination(comb: &mut [usize], n: usize) -> bool {
    let k = comb.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if comb[i] < n - k + i {
            comb[i] += 1;
            for j in i + 1..k {
                comb[j] = comb[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

fn is_early_atom(z: u8) -> bool {
    // RDKit's isEarlyAtom table (Atom.cpp).
    matches!(
        z,
        3 | 4 | 5 | 11 | 12 | 13 | 19..=22 | 30..=32 | 37..=41 | 48..=51 | 55..=61 | 72 | 73 | 80..=83
            | 87..=93 | 104..=118
    )
}

/// Backtracking perfect matching over candidate atoms, most-constrained first.
fn perfect_matching(
    cands: &[bool],
    edges: &[Vec<(usize, usize)>],
    mate: &mut Vec<Option<usize>>,
    budget: &mut u64,
) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    let mut best: Option<(usize, usize)> = None;
    for i in 0..cands.len() {
        if !cands[i] || mate[i].is_some() {
            continue;
        }
        let options = edges[i].iter().filter(|&&(m, _)| mate[m].is_none()).count();
        if options == 0 {
            return false;
        }
        if best.is_none_or(|(_, o)| options < o) {
            best = Some((i, options));
        }
    }
    let Some((i, _)) = best else { return true };
    for &(m, b) in &edges[i] {
        if mate[m].is_some() {
            continue;
        }
        mate[i] = Some(b);
        mate[m] = Some(b);
        if perfect_matching(cands, edges, mate, budget) {
            return true;
        }
        mate[i] = None;
        mate[m] = None;
    }
    false
}

// -------------------------------------------------------------------- SMILES

const ORGANIC: [&str; 10] = ["Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I"];
const AROMATIC_ORGANIC: [&str; 6] = ["b", "c", "n", "o", "p", "s"];
const AROMATIC_BRACKET: [&str; 10] = ["se", "te", "as", "si", "b", "c", "n", "o", "p", "s"];

fn element_z(symbol: &str) -> Result<u8, String> {
    crate::periodic::symbol_to_z(symbol).ok_or_else(|| format!("unknown element {symbol}"))
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
}

#[derive(Clone, Copy)]
struct Pending {
    kind: BondType,
    directional: bool,
    reversed: bool,
}

/// Parse a SMILES string and sanitize it like `Chem.MolFromSmiles`.
pub fn parse_smiles(smiles: &str) -> Result<Mol, String> {
    let s = smiles.split_whitespace().next().unwrap_or("").as_bytes();
    let mut mol = Mol::default();
    // Bonds whose order was not written: fixed after parsing.
    let mut unspecified: Vec<bool> = Vec::new();
    let mut prev: Option<usize> = None;
    let mut stack: Vec<Option<usize>> = Vec::new();
    let mut pending: Option<Pending> = None;
    let mut ring_open: std::collections::HashMap<u32, (usize, Option<Pending>)> = Default::default();
    // RDKit adds ring-closure bonds after the whole string is parsed (CloseMolRings),
    // by ascending ring number, from the closing atom to the opening one. Bond
    // order drives ring perception, so this matters for aromaticity.
    let mut closures: Vec<(u32, usize, usize, Option<Pending>)> = Vec::new();
    let mut i = 0;
    let text = std::str::from_utf8(s).map_err(|e| e.to_string())?;
    // `a` is the atom written first; `<-` makes the later atom the dative donor.
    let connect = |mol: &mut Mol, unspecified: &mut Vec<bool>, a: usize, b: usize, kind: Option<Pending>| {
        let p = kind.unwrap_or(Pending { kind: BondType::Single, directional: false, reversed: false });
        let (a, b) = if p.reversed { (b, a) } else { (a, b) };
        mol.add_bond(a, b, p.kind, p.kind == BondType::Aromatic, p.directional);
        unspecified.push(kind.is_none());
    };
    while i < s.len() {
        let c = s[i] as char;
        match c {
            '(' => {
                stack.push(prev);
                i += 1;
            }
            ')' => {
                prev = stack.pop().ok_or("unbalanced )")?;
                i += 1;
            }
            '.' => {
                prev = None;
                i += 1;
            }
            '-' if s.get(i + 1) == Some(&b'>') => {
                pending = Some(Pending { kind: BondType::Dative, directional: false, reversed: false });
                i += 2;
            }
            '<' if s.get(i + 1) == Some(&b'-') => {
                pending = Some(Pending { kind: BondType::Dative, directional: false, reversed: true });
                i += 2;
            }
            '-' | '=' | '#' | '$' | ':' | '/' | '\\' => {
                let kind = match c {
                    '-' | '/' | '\\' => BondType::Single,
                    '=' => BondType::Double,
                    '#' => BondType::Triple,
                    '$' => BondType::Quadruple,
                    _ => BondType::Aromatic,
                };
                pending = Some(Pending { kind, directional: c == '/' || c == '\\', reversed: false });
                i += 1;
            }
            '0'..='9' | '%' => {
                let num = if c == '%' {
                    if s.get(i + 1) == Some(&b'(') {
                        let end = text[i..].find(')').ok_or("bad %(")? + i;
                        let v = text[i + 2..end].parse::<u32>().map_err(|e| e.to_string())?;
                        i = end + 1;
                        v
                    } else {
                        let v = text.get(i + 1..i + 3).ok_or("bad %nn")?.parse::<u32>().map_err(|e| e.to_string())?;
                        i += 3;
                        v
                    }
                } else {
                    i += 1;
                    c.to_digit(10).unwrap()
                };
                let cur = prev.ok_or("ring closure without atom")?;
                if let Some((other, kind)) = ring_open.remove(&num) {
                    // A bond written at the opening digit points from the opening atom.
                    let kind = pending.take().or(kind.map(|p| Pending { reversed: !p.reversed, ..p }));
                    closures.push((num, cur, other, kind));
                } else {
                    ring_open.insert(num, (cur, pending.take()));
                }
            }
            _ => {
                let (atom, len) = if c == '[' {
                    let end = text[i..].find(']').ok_or("unclosed [")? + i;
                    (parse_bracket_atom(&text[i + 1..end])?, end + 1 - i)
                } else {
                    parse_organic_atom(&text[i..])?
                };
                i += len;
                let idx = mol.atoms.len();
                mol.atoms.push(atom);
                mol.nbrs.push(Vec::new());
                if let Some(p) = prev {
                    connect(&mut mol, &mut unspecified, p, idx, pending.take());
                }
                pending = None;
                prev = Some(idx);
            }
        }
    }
    if !ring_open.is_empty() || !stack.is_empty() {
        return Err("unclosed ring or branch".into());
    }
    closures.sort_by_key(|c| c.0); // stable: reused digits keep their order
    for (_, a, b, kind) in closures {
        connect(&mut mol, &mut unspecified, a, b, kind);
    }
    for (b, &unspec) in unspecified.iter().enumerate() {
        if unspec {
            let (x, y) = (mol.bonds[b].a, mol.bonds[b].b);
            if mol.atoms[x].aromatic && mol.atoms[y].aromatic {
                mol.bonds[b].kind = BondType::Aromatic;
                mol.bonds[b].aromatic = true;
            }
        }
    }
    mol.sanitize()?;
    Ok(mol)
}

fn new_atom(z: u8, aromatic: bool) -> Atom {
    Atom {
        z,
        charge: 0,
        isotope: 0,
        aromatic,
        explicit_h: 0,
        no_implicit: false,
        implicit_h: 0,
        radicals: 0,
        explicit_valence: 0,
    }
}

fn parse_organic_atom(s: &str) -> Result<(Atom, usize), String> {
    if s.starts_with('*') {
        return Ok((new_atom(0, false), 1));
    }
    for sym in ORGANIC {
        if s.starts_with(sym) {
            return Ok((new_atom(element_z(sym)?, false), sym.len()));
        }
    }
    for sym in AROMATIC_ORGANIC {
        if s.starts_with(sym) {
            return Ok((new_atom(element_z(&capitalize(sym))?, true), 1));
        }
    }
    Err(format!("unexpected SMILES character in {s:?}"))
}

fn parse_bracket_atom(s: &str) -> Result<Atom, String> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let isotope = if i > 0 { s[..i].parse::<u16>().map_err(|e| e.to_string())? } else { 0 };
    let rest = &s[i..];
    let (z, aromatic, len) = if rest.starts_with('*') {
        (0, false, 1)
    } else if let Some(sym) = AROMATIC_BRACKET.iter().find(|sym| rest.starts_with(**sym)) {
        (element_z(&capitalize(sym))?, true, sym.len())
    } else {
        let two = rest.get(..2).filter(|t| t.as_bytes()[1].is_ascii_lowercase());
        match two.and_then(crate::periodic::symbol_to_z) {
            Some(z) => (z, false, 2),
            None => (element_z(rest.get(..1).ok_or("empty bracket atom")?)?, false, 1),
        }
    };
    i += len;
    let mut atom = new_atom(z, aromatic);
    atom.isotope = isotope;
    atom.no_implicit = true;
    // Chirality: @, @@, @TH1, @AL2, @SP3, @TB12, @OH30 ...
    if b.get(i) == Some(&b'@') {
        i += 1;
        if b.get(i) == Some(&b'@') {
            i += 1;
        } else if i + 1 < b.len() && b[i].is_ascii_uppercase() && b[i + 1].is_ascii_uppercase() {
            i += 2;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
    }
    if b.get(i) == Some(&b'H') {
        i += 1;
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        atom.explicit_h = if start == i { 1 } else { s[start..i].parse().map_err(|_| "bad H count")? };
    }
    if let Some(&sign) = b.get(i).filter(|&&c| c == b'+' || c == b'-') {
        let unit: i8 = if sign == b'+' { 1 } else { -1 };
        i += 1;
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        atom.charge = if start < i {
            unit * s[start..i].parse::<i8>().map_err(|_| "bad charge")?
        } else {
            let mut q = unit;
            while b.get(i) == Some(&sign) {
                q += unit;
                i += 1;
            }
            q
        };
    }
    if b.get(i) == Some(&b':') {
        i = b.len();
    }
    if i != b.len() {
        return Err(format!("unsupported bracket atom [{s}]"));
    }
    Ok(atom)
}

//! Subgraph monomorphism (non-induced, like RDKit's VF2 matcher).
//!
//! Author: Marcin Kowiel + Claude

use crate::mol::{BondType, Mol, TargetAtom};
use crate::smarts::{BondQuery, Query};

/// A reactant prepared for repeated matching.
pub struct Target {
    pub atoms: Vec<TargetAtom>,
    /// Per atom: (neighbor, bond type).
    pub nbrs: Vec<Vec<(u32, BondType)>>,
}

impl Target {
    pub fn new(mol: &Mol) -> Self {
        let nbrs = mol
            .nbrs
            .iter()
            .map(|ns| ns.iter().map(|&(n, b)| (n as u32, mol.bonds[b].kind)).collect())
            .collect();
        Target { atoms: mol.target_atoms(), nbrs }
    }

    #[inline]
    fn bond(&self, a: u32, b: u32) -> Option<BondType> {
        self.nbrs[a as usize].iter().find(|&&(n, _)| n == b).map(|&(_, k)| k)
    }
}

/// Query atoms in search order; each step lists bonds back to earlier atoms.
pub struct Plan {
    order: Vec<usize>,
    /// For step k: (earlier step index, bond query); the first is the anchor when present.
    back: Vec<Vec<(usize, BondQuery)>>,
    anchored: Vec<bool>,
}

impl Plan {
    pub fn new(q: &Query) -> Self {
        let n = q.atoms.len();
        let score = |i: usize| {
            let k = &q.atoms[i].known;
            let mut s = q.nbrs[i].len() as i32 * 2;
            s += [k.total_h.is_some(), k.degree.is_some(), k.charge.is_some(), k.aromatic.is_some()]
                .iter()
                .filter(|&&x| x)
                .count() as i32;
            // Rare elements first.
            if let Some(z) = k.z {
                if z != 6 {
                    s += 4;
                }
            }
            s
        };
        let mut step_of = vec![usize::MAX; n];
        let mut order = Vec::with_capacity(n);
        while order.len() < n {
            // Start a new component at its best-scored atom.
            let start = (0..n).filter(|&i| step_of[i] == usize::MAX).max_by_key(|&i| score(i)).unwrap();
            step_of[start] = order.len();
            order.push(start);
            // Grow by the frontier atom with the most bonds to placed atoms, then score.
            loop {
                let next = (0..n)
                    .filter(|&i| step_of[i] == usize::MAX)
                    .filter_map(|i| {
                        let placed = q.nbrs[i].iter().filter(|&&(m, _)| step_of[m] != usize::MAX).count();
                        (placed > 0).then_some((placed, score(i), i))
                    })
                    .max();
                let Some((_, _, i)) = next else { break };
                step_of[i] = order.len();
                order.push(i);
            }
        }
        let mut back = Vec::with_capacity(n);
        let mut anchored = Vec::with_capacity(n);
        for (k, &i) in order.iter().enumerate() {
            let mut b: Vec<(usize, BondQuery)> = q.nbrs[i]
                .iter()
                .filter(|&&(m, _)| step_of[m] < k)
                .map(|&(m, bi)| (step_of[m], q.bonds[bi].query))
                .collect();
            b.sort_by_key(|&(s, _)| s);
            anchored.push(!b.is_empty());
            back.push(b);
        }
        Plan { order, back, anchored }
    }
}

/// Does `q` occur in `t`?
pub fn has_match(q: &Query, plan: &Plan, t: &Target) -> bool {
    let n = plan.order.len();
    if n > t.atoms.len() {
        return false;
    }
    let mut mapped = vec![u32::MAX; n];
    let mut used = vec![false; t.atoms.len()];
    search(q, plan, t, 0, &mut mapped, &mut used)
}

fn search(q: &Query, plan: &Plan, t: &Target, k: usize, mapped: &mut [u32], used: &mut [bool]) -> bool {
    if k == plan.order.len() {
        return true;
    }
    let qa = &q.atoms[plan.order[k]];
    let back = &plan.back[k];
    let try_atom = |cand: u32, mapped: &mut [u32], used: &mut [bool]| -> bool {
        if used[cand as usize] || !qa.expr.matches(&t.atoms[cand as usize]) {
            return false;
        }
        let skip = usize::from(plan.anchored[k]);
        for &(s, bq) in &back[skip..] {
            match t.bond(mapped[s], cand) {
                Some(kind) if bq.matches(kind) => {}
                _ => return false,
            }
        }
        mapped[k] = cand;
        used[cand as usize] = true;
        if search(q, plan, t, k + 1, mapped, used) {
            return true;
        }
        used[cand as usize] = false;
        false
    };
    if plan.anchored[k] {
        let (s, bq) = back[0];
        let anchor = mapped[s];
        for &(cand, kind) in &t.nbrs[anchor as usize] {
            if bq.matches(kind) && try_atom(cand, mapped, used) {
                return true;
            }
        }
        false
    } else {
        (0..t.atoms.len() as u32).any(|cand| try_atom(cand, mapped, used))
    }
}

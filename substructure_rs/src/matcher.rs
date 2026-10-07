//! Subgraph monomorphism (non-induced, like RDKit's VF2 matcher).
//!
//! Author: Marcin Kowiel + Claude

use crate::mol::{BondType, Mol, TargetAtom};
use crate::smarts::{compile_atom, BondQuery, Expr, Query};

/// A reactant prepared for repeated matching.
pub struct Target {
    pub atoms: Vec<TargetAtom>,
    /// `TargetAtom::packed` per atom.
    packed: Vec<u64>,
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
        let atoms = mol.target_atoms();
        let packed = atoms.iter().map(|a| a.packed()).collect();
        Target { atoms, packed, nbrs }
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
    /// For step k: the query atom's test.
    tests: Vec<AtomTest>,
}

/// A query atom's test against a target atom.
enum AtomTest {
    /// `packed & mask == value`.
    One(u64, u64),
    /// Any of these terms.
    Any(Vec<(u64, u64)>),
    /// Evaluate the expression (it did not compile).
    Expr(Expr),
}

impl AtomTest {
    fn new(e: &Expr) -> Self {
        match compile_atom(e) {
            Some(terms) if terms.len() == 1 => AtomTest::One(terms[0].0, terms[0].1),
            Some(terms) => AtomTest::Any(terms),
            None => AtomTest::Expr(e.clone()),
        }
    }

    #[inline]
    fn matches(&self, t: &Target, atom: u32) -> bool {
        match self {
            AtomTest::One(mask, value) => t.packed[atom as usize] & mask == *value,
            AtomTest::Any(terms) => {
                let p = t.packed[atom as usize];
                terms.iter().any(|&(mask, value)| p & mask == value)
            }
            AtomTest::Expr(e) => e.matches(&t.atoms[atom as usize]),
        }
    }
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
        let tests = order.iter().map(|&i| AtomTest::new(&q.atoms[i].expr)).collect();
        Plan { order, back, anchored, tests }
    }
}

/// Does `q` occur in `t`?
pub fn has_match(_q: &Query, plan: &Plan, t: &Target) -> bool {
    let n = plan.order.len();
    if n > t.atoms.len() {
        return false;
    }
    // Small molecules keep the search state on the stack.
    const STACK_ATOMS: usize = 64;
    const STACK_WORDS: usize = 8;
    let words = t.atoms.len().div_ceil(64);
    if n <= STACK_ATOMS && words <= STACK_WORDS {
        let mut mapped = [u32::MAX; STACK_ATOMS];
        let mut used = [0u64; STACK_WORDS];
        search(plan, t, 0, &mut mapped[..n], &mut used[..words])
    } else {
        search(plan, t, 0, &mut vec![u32::MAX; n], &mut vec![0u64; words])
    }
}

/// `used` is a bitmap of target atoms already mapped.
fn search(plan: &Plan, t: &Target, k: usize, mapped: &mut [u32], used: &mut [u64]) -> bool {
    if k == plan.order.len() {
        return true;
    }
    let test = &plan.tests[k];
    let back = &plan.back[k];
    let skip = usize::from(plan.anchored[k]);
    let try_atom = |cand: u32, mapped: &mut [u32], used: &mut [u64]| -> bool {
        let (w, bit) = (cand as usize / 64, 1u64 << (cand % 64));
        if used[w] & bit != 0 || !test.matches(t, cand) {
            return false;
        }
        for &(s, bq) in &back[skip..] {
            match t.bond(mapped[s], cand) {
                Some(kind) if bq.matches(kind) => {}
                _ => return false,
            }
        }
        mapped[k] = cand;
        used[w] |= bit;
        if search(plan, t, k + 1, mapped, used) {
            return true;
        }
        used[w] &= !bit;
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

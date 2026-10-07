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
    steps: Vec<Step>,
}

/// Everything the search needs at one step, in one place.
struct Step {
    test: AtomTest,
    /// The earlier step this atom is bonded to and the bond, or `None` at a
    /// component root, where every target atom is a candidate.
    anchor: Option<(u32, BondQuery)>,
    /// Further bonds back to earlier steps (ring closures).
    closures: Box<[(u32, BondQuery)]>,
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

/// How a plan orders the query atoms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Root at the best-scored atom (bonds, fixed properties, non-carbon), grow
    /// by most bonds to placed atoms, then score. The original order.
    Score,
    /// Root at the atom whose label is rarest, then grow as `Score`. The default:
    /// the root scan tests every target atom, so a rare root leaves few to extend.
    RareRoot,
    /// Root at the rarest label; grow by most bonds to placed atoms, then rarity.
    Rare,
}

pub const ORDERS: [(&str, Order); 3] = [("rare-root", Order::RareRoot), ("score", Order::Score), ("rare", Order::Rare)];
pub const DEFAULT_ORDER: &str = "rare-root";

pub fn order(name: &str) -> Result<Order, String> {
    ORDERS.iter().find(|(n, _)| *n == name).map(|&(_, o)| o).ok_or_else(|| {
        let names: Vec<&str> = ORDERS.iter().map(|(n, _)| *n).collect();
        format!("unknown order {name:?}; choose one of {}", names.join(", "))
    })
}

/// How often atom labels occur, estimated from fully specified query atoms
/// (the queries are fragments of reactants, so they sample the same atoms).
pub struct LabelFreq {
    /// (packed label, count), one per distinct label.
    labels: Vec<(u64, u64)>,
    total: u64,
}

impl LabelFreq {
    pub fn new(queries: &[Query]) -> Self {
        let full = full_mask();
        let mut counts = std::collections::HashMap::new();
        for q in queries {
            for a in &q.atoms {
                if let Some(terms) = compile_atom(&a.expr) {
                    if let [(mask, value)] = terms[..] {
                        if mask == full {
                            *counts.entry(value).or_insert(0u64) += 1;
                        }
                    }
                }
            }
        }
        let total = counts.values().sum::<u64>().max(1);
        LabelFreq { labels: counts.into_iter().collect(), total }
    }

    /// Estimated share of atoms that pass `test`.
    fn share(&self, test: &AtomTest) -> f64 {
        let terms: &[(u64, u64)] = match test {
            AtomTest::One(m, v) => &[(*m, *v)],
            AtomTest::Any(terms) => terms,
            AtomTest::Expr(_) => return 1.0,
        };
        let hits: u64 = self
            .labels
            .iter()
            .filter(|&&(l, _)| terms.iter().any(|&(m, v)| l & m == v))
            .map(|&(_, c)| c)
            .sum();
        // Unseen labels are rare but possible.
        (hits as f64 + 0.5) / self.total as f64
    }
}

fn full_mask() -> u64 {
    let t = crate::mol::TargetAtom { z: 0xff, aromatic: true, total_h: 0xff, degree: 0xff, charge: -1 };
    // Every field's bits, whatever the layout.
    let mut mask = 0;
    for shift in (0..64).step_by(8) {
        if t.packed() >> shift & 0xff != 0 {
            mask |= 0xff << shift;
        }
    }
    mask
}

impl Plan {
    pub fn new(q: &Query) -> Self {
        Self::build(q, Order::Score, None)
    }

    pub fn with_order(q: &Query, order: Order, freq: &LabelFreq) -> Self {
        Self::build(q, order, Some(freq))
    }

    fn build(q: &Query, how: Order, freq: Option<&LabelFreq>) -> Self {
        let n = q.atoms.len();
        let atom_tests: Vec<AtomTest> = q.atoms.iter().map(|a| AtomTest::new(&a.expr)).collect();
        // Rarity as an integer key: higher = rarer.
        let rarity: Vec<i64> = match freq {
            Some(f) if how != Order::Score => {
                atom_tests.iter().map(|t| (-f.share(t).ln() * 1000.0) as i64).collect()
            }
            _ => vec![0; n],
        };
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
            let start = (0..n)
                .filter(|&i| step_of[i] == usize::MAX)
                .max_by_key(|&i| match how {
                    Order::Score => (0, score(i)),
                    Order::RareRoot | Order::Rare => (rarity[i], q.nbrs[i].len() as i32),
                })
                .unwrap();
            step_of[start] = order.len();
            order.push(start);
            // Grow by the frontier atom with the most bonds to placed atoms, then score.
            loop {
                let next = (0..n)
                    .filter(|&i| step_of[i] == usize::MAX)
                    .filter_map(|i| {
                        let placed = q.nbrs[i].iter().filter(|&&(m, _)| step_of[m] != usize::MAX).count();
                        let key = match how {
                            Order::Score | Order::RareRoot => score(i) as i64,
                            Order::Rare => rarity[i],
                        };
                        (placed > 0).then_some((placed, key, i))
                    })
                    .max();
                let Some((_, _, i)) = next else { break };
                step_of[i] = order.len();
                order.push(i);
            }
        }
        let mut atom_tests: Vec<Option<AtomTest>> = atom_tests.into_iter().map(Some).collect();
        let mut steps = Vec::with_capacity(n);
        for (k, &i) in order.iter().enumerate() {
            let mut b: Vec<(usize, BondQuery)> = q.nbrs[i]
                .iter()
                .filter(|&&(m, _)| step_of[m] < k)
                .map(|&(m, bi)| (step_of[m], q.bonds[bi].query))
                .collect();
            b.sort_by_key(|&(s, _)| s);
            let mut b = b.into_iter().map(|(s, bq)| (s as u32, bq));
            let anchor = b.next();
            steps.push(Step { test: atom_tests[i].take().unwrap(), anchor, closures: b.collect() });
        }
        Plan { order, steps }
    }
}

/// Search work, for `--search-stats`; `()` counts nothing.
pub trait Counter {
    /// A target atom offered for step `k` (`root` = unanchored step, which scans every atom).
    fn offered(&mut self, k: usize, root: bool);
    /// A target atom accepted for step `k`.
    fn placed(&mut self, k: usize);
}

impl Counter for () {
    #[inline(always)]
    fn offered(&mut self, _: usize, _: bool) {}
    #[inline(always)]
    fn placed(&mut self, _: usize) {}
}

/// Counts for one (query, target) pair.
#[derive(Clone, Default, Debug)]
pub struct SearchCounts {
    pub offered: u64,
    pub root_offered: u64,
    pub placed: u64,
    /// Deepest step reached (number of query atoms placed at once).
    pub depth: usize,
}

impl Counter for SearchCounts {
    fn offered(&mut self, _: usize, root: bool) {
        self.offered += 1;
        self.root_offered += root as u64;
    }
    fn placed(&mut self, k: usize) {
        self.placed += 1;
        self.depth = self.depth.max(k + 1);
    }
}

impl Plan {
    pub fn len(&self) -> usize {
        self.order.len()
    }
}

/// Does `q` occur in `t`?
pub fn has_match(q: &Query, plan: &Plan, t: &Target) -> bool {
    has_match_counted(q, plan, t, &mut ())
}

/// `has_match`, reporting the search work to `c`.
pub fn has_match_counted<C: Counter>(_q: &Query, plan: &Plan, t: &Target, c: &mut C) -> bool {
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
        search(plan, t, 0, &mut mapped[..n], &mut used[..words], c)
    } else {
        search(plan, t, 0, &mut vec![u32::MAX; n], &mut vec![0u64; words], c)
    }
}

/// `used` is a bitmap of target atoms already mapped.
fn search<C: Counter>(plan: &Plan, t: &Target, k: usize, mapped: &mut [u32], used: &mut [u64], c: &mut C) -> bool {
    let Some(step) = plan.steps.get(k) else { return true };
    let root = step.anchor.is_none();
    let mut try_atom = |cand: u32, mapped: &mut [u32], used: &mut [u64]| -> bool {
        c.offered(k, root);
        let (w, bit) = (cand as usize / 64, 1u64 << (cand % 64));
        if used[w] & bit != 0 || !step.test.matches(t, cand) {
            return false;
        }
        for &(s, bq) in step.closures.iter() {
            match t.bond(mapped[s as usize], cand) {
                Some(kind) if bq.matches(kind) => {}
                _ => return false,
            }
        }
        c.placed(k);
        mapped[k] = cand;
        used[w] |= bit;
        if search(plan, t, k + 1, mapped, used, c) {
            return true;
        }
        used[w] &= !bit;
        false
    };
    match (step.anchor, &step.test) {
        (Some((s, bq)), _) => {
            for &(cand, kind) in &t.nbrs[mapped[s as usize] as usize] {
                if bq.matches(kind) && try_atom(cand, mapped, used) {
                    return true;
                }
            }
            false
        }
        // Root scan: a tight pass over the packed labels; only hits go further.
        (None, &AtomTest::One(mask, value)) => {
            for (cand, &p) in t.packed.iter().enumerate() {
                if p & mask == value && try_atom(cand as u32, mapped, used) {
                    return true;
                }
            }
            false
        }
        (None, _) => (0..t.atoms.len() as u32).any(|cand| try_atom(cand, mapped, used)),
    }
}

//! Subgraph monomorphism (non-induced, like RDKit's VF2 matcher).
//!
//! Author: Marcin Kowiel + Claude

use crate::mol::{BondType, Mol, TargetAtom};
use crate::smarts::{bond_code, compile_atom, BondQuery, Expr, Packed, Query, RecNode, LABEL_MASK};

/// A reactant prepared for repeated matching.
pub struct Target {
    pub atoms: Vec<TargetAtom>,
    /// `TargetAtom::packed` per atom.
    pub(crate) packed: Vec<Packed>,
    /// Per atom: its bonds.
    pub nbrs: Vec<Vec<TargetBond>>,
}

/// A bond as seen from one end.
#[derive(Clone, Copy, Debug)]
pub struct TargetBond {
    pub atom: u32,
    pub kind: BondType,
    /// `bond_code(kind, in ring of the symmetrized SSSR)`: the bit a `BondQuery` tests.
    pub code: u8,
}

impl Target {
    pub fn new(mol: &Mol) -> Self {
        let nbrs = mol
            .nbrs
            .iter()
            .map(|ns| {
                ns.iter()
                    .map(|&(n, b)| {
                        let (kind, in_ring) = (mol.bonds[b].kind, mol.bonds[b].in_ring);
                        TargetBond { atom: n as u32, kind, code: bond_code(kind, in_ring) }
                    })
                    .collect()
            })
            .collect();
        let atoms = mol.target_atoms();
        let packed = atoms.iter().map(|a| a.packed()).collect();
        Target { atoms, packed, nbrs }
    }

    #[inline]
    fn bond(&self, a: u32, b: u32) -> Option<u8> {
        self.nbrs[a as usize].iter().find(|n| n.atom == b).map(|n| n.code)
    }
}

/// Query atoms in search order; each step lists bonds back to earlier atoms.
#[derive(Debug)]
pub struct Plan {
    order: Vec<usize>,
    steps: Vec<Step>,
    /// Recursive SMARTS nodes in the whole query tree (the size of a `RecCache`).
    n_rec: usize,
}

/// Everything the search needs at one step, in one place.
#[derive(Debug)]
struct Step {
    test: AtomTest,
    /// The earlier step this atom is bonded to and the bond, or `None` at a
    /// component root, where every target atom is a candidate.
    anchor: Option<(u32, BondQuery)>,
    /// Further bonds back to earlier steps (ring closures).
    closures: Box<[(u32, BondQuery)]>,
}

/// A query atom's test against a target atom.
#[derive(Debug)]
enum AtomTest {
    /// `packed & mask == value`.
    One(Packed, Packed),
    /// Any of these terms.
    Any(Vec<(Packed, Packed)>),
    /// The conjuncts that compile (any of these terms), then the rest (recursive
    /// SMARTS and whatever else did not compile) evaluated only for atoms that pass.
    Split(Vec<(Packed, Packed)>, Expr),
    /// Evaluate the expression (it did not compile).
    Expr(Expr),
}

/// Memo of recursive SMARTS results for one target: per recursive node, per
/// target atom, 0 = not computed, 1 = no match, 2 = match. It lives for one
/// `has_match` call, so threads never share one, and it costs nothing when the
/// query has no recursive SMARTS.
pub struct RecCache(Vec<Vec<u8>>);

impl RecCache {
    fn new(n: usize) -> Self {
        RecCache(vec![Vec::new(); n])
    }
}

/// Does the target atom match the inner query of `rec`, with the inner query's
/// first atom on it?
pub fn recursive_matches(rec: &RecNode, t: &Target, atom: u32, cache: &mut RecCache) -> bool {
    let slot = &mut cache.0[rec.id];
    if slot.is_empty() {
        slot.resize(t.atoms.len(), 0);
    }
    match slot[atom as usize] {
        1 => return false,
        2 => return true,
        _ => {}
    }
    let hit = run(&rec.plan, t, Some(atom), cache, &mut ());
    cache.0[rec.id][atom as usize] = 1 + hit as u8;
    hit
}

fn has_recursive(e: &Expr) -> bool {
    match e {
        Expr::Prim(_) => false,
        Expr::Recursive(_) => true,
        Expr::Not(x) => has_recursive(x),
        Expr::And(es) | Expr::Or(es) => es.iter().any(has_recursive),
    }
}

impl AtomTest {
    fn new(e: &Expr) -> Self {
        match compile_atom(e) {
            Some(terms) if terms.len() == 1 => AtomTest::One(terms[0].0, terms[0].1),
            Some(terms) => AtomTest::Any(terms),
            None => Self::split(e).unwrap_or_else(|| AtomTest::Expr(e.clone())),
        }
    }

    /// Split a conjunction into the part that compiles and the rest.
    fn split(e: &Expr) -> Option<Self> {
        let Expr::And(es) = e else { return None };
        let (fast, mut slow): (Vec<&Expr>, Vec<&Expr>) = es.iter().partition(|x| compile_atom(x).is_some());
        if fast.is_empty() || slow.is_empty() {
            return None;
        }
        let terms = compile_atom(&Expr::And(fast.into_iter().cloned().collect()))?;
        // Cheap tests before recursive ones.
        slow.sort_by_key(|x| has_recursive(x));
        let rest = if slow.len() == 1 { slow[0].clone() } else { Expr::And(slow.into_iter().cloned().collect()) };
        Some(AtomTest::Split(terms, rest))
    }

    #[inline]
    fn matches(&self, t: &Target, atom: u32, cache: &mut RecCache) -> bool {
        match self {
            AtomTest::One(mask, value) => t.packed[atom as usize] & mask == *value,
            AtomTest::Any(terms) => {
                let p = t.packed[atom as usize];
                terms.iter().any(|&(mask, value)| p & mask == value)
            }
            AtomTest::Split(terms, rest) => {
                let p = t.packed[atom as usize];
                terms.iter().any(|&(mask, value)| p & mask == value) && rest.eval(t, atom, cache)
            }
            AtomTest::Expr(e) => e.eval(t, atom, cache),
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
    /// (packed label, count), one per distinct label; labels hold only the
    /// `LABEL_MASK` fields.
    labels: Vec<(Packed, u64)>,
    total: u64,
}

impl LabelFreq {
    pub fn new(queries: &[Query]) -> Self {
        let mut counts = std::collections::HashMap::new();
        for q in queries {
            for a in &q.atoms {
                if let Some(terms) = compile_atom(&a.expr) {
                    // Extra constraints (isotope, rings...) only narrow the label.
                    if let [(mask, value)] = terms[..] {
                        if mask & LABEL_MASK == LABEL_MASK {
                            *counts.entry(value & LABEL_MASK).or_insert(0u64) += 1;
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
        let terms: &[(Packed, Packed)] = match test {
            AtomTest::One(m, v) => &[(*m, *v)],
            AtomTest::Any(terms) | AtomTest::Split(terms, _) => terms,
            AtomTest::Expr(_) => return 1.0,
        };
        let hits: u64 = self
            .labels
            .iter()
            .filter(|&&(l, _)| terms.iter().any(|&(m, v)| l & m & LABEL_MASK == v & LABEL_MASK))
            .map(|&(_, c)| c)
            .sum();
        // Unseen labels are rare but possible.
        (hits as f64 + 0.5) / self.total as f64
    }
}

impl Plan {
    pub fn new(q: &Query) -> Self {
        Self::build(q, Order::Score, None, None)
    }

    /// A plan that starts at query atom 0 (the atom a recursive SMARTS pins).
    pub fn rooted(q: &Query) -> Self {
        Self::build(q, Order::Score, None, Some(0))
    }

    pub fn with_order(q: &Query, order: Order, freq: &LabelFreq) -> Self {
        Self::build(q, order, Some(freq), None)
    }

    fn build(q: &Query, how: Order, freq: Option<&LabelFreq>, root: Option<usize>) -> Self {
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
                .count() as i32
                + k.other as i32;
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
            let best = || {
                (0..n)
                    .filter(|&i| step_of[i] == usize::MAX)
                    .max_by_key(|&i| match how {
                        Order::Score => (0, score(i)),
                        Order::RareRoot | Order::Rare => (rarity[i], q.nbrs[i].len() as i32),
                    })
                    .unwrap()
            };
            let start = match root {
                Some(r) if order.is_empty() => r,
                _ => best(),
            };
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
        Plan { order, steps, n_rec: q.n_rec }
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
    run(plan, t, None, &mut RecCache::new(plan.n_rec), c)
}

/// Search `plan` in `t`; with `root`, step 0 is fixed on that target atom.
fn run<C: Counter>(plan: &Plan, t: &Target, root: Option<u32>, cache: &mut RecCache, c: &mut C) -> bool {
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
        start(plan, t, root, &mut mapped[..n], &mut used[..words], cache, c)
    } else {
        start(plan, t, root, &mut vec![u32::MAX; n], &mut vec![0u64; words], cache, c)
    }
}

fn start<C: Counter>(
    plan: &Plan,
    t: &Target,
    root: Option<u32>,
    mapped: &mut [u32],
    used: &mut [u64],
    cache: &mut RecCache,
    c: &mut C,
) -> bool {
    let Some(atom) = root else { return search(plan, t, 0, mapped, used, cache, c) };
    if !plan.steps[0].test.matches(t, atom, cache) {
        return false;
    }
    mapped[0] = atom;
    used[atom as usize / 64] |= 1 << (atom % 64);
    search(plan, t, 1, mapped, used, cache, c)
}

/// `used` is a bitmap of target atoms already mapped.
fn search<C: Counter>(
    plan: &Plan,
    t: &Target,
    k: usize,
    mapped: &mut [u32],
    used: &mut [u64],
    cache: &mut RecCache,
    c: &mut C,
) -> bool {
    let Some(step) = plan.steps.get(k) else { return true };
    let root = step.anchor.is_none();
    let mut try_atom = |cand: u32, mapped: &mut [u32], used: &mut [u64]| -> bool {
        c.offered(k, root);
        let (w, bit) = (cand as usize / 64, 1u64 << (cand % 64));
        if used[w] & bit != 0 || !step.test.matches(t, cand, cache) {
            return false;
        }
        for &(s, bq) in step.closures.iter() {
            match t.bond(mapped[s as usize], cand) {
                Some(code) if bq.matches(code) => {}
                _ => return false,
            }
        }
        c.placed(k);
        mapped[k] = cand;
        used[w] |= bit;
        if search(plan, t, k + 1, mapped, used, cache, c) {
            return true;
        }
        used[w] &= !bit;
        false
    };
    match (step.anchor, &step.test) {
        (Some((s, bq)), _) => {
            for &TargetBond { atom: cand, code, .. } in &t.nbrs[mapped[s as usize] as usize] {
                if bq.matches(code) && try_atom(cand, mapped, used) {
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

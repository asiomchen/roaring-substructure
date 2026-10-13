//! SMARTS queries: the subset used by reaction-template fragments.
//!
//! Atom primitives: element symbols (aliphatic/aromatic), `#n`, `a`, `A`, `*`,
//! isotopes (`13C`, `13`), `Hn`, `Dn`, `Xn`, `vn`, `hn`, `zn`, `Zn`, `Rn`, `rn`,
//! `xn`, charges, atom maps (ignored) and chirality (ignored, as
//! `HasSubstructMatch` does by default). Operators: `!`, `&`, implicit and, `,`,
//! `;`. Ranges `{n-m}`, `{-m}`, `{n-}` follow `D h R r v X x z Z` and a lone
//! `+` or `-` (not `H`, as in RDKit).
//!
//! Bonds: `- = # $ : ~ @ / \` and the unwritten single-or-aromatic bond, combined
//! with `!`, `&`, implicit and, `,`, `;` and reduced to a `BondQuery` bitmask.
//! Recursive SMARTS `$(...)` is an atom primitive: the atom matches when the
//! inner query matches with its first atom on it (see `RecNode`).
//! Dative bonds (`->`) are rejected. Anything else is rejected.
//!
//! Semantics follow RDKit 2026.03: a two-letter element symbol always wins
//! over one-letter primitives (`[Dy]` is dysprosium, `[Rh]` rhodium), except
//! `Nh`, `Mc`, `Ts`, `Og`, which RDKit's lexer lacks.
//!
//! Author: Marcin Kowiel + Claude

use crate::matcher::{recursive_matches, Plan, RecCache, Target};
use crate::mol::{BondType, TargetAtom};
use std::sync::Arc;
use crate::periodic::symbol_to_z;

/// All the atom properties of a `TargetAtom` in one word, for mask tests.
pub type Packed = u128;

/// Counted atom properties that `Prim::Num` compares against a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prop {
    /// `D`: explicit connections (heavy-atom degree).
    Degree,
    /// `h`: implicit (and explicit-count) hydrogens.
    ImplicitH,
    /// `v`: total valence.
    Valence,
    /// `X`: total connections.
    Connections,
    /// `z`: heteroatom neighbours.
    HeteroNbrs,
    /// `Z`: aliphatic heteroatom neighbours.
    AliphaticHeteroNbrs,
    /// `R`: SSSR rings containing the atom.
    Rings,
    /// `r`: smallest SSSR ring size.
    RingSize,
    /// `x`: ring bonds.
    RingBonds,
}

impl Prop {
    #[inline]
    fn get(self, t: &TargetAtom) -> u8 {
        match self {
            Prop::Degree => t.degree,
            Prop::ImplicitH => t.implicit_h,
            Prop::Valence => t.valence,
            Prop::Connections => t.connections,
            Prop::HeteroNbrs => t.hetero_nbrs,
            Prop::AliphaticHeteroNbrs => t.aliphatic_hetero_nbrs,
            Prop::Rings => t.rings,
            Prop::RingSize => t.ring_size,
            Prop::RingBonds => t.ring_bonds,
        }
    }

    /// Bit offset of the property's byte in `Packed`.
    fn shift(self) -> u32 {
        match self {
            Prop::Degree => DEGREE,
            Prop::ImplicitH => IMPLICIT_H,
            Prop::Valence => VALENCE,
            Prop::Connections => CONNECTIONS,
            Prop::HeteroNbrs => HETERO,
            Prop::AliphaticHeteroNbrs => ALIPHATIC_HETERO,
            Prop::Rings => RINGS,
            Prop::RingSize => RING_SIZE,
            Prop::RingBonds => RING_BONDS,
        }
    }

    /// Bit in `Packed` that is set when the property is above zero, if there is one.
    fn positive_flag(self) -> Option<u32> {
        match self {
            Prop::ImplicitH => Some(FLAG_IMPLICIT_H),
            Prop::HeteroNbrs => Some(FLAG_HETERO),
            Prop::AliphaticHeteroNbrs => Some(FLAG_ALIPHATIC_HETERO),
            // In a ring if and only if on a ring bond if and only if the smallest ring is above 0.
            Prop::Rings | Prop::RingSize => Some(FLAG_RINGS),
            Prop::RingBonds => Some(FLAG_RING_BONDS),
            Prop::Degree | Prop::Valence | Prop::Connections => None,
        }
    }
}

/// Query numbers saturate here; no target property reaches it (`mol::PROP_MAX`),
/// so a query asking for it matches nothing, as in RDKit.
pub const UNBOUNDED: u8 = u8::MAX;

#[derive(Clone, Debug)]
pub enum Prim {
    Element { z: u8, aromatic: Option<bool> },
    Aromatic(bool),
    TotalH(u8),
    Charge(i8),
    /// `lo <= charge <= hi` (`+{1-2}`, `-{2-}`), bounds clamped to the `i16` range.
    ChargeRange { lo: i16, hi: i16 },
    /// Isotope equals this (`0` = no isotope given).
    Isotope(u16),
    /// `lo <= prop <= hi`. `hi == UNBOUNDED` has no upper limit, so `{n-}` is
    /// `{ lo: n, hi: UNBOUNDED }`, an exact count is `lo == hi` and the bare
    /// `R`, `h`, `z`, `Z`, `r`, `x` ("at least one") are `{ lo: 1, hi: UNBOUNDED }`.
    Num { prop: Prop, lo: u8, hi: u8 },
    True,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Prim(Prim),
    Not(Box<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
    /// `$(...)`.
    Recursive(Arc<RecNode>),
}

/// A recursive SMARTS: the atom matches when `query` matches with its first
/// atom (the atom `plan` starts at) on it. The inner query is independent of the
/// outer match, so it may reuse the outer match's atoms.
#[derive(Debug)]
pub struct RecNode {
    /// Index into the `RecCache`; unique within the outermost query, nested
    /// nodes included.
    pub id: usize,
    pub query: Query,
    pub plan: Plan,
}

/// A bond expression, reduced to the set of target bonds it accepts: bit
/// `bond_code(kind, in_ring)` is set when a bond of that kind, in a ring or
/// not, passes. Any boolean expression over bond kinds and `@` reduces to
/// such a set exactly, so a test is one shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BondQuery(u16);

/// Number of (kind, ring) combinations: six kinds, ring or not.
const BOND_CODES: u32 = 12;
const ALL_BONDS: u16 = (1 << BOND_CODES) - 1;
/// The ring bit of every kind.
const RING_BONDS_MASK: u16 = 0b1010_1010_1010;

/// The bit of a target bond in a `BondQuery`.
#[inline]
pub fn bond_code(kind: BondType, in_ring: bool) -> u8 {
    kind.index() * 2 + in_ring as u8
}

const fn kind_bits(kind: BondType) -> u16 {
    0b11 << (kind as u8 * 2)
}

impl BondQuery {
    pub const SINGLE: BondQuery = BondQuery(kind_bits(BondType::Single));
    pub const DOUBLE: BondQuery = BondQuery(kind_bits(BondType::Double));
    pub const TRIPLE: BondQuery = BondQuery(kind_bits(BondType::Triple));
    pub const QUADRUPLE: BondQuery = BondQuery(kind_bits(BondType::Quadruple));
    pub const AROMATIC: BondQuery = BondQuery(kind_bits(BondType::Aromatic));
    /// The unwritten bond, and `/`, `\`.
    pub const SINGLE_OR_AROMATIC: BondQuery =
        BondQuery(kind_bits(BondType::Single) | kind_bits(BondType::Aromatic));
    /// `~`.
    pub const ANY: BondQuery = BondQuery(ALL_BONDS);
    /// `@`.
    pub const RING: BondQuery = BondQuery(RING_BONDS_MASK);

    /// Does a target bond with this `bond_code` pass?
    #[inline(always)]
    pub fn matches(self, code: u8) -> bool {
        self.0 >> code & 1 != 0
    }

    /// The bond kinds that can pass, in any ring state.
    pub fn kinds(self) -> impl Iterator<Item = BondType> {
        use BondType::*;
        [Single, Double, Triple, Quadruple, Aromatic, Dative].into_iter().filter(move |&k| self.0 & kind_bits(k) != 0)
    }
}

/// Properties every matching target atom must have, when the query fixes them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Known {
    pub z: Option<u8>,
    pub aromatic: Option<bool>,
    pub total_h: Option<u8>,
    pub degree: Option<u8>,
    pub charge: Option<i8>,
    /// Other conjuncts (isotope, counts) that narrow the atom without being a label.
    pub other: u8,
}

#[derive(Clone, Debug)]
pub struct QueryAtom {
    pub expr: Expr,
    pub known: Known,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct QueryBond {
    pub a: usize,
    pub b: usize,
    pub query: BondQuery,
}

#[derive(Clone, Debug, Default)]
pub struct Query {
    pub atoms: Vec<QueryAtom>,
    pub bonds: Vec<QueryBond>,
    pub nbrs: Vec<Vec<(usize, usize)>>,
    /// Recursive SMARTS nodes in this query and below (`RecNode::id` < this).
    pub n_rec: usize,
}

impl Query {
    /// The inner queries that must also occur in any target this query matches
    /// (recursive SMARTS that are top-level conjuncts of an atom, at any depth).
    /// Each is a subgraph of the target, so its fingerprint features are required
    /// too, though they may overlap this query's: take their union, not their sum.
    pub fn required_subqueries(&self) -> Vec<&Query> {
        let mut found: Vec<&RecNode> = Vec::new();
        for a in &self.atoms {
            a.expr.required_recursives(&mut found);
        }
        let mut i = 0;
        while i < found.len() {
            let inner = &found[i].query;
            for a in &inner.atoms {
                a.expr.required_recursives(&mut found);
            }
            i += 1;
        }
        found.into_iter().map(|r| &r.query).collect()
    }
}

impl Prim {
    #[inline]
    fn matches(&self, t: &TargetAtom) -> bool {
        match *self {
            Prim::Element { z, aromatic } => t.z == z && aromatic.is_none_or(|a| a == t.aromatic),
            Prim::Aromatic(a) => t.aromatic == a,
            Prim::TotalH(h) => t.total_h == h,
            Prim::Charge(q) => t.charge == q,
            Prim::ChargeRange { lo, hi } => (lo..=hi).contains(&(t.charge as i16)),
            Prim::Isotope(n) => t.isotope == n,
            Prim::Num { prop, lo, hi } => (lo..=hi).contains(&prop.get(t)),
            Prim::True => true,
        }
    }
}

impl Expr {
    #[cfg(test)]
    /// The expression on one atom's properties; only for expressions without
    /// recursive SMARTS (a recursive one needs the whole target: use `eval`).
    pub fn matches(&self, t: &TargetAtom) -> bool {
        match self {
            Expr::Prim(p) => p.matches(t),
            Expr::Not(e) => !e.matches(t),
            Expr::And(es) => es.iter().all(|e| e.matches(t)),
            Expr::Or(es) => es.iter().any(|e| e.matches(t)),
            Expr::Recursive(_) => panic!("recursive SMARTS needs a target: use Expr::eval"),
        }
    }

    /// The expression on atom `atom` of `t`.
    pub fn eval(&self, t: &Target, atom: u32, cache: &mut RecCache) -> bool {
        match self {
            Expr::Prim(p) => p.matches(&t.atoms[atom as usize]),
            Expr::Not(e) => !e.eval(t, atom, cache),
            Expr::And(es) => es.iter().all(|e| e.eval(t, atom, cache)),
            Expr::Or(es) => es.iter().any(|e| e.eval(t, atom, cache)),
            Expr::Recursive(r) => recursive_matches(r, t, atom, cache),
        }
    }

    /// The recursive SMARTS every matching atom must satisfy (top-level conjuncts).
    fn required_recursives<'a>(&'a self, out: &mut Vec<&'a RecNode>) {
        match self {
            Expr::Recursive(r) => out.push(r),
            Expr::And(es) => es.iter().for_each(|e| e.required_recursives(out)),
            _ => {}
        }
    }

    /// Facts implied by the expression (only through top-level conjunctions).
    fn known(&self, k: &mut Known) {
        match self {
            Expr::Prim(Prim::Element { z, aromatic }) => {
                k.z = Some(*z);
                if aromatic.is_some() {
                    k.aromatic = *aromatic;
                }
            }
            Expr::Prim(Prim::Aromatic(a)) => k.aromatic = Some(*a),
            Expr::Prim(Prim::TotalH(h)) => k.total_h = Some(*h),
            Expr::Prim(Prim::Num { prop: Prop::Degree, lo, hi }) if lo == hi => k.degree = Some(*lo),
            Expr::Prim(Prim::Charge(q)) => k.charge = Some(*q),
            Expr::Prim(Prim::Isotope(_) | Prim::Num { .. } | Prim::ChargeRange { .. }) => {
                k.other = k.other.saturating_add(1)
            }
            Expr::And(es) => es.iter().for_each(|e| e.known(k)),
            // The atom must match the inner query's first atom, so it has its facts.
            Expr::Recursive(r) => {
                let i = &r.query.atoms[0].known;
                k.z = k.z.or(i.z);
                k.aromatic = k.aromatic.or(i.aromatic);
                k.total_h = k.total_h.or(i.total_h);
                k.degree = k.degree.or(i.degree);
                k.charge = k.charge.or(i.charge);
                k.other = k.other.saturating_add(i.other);
            }
            _ => {}
        }
    }
}

/// Field offsets in `TargetAtom::packed`: the label fields (one byte each,
/// `LABEL_MASK`), then the isotope, then the counted properties.
const Z: u32 = 0;
const AROM: u32 = 8;
const H: u32 = 16;
const DEGREE: u32 = 24;
const CHARGE: u32 = 32;
const ISOTOPE: u32 = 40;
const IMPLICIT_H: u32 = 56;
const VALENCE: u32 = 64;
const CONNECTIONS: u32 = 72;
const HETERO: u32 = 80;
const ALIPHATIC_HETERO: u32 = 88;
const RINGS: u32 = 96;
const RING_SIZE: u32 = 104;
const RING_BONDS: u32 = 112;
/// One bit per property that has a bare "at least one" query.
const FLAG_IMPLICIT_H: u32 = 120;
const FLAG_HETERO: u32 = 121;
const FLAG_ALIPHATIC_HETERO: u32 = 122;
const FLAG_RINGS: u32 = 123;
const FLAG_RING_BONDS: u32 = 124;

/// The fields every fully specified reaction-template atom pins down
/// (element, aromaticity, H count, degree, charge); `LabelFreq` keys on them.
pub const LABEL_MASK: Packed = (1 << ISOTOPE) - 1;

impl TargetAtom {
    /// Every property in one word, for `AtomTest`.
    #[inline]
    pub fn packed(&self) -> Packed {
        (self.z as Packed) << Z
            | (self.aromatic as Packed) << AROM
            | (self.total_h as Packed) << H
            | (self.degree as Packed) << DEGREE
            | (self.charge as u8 as Packed) << CHARGE
            | (self.isotope as Packed) << ISOTOPE
            | (self.implicit_h as Packed) << IMPLICIT_H
            | (self.valence as Packed) << VALENCE
            | (self.connections as Packed) << CONNECTIONS
            | (self.hetero_nbrs as Packed) << HETERO
            | (self.aliphatic_hetero_nbrs as Packed) << ALIPHATIC_HETERO
            | (self.rings as Packed) << RINGS
            | (self.ring_size as Packed) << RING_SIZE
            | (self.ring_bonds as Packed) << RING_BONDS
            | ((self.implicit_h > 0) as Packed) << FLAG_IMPLICIT_H
            | ((self.hetero_nbrs > 0) as Packed) << FLAG_HETERO
            | ((self.aliphatic_hetero_nbrs > 0) as Packed) << FLAG_ALIPHATIC_HETERO
            | ((self.rings > 0) as Packed) << FLAG_RINGS
            | ((self.ring_bonds > 0) as Packed) << FLAG_RING_BONDS
    }
}

type Term = (Packed, Packed);

/// `lo <= prop <= hi` as `packed & mask == value` terms, any of which may hold;
/// `None` when it needs more than `max_terms`.
fn num_terms(prop: Prop, lo: u8, hi: u8, max_terms: usize) -> Option<Vec<Term>> {
    let field = |v: u8| ((0xff as Packed) << prop.shift(), (v as Packed) << prop.shift());
    if lo > hi {
        return Some(Vec::new());
    }
    if lo == 0 && hi == UNBOUNDED {
        return Some(vec![(0, 0)]);
    }
    if lo == hi {
        return Some(vec![field(lo)]);
    }
    if lo == 1 && hi == UNBOUNDED {
        if let Some(bit) = prop.positive_flag() {
            return Some(vec![(1 << bit, 1 << bit)]);
        }
    }
    // `hi == UNBOUNDED` is open ended, but targets stay below it.
    if (hi as usize - lo as usize) < max_terms && hi != UNBOUNDED {
        return Some((lo..=hi).map(field).collect());
    }
    None
}

/// `lo <= charge <= hi` as terms, `None` when it needs more than `max_terms`.
fn charge_terms(lo: i16, hi: i16, max_terms: usize) -> Option<Vec<Term>> {
    let (lo, hi) = (lo.max(i8::MIN as i16), hi.min(i8::MAX as i16));
    if lo > hi {
        return Some(Vec::new());
    }
    if lo == i8::MIN as i16 && hi == i8::MAX as i16 {
        return Some(vec![(0, 0)]);
    }
    if ((hi - lo) as usize) < max_terms {
        return Some((lo..=hi).map(|q| ((0xff as Packed) << CHARGE, (q as i8 as u8 as Packed) << CHARGE)).collect());
    }
    None
}

/// The values a `Num` rejects, when that is one range too.
fn complement(lo: u8, hi: u8) -> Option<(u8, u8)> {
    match (lo, hi) {
        (0, UNBOUNDED) => Some((1, 0)),
        (1.., UNBOUNDED) => Some((0, lo - 1)),
        (0, _) => Some((hi + 1, UNBOUNDED)),
        _ => None,
    }
}

/// An atom expression compiled to `packed & mask == value` terms, any of which
/// may hold; `None` when the expression does not fit (negations, wide ORs).
pub fn compile_atom(e: &Expr) -> Option<Vec<Term>> {
    const MAX_TERMS: usize = 8;
    let field = |shift: u32, v: Packed| ((0xff as Packed) << shift, (v & 0xff) << shift);
    match e {
        Expr::Prim(p) => Some(vec![match *p {
            Prim::Element { z, aromatic: None } => field(Z, z as Packed),
            Prim::Element { z, aromatic: Some(a) } => {
                let (m1, v1) = field(Z, z as Packed);
                let (m2, v2) = field(AROM, a as Packed);
                (m1 | m2, v1 | v2)
            }
            Prim::Aromatic(a) => field(AROM, a as Packed),
            Prim::TotalH(h) => field(H, h as Packed),
            Prim::Charge(q) => field(CHARGE, q as u8 as Packed),
            Prim::ChargeRange { lo, hi } => return charge_terms(lo, hi, MAX_TERMS),
            Prim::Isotope(n) => ((0xffff as Packed) << ISOTOPE, (n as Packed) << ISOTOPE),
            Prim::Num { prop, lo, hi } => return num_terms(prop, lo, hi, MAX_TERMS),
            Prim::True => (0, 0),
        }]),
        Expr::Not(inner) => match **inner {
            Expr::Prim(Prim::Aromatic(a)) => Some(vec![field(AROM, !a as Packed)]),
            Expr::Prim(Prim::Num { prop, lo, hi }) => {
                let (lo, hi) = complement(lo, hi)?;
                num_terms(prop, lo, hi, MAX_TERMS)
            }
            _ => None,
        },
        Expr::Recursive(_) => None,
        Expr::Or(es) => {
            let mut terms = Vec::new();
            for x in es {
                terms.extend(compile_atom(x)?);
            }
            (terms.len() <= MAX_TERMS).then_some(terms)
        }
        Expr::And(es) => {
            let mut terms = vec![(0, 0)];
            for x in es {
                let rhs = compile_atom(x)?;
                let mut next = Vec::new();
                for &(m1, v1) in &terms {
                    for &(m2, v2) in &rhs {
                        // Both constrain a field to different values: never true.
                        if (v1 ^ v2) & m1 & m2 == 0 {
                            next.push((m1 | m2, v1 | v2));
                        }
                    }
                }
                if next.len() > MAX_TERMS {
                    return None;
                }
                terms = next;
            }
            Some(terms)
        }
    }
}

const ORGANIC: [&str; 10] = ["Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I"];
const AROMATIC: [&str; 10] = ["se", "te", "as", "si", "b", "c", "n", "o", "p", "s"];
/// Symbols RDKit's SMARTS lexer lacks; `[Nh]` is nitrogen with an implicit H,
/// and `[Mc]`, `[Ts]`, `[Og]` are errors.
const NOT_LEXED: [&str; 4] = ["Nh", "Mc", "Ts", "Og"];

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
}

/// An aromatic symbol (`c`, `se`) or, with `aromatic` false, an organic-subset one.
fn element(sym: &str, aromatic: bool) -> Result<Expr, String> {
    let z = symbol_to_z(&capitalize(sym)).ok_or_else(|| format!("unknown element {sym}"))?;
    Ok(Expr::Prim(Prim::Element { z, aromatic: Some(aromatic) }))
}

/// A number from the query, saturating at `UNBOUNDED`.
fn sat(n: u32) -> u8 {
    n.min(UNBOUNDED as u32) as u8
}

/// The largest number RDKit's SMARTS lexer accepts in an atom primitive.
const MAX_NUMBER: u32 = 2_147_483_639;

/// Deepest `$(...)` nesting accepted.
const MAX_DEPTH: usize = 32;

struct BracketParser<'a> {
    s: &'a [u8],
    i: usize,
    first: bool,
    /// Recursive SMARTS ids handed out so far, and how deep this bracket is nested.
    ids: &'a mut usize,
    depth: usize,
    /// A number past `MAX_NUMBER` was read: RDKit rejects the SMARTS.
    too_big: bool,
}

impl BracketParser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn number(&mut self) -> Option<u32> {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        let Ok(n) = std::str::from_utf8(&self.s[start..self.i]).ok()?.parse::<u32>() else {
            self.too_big = true;
            return None;
        };
        self.too_big |= n > MAX_NUMBER;
        Some(n)
    }

    /// The `{lo-hi}`, `{-hi}` or `{lo-}` after a ranged primitive, as RDKit
    /// writes it: no spaces, no leading zeros, at least one bound.
    fn range(&mut self) -> Result<(Option<u32>, Option<u32>), String> {
        self.i += 1; // {
        let bound = |p: &mut Self| -> Result<Option<u32>, String> {
            if !p.at_digit() {
                return Ok(None);
            }
            let leading_zero = p.peek() == Some(b'0') && p.s.get(p.i + 1).is_some_and(|d| d.is_ascii_digit());
            if leading_zero {
                return Err("leading zero in range".into());
            }
            Ok(p.number())
        };
        let lo = bound(self)?;
        if self.peek() != Some(b'-') {
            return Err("range needs a '-'".into());
        }
        self.i += 1;
        let hi = bound(self)?;
        if self.peek() != Some(b'}') || (lo.is_none() && hi.is_none()) {
            return Err("bad range".into());
        }
        self.i += 1;
        Ok((lo, hi))
    }

    fn at_digit(&self) -> bool {
        self.peek().is_some_and(|d| d.is_ascii_digit())
    }

    // low_and := or (';' or)*
    fn low_and(&mut self) -> Result<Expr, String> {
        let mut parts = vec![self.or()?];
        while self.peek() == Some(b';') {
            self.i += 1;
            parts.push(self.or()?);
        }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { Expr::And(parts) })
    }

    // or := and (',' and)*
    fn or(&mut self) -> Result<Expr, String> {
        let mut parts = vec![self.and()?];
        while self.peek() == Some(b',') {
            self.i += 1;
            parts.push(self.and()?);
        }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { Expr::Or(parts) })
    }

    // and := unary ('&'? unary)*
    fn and(&mut self) -> Result<Expr, String> {
        let mut parts = vec![self.unary()?];
        loop {
            match self.peek() {
                Some(b'&') => {
                    self.i += 1;
                    parts.push(self.unary()?);
                }
                Some(b',') | Some(b';') | None => break,
                Some(_) => parts.push(self.unary()?),
            }
        }
        Ok(if parts.len() == 1 { parts.pop().unwrap() } else { Expr::And(parts) })
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if self.peek() == Some(b'!') {
            self.i += 1;
            self.first = false;
            return Ok(Expr::Not(Box::new(self.unary()?)));
        }
        let first = std::mem::replace(&mut self.first, false);
        if self.at_digit() {
            return self.isotope();
        }
        self.primitive(first)
    }

    /// A number is an isotope. Directly before an atom symbol (`13C`, `2H`,
    /// `13#6`) the two are one primitive, which a leading `!` negates whole.
    fn isotope(&mut self) -> Result<Expr, String> {
        let n = self.number().ok_or("bad isotope")?;
        let n = u16::try_from(n).map_err(|_| format!("isotope {n} out of range"))?;
        let iso = Expr::Prim(Prim::Isotope(n));
        Ok(match self.atom_symbol(self.h_is_atom())? {
            Some(atom) => Expr::And(vec![iso, atom]),
            None => iso,
        })
    }

    /// Is the `H` at the cursor a hydrogen atom? RDKit reads `[H]`, `[H+]`,
    /// `[H-]` and `[H:1]` that way and every other `H` (`[H;R]`, `[Hx]`) as a count.
    fn h_is_atom(&self) -> bool {
        matches!(self.s.get(self.i + 1), None | Some(b':' | b'+' | b'-'))
    }

    /// `n`, or `default` when there is no number (RDKit's default for the letter).
    fn count(&mut self, default: u32) -> u32 {
        if self.at_digit() {
            self.number().unwrap_or(u32::MAX)
        } else {
            default
        }
    }

    /// `X3`, or the bare letter with RDKit's default: exactly `one`, or at least one.
    fn num(&mut self, prop: Prop, bare: Bare) -> Result<Expr, String> {
        self.i += 1;
        let (lo, hi) = if self.peek() == Some(b'{') {
            let (lo, hi) = self.range()?;
            (lo.map_or(0, sat), hi.map_or(UNBOUNDED, sat))
        } else if self.at_digit() {
            let n = sat(self.number().unwrap_or(u32::MAX));
            (n, n)
        } else {
            match bare {
                Bare::Exactly(n) => (n, n),
                Bare::AtLeastOne => (1, UNBOUNDED),
            }
        };
        Ok(Expr::Prim(Prim::Num { prop, lo, hi }))
    }

    /// An atom symbol: `#n`, `*`, an element (the two-letter symbols first),
    /// aromatic `c`, `se`... Not a property letter such as `D`, `R` or `a`.
    fn atom_symbol(&mut self, h_is_element: bool) -> Result<Option<Expr>, String> {
        let rest = std::str::from_utf8(&self.s[self.i..]).map_err(|e| e.to_string())?;
        let Some(c) = rest.bytes().next() else { return Ok(None) };
        if c == b'#' {
            self.i += 1;
            let z = self.number().ok_or("bad #n")?;
            // RDKit accepts numbers past 118; they match nothing.
            let z = if z > 118 { UNBOUNDED } else { z as u8 };
            return Ok(Some(Expr::Prim(Prim::Element { z, aromatic: None })));
        }
        if c == b'*' {
            self.i += 1;
            return Ok(Some(Expr::Prim(Prim::True)));
        }
        if let Some(sym) = AROMATIC.iter().find(|s| rest.starts_with(**s)) {
            self.i += sym.len();
            return element(sym, true).map(Some);
        }
        if !c.is_ascii_uppercase() {
            return Ok(None);
        }
        if let Some(e) = self.two_letter_element(rest) {
            return Ok(Some(e));
        }
        if c == b'H' && !h_is_element {
            return Ok(None);
        }
        // Letters that start a property (`D`, `R`, `X`, `Z`) are not symbols.
        if let Some(z) = rest.get(..1).and_then(symbol_to_z) {
            self.i += 1;
            return Ok(Some(Expr::Prim(Prim::Element { z, aromatic: ORGANIC.contains(&&rest[..1]).then_some(false) })));
        }
        Ok(None)
    }

    fn primitive(&mut self, first: bool) -> Result<Expr, String> {
        let rest = std::str::from_utf8(&self.s[self.i..]).map_err(|e| e.to_string())?;
        let c = self.peek().ok_or("empty primitive")?;
        // A two-letter element symbol beats the one-letter primitives, as in
        // RDKit's lexer (`[Dy]`, `[Rh]`, `[Xe]`, `[Zn]`, `[Hg]`...).
        if c.is_ascii_uppercase() {
            if let Some(e) = self.two_letter_element(rest) {
                return Ok(e);
            }
        }
        match c {
            b'@' => {
                // Chirality is ignored by HasSubstructMatch(useChirality=False).
                self.i += 1;
                if self.peek() == Some(b'@') {
                    self.i += 1;
                }
                if self.peek() == Some(b'?') {
                    self.i += 1;
                }
                Ok(Expr::Prim(Prim::True))
            }
            b':' => {
                // Atom map number: no constraint.
                self.i += 1;
                self.number().ok_or("bad atom map")?;
                Ok(Expr::Prim(Prim::True))
            }
            b'+' | b'-' => {
                self.i += 1;
                let unit: i8 = if c == b'+' { 1 } else { -1 };
                if self.peek() == Some(b'{') {
                    // `-{lo-hi}` is charge -hi..=-lo, so a missing bound opens that end.
                    let (lo, hi) = self.range()?;
                    let open = 1_000_000_000i64;
                    let lo = lo.map_or(-open, i64::from);
                    let hi = hi.map_or(open, i64::from);
                    let (lo, hi) = if unit > 0 { (lo, hi) } else { (-hi, -lo) };
                    let clamp = |n: i64| n.clamp(-1000, 1000) as i16;
                    return Ok(Expr::Prim(Prim::ChargeRange { lo: clamp(lo), hi: clamp(hi) }));
                }
                let q = if self.at_digit() {
                    let n = self.number().ok_or("bad charge")?;
                    // Past any charge a molecule can have: matches nothing.
                    if n > 127 { i8::MIN } else { unit * n as i8 }
                } else {
                    let mut q = unit;
                    while self.peek() == Some(c) {
                        self.i += 1;
                        q += unit;
                    }
                    q
                };
                Ok(Expr::Prim(Prim::Charge(q)))
            }
            b'H' => {
                self.i += 1;
                if first && matches!(self.peek(), None | Some(b':' | b'+' | b'-')) {
                    // `[H]`, `[H+]`: a hydrogen atom, not a hydrogen count.
                    return Ok(Expr::Prim(Prim::Element { z: 1, aromatic: None }));
                }
                let h = sat(self.count(1));
                Ok(Expr::Prim(Prim::TotalH(h)))
            }
            b'$' => self.recursive(),
            b'D' => self.num(Prop::Degree, Bare::Exactly(1)),
            b'X' => self.num(Prop::Connections, Bare::Exactly(1)),
            b'v' => self.num(Prop::Valence, Bare::Exactly(1)),
            b'h' => self.num(Prop::ImplicitH, Bare::AtLeastOne),
            b'z' => self.num(Prop::HeteroNbrs, Bare::AtLeastOne),
            b'Z' => self.num(Prop::AliphaticHeteroNbrs, Bare::AtLeastOne),
            b'R' => self.num(Prop::Rings, Bare::AtLeastOne),
            b'r' => self.num(Prop::RingSize, Bare::AtLeastOne),
            b'x' => self.num(Prop::RingBonds, Bare::AtLeastOne),
            b'a' if !rest.starts_with("as") => {
                self.i += 1;
                Ok(Expr::Prim(Prim::Aromatic(true)))
            }
            b'A' => {
                self.i += 1;
                Ok(Expr::Prim(Prim::Aromatic(false)))
            }
            _ => match self.atom_symbol(false)? {
                Some(e) => Ok(e),
                None => Err(format!("unsupported SMARTS primitive at {rest:?}")),
            },
        }
    }

    /// `$(smarts)`, with the cursor on the `$`.
    fn recursive(&mut self) -> Result<Expr, String> {
        if self.s.get(self.i + 1) != Some(&b'(') {
            return Err("$ needs a ( in an atom".into());
        }
        let open = self.i + 1;
        let mut depth = 0usize;
        let close = self.s[open..]
            .iter()
            .position(|&c| {
                depth += (c == b'(') as usize;
                depth -= (c == b')') as usize;
                depth == 0
            })
            .ok_or("unclosed $(")?
            + open;
        let inner = std::str::from_utf8(&self.s[open + 1..close]).map_err(|e| e.to_string())?;
        if self.depth >= MAX_DEPTH {
            return Err("recursive SMARTS nested too deeply".into());
        }
        let query = parse_inner(inner, self.ids, self.depth + 1)?;
        if query.atoms.is_empty() {
            return Err("empty recursive SMARTS".into());
        }
        self.i = close + 1;
        // RDKit's lexer swallows the digits right after a recursive primitive
        // (`[$(C)13]` is `[$(C)]`, not an isotope).
        while self.at_digit() {
            self.i += 1;
        }
        let id = *self.ids;
        *self.ids += 1;
        let plan = Plan::rooted(&query);
        Ok(Expr::Recursive(Arc::new(RecNode { id, query, plan })))
    }

    fn two_letter_element(&mut self, rest: &str) -> Option<Expr> {
        let two = rest.get(..2).filter(|t| t.as_bytes()[1].is_ascii_lowercase() && !NOT_LEXED.contains(t))?;
        let z = symbol_to_z(two)?;
        self.i += 2;
        Some(Expr::Prim(Prim::Element { z, aromatic: ORGANIC.contains(&two).then_some(false) }))
    }
}

/// What a bare property letter means.
#[derive(Clone, Copy)]
enum Bare {
    Exactly(u8),
    AtLeastOne,
}

/// Is `c` the start of a bond primitive (`- = # $ : ~ @ / \\`) or a negation?
fn starts_bond(c: u8) -> bool {
    matches!(c, b'-' | b'=' | b'#' | b'$' | b':' | b'~' | b'@' | b'/' | b'\\' | b'!')
}

/// A bond expression with the atom operators' precedence (`!`, then `&` or
/// nothing, then `,`, then `;`), evaluated straight to the set of bonds it passes.
struct BondParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl BondParser<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn low_and(&mut self) -> Result<u16, String> {
        let mut m = self.or()?;
        while self.peek() == Some(b';') {
            self.i += 1;
            m &= self.or()?;
        }
        Ok(m)
    }

    fn or(&mut self) -> Result<u16, String> {
        let mut m = self.and()?;
        while self.peek() == Some(b',') {
            self.i += 1;
            m |= self.and()?;
        }
        Ok(m)
    }

    fn and(&mut self) -> Result<u16, String> {
        let mut m = self.unary()?;
        loop {
            match self.peek() {
                Some(b'&') => {
                    self.i += 1;
                    m &= self.unary()?;
                }
                Some(c) if starts_bond(c) => m &= self.unary()?,
                _ => return Ok(m),
            }
        }
    }

    fn unary(&mut self) -> Result<u16, String> {
        let c = self.peek().ok_or("bond expression ends early")?;
        self.i += 1;
        Ok(match c {
            b'!' => !self.unary()? & ALL_BONDS,
            b'-' => BondQuery::SINGLE.0,
            b'=' => BondQuery::DOUBLE.0,
            b'#' => BondQuery::TRIPLE.0,
            b'$' => BondQuery::QUADRUPLE.0,
            b':' => BondQuery::AROMATIC.0,
            b'~' => BondQuery::ANY.0,
            b'@' => BondQuery::RING.0,
            // Direction is ignored: RDKit matches them as single-or-aromatic.
            b'/' | b'\\' => BondQuery::SINGLE_OR_AROMATIC.0,
            _ => return Err(format!("unsupported bond primitive {:?}", c as char)),
        })
    }
}

/// Parse the bond expression starting at `s[*i]`, advancing `*i` past it.
fn parse_bond(s: &[u8], i: &mut usize) -> Result<BondQuery, String> {
    let mut p = BondParser { s, i: *i };
    let m = p.low_and()?;
    *i = p.i;
    Ok(BondQuery(m))
}

fn parse_bracket(s: &str, ids: &mut usize, depth: usize) -> Result<Expr, String> {
    let mut p = BracketParser { s: s.as_bytes(), i: 0, first: true, ids, depth, too_big: false };
    let e = p.low_and()?;
    if p.too_big {
        return Err(format!("number too large in [{s}]"));
    }
    if p.i != s.len() {
        return Err(format!("unsupported bracket atom [{s}]"));
    }
    Ok(e)
}

pub fn parse_smarts(smarts: &str) -> Result<Query, String> {
    let mut ids = 0;
    let mut q = parse_inner(smarts, &mut ids, 0)?;
    q.n_rec = ids;
    Ok(q)
}

/// The end of the bracket atom opening at `s[0]`: brackets nest inside `$(...)`.
fn bracket_end(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in s.bytes().enumerate() {
        match c {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// `parse_smarts` for a query at `depth` levels of `$(...)`; `ids` numbers the recursive nodes.
fn parse_inner(smarts: &str, ids: &mut usize, depth: usize) -> Result<Query, String> {
    let s = smarts.as_bytes();
    let mut q = Query::default();
    let mut prev: Option<usize> = None;
    let mut stack = Vec::new();
    let mut pending: Option<BondQuery> = None;
    let mut ring_open: std::collections::HashMap<u32, (usize, Option<BondQuery>)> = Default::default();
    let mut i = 0;
    let add_bond = |q: &mut Query, a: usize, b: usize, bq: Option<BondQuery>| {
        let idx = q.bonds.len();
        q.bonds.push(QueryBond { a, b, query: bq.unwrap_or(BondQuery::SINGLE_OR_AROMATIC) });
        q.nbrs[a].push((b, idx));
        q.nbrs[b].push((a, idx));
    };
    while i < s.len() {
        let c = s[i];
        match c {
            b'(' => {
                if pending.is_some() {
                    return Err("bond before a branch".into());
                }
                if prev.is_none() {
                    return Err("branch without an atom".into());
                }
                stack.push(prev);
                i += 1;
            }
            b')' => {
                if pending.is_some() {
                    return Err("bond before )".into());
                }
                prev = stack.pop().ok_or("unbalanced )")?;
                i += 1;
            }
            b'.' => {
                if pending.is_some() {
                    return Err("bond before .".into());
                }
                // A component needs an atom on each side, as RDKit insists.
                let before = i.checked_sub(1).map(|j| s[j]);
                if matches!(before, None | Some(b'(' | b'.')) || matches!(s.get(i + 1), None | Some(b'.' | b')')) {
                    return Err("empty component".into());
                }
                prev = None;
                i += 1;
            }
            c if starts_bond(c) => {
                if pending.is_some() {
                    return Err("two bonds in a row".into());
                }
                pending = Some(parse_bond(s, &mut i)?);
            }
            b'0'..=b'9' | b'%' => {
                let num = if c == b'%' {
                    let v = smarts.get(i + 1..i + 3).ok_or("bad %nn")?.parse::<u32>().map_err(|e| e.to_string())?;
                    i += 3;
                    v
                } else {
                    i += 1;
                    (c - b'0') as u32
                };
                let cur = prev.ok_or("ring closure without atom")?;
                if let Some((other, bq)) = ring_open.remove(&num) {
                    // The bond written at the opening digit wins over the closing one.
                    let bq = bq.or(pending.take());
                    add_bond(&mut q, other, cur, bq);
                } else {
                    ring_open.insert(num, (cur, pending.take()));
                }
            }
            _ => {
                let (expr, len) = if c == b'[' {
                    let end = bracket_end(&smarts[i..]).ok_or("unclosed [")? + i;
                    (parse_bracket(&smarts[i + 1..end], ids, depth)?, end + 1 - i)
                } else if c == b'*' {
                    (Expr::Prim(Prim::True), 1)
                } else if let Some(sym) = ORGANIC.iter().find(|sym| smarts[i..].starts_with(**sym)) {
                    (element(sym, false)?, sym.len())
                } else if let Some(sym) = ["b", "c", "n", "o", "p", "s"].iter().find(|sym| smarts[i..].starts_with(**sym)) {
                    (element(sym, true)?, 1)
                } else if c == b'A' {
                    (Expr::Prim(Prim::Aromatic(false)), 1)
                } else if c == b'a' {
                    (Expr::Prim(Prim::Aromatic(true)), 1)
                } else {
                    return Err(format!("unsupported SMARTS at {:?}", &smarts[i..]));
                };
                i += len;
                let mut known = Known::default();
                expr.known(&mut known);
                let idx = q.atoms.len();
                q.atoms.push(QueryAtom { expr, known });
                q.nbrs.push(Vec::new());
                if let Some(p) = prev {
                    add_bond(&mut q, p, idx, pending.take());
                } else if pending.is_some() {
                    return Err("bond without preceding atom".into());
                }
                prev = Some(idx);
            }
        }
    }
    if pending.is_some() {
        return Err("bond at the end of the SMARTS".into());
    }
    if !ring_open.is_empty() || !stack.is_empty() {
        return Err("unclosed ring or branch".into());
    }
    Ok(q)
}

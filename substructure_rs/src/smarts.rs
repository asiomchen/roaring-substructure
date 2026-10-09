//! SMARTS queries: the subset used by reaction-template fragments.
//!
//! Atom primitives: element symbols (aliphatic/aromatic), `#n`, `a`, `A`, `*`,
//! `Hn`, `Dn`, charges, and chirality (ignored, as `HasSubstructMatch` does by
//! default). Operators: `!`, `&`, implicit and, `,`, `;`. Bonds: `- = # : ~ / \`
//! and the unwritten single-or-aromatic bond. Anything else is rejected.
//!
//! Author: Marcin Kowiel + Claude

use crate::mol::{BondType, TargetAtom};
use crate::periodic::symbol_to_z;

#[derive(Clone, Debug)]
pub enum Prim {
    Element { z: u8, aromatic: Option<bool> },
    Aromatic(bool),
    TotalH(u8),
    Degree(u8),
    Charge(i8),
    True,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Prim(Prim),
    Not(Box<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondQuery {
    Single,
    Double,
    Triple,
    Aromatic,
    SingleOrAromatic,
    Any,
}

impl BondQuery {
    #[inline]
    pub fn matches(self, kind: BondType) -> bool {
        match self {
            BondQuery::Single => kind == BondType::Single,
            BondQuery::Double => kind == BondType::Double,
            BondQuery::Triple => kind == BondType::Triple,
            BondQuery::Aromatic => kind == BondType::Aromatic,
            BondQuery::SingleOrAromatic => matches!(kind, BondType::Single | BondType::Aromatic),
            BondQuery::Any => true,
        }
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
}

impl Prim {
    #[inline]
    fn matches(&self, t: &TargetAtom) -> bool {
        match *self {
            Prim::Element { z, aromatic } => t.z == z && aromatic.is_none_or(|a| a == t.aromatic),
            Prim::Aromatic(a) => t.aromatic == a,
            Prim::TotalH(h) => t.total_h == h,
            Prim::Degree(d) => t.degree == d,
            Prim::Charge(q) => t.charge == q,
            Prim::True => true,
        }
    }
}

impl Expr {
    #[inline]
    pub fn matches(&self, t: &TargetAtom) -> bool {
        match self {
            Expr::Prim(p) => p.matches(t),
            Expr::Not(e) => !e.matches(t),
            Expr::And(es) => es.iter().all(|e| e.matches(t)),
            Expr::Or(es) => es.iter().any(|e| e.matches(t)),
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
            Expr::Prim(Prim::Degree(d)) => k.degree = Some(*d),
            Expr::Prim(Prim::Charge(q)) => k.charge = Some(*q),
            Expr::And(es) => es.iter().for_each(|e| e.known(k)),
            _ => {}
        }
    }
}

const ORGANIC: [&str; 10] = ["Cl", "Br", "B", "C", "N", "O", "P", "S", "F", "I"];
const AROMATIC: [&str; 10] = ["se", "te", "as", "si", "b", "c", "n", "o", "p", "s"];

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
}

fn element(sym: &str, aromatic: bool) -> Result<Expr, String> {
    let z = symbol_to_z(&capitalize(sym)).ok_or_else(|| format!("unknown element {sym}"))?;
    Ok(Expr::Prim(Prim::Element { z, aromatic: Some(aromatic) }))
}

struct BracketParser<'a> {
    s: &'a [u8],
    i: usize,
    first: bool,
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
        std::str::from_utf8(&self.s[start..self.i]).ok()?.parse().ok()
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
            return Ok(Expr::Not(Box::new(self.unary()?)));
        }
        let first = std::mem::replace(&mut self.first, false);
        self.primitive(first)
    }

    fn primitive(&mut self, first: bool) -> Result<Expr, String> {
        let rest = std::str::from_utf8(&self.s[self.i..]).map_err(|e| e.to_string())?;
        let c = self.peek().ok_or("empty primitive")?;
        match c {
            b'#' => {
                self.i += 1;
                let z = self.number().ok_or("bad #n")?;
                Ok(Expr::Prim(Prim::Element { z: z as u8, aromatic: None }))
            }
            b'*' => {
                self.i += 1;
                Ok(Expr::Prim(Prim::True))
            }
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
            b'+' | b'-' => {
                self.i += 1;
                let unit: i8 = if c == b'+' { 1 } else { -1 };
                let q = if self.peek().is_some_and(|d| d.is_ascii_digit()) {
                    unit * self.number().ok_or("bad charge")? as i8
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
            b'H' if !rest.get(..2).and_then(symbol_to_z).is_some_and(|z| z != 1) => {
                self.i += 1;
                let has_digit = self.peek().is_some_and(|d| d.is_ascii_digit());
                if first && !has_digit {
                    // `[H]`, `[H+]`: a hydrogen atom, not a hydrogen count.
                    return Ok(Expr::Prim(Prim::Element { z: 1, aromatic: Some(false) }));
                }
                let h = if has_digit { self.number().unwrap() } else { 1 };
                Ok(Expr::Prim(Prim::TotalH(h as u8)))
            }
            b'D' => {
                self.i += 1;
                let d = if self.peek().is_some_and(|d| d.is_ascii_digit()) { self.number().unwrap() } else { 1 };
                Ok(Expr::Prim(Prim::Degree(d as u8)))
            }
            b'a' if !rest.starts_with("as") => {
                self.i += 1;
                Ok(Expr::Prim(Prim::Aromatic(true)))
            }
            b'A' if !rest.starts_with("Al") && !rest.starts_with("Ar") && !rest.starts_with("As") && !rest.starts_with("Ag") && !rest.starts_with("Au") && !rest.starts_with("Am") && !rest.starts_with("Ac") && !rest.starts_with("At") => {
                self.i += 1;
                Ok(Expr::Prim(Prim::Aromatic(false)))
            }
            _ => {
                if let Some(sym) = AROMATIC.iter().find(|s| rest.starts_with(**s)) {
                    self.i += sym.len();
                    return element(sym, true);
                }
                let two = rest.get(..2).filter(|t| t.as_bytes()[1].is_ascii_lowercase());
                if let Some(z) = two.and_then(symbol_to_z) {
                    self.i += 2;
                    return Ok(Expr::Prim(Prim::Element { z, aromatic: Some(false) }));
                }
                if let Some(z) = rest.get(..1).filter(|t| t.as_bytes()[0].is_ascii_uppercase()).and_then(symbol_to_z) {
                    self.i += 1;
                    return Ok(Expr::Prim(Prim::Element { z, aromatic: Some(false) }));
                }
                Err(format!("unsupported SMARTS primitive at {rest:?}"))
            }
        }
    }
}

fn parse_bracket(s: &str) -> Result<Expr, String> {
    let mut p = BracketParser { s: s.as_bytes(), i: 0, first: true };
    if s.starts_with(|c: char| c.is_ascii_digit()) {
        return Err(format!("isotope queries unsupported: [{s}]"));
    }
    let e = p.low_and()?;
    if p.i != s.len() {
        return Err(format!("unsupported bracket atom [{s}]"));
    }
    Ok(e)
}

pub fn parse_smarts(smarts: &str) -> Result<Query, String> {
    let s = smarts.as_bytes();
    let mut q = Query::default();
    let mut prev: Option<usize> = None;
    let mut stack = Vec::new();
    let mut pending: Option<BondQuery> = None;
    let mut ring_open: std::collections::HashMap<u32, (usize, Option<BondQuery>)> = Default::default();
    let mut i = 0;
    let add_bond = |q: &mut Query, a: usize, b: usize, bq: Option<BondQuery>| {
        let idx = q.bonds.len();
        q.bonds.push(QueryBond { a, b, query: bq.unwrap_or(BondQuery::SingleOrAromatic) });
        q.nbrs[a].push((b, idx));
        q.nbrs[b].push((a, idx));
    };
    while i < s.len() {
        let c = s[i];
        match c {
            b'(' => {
                stack.push(prev);
                i += 1;
            }
            b')' => {
                prev = stack.pop().ok_or("unbalanced )")?;
                i += 1;
            }
            b'.' => {
                prev = None;
                i += 1;
            }
            b'-' | b'=' | b'#' | b':' | b'~' | b'/' | b'\\' => {
                if pending.is_some() {
                    return Err("compound bond queries unsupported".into());
                }
                pending = Some(match c {
                    b'-' => BondQuery::Single,
                    b'=' => BondQuery::Double,
                    b'#' => BondQuery::Triple,
                    b':' => BondQuery::Aromatic,
                    b'~' => BondQuery::Any,
                    _ => BondQuery::SingleOrAromatic,
                });
                i += 1;
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
                    let bq = pending.take().or(bq);
                    add_bond(&mut q, other, cur, bq);
                } else {
                    ring_open.insert(num, (cur, pending.take()));
                }
            }
            _ => {
                let (expr, len) = if c == b'[' {
                    let end = smarts[i..].find(']').ok_or("unclosed [")? + i;
                    (parse_bracket(&smarts[i + 1..end])?, end + 1 - i)
                } else if c == b'*' {
                    (Expr::Prim(Prim::True), 1)
                } else if let Some(sym) = ORGANIC.iter().find(|sym| smarts[i..].starts_with(**sym)) {
                    (element(sym, false)?, sym.len())
                } else if let Some(sym) = ["b", "c", "n", "o", "p", "s"].iter().find(|sym| smarts[i..].starts_with(**sym)) {
                    (element(sym, true)?, 1)
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
    if !ring_open.is_empty() || !stack.is_empty() {
        return Err("unclosed ring or branch".into());
    }
    Ok(q)
}

//! The posting-list index as a file (benchmark 11): build it once, save it,
//! and load it in another process to match reactants.
//!
//! The file holds the SMARTS, the screening settings and the index arrays
//! (query fingerprints, postings, ranked bits), little-endian:
//!
//! ```text
//! "SSRSIDX1"
//! fp kind, order                 u32 length + UTF-8 each
//! fp bits, queries               u64 each
//! SMARTS                         u32 length + UTF-8, per query
//! query fingerprints             queries x fp bits / 64 u64
//! postings                       fp bits x ceil(queries / 64) u64
//! ranked bits                    fp bits u16
//! ```
//!
//! The parsed query graphs and exact-match plans are not stored: loading
//! parses the SMARTS and plans them again, which needs no fingerprints.
//!
//! Author: Marcin Kowiel + Claude

use crate::fingerprint::{fp_kind, target_fp, Fp, FpKind};
use crate::index::Index;
use crate::matcher::{self, Order};
use crate::smarts::{self, Query};
use crate::{parse_all, Targets};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SSRSIDX1";

/// Parsed queries, with the SMARTS kept for saving.
pub struct Queries {
    smarts: Vec<String>,
    parsed: Vec<Query>,
}

impl Queries {
    pub fn parse(smarts: Vec<String>) -> Result<Self, String> {
        let parsed = parse_all(&smarts, "query", smarts::parse_smarts)?;
        Ok(Queries { smarts, parsed })
    }

    pub fn len(&self) -> usize {
        self.smarts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.smarts.is_empty()
    }
}

/// One reactant's matches: query IDs ascending, and the screen counts.
pub struct Found {
    pub found: Vec<u32>,
    pub posted: usize,
    pub candidates: usize,
}

/// An index of any supported fingerprint width.
pub struct StoredIndex {
    kind: FpKind,
    order: String,
    fp_bits: usize,
    smarts: Vec<String>,
    inner: Box<dyn Width>,
}

/// What `StoredIndex` needs from an `Index<W>`, whatever `W` is.
trait Width: Send + Sync {
    fn match_many(&self, kind: &FpKind, targets: &Targets, postings: usize) -> Vec<Found>;
    fn write_arrays(&self, out: &mut dyn Write) -> std::io::Result<()>;
    fn nbytes(&self) -> usize;
}

impl<const W: usize> Width for Index<W> {
    fn match_many(&self, kind: &FpKind, targets: &Targets, postings: usize) -> Vec<Found> {
        targets
            .0
            .par_iter()
            .map(|t| {
                let m = self.match_one(t, &target_fp::<W>(kind, t), postings);
                Found { found: m.found, posted: m.posted, candidates: m.candidates }
            })
            .collect()
    }

    fn write_arrays(&self, out: &mut dyn Write) -> std::io::Result<()> {
        let (fps, postings, ranked_bits) = self.parts();
        for &word in fps.iter().flatten().chain(postings) {
            out.write_all(&word.to_le_bytes())?;
        }
        for &bit in ranked_bits {
            out.write_all(&bit.to_le_bytes())?;
        }
        Ok(())
    }

    fn nbytes(&self) -> usize {
        Index::nbytes(self)
    }
}

/// Calls `$f::<W>($args)` for the width of `$bits`, or returns an error.
macro_rules! by_width {
    ($bits:expr, $f:ident($($args:expr),*)) => {
        match $bits {
            512 => $f::<8>($($args),*),
            1024 => $f::<16>($($args),*),
            2048 => $f::<32>($($args),*),
            4096 => $f::<64>($($args),*),
            8192 => $f::<128>($($args),*),
            16384 => $f::<256>($($args),*),
            other => return Err(format!("fingerprint size {other} not supported")),
        }
    };
}

fn build_width<const W: usize>(kind: &FpKind, order: Order, queries: Vec<Query>) -> Box<dyn Width> {
    Box::new(Index::<W>::build(kind, order, queries))
}

fn load_width<const W: usize>(
    order: Order,
    queries: Vec<Query>,
    reader: &mut Reader,
) -> Result<Box<dyn Width>, String> {
    let n = queries.len();
    let fps: Vec<Fp<W>> = reader
        .u64s(n * W)?
        .chunks_exact(W)
        .map(|c| c.try_into().expect("W words"))
        .collect();
    let postings = reader.u64s(64 * W * n.div_ceil(64))?;
    let ranked_bits = reader.u16s(64 * W)?;
    Ok(Box::new(Index::<W>::from_parts(order, queries, fps, postings, ranked_bits)?))
}

impl StoredIndex {
    /// Query fingerprints, postings and plans; `queries` is consumed.
    pub fn build(queries: Queries, kind: &str, fp_bits: usize, order: &str) -> Result<Self, String> {
        let kind = fp_kind(kind)?;
        let inner = by_width!(fp_bits, build_width(&kind, matcher::order(order)?, queries.parsed));
        Ok(StoredIndex { kind, order: order.into(), fp_bits, smarts: queries.smarts, inner })
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut out = BufWriter::new(File::create(path).map_err(err)?);
        out.write_all(MAGIC).map_err(err)?;
        for text in [self.kind.name, self.order.as_str()] {
            write_str(&mut out, text).map_err(err)?;
        }
        for value in [self.fp_bits, self.smarts.len()] {
            out.write_all(&(value as u64).to_le_bytes()).map_err(err)?;
        }
        for text in &self.smarts {
            write_str(&mut out, text).map_err(err)?;
        }
        self.inner.write_arrays(&mut out).map_err(err)?;
        out.flush().map_err(err)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut reader = Reader { bytes: &bytes, at: 0 };
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(format!("{}: not a substructure_rs index", path.display()));
        }
        let kind = fp_kind(&reader.string()?)?;
        let order = reader.string()?;
        let fp_bits = reader.u64()? as usize;
        let n = reader.u64()? as usize;
        let smarts = (0..n).map(|_| reader.string()).collect::<Result<Vec<_>, _>>()?;
        let queries = parse_all(&smarts, "query", smarts::parse_smarts)?;
        let inner = by_width!(fp_bits, load_width(matcher::order(&order)?, queries, &mut reader))?;
        if reader.at != bytes.len() {
            return Err(format!("{}: {} trailing bytes", path.display(), bytes.len() - reader.at));
        }
        Ok(StoredIndex { kind, order, fp_bits, smarts, inner })
    }

    /// Screen and exactly match every reactant, on the current rayon pool.
    pub fn match_many(&self, targets: &Targets, postings: usize) -> Result<Vec<Found>, String> {
        if !(1..=self.fp_bits).contains(&postings) {
            return Err(format!("postings must be between 1 and {}", self.fp_bits));
        }
        Ok(self.inner.match_many(&self.kind, targets, postings))
    }

    pub fn queries(&self) -> usize {
        self.smarts.len()
    }

    pub fn fp_kind(&self) -> &str {
        self.kind.name
    }

    pub fn fp_bits(&self) -> usize {
        self.fp_bits
    }

    pub fn order(&self) -> &str {
        &self.order
    }

    /// Bytes of the fingerprint and posting arrays in memory.
    pub fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }
}

fn write_str(out: &mut impl Write, text: &str) -> std::io::Result<()> {
    out.write_all(&(text.len() as u32).to_le_bytes())?;
    out.write_all(text.as_bytes())
}

/// A cursor over the file's bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&end| end <= self.bytes.len()).ok_or("index file is truncated")?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")) as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|e| e.to_string())
    }

    fn u64s(&mut self, n: usize) -> Result<Vec<u64>, String> {
        let bytes = self.take(n.checked_mul(8).ok_or("index file is corrupt")?)?;
        Ok(bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect())
    }

    fn u16s(&mut self, n: usize) -> Result<Vec<u16>, String> {
        let bytes = self.take(n.checked_mul(2).ok_or("index file is corrupt")?)?;
        Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().expect("2 bytes"))).collect())
    }
}

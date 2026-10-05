//! Posting-list substructure screen over packed 2048-bit pattern fingerprints.
//!
//! Author: Marcin Kowiel + Claude

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

const FP_SIZE: usize = 2048;
const FP_WORDS: usize = FP_SIZE / 64;

type Fp = [u64; FP_WORDS];

fn pack(bits: &[u32]) -> PyResult<Fp> {
    let mut fp = [0u64; FP_WORDS];
    for &bit in bits {
        let bit = bit as usize;
        if bit >= FP_SIZE {
            return Err(PyValueError::new_err(format!("bit {bit} out of range")));
        }
        fp[bit / 64] |= 1 << (bit % 64);
    }
    Ok(fp)
}

#[pyclass(frozen)]
struct PostingIndex {
    query_fps: Vec<Fp>,
    /// `FP_SIZE` rows of `words` u64s, row-major by fingerprint bit.
    postings: Vec<u64>,
    words: usize,
    ranked_bits: Vec<u16>,
    all_ids: Vec<u64>,
}

impl PostingIndex {
    fn screen(&self, fp: &Fp, posting_limit: usize, out: &mut Vec<u32>) {
        let mut remaining = self.all_ids.clone();
        let mut selected = 0;
        for &bit in &self.ranked_bits {
            let bit = bit as usize;
            if fp[bit / 64] >> (bit % 64) & 1 == 0 {
                let row = &self.postings[bit * self.words..(bit + 1) * self.words];
                for (r, &p) in remaining.iter_mut().zip(row) {
                    *r &= !p;
                }
                selected += 1;
                if selected == posting_limit {
                    break;
                }
            }
        }
        for (w, &word) in remaining.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let idx = w * 64 + word.trailing_zeros() as usize;
                word &= word - 1;
                let q = &self.query_fps[idx];
                if q.iter().zip(fp).all(|(&a, &b)| a & !b == 0) {
                    out.push(idx as u32);
                }
            }
        }
    }
}

#[pymethods]
impl PostingIndex {
    /// Build from each query's pattern-fingerprint on-bits.
    #[new]
    fn new(query_bits: Vec<Vec<u32>>) -> PyResult<Self> {
        let n = query_bits.len();
        let words = n.div_ceil(64);
        let query_fps = query_bits
            .iter()
            .map(|bits| pack(bits))
            .collect::<PyResult<Vec<_>>>()?;
        let mut postings = vec![0u64; FP_SIZE * words];
        let mut counts = [0usize; FP_SIZE];
        for (idx, bits) in query_bits.iter().enumerate() {
            for &bit in bits {
                let bit = bit as usize;
                postings[bit * words + idx / 64] |= 1 << (idx % 64);
                counts[bit] += 1;
            }
        }
        // Stable sort matches Python's sorted(..., reverse=True) tie order.
        let mut ranked_bits: Vec<u16> = (0..FP_SIZE as u16).collect();
        ranked_bits.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]));
        let mut all_ids = vec![u64::MAX; words];
        if n % 64 != 0 {
            all_ids[words - 1] = (1u64 << (n % 64)) - 1;
        }
        Ok(Self { query_fps, postings, words, ranked_bits, all_ids })
    }

    fn __len__(&self) -> usize {
        FP_SIZE
    }

    /// Bytes held by fingerprints, postings and the all-ids mask.
    fn nbytes(&self) -> usize {
        8 * (self.query_fps.len() * FP_WORDS + self.postings.len() + self.all_ids.len())
    }

    /// Screen reactants in parallel. Returns (offsets, ids) as little-endian
    /// u64 / u32 bytes: reactant i's candidates are ids[offsets[i]:offsets[i+1]].
    fn screen_many<'py>(
        &self,
        py: Python<'py>,
        reactant_bits: Vec<Vec<u32>>,
        posting_limit: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        if !(1..=FP_SIZE).contains(&posting_limit) {
            return Err(PyValueError::new_err("posting_limit out of range"));
        }
        let fps = reactant_bits
            .iter()
            .map(|bits| pack(bits))
            .collect::<PyResult<Vec<_>>>()?;
        let per: Vec<Vec<u32>> = py.detach(|| {
            fps.par_iter()
                .map(|fp| {
                    let mut out = Vec::new();
                    self.screen(fp, posting_limit, &mut out);
                    out
                })
                .collect()
        });
        let mut offsets = Vec::with_capacity(8 * (per.len() + 1));
        let mut ids = Vec::new();
        let mut total = 0u64;
        offsets.extend_from_slice(&total.to_le_bytes());
        for c in &per {
            total += c.len() as u64;
            offsets.extend_from_slice(&total.to_le_bytes());
            for id in c {
                ids.extend_from_slice(&id.to_le_bytes());
            }
        }
        Ok((PyBytes::new(py, &offsets), PyBytes::new(py, &ids)))
    }
}

#[pymodule]
fn rust_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PostingIndex>()
}

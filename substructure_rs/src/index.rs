//! Posting-list screen, as in benchmarks 05a/06, over this crate's fingerprint.
//!
//! Author: Marcin Kowiel + Claude

use crate::fingerprint::{query_fp, Fp, FP_BITS};
use crate::matcher::{has_match, Plan, Target};
use crate::smarts::Query;
use rayon::prelude::*;

pub struct Index {
    pub queries: Vec<Query>,
    plans: Vec<Plan>,
    fps: Vec<Fp>,
    /// `FP_BITS` rows of `words` u64s.
    postings: Vec<u64>,
    words: usize,
    ranked_bits: Vec<u16>,
    all_ids: Vec<u64>,
}

impl Index {
    pub fn build(queries: Vec<Query>) -> Self {
        let n = queries.len();
        let words = n.div_ceil(64);
        let (fps, plans): (Vec<Fp>, Vec<Plan>) = queries.par_iter().map(|q| (query_fp(q), Plan::new(q))).unzip();
        let mut postings = vec![0u64; FP_BITS * words];
        let mut counts = vec![0usize; FP_BITS];
        for (idx, fp) in fps.iter().enumerate() {
            for (w, &word) in fp.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let bit = w * 64 + word.trailing_zeros() as usize;
                    word &= word - 1;
                    postings[bit * words + idx / 64] |= 1 << (idx % 64);
                    counts[bit] += 1;
                }
            }
        }
        let mut ranked_bits: Vec<u16> = (0..FP_BITS as u16).collect();
        ranked_bits.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]));
        let mut all_ids = vec![u64::MAX; words];
        if n % 64 != 0 {
            all_ids[words - 1] = (1u64 << (n % 64)) - 1;
        }
        Index { queries, plans, fps, postings, words, ranked_bits, all_ids }
    }

    pub fn nbytes(&self) -> usize {
        8 * (self.fps.len() * self.fps.first().map_or(0, |f| f.len()) + self.postings.len() + self.all_ids.len())
    }

    /// Screened candidate query IDs, ascending.
    pub fn screen(&self, fp: &Fp, posting_limit: usize, out: &mut Vec<u32>) {
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
                if self.fps[idx].iter().zip(fp).all(|(&q, &t)| q & !t == 0) {
                    out.push(idx as u32);
                }
            }
        }
    }

    /// Candidates and matching query IDs for one reactant.
    pub fn match_one(&self, target: &Target, fp: &Fp, posting_limit: usize) -> (usize, Vec<u32>) {
        let mut cands = Vec::new();
        self.screen(fp, posting_limit, &mut cands);
        let n = cands.len();
        cands.retain(|&q| has_match(&self.queries[q as usize], &self.plans[q as usize], target));
        (n, cands)
    }
}

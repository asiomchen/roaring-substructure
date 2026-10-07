//! Posting-list screen, as in benchmarks 05a/06, over this crate's fingerprint.
//!
//! Author: Marcin Kowiel + Claude

use crate::fingerprint::{query_fp, Fp, FpKind};
use crate::matcher::{has_match, Plan, Target};
use crate::smarts::Query;
use rayon::prelude::*;
use std::time::Instant;

/// `W` is the fingerprint width in 64-bit words.
pub struct Index<const W: usize> {
    pub queries: Vec<Query>,
    plans: Vec<Plan>,
    fps: Vec<Fp<W>>,
    /// `64 * W` rows of `words` u64s.
    postings: Vec<u64>,
    words: usize,
    ranked_bits: Vec<u16>,
    all_ids: Vec<u64>,
}

impl<const W: usize> Index<W> {
    const BITS: usize = 64 * W;

    pub fn build(kind: &FpKind, queries: Vec<Query>) -> Self {
        let n = queries.len();
        let words = n.div_ceil(64);
        let (fps, plans): (Vec<Fp<W>>, Vec<Plan>) =
            queries.par_iter().map(|q| (query_fp(kind, q), Plan::new(q))).unzip();
        let mut postings = vec![0u64; Self::BITS * words];
        let mut counts = vec![0usize; Self::BITS];
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
        let mut ranked_bits: Vec<u16> = (0..Self::BITS).map(|b| b as u16).collect();
        ranked_bits.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]));
        let mut all_ids = vec![u64::MAX; words];
        if n % 64 != 0 {
            all_ids[words - 1] = (1u64 << (n % 64)) - 1;
        }
        Index { queries, plans, fps, postings, words, ranked_bits, all_ids }
    }

    pub fn nbytes(&self) -> usize {
        8 * (self.fps.len() * W + self.postings.len() + self.all_ids.len())
    }

    /// Mean fraction of bits set in the query fingerprints.
    pub fn query_density(&self) -> f64 {
        let ones: u64 = self.fps.iter().flatten().map(|w| w.count_ones() as u64).sum();
        ones as f64 / (self.fps.len().max(1) * Self::BITS) as f64
    }

    /// Query IDs left after removing the postings of up to `posting_limit`
    /// of the reactant's absent bits, as a bitmap of `words` u64s.
    pub fn posting_filter(&self, fp: &Fp<W>, posting_limit: usize) -> Vec<u64> {
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
        remaining
    }

    /// IDs from `remaining` whose fingerprint is a subset of `fp`, ascending.
    pub fn subset_filter(&self, fp: &Fp<W>, remaining: &[u64], out: &mut Vec<u32>) {
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

    /// Screen and exactly match one reactant, timing each step.
    pub fn match_one(&self, target: &Target, fp: &Fp<W>, posting_limit: usize) -> Matched {
        let start = Instant::now();
        let remaining = self.posting_filter(fp, posting_limit);
        let posted = remaining.iter().map(|w| w.count_ones() as usize).sum();
        let t_postings = start.elapsed();
        let mut found = Vec::new();
        self.subset_filter(fp, &remaining, &mut found);
        let candidates = found.len();
        let t_subset = start.elapsed();
        found.retain(|&q| has_match(&self.queries[q as usize], &self.plans[q as usize], target));
        let t_exact = start.elapsed();
        Matched {
            posted,
            candidates,
            found,
            seconds: [
                t_postings.as_secs_f64(),
                (t_subset - t_postings).as_secs_f64(),
                (t_exact - t_subset).as_secs_f64(),
            ],
        }
    }
}

/// One reactant's result.
pub struct Matched {
    /// Queries left after the posting filter.
    pub posted: usize,
    /// Queries left after the full fingerprint check, i.e. sent to exact matching.
    pub candidates: usize,
    /// Matching query IDs, ascending.
    pub found: Vec<u32>,
    /// Seconds in the posting filter, the fingerprint check and exact matching.
    pub seconds: [f64; 3],
}

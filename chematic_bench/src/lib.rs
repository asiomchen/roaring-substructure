//! Benchmark 10: substructure search with the chematic crates.
//!
//! chematic (<https://github.com/kent-tokyo/chematic>) is a pure-Rust
//! cheminformatics toolkit with its own SMILES parser, SMARTS parser and VF2
//! matcher. It has no fingerprint for SMARTS queries (its Pattern fingerprint
//! only covers concrete molecules), so it cannot screen this workload itself.
//! Two modes:
//!
//! - `naive`: chematic parses everything and runs `has_match_with_config` on
//!   every (reactant, query) pair, like benchmark 01 does with RDKit.
//! - `screened`: `substructure_rs` screens the pairs with its fingerprint and
//!   postings (outside the timer), and chematic checks only the candidates,
//!   so the timer measures chematic's exact matcher on the pairs that
//!   benchmark 08's own matcher sees.
//!
//! The digest uses the same encoding as benchmarks 05a-09.
//!
//! Author: Marcin Kowiel + Claude

#[cfg(feature = "python")]
mod python;

use chematic_core::Molecule;
use chematic_smarts::{has_match_with_config, parse_smarts, MatchConfig, QueryMolecule};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

pub const USAGE: &str = "usage: chematic_bench [--mode naive|screened] [--queries-file F] [--reactants-file F] \
[--runs N] [--threads N] [--fp-kind K] [--fp-bits N] [--postings N] [--reference pairs.bin]\n\
Files may be .parquet or .csv; queries use column `substructure`, reactants `smiles`.\n\
--fp-kind, --fp-bits and --postings set the substructure_rs screen of `screened` mode\n\
(default paths4+branches at 2048 bits, 128 postings).";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Naive,
    Screened,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "naive" => Ok(Mode::Naive),
            "screened" => Ok(Mode::Screened),
            other => Err(format!("unknown mode {other:?}; choose naive or screened")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Naive => "naive",
            Mode::Screened => "screened",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub mode: Mode,
    pub queries_file: PathBuf,
    pub reactants_file: PathBuf,
    pub runs: usize,
    /// 0 = all cores.
    pub threads: usize,
    pub fp_kind: String,
    pub fp_bits: usize,
    pub postings: usize,
    /// RDKit match pairs to compare against (from `substructure_rs/tools/rdkit_reference.py`).
    pub reference: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            mode: Mode::Screened,
            queries_file: "data/substructure_sample.parquet".into(),
            reactants_file: "data/reactant_sample.csv".into(),
            runs: 3,
            threads: 0,
            fp_kind: "paths4+branches".into(),
            fp_bits: 2048,
            postings: 128,
            reference: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub mode: Mode,
    pub queries: usize,
    pub reactants: usize,
    pub threads: usize,
    pub parse_queries_seconds: f64,
    pub parse_reactants_seconds: f64,
    /// `screened` only: substructure_rs's reactant fingerprints and screen (not in the timer).
    pub screen_seconds: Option<f64>,
    /// Pairs given to chematic's matcher per run.
    pub pairs: usize,
    pub match_seconds: Vec<f64>,
    pub matches: usize,
    pub digest: String,
    /// (reference pairs, missing, extra) against `--reference`.
    pub comparison: Option<(usize, usize, usize)>,
}

impl Report {
    pub fn match_median(&self) -> f64 {
        let mut sorted = self.match_seconds.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    }
}

pub fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Option<Options>, String> {
    let mut opts = Options::default();
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        let num = |v: String, f: &str| v.parse::<usize>().map_err(|e| format!("{f}: {e}"));
        match flag.as_str() {
            "--mode" => opts.mode = Mode::parse(&value()?)?,
            "--queries-file" => opts.queries_file = value()?.into(),
            "--reactants-file" => opts.reactants_file = value()?.into(),
            "--runs" => opts.runs = num(value()?, "--runs")?,
            "--threads" => opts.threads = num(value()?, "--threads")?,
            "--fp-kind" => opts.fp_kind = value()?,
            "--fp-bits" => opts.fp_bits = num(value()?, "--fp-bits")?,
            "--postings" => opts.postings = num(value()?, "--postings")?,
            "--reference" => opts.reference = Some(value()?.into()),
            "-h" | "--help" => return Ok(None),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if opts.runs < 1 {
        return Err("runs must be at least 1".into());
    }
    Ok(Some(opts))
}

pub fn cli<I: IntoIterator<Item = String>>(args: I) -> i32 {
    let result = parse_args(args).and_then(|opts| match opts {
        None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(opts) => run(&opts).map(|r| print_report(&r)),
    });
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

/// Parse everything with chematic, reporting the first failures by row.
fn parse_all<T: Send, E: std::fmt::Debug + Send>(
    items: &[String],
    what: &str,
    parse: impl Fn(&str) -> Result<T, E> + Sync,
) -> Result<Vec<T>, String> {
    let parsed: Vec<Result<T, E>> = items.par_iter().map(|s| parse(s)).collect();
    let bad: Vec<String> = parsed
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.as_ref().err().map(|e| format!("{what} {i} {:?}: {e:?}", items[i])))
        .collect();
    if !bad.is_empty() {
        let shown = bad.iter().take(10).cloned().collect::<Vec<_>>().join("\n");
        return Err(format!("{shown}\n{} {what}s failed to parse with chematic", bad.len()));
    }
    Ok(parsed.into_iter().map(|r| r.ok().unwrap()).collect())
}

pub fn run(opts: &Options) -> Result<Report, String> {
    if opts.threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(opts.threads).build().map_err(|e| e.to_string())?;
        pool.install(|| run_in_pool(opts))
    } else {
        run_in_pool(opts)
    }
}

fn run_in_pool(opts: &Options) -> Result<Report, String> {
    // Candidate query IDs per reactant: every query, or substructure_rs's screen.
    let (smarts, smiles, candidates, screen_seconds) = match opts.mode {
        Mode::Naive => {
            let smarts = substructure_rs::read_column(&opts.queries_file, "substructure")?;
            let smiles = substructure_rs::read_column(&opts.reactants_file, "smiles")?;
            (smarts, smiles, None, None)
        }
        Mode::Screened => {
            let screen_opts = substructure_rs::Options {
                queries_file: opts.queries_file.clone(),
                reactants_file: opts.reactants_file.clone(),
                fp_kind: opts.fp_kind.clone(),
                fp_bits: opts.fp_bits,
                postings: opts.postings,
                threads: opts.threads,
                ..Default::default()
            };
            let c = substructure_rs::candidates(&screen_opts)?;
            (c.smarts, c.smiles, Some(c.per_reactant), Some(c.screen_seconds))
        }
    };

    let start = Instant::now();
    let queries: Vec<QueryMolecule> = parse_all(&smarts, "query", parse_smarts)?;
    let parse_queries_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mols: Vec<Molecule> = parse_all(&smiles, "reactant", chematic_smiles::parse)?;
    let parse_reactants_seconds = start.elapsed().as_secs_f64();

    let all: Vec<u32> = (0..queries.len() as u32).collect();
    let pairs = match &candidates {
        Some(c) => c.iter().map(Vec::len).sum(),
        None => queries.len() * mols.len(),
    };
    let config = MatchConfig::default();
    let mut match_seconds = Vec::new();
    let mut result = None;
    for _ in 0..opts.runs {
        let start = Instant::now();
        let per: Vec<Vec<u32>> = mols
            .par_iter()
            .enumerate()
            .map(|(r, mol)| {
                let cands = candidates.as_ref().map_or(&all, |c| &c[r]);
                cands.iter().copied().filter(|&q| has_match_with_config(&queries[q as usize], mol, &config)).collect()
            })
            .collect();
        match_seconds.push(start.elapsed().as_secs_f64());
        result.get_or_insert(per);
    }
    let per = result.expect("at least one run");

    let matches = per.iter().map(Vec::len).sum();
    let mut hasher = Sha256::new();
    for (r, found) in per.iter().enumerate() {
        for &q in found {
            hasher.update((r as u32).to_le_bytes());
            hasher.update(q.to_le_bytes());
        }
    }
    let digest = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();

    let comparison = match &opts.reference {
        None => None,
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let reference: HashSet<(u32, u32)> = bytes
                .chunks_exact(8)
                .map(|c| (u32::from_le_bytes(c[..4].try_into().unwrap()), u32::from_le_bytes(c[4..].try_into().unwrap())))
                .collect();
            let ours: HashSet<(u32, u32)> =
                per.iter().enumerate().flat_map(|(r, f)| f.iter().map(move |&q| (r as u32, q))).collect();
            Some((reference.len(), reference.difference(&ours).count(), ours.difference(&reference).count()))
        }
    };

    Ok(Report {
        mode: opts.mode,
        queries: queries.len(),
        reactants: mols.len(),
        threads: rayon::current_num_threads(),
        parse_queries_seconds,
        parse_reactants_seconds,
        screen_seconds,
        pairs,
        match_seconds,
        matches,
        digest,
        comparison,
    })
}

pub fn print_report(r: &Report) {
    println!(
        "{} queries x {} reactants; chematic {}; {} matching runs; {} threads",
        r.queries,
        r.reactants,
        r.mode.name(),
        r.match_seconds.len(),
        r.threads
    );
    println!("parse SMARTS    {:>8.3} s  (chematic)", r.parse_queries_seconds);
    println!("parse SMILES    {:>8.3} s  (chematic)", r.parse_reactants_seconds);
    if let Some(s) = r.screen_seconds {
        println!("screen          {:>8.3} s  (substructure_rs fingerprints + postings, not in the timer)", s);
    }
    println!(
        "match median    {:>8.3} s  runs: {:?}  ({:.0} ns per pair per thread)",
        r.match_median(),
        r.match_seconds.iter().map(|t| (t * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        1e9 * r.match_median() * r.threads as f64 / r.pairs.max(1) as f64
    );
    println!("pairs checked   {:>8}  matches: {}", r.pairs, r.matches);
    println!("digest          {}", r.digest);
    if let Some((n, missing, extra)) = r.comparison {
        println!("reference       {n} pairs; missing {missing}; extra {extra}");
    }
}

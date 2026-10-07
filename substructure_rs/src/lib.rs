//! Benchmark 08: substructure search without RDKit.
//!
//! Reads reaction-template SMARTS and reactant SMILES, builds the posting-list
//! index, and matches every reactant against every screened query in parallel.
//! The digest uses the same encoding as benchmarks 05a-07, so it can be
//! compared with RDKit's result directly. Benchmark 09 reruns it over the
//! fingerprint kinds and sizes of `--fp-kind` / `--fp-bits`.
//!
//! Used three ways: the `substructure_rs` binary (`src/main.rs`), the same
//! command installed by uv as a Python entry point, and `substructure_rs.run()`
//! from Python (`src/python.rs`, feature `python`).
//!
//! Author: Marcin Kowiel + Claude

mod analysis;
mod fingerprint;
mod index;
mod matcher;
mod mol;
mod periodic;
mod rings;
mod smarts;
#[cfg(test)]
mod tests;
#[cfg(feature = "python")]
mod python;

use fingerprint::{fp_kind, target_fp, FpKind};
use index::{Index, Matched};
use matcher::Target;
use smarts::Query;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const USAGE: &str = "usage: substructure_rs [--queries-file F] [--reactants-file F] [--postings N] \
[--runs N] [--threads N] [--fp-kind K] [--fp-bits N] [--reference pairs.bin] [--dump-atoms out.txt]\n\
[--analyze-false-positives]\n\
Files may be .parquet or .csv; queries use column `substructure`, reactants `smiles`.\n\
--fp-kind: atoms, paths2, paths4 (default), paths4-nocount, paths6, paths4+branches,\n\
paths4+cycles, paths4+branches+cycles. --fp-bits: 512, 1024, 2048, 4096 (default), 8192, 16384.";

/// Benchmark settings; `Options::default()` matches the command-line defaults.
#[derive(Clone, Debug)]
pub struct Options {
    pub queries_file: PathBuf,
    pub reactants_file: PathBuf,
    pub postings: usize,
    pub runs: usize,
    /// 0 = all cores.
    pub threads: usize,
    /// Screening fingerprint features, a name from `fingerprint::FP_KINDS`.
    pub fp_kind: String,
    /// Screening fingerprint size, one of `fingerprint::FP_SIZES`.
    pub fp_bits: usize,
    /// RDKit match pairs to compare against (from `tools/rdkit_reference.py`).
    pub reference: Option<PathBuf>,
    /// Write per-atom properties in the format of `tools/rdkit_reference.py`.
    pub dump_atoms: Option<PathBuf>,
    /// Report why screened pairs fail the exact match (see `analysis.rs`).
    pub analyze: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            queries_file: "data/substructure_sample.parquet".into(),
            reactants_file: "data/reactant_sample.csv".into(),
            postings: 128,
            runs: 3,
            threads: 0,
            fp_kind: fingerprint::DEFAULT_FP_KIND.into(),
            fp_bits: fingerprint::DEFAULT_FP_BITS,
            reference: None,
            dump_atoms: None,
            analyze: false,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<(), String> {
        fp_kind(&self.fp_kind)?;
        fingerprint::check_fp_bits(self.fp_bits)?;
        if !(1..=self.fp_bits).contains(&self.postings) {
            return Err(format!("postings must be between 1 and {}", self.fp_bits));
        }
        if self.runs < 1 {
            return Err("runs must be at least 1".into());
        }
        Ok(())
    }
}

/// Differences from a reference match set; examples are (reactant SMILES, query SMARTS).
#[derive(Clone, Debug)]
pub struct Comparison {
    pub reference_pairs: usize,
    pub missing: usize,
    pub extra: usize,
    pub missing_examples: Vec<(u32, u32, String, String)>,
    pub extra_examples: Vec<(u32, u32, String, String)>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub queries: usize,
    pub reactants: usize,
    pub postings: usize,
    pub threads: usize,
    pub fp_kind: String,
    pub fp_bits: usize,
    pub read_seconds: f64,
    pub parse_queries_seconds: f64,
    pub parse_reactants_seconds: f64,
    pub build_seconds: f64,
    pub match_seconds: Vec<f64>,
    /// Time inside matching by step, summed over reactants (CPU seconds, so
    /// it exceeds the wall clock with several threads), mean over runs:
    /// reactant fingerprint, posting filter, fingerprint check, exact match.
    pub phase_seconds: [f64; 4],
    pub index_bytes: usize,
    /// Mean fraction of fingerprint bits set, over queries and over reactants.
    pub query_density: f64,
    pub target_density: f64,
    /// Pairs left after the posting filter.
    pub posted: usize,
    /// Pairs left after the full fingerprint check, i.e. sent to exact matching.
    pub candidates: usize,
    pub matches: usize,
    pub digest: String,
    pub comparison: Option<Comparison>,
    pub analysis: Option<String>,
}

impl Report {
    pub fn match_median(&self) -> f64 {
        let mut sorted = self.match_seconds.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    }
}

/// Parse command-line arguments (without the program name).
/// `Ok(None)` means `--help` was requested.
pub fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Option<Options>, String> {
    let mut opts = Options::default();
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--queries-file" => opts.queries_file = value()?.into(),
            "--reactants-file" => opts.reactants_file = value()?.into(),
            "--postings" => opts.postings = value()?.parse().map_err(|e| format!("--postings: {e}"))?,
            "--runs" => opts.runs = value()?.parse().map_err(|e| format!("--runs: {e}"))?,
            "--threads" => opts.threads = value()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--fp-kind" => opts.fp_kind = value()?,
            "--fp-bits" => opts.fp_bits = value()?.parse().map_err(|e| format!("--fp-bits: {e}"))?,
            "--reference" => opts.reference = Some(value()?.into()),
            "--dump-atoms" => opts.dump_atoms = Some(value()?.into()),
            "--analyze-false-positives" => opts.analyze = true,
            "-h" | "--help" => return Ok(None),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    opts.validate()?;
    Ok(Some(opts))
}

/// Run the command line: parse, run, print. Returns the process exit code.
pub fn cli<I: IntoIterator<Item = String>>(args: I) -> i32 {
    let result = parse_args(args).and_then(|opts| match opts {
        None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(opts) => run(&opts).map(|report| print_report(&report)),
    });
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

fn read_column(path: &Path, column: &str) -> Result<Vec<String>, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "parquet" {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use parquet::record::Field;
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let reader = SerializedFileReader::new(file).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut out = Vec::with_capacity(reader.metadata().file_metadata().num_rows() as usize);
        for row in reader.get_row_iter(None).map_err(|e| e.to_string())? {
            let row = row.map_err(|e| e.to_string())?;
            let (_, field) = row
                .get_column_iter()
                .find(|(name, _)| name.as_str() == column)
                .ok_or_else(|| format!("{}: no column {column}", path.display()))?;
            match field {
                Field::Str(s) => out.push(s.clone()),
                other => return Err(format!("{}: column {column} is not a string: {other:?}", path.display())),
            }
        }
        Ok(out)
    } else {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut lines = BufReader::new(file).lines();
        let header = lines.next().ok_or("empty CSV")?.map_err(|e| e.to_string())?;
        let col = header
            .split(',')
            .position(|h| h.trim() == column)
            .ok_or_else(|| format!("{}: no column {column}", path.display()))?;
        let mut out = Vec::new();
        for line in lines {
            let line = line.map_err(|e| e.to_string())?;
            if line.contains('"') {
                return Err(format!("{}: quoted CSV fields are not supported", path.display()));
            }
            out.push(line.split(',').nth(col).ok_or("short CSV row")?.to_string());
        }
        Ok(out)
    }
}

fn dump_atoms(path: &Path, mols: &[mol::Mol]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(File::create(path)?);
    for m in mols {
        let atoms: Vec<String> = m
            .target_atoms()
            .iter()
            .map(|a| format!("{},{},{},{},{}", a.z, a.aromatic as u8, a.total_h, a.degree, a.charge))
            .collect();
        let bonds: Vec<String> = m
            .bonds
            .iter()
            .map(|b| {
                let t = match b.kind {
                    mol::BondType::Single | mol::BondType::Dative => 2,
                    mol::BondType::Double => 4,
                    mol::BondType::Triple => 6,
                    mol::BondType::Quadruple => 8,
                    mol::BondType::Aromatic => 3,
                };
                format!("{}-{}-{}", b.a.min(b.b), b.a.max(b.b), t)
            })
            .collect();
        writeln!(f, "{}|{}", atoms.join(";"), bonds.join(";"))?;
    }
    Ok(())
}

/// Parse everything with `parse`, reporting the first failures by row.
fn parse_all<T: Send>(
    items: &[String],
    what: &str,
    parse: impl Fn(&str) -> Result<T, String> + Sync,
) -> Result<Vec<T>, String> {
    let parsed: Vec<Result<T, String>> = items.par_iter().map(|s| parse(s)).collect();
    let bad: Vec<String> = parsed
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.as_ref().err().map(|e| format!("{what} {i} {:?}: {e}", items[i])))
        .collect();
    if !bad.is_empty() {
        let shown = bad.iter().take(10).cloned().collect::<Vec<_>>().join("\n");
        return Err(format!("{shown}\n{} {what}s failed to parse", bad.len()));
    }
    Ok(parsed.into_iter().map(|r| r.ok().unwrap()).collect())
}

/// Run the benchmark. Uses a private thread pool when `threads > 0`, so it can
/// be called repeatedly with different thread counts.
pub fn run(opts: &Options) -> Result<Report, String> {
    opts.validate()?;
    if opts.threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(opts.threads).build().map_err(|e| e.to_string())?;
        pool.install(|| run_in_pool(opts))
    } else {
        run_in_pool(opts)
    }
}

fn run_in_pool(opts: &Options) -> Result<Report, String> {
    let start = Instant::now();
    let smarts = read_column(&opts.queries_file, "substructure")?;
    let smiles = read_column(&opts.reactants_file, "smiles")?;
    let read_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let queries = parse_all(&smarts, "query", smarts::parse_smarts)?;
    let parse_queries_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let mols = parse_all(&smiles, "reactant", mol::parse_smiles)?;
    let targets: Vec<Target> = mols.par_iter().map(Target::new).collect();
    let parse_reactants_seconds = start.elapsed().as_secs_f64();
    if let Some(path) = &opts.dump_atoms {
        dump_atoms(path, &mols).map_err(|e| format!("{}: {e}", path.display()))?;
    }

    let kind = fp_kind(&opts.fp_kind)?;
    let screened = match opts.fp_bits {
        512 => screen_and_match::<8>(&kind, queries, &targets, opts),
        1024 => screen_and_match::<16>(&kind, queries, &targets, opts),
        2048 => screen_and_match::<32>(&kind, queries, &targets, opts),
        4096 => screen_and_match::<64>(&kind, queries, &targets, opts),
        8192 => screen_and_match::<128>(&kind, queries, &targets, opts),
        16384 => screen_and_match::<256>(&kind, queries, &targets, opts),
        other => return Err(format!("fingerprint size {other} not supported")),
    };
    let Screened { build_seconds, match_seconds, phase_seconds, index_bytes, query_density, target_density, per, analysis } =
        screened;

    let posted = per.iter().map(|m| m.posted).sum();
    let candidates = per.iter().map(|m| m.candidates).sum();
    let matches = per.iter().map(|m| m.found.len()).sum();
    let mut hasher = Sha256::new();
    for (r, m) in per.iter().enumerate() {
        for &q in &m.found {
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
                per.iter().enumerate().flat_map(|(r, m)| m.found.iter().map(move |&q| (r as u32, q))).collect();
            let examples = |pairs: HashSet<&(u32, u32)>| {
                let mut v: Vec<(u32, u32)> = pairs.into_iter().copied().collect();
                v.sort_unstable();
                let n = v.len();
                let shown = v
                    .into_iter()
                    .take(15)
                    .map(|(r, q)| (r, q, smiles[r as usize].clone(), smarts[q as usize].clone()))
                    .collect();
                (n, shown)
            };
            let (missing, missing_examples) = examples(reference.difference(&ours).collect());
            let (extra, extra_examples) = examples(ours.difference(&reference).collect());
            Some(Comparison { reference_pairs: reference.len(), missing, extra, missing_examples, extra_examples })
        }
    };

    Ok(Report {
        queries: smarts.len(),
        reactants: smiles.len(),
        postings: opts.postings,
        threads: rayon::current_num_threads(),
        fp_kind: opts.fp_kind.clone(),
        fp_bits: opts.fp_bits,
        read_seconds,
        parse_queries_seconds,
        parse_reactants_seconds,
        build_seconds,
        match_seconds,
        phase_seconds,
        index_bytes,
        query_density,
        target_density,
        posted,
        candidates,
        matches,
        digest,
        comparison,
        analysis,
    })
}

/// Index build and timed matching runs for one fingerprint width.
struct Screened {
    build_seconds: f64,
    match_seconds: Vec<f64>,
    phase_seconds: [f64; 4],
    index_bytes: usize,
    query_density: f64,
    target_density: f64,
    per: Vec<Matched>,
    analysis: Option<String>,
}

fn screen_and_match<const W: usize>(kind: &FpKind, queries: Vec<Query>, targets: &[Target], opts: &Options) -> Screened {
    let start = Instant::now();
    let index = Index::<W>::build(kind, queries);
    let build_seconds = start.elapsed().as_secs_f64();

    let mut match_seconds = Vec::new();
    let mut phase_seconds = [0.0; 4];
    let mut result = None;
    for _ in 0..opts.runs {
        let start = Instant::now();
        let per: Vec<(f64, Matched)> = targets
            .par_iter()
            .map(|t| {
                let fp_start = Instant::now();
                let fp = target_fp::<W>(kind, t);
                let fp_seconds = fp_start.elapsed().as_secs_f64();
                (fp_seconds, index.match_one(t, &fp, opts.postings))
            })
            .collect();
        match_seconds.push(start.elapsed().as_secs_f64());
        for (fp_seconds, m) in &per {
            phase_seconds[0] += fp_seconds / opts.runs as f64;
            for (p, s) in phase_seconds[1..].iter_mut().zip(m.seconds) {
                *p += s / opts.runs as f64;
            }
        }
        result.get_or_insert_with(|| per.into_iter().map(|(_, m)| m).collect());
    }

    let per: Vec<Matched> = result.expect("at least one run");
    let analysis = opts.analyze.then(|| {
        let candidates: Vec<Vec<u32>> = targets
            .par_iter()
            .map(|t| {
                let fp = target_fp::<W>(kind, t);
                let mut out = Vec::new();
                index.subset_filter(&fp, &index.posting_filter(&fp, opts.postings), &mut out);
                out
            })
            .collect();
        let found: Vec<Vec<u32>> = per.iter().map(|m| m.found.clone()).collect();
        analysis::false_positives(kind, &index.queries, targets, &candidates, &found)
    });

    let target_ones: u64 = targets
        .par_iter()
        .map(|t| target_fp::<W>(kind, t).iter().map(|w| w.count_ones() as u64).sum::<u64>())
        .sum();
    Screened {
        build_seconds,
        match_seconds,
        phase_seconds,
        index_bytes: index.nbytes(),
        query_density: index.query_density(),
        target_density: target_ones as f64 / (targets.len().max(1) * 64 * W) as f64,
        per,
        analysis,
    }
}

pub fn print_report(r: &Report) {
    println!(
        "{} queries x {} reactants; {} postings; {} matching runs; {} threads",
        r.queries,
        r.reactants,
        r.postings,
        r.match_seconds.len(),
        r.threads
    );
    println!(
        "fingerprint     {} at {} bits; {:.1}% of bits set in queries, {:.1}% in reactants",
        r.fp_kind,
        r.fp_bits,
        100.0 * r.query_density,
        100.0 * r.target_density
    );
    println!("read files      {:>8.3} s", r.read_seconds);
    println!("parse SMARTS    {:>8.3} s", r.parse_queries_seconds);
    println!("parse SMILES    {:>8.3} s  (includes sanitization)", r.parse_reactants_seconds);
    println!(
        "build index     {:>8.3} s  ({:.1} MiB fingerprints + postings)",
        r.build_seconds,
        r.index_bytes as f64 / 1024f64.powi(2)
    );
    println!(
        "match median    {:>8.3} s  runs: {:?}",
        r.match_median(),
        r.match_seconds.iter().map(|t| (t * 1000.0).round() / 1000.0).collect::<Vec<_>>()
    );
    let [fp, postings, subset, exact] = r.phase_seconds;
    println!(
        "  of which      {fp:.3} s reactant fingerprints, {postings:.3} s postings, \
{subset:.3} s fingerprint check, {exact:.3} s exact match (summed over threads)"
    );
    println!("after postings  {:>8}", r.posted);
    println!("candidates      {:>8}  matches: {}", r.candidates, r.matches);
    println!("digest          {}", r.digest);
    if let Some(c) = &r.comparison {
        println!("reference       {} pairs; missing {}; extra {}", c.reference_pairs, c.missing, c.extra);
        for (label, pairs) in [("missing", &c.missing_examples), ("extra", &c.extra_examples)] {
            for (reactant, query, smiles, smarts) in pairs {
                println!("  {label} r{reactant} q{query}  {smiles}  {smarts}");
            }
        }
    }
    if let Some(a) = &r.analysis {
        print!("{a}");
    }
}

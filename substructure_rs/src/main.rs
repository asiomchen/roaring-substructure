//! Benchmark 09: substructure search without RDKit.
//!
//! Reads reaction-template SMARTS and reactant SMILES, builds the posting-list
//! index, and matches every reactant against every screened query in parallel.
//! The printed digest uses the same encoding as benchmarks 05a-07, so it can be
//! compared with RDKit's result directly.
//!
//! Author: Marcin Kowiel + Claude

mod fingerprint;
mod index;
mod matcher;
mod mol;
mod periodic;
mod rings;
mod smarts;
#[cfg(test)]
mod tests;

use fingerprint::target_fp;
use index::Index;
use matcher::Target;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    queries_file: PathBuf,
    reactants_file: PathBuf,
    postings: usize,
    runs: usize,
    threads: usize,
    reference: Option<PathBuf>,
    dump_atoms: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        queries_file: "data/substructure_sample.parquet".into(),
        reactants_file: "data/reactant_sample.csv".into(),
        postings: 128,
        runs: 3,
        threads: 0,
        reference: None,
        dump_atoms: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--queries-file" => args.queries_file = value()?.into(),
            "--reactants-file" => args.reactants_file = value()?.into(),
            "--postings" => args.postings = value()?.parse().map_err(|e| format!("--postings: {e}"))?,
            "--runs" => args.runs = value()?.parse().map_err(|e| format!("--runs: {e}"))?,
            "--threads" => args.threads = value()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--reference" => args.reference = Some(value()?.into()),
            "--dump-atoms" => args.dump_atoms = Some(value()?.into()),
            "-h" | "--help" => {
                println!(
                    "usage: substructure_rs [--queries-file F] [--reactants-file F] [--postings N] \
                     [--runs N] [--threads N] [--reference pairs.bin] [--dump-atoms out.txt]\n\
                     Files may be .parquet or .csv; queries use column `substructure`, reactants `smiles`."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if !(1..=fingerprint::FP_BITS).contains(&args.postings) {
        return Err(format!("--postings must be between 1 and {}", fingerprint::FP_BITS));
    }
    if args.runs < 1 {
        return Err("--runs must be at least 1".into());
    }
    Ok(args)
}

fn read_column(path: &Path, column: &str) -> Result<Vec<String>, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "parquet" {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use parquet::record::Field;
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let reader = SerializedFileReader::new(file).map_err(|e| e.to_string())?;
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
                    mol::BondType::Single => 2,
                    mol::BondType::Double => 4,
                    mol::BondType::Triple => 6,
                    mol::BondType::Quadruple => 8,
                    mol::BondType::Aromatic => 3,
                    mol::BondType::Dative => 2,
                };
                format!("{}-{}-{}", b.a.min(b.b), b.a.max(b.b), t)
            })
            .collect();
        writeln!(f, "{}|{}", atoms.join(";"), bonds.join(";"))?;
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().map_err(|e| e.to_string())?;
    }

    let start = Instant::now();
    let smarts = read_column(&args.queries_file, "substructure")?;
    let smiles = read_column(&args.reactants_file, "smiles")?;
    let read_s = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let queries: Vec<Result<smarts::Query, String>> = smarts.par_iter().map(|s| smarts::parse_smarts(s)).collect();
    let bad: Vec<(usize, &String)> = queries.iter().enumerate().filter_map(|(i, q)| q.as_ref().err().map(|e| (i, e))).collect();
    if !bad.is_empty() {
        for (i, e) in bad.iter().take(10) {
            eprintln!("query {i} {:?}: {e}", smarts[*i]);
        }
        return Err(format!("{} queries failed to parse", bad.len()));
    }
    let queries: Vec<smarts::Query> = queries.into_iter().map(Result::unwrap).collect();
    let parse_queries_s = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let mols: Vec<Result<mol::Mol, String>> = smiles.par_iter().map(|s| mol::parse_smiles(s)).collect();
    let bad: Vec<(usize, &String)> = mols.iter().enumerate().filter_map(|(i, m)| m.as_ref().err().map(|e| (i, e))).collect();
    if !bad.is_empty() {
        for (i, e) in bad.iter().take(10) {
            eprintln!("reactant {i} {:?}: {e}", smiles[*i]);
        }
        return Err(format!("{} reactants failed to parse", bad.len()));
    }
    let mols: Vec<mol::Mol> = mols.into_iter().map(Result::unwrap).collect();
    let targets: Vec<Target> = mols.par_iter().map(Target::new).collect();
    let parse_reactants_s = start.elapsed().as_secs_f64();
    if let Some(path) = &args.dump_atoms {
        dump_atoms(path, &mols).map_err(|e| e.to_string())?;
    }

    let start = Instant::now();
    let index = Index::build(queries);
    let build_s = start.elapsed().as_secs_f64();

    let mut times = Vec::new();
    let mut result = None;
    for _ in 0..args.runs {
        let start = Instant::now();
        let per: Vec<(usize, Vec<u32>)> = targets
            .par_iter()
            .map(|t| index.match_one(t, &target_fp(t), args.postings))
            .collect();
        times.push(start.elapsed().as_secs_f64());
        result.get_or_insert(per);
    }
    let per = result.unwrap();

    let candidates: usize = per.iter().map(|(c, _)| c).sum();
    let matches: usize = per.iter().map(|(_, m)| m.len()).sum();
    let mut hasher = Sha256::new();
    for (r, (_, found)) in per.iter().enumerate() {
        for &q in found {
            hasher.update((r as u32).to_le_bytes());
            hasher.update(q.to_le_bytes());
        }
    }
    let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();

    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    println!(
        "{} queries x {} reactants; {} postings; {} matching runs; {} threads",
        smarts.len(),
        smiles.len(),
        args.postings,
        args.runs,
        rayon::current_num_threads()
    );
    println!("read files      {read_s:>8.3} s");
    println!("parse SMARTS    {parse_queries_s:>8.3} s");
    println!("parse SMILES    {parse_reactants_s:>8.3} s  (includes sanitization)");
    println!("build index     {build_s:>8.3} s  ({:.1} MiB fingerprints + postings)", index.nbytes() as f64 / 1024f64.powi(2));
    println!("match median    {median:>8.3} s  runs: {:?}", times.iter().map(|t| (t * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    println!("candidates      {candidates:>8}  matches: {matches}");
    println!("digest          {digest}");

    if let Some(path) = &args.reference {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let reference: HashSet<(u32, u32)> = bytes
            .chunks_exact(8)
            .map(|c| (u32::from_le_bytes(c[..4].try_into().unwrap()), u32::from_le_bytes(c[4..].try_into().unwrap())))
            .collect();
        let ours: HashSet<(u32, u32)> =
            per.iter().enumerate().flat_map(|(r, (_, f))| f.iter().map(move |&q| (r as u32, q))).collect();
        let mut missing: Vec<_> = reference.difference(&ours).copied().collect();
        let mut extra: Vec<_> = ours.difference(&reference).copied().collect();
        missing.sort_unstable();
        extra.sort_unstable();
        println!("reference       {} pairs; missing {}; extra {}", reference.len(), missing.len(), extra.len());
        for (label, pairs) in [("missing", &missing), ("extra", &extra)] {
            for &(r, q) in pairs.iter().take(15) {
                println!("  {label} r{r} q{q}  {}  {}", smiles[r as usize], smarts[q as usize]);
            }
        }
    }
    Ok(())
}

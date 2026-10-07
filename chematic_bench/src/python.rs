//! Python bindings: `import chematic_bench`.
//!
//! `run(...)` takes the same settings as the command line and returns the
//! report as a dict; `cli(args)` runs the command line and returns its exit
//! code; `main()` is the `chematic_bench` console script installed by uv.
//!
//! Author: Marcin Kowiel + Claude

use crate::{Mode, Options, Report};
use pyo3::exceptions::{PySystemExit, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

fn report_dict<'py>(py: Python<'py>, r: &Report) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("mode", r.mode.name())?;
    d.set_item("queries", r.queries)?;
    d.set_item("reactants", r.reactants)?;
    d.set_item("threads", r.threads)?;
    d.set_item("parse_queries_seconds", r.parse_queries_seconds)?;
    d.set_item("parse_reactants_seconds", r.parse_reactants_seconds)?;
    d.set_item("screen_seconds", r.screen_seconds)?;
    d.set_item("pairs", r.pairs)?;
    d.set_item("match_seconds", r.match_seconds.clone())?;
    d.set_item("match_median_seconds", r.match_median())?;
    d.set_item("matches", r.matches)?;
    d.set_item("digest", &r.digest)?;
    if let Some((n, missing, extra)) = r.comparison {
        let cd = PyDict::new(py);
        cd.set_item("reference_pairs", n)?;
        cd.set_item("missing", missing)?;
        cd.set_item("extra", extra)?;
        d.set_item("comparison", cd)?;
    }
    Ok(d)
}

/// Run benchmark 10 and return its report as a dict.
///
/// `mode` is "naive" (chematic on every pair) or "screened" (chematic on the
/// pairs substructure_rs's screen passes). `threads=0` uses all cores.
#[pyfunction]
#[pyo3(signature = (
    mode = "screened".to_string(),
    queries_file = PathBuf::from("data/substructure_sample.parquet"),
    reactants_file = PathBuf::from("data/reactant_sample.csv"),
    runs = 3,
    threads = 0,
    fp_kind = "paths4+branches".to_string(),
    fp_bits = 2048,
    postings = 128,
    reference = None,
    print_report = false,
))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    mode: String,
    queries_file: PathBuf,
    reactants_file: PathBuf,
    runs: usize,
    threads: usize,
    fp_kind: String,
    fp_bits: usize,
    postings: usize,
    reference: Option<PathBuf>,
    print_report: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mode = Mode::parse(&mode).map_err(PyValueError::new_err)?;
    if runs < 1 {
        return Err(PyValueError::new_err("runs must be at least 1"));
    }
    let opts = Options { mode, queries_file, reactants_file, runs, threads, fp_kind, fp_bits, postings, reference };
    let report = py.detach(|| crate::run(&opts)).map_err(PyValueError::new_err)?;
    if print_report {
        crate::print_report(&report);
    }
    report_dict(py, &report)
}

/// Run the command line with `args` (without the program name); returns the exit code.
#[pyfunction]
fn cli(py: Python<'_>, args: Vec<String>) -> i32 {
    py.detach(|| crate::cli(args))
}

/// Console-script entry point: `chematic_bench [options]`.
#[pyfunction]
fn main(py: Python<'_>) -> PyResult<()> {
    let argv: Vec<String> = py.import("sys")?.getattr("argv")?.extract()?;
    let code = py.detach(|| crate::cli(argv.into_iter().skip(1)));
    if code != 0 {
        return Err(PySystemExit::new_err(code));
    }
    Ok(())
}

#[pymodule]
fn chematic_bench(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(cli, m)?)?;
    m.add_function(wrap_pyfunction!(main, m)?)?;
    m.add("USAGE", crate::USAGE)?;
    Ok(())
}

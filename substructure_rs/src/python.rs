//! Python bindings: `import substructure_rs`.
//!
//! `run(...)` takes the same settings as the command line and returns the
//! report as a dict; `cli(args)` runs the command line and returns its exit
//! code; `main()` is the `substructure_rs` console script installed by uv.
//!
//! Author: Marcin Kowiel + Claude

use crate::{Options, Report};
use pyo3::exceptions::{PySystemExit, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

fn report_dict<'py>(py: Python<'py>, r: &Report) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("queries", r.queries)?;
    d.set_item("reactants", r.reactants)?;
    d.set_item("postings", r.postings)?;
    d.set_item("threads", r.threads)?;
    d.set_item("read_seconds", r.read_seconds)?;
    d.set_item("parse_queries_seconds", r.parse_queries_seconds)?;
    d.set_item("parse_reactants_seconds", r.parse_reactants_seconds)?;
    d.set_item("build_seconds", r.build_seconds)?;
    d.set_item("match_seconds", r.match_seconds.clone())?;
    d.set_item("match_median_seconds", r.match_median())?;
    d.set_item("index_bytes", r.index_bytes)?;
    d.set_item("candidates", r.candidates)?;
    d.set_item("matches", r.matches)?;
    d.set_item("digest", &r.digest)?;
    if let Some(c) = &r.comparison {
        let cd = PyDict::new(py);
        cd.set_item("reference_pairs", c.reference_pairs)?;
        cd.set_item("missing", c.missing)?;
        cd.set_item("extra", c.extra)?;
        cd.set_item("missing_examples", c.missing_examples.clone())?;
        cd.set_item("extra_examples", c.extra_examples.clone())?;
        d.set_item("comparison", cd)?;
    }
    Ok(d)
}

/// Run benchmark 08 and return its report as a dict.
///
/// Paths are relative to the current directory. `threads=0` uses all cores.
/// With `print_report=True` the command-line summary is printed as well.
#[pyfunction]
#[pyo3(signature = (
    queries_file = PathBuf::from("data/substructure_sample.parquet"),
    reactants_file = PathBuf::from("data/reactant_sample.csv"),
    postings = 128,
    runs = 3,
    threads = 0,
    reference = None,
    dump_atoms = None,
    print_report = false,
))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    queries_file: PathBuf,
    reactants_file: PathBuf,
    postings: usize,
    runs: usize,
    threads: usize,
    reference: Option<PathBuf>,
    dump_atoms: Option<PathBuf>,
    print_report: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = Options { queries_file, reactants_file, postings, runs, threads, reference, dump_atoms };
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

/// Console-script entry point: `substructure_rs [options]`.
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
fn substructure_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(cli, m)?)?;
    m.add_function(wrap_pyfunction!(main, m)?)?;
    m.add("USAGE", crate::USAGE)?;
    Ok(())
}

//! Python bindings: `import substructure_rs`.
//!
//! `run(...)` takes the same settings as the command line and returns the
//! report as a dict; `cli(args)` runs the command line and returns its exit
//! code; `main()` is the `substructure_rs` console script installed by uv.
//!
//! For an index kept on disk (benchmark 11), each step is its own call, so
//! Python can time it: `read_queries`, `Index.build`, `Index.save`,
//! `Index.load`, `read_reactants` and `Index.match_many`. Each runs on the
//! global rayon pool (`RAYON_NUM_THREADS`) without holding the GIL.
//!
//! Author: Marcin Kowiel + Claude

use crate::store::{self, StoredIndex};
use crate::{read_column, Options, Report, Screen, Targets};
use pyo3::exceptions::{PySystemExit, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::sync::Mutex;

fn report_dict<'py>(py: Python<'py>, r: &Report) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("queries", r.queries)?;
    d.set_item("reactants", r.reactants)?;
    d.set_item("postings", r.postings)?;
    d.set_item("threads", r.threads)?;
    d.set_item("fp_kind", &r.fp_kind)?;
    d.set_item("fp_bits", r.fp_bits)?;
    d.set_item("order", &r.order)?;
    d.set_item("query_prep_seconds", r.query_prep_seconds)?;
    d.set_item("reactant_prep_seconds", r.reactant_prep_seconds)?;
    d.set_item("build_seconds", r.build_seconds)?;
    d.set_item("match_seconds", r.match_seconds.clone())?;
    d.set_item("match_median_seconds", r.match_median())?;
    d.set_item("phase_seconds", r.phase_seconds.to_vec())?;
    d.set_item("index_bytes", r.index_bytes)?;
    d.set_item("query_density", r.query_density)?;
    d.set_item("target_density", r.target_density)?;
    d.set_item("posted", r.posted)?;
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
/// `fp_kind` and `fp_bits` pick the screening fingerprint (benchmark 09);
/// `FP_KINDS` and `FP_SIZES` list the choices.
/// With `print_report=True` the command-line summary is printed as well.
#[pyfunction]
#[pyo3(signature = (
    queries_file = PathBuf::from("data/substructure_sample.parquet"),
    reactants_file = PathBuf::from("data/reactant_sample.csv"),
    postings = 128,
    runs = 3,
    threads = 0,
    fp_kind = crate::fingerprint::DEFAULT_FP_KIND.to_string(),
    fp_bits = crate::fingerprint::DEFAULT_FP_BITS,
    order = crate::matcher::DEFAULT_ORDER.to_string(),
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
    fp_kind: String,
    fp_bits: usize,
    order: String,
    reference: Option<PathBuf>,
    dump_atoms: Option<PathBuf>,
    print_report: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = Options { queries_file, reactants_file, postings, runs, threads, fp_kind, fp_bits, order, reference, dump_atoms, analyze: false, search_stats: false };
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

/// Parsed SMARTS queries; `Index.build` consumes them.
#[pyclass(frozen, module = "substructure_rs")]
struct Queries(Mutex<Option<store::Queries>>);

#[pymethods]
impl Queries {
    fn __len__(&self) -> usize {
        self.0.lock().unwrap().as_ref().map_or(0, store::Queries::len)
    }
}

/// Reactants parsed and sanitized by this crate.
#[pyclass(frozen, module = "substructure_rs")]
struct Reactants(Targets);

#[pymethods]
impl Reactants {
    fn __len__(&self) -> usize {
        self.0 .0.len()
    }
}

/// Read and parse the `substructure` column of a `.parquet` or `.csv` file.
#[pyfunction]
fn read_queries(py: Python<'_>, path: PathBuf) -> PyResult<Queries> {
    let queries = py
        .detach(|| store::Queries::parse(read_column(&path, "substructure")?))
        .map_err(PyValueError::new_err)?;
    Ok(Queries(Mutex::new(Some(queries))))
}

/// Read, parse and sanitize the `smiles` column of a `.parquet` or `.csv` file.
#[pyfunction]
fn read_reactants(py: Python<'_>, path: PathBuf) -> PyResult<Reactants> {
    let targets = py
        .detach(|| Screen::targets(&read_column(&path, "smiles")?))
        .map_err(PyValueError::new_err)?;
    Ok(Reactants(targets))
}

/// The posting-list index, built from queries or loaded from a file.
#[pyclass(frozen, module = "substructure_rs")]
struct Index(StoredIndex);

#[pymethods]
impl Index {
    /// Fingerprint the queries and build the postings; consumes `queries`.
    #[staticmethod]
    #[pyo3(signature = (
        queries,
        fp_kind = crate::fingerprint::DEFAULT_FP_KIND.to_string(),
        fp_bits = crate::fingerprint::DEFAULT_FP_BITS,
        order = crate::matcher::DEFAULT_ORDER.to_string(),
    ))]
    fn build(py: Python<'_>, queries: &Queries, fp_kind: String, fp_bits: usize, order: String) -> PyResult<Self> {
        let queries = queries.0.lock().unwrap().take().ok_or_else(|| PyValueError::new_err("queries already used"))?;
        let index = py.detach(|| StoredIndex::build(queries, &fp_kind, fp_bits, &order)).map_err(PyValueError::new_err)?;
        Ok(Index(index))
    }

    /// Read an index file written by `save`; parses its SMARTS again.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let index = py.detach(|| StoredIndex::load(&path)).map_err(PyValueError::new_err)?;
        Ok(Index(index))
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| self.0.save(&path)).map_err(PyValueError::new_err)
    }

    /// Per reactant, the matching query IDs, ascending; and the pairs left
    /// after the posting filter and after the full fingerprint check.
    #[pyo3(signature = (reactants, postings = 128))]
    fn match_many(&self, py: Python<'_>, reactants: &Reactants, postings: usize) -> PyResult<(Vec<Vec<u32>>, usize, usize)> {
        let per = py.detach(|| self.0.match_many(&reactants.0, postings)).map_err(PyValueError::new_err)?;
        let posted = per.iter().map(|f| f.posted).sum();
        let candidates = per.iter().map(|f| f.candidates).sum();
        Ok((per.into_iter().map(|f| f.found).collect(), posted, candidates))
    }

    fn __len__(&self) -> usize {
        self.0.queries()
    }

    #[getter]
    fn fp_kind(&self) -> &str {
        self.0.fp_kind()
    }

    #[getter]
    fn fp_bits(&self) -> usize {
        self.0.fp_bits()
    }

    #[getter]
    fn order(&self) -> &str {
        self.0.order()
    }

    /// Bytes of the fingerprint and posting arrays in memory.
    #[getter]
    fn nbytes(&self) -> usize {
        self.0.nbytes()
    }
}

#[pymodule]
fn substructure_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(read_queries, m)?)?;
    m.add_function(wrap_pyfunction!(read_reactants, m)?)?;
    m.add_class::<Index>()?;
    m.add_class::<Queries>()?;
    m.add_class::<Reactants>()?;
    m.add_function(wrap_pyfunction!(cli, m)?)?;
    m.add_function(wrap_pyfunction!(main, m)?)?;
    m.add("USAGE", crate::USAGE)?;
    m.add("FP_KINDS", crate::fingerprint::FP_KINDS.iter().map(|k| k.name).collect::<Vec<_>>())?;
    m.add("FP_SIZES", crate::fingerprint::FP_SIZES.to_vec())?;
    m.add("ORDERS", crate::matcher::ORDERS.iter().map(|(n, _)| *n).collect::<Vec<_>>())?;
    Ok(())
}

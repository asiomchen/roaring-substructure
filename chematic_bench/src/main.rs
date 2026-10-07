//! Command-line entry point; the benchmark itself lives in `lib.rs`.
//!
//! Author: Marcin Kowiel + Claude

fn main() {
    std::process::exit(chematic_bench::cli(std::env::args().skip(1)));
}

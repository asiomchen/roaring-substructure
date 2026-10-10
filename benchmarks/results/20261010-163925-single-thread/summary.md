# Single-thread benchmark run

- **Date:** 2026-10-10T16:39:25+02:00
- **Machine:** Apple M4, 10 cores, 24.0 GiB, Darwin 27.0 (arm64)
- **Software:** Python 3.13.14, RDKit 2026.3.6, NumPy 2.5.3, pyroaring 1.1.0, rustc 1.98.1 (48a229cea 2026-09-01)
- **Threads:** one thread of work per benchmark, run one after another; `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `MKL_NUM_THREADS=1`. No CPU pinning (macOS).
- **Data:** 50,000 SMARTS queries x 1,000 reactants = 50M pairs.
- **Match set:** every report returned the same matches, digest `72f6936d8ee5f05b`.

Times are seconds. Prep steps are outside the matching timer; matching is the median of the runs; end to end is prep plus median matching. "Exact matches" counts the pairs sent to the exact matcher.

| benchmark | variant | query prep | index prep | reactant prep | matching | end to end | matches | exact matches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01_naive |  | 0.7695 | - | 0.0872 | **76.9433** | 77.8001 | 27,723 | 50,000,000 |
| 02_pattern |  | 0.6626 | 2.6830 | 0.0990 | **22.7681** | 26.2127 | 27,723 | 350,393 |
| 03_roaring_approach |  | 0.6819 | 4.2408 | 0.0912 | **5.7220** | 10.7358 | 27,723 | 350,393 |
| 04_roaring_multiprocessing |  | 0.6418 | 4.2280 | 0.0855 | **6.0045** | 10.9599 | 27,723 | 350,393 |
| 05_postings | roaring pattern | 0.7230 | 4.8787 | 0.1110 | **1.5352** | 7.2479 | 27,723 | 350,393 |
| 05_postings | roaring paths | 0.6665 | 15.6294 | 0.1019 | **1.0226** | 17.4203 | 27,723 | 90,664 |
| 05_postings | numpy pattern | 0.6503 | 4.7817 | 0.0902 | **1.3591** | 6.8814 | 27,723 | 350,393 |
| 05_postings | numpy paths | 0.6659 | 15.6524 | 0.0915 | **1.0017** | 17.4115 | 27,723 | 90,664 |
| 06_rust_postings | numpy | 0.6497 | 4.7464 | 0.0906 | **1.2190** | 6.7057 | 27,723 | 350,393 |
| 06_rust_postings | rust | 0.6434 | 4.3745 | 0.0926 | **1.0218** | 6.1323 | 27,723 | 350,393 |
| 07_rust_postings_process | rust | 0.6565 | 4.3196 | 0.0932 | **1.0177** | 6.0870 | 27,723 | 350,393 |
| 08_rust_native |  | 0.1760 | 0.4266 | 0.0102 | **0.0566** | 0.6695 | 27,723 | 90,417 |
| 10_chematic | naive | 0.1167 | - | 0.0019 | **19.7089** | 19.8274 | 27,723 | 50,000,000 |
| 10_chematic | screened | 0.0940 | 0.5857 | 0.0122 | **0.1599** | 0.8518 | 27,723 | 82,461 |
| 11_index_on_disk | numpy-pattern | 0.7512 | 4.8399 | 0.3128 | **1.2221** | 7.1260 | 27,723 | 350,393 |
| 11_index_on_disk | numpy-paths | 0.6702 | 15.4583 | 0.3113 | **0.9707** | 17.4105 | 27,723 | 90,664 |
| 11_index_on_disk | rust-paths4-branches | 0.1774 | 0.4265 | 0.0105 | **0.0482** | 0.6626 | 27,723 | 82,461 |
| postgres | PostgreSQL 18 | 3.6088 | 4.6567 | 0.0014 | **8.5959** | 16.8628 | 27,723 | - |

Settings per report:

- 01_naive: defaults
- 02_pattern: defaults
- 03_roaring_approach: defaults
- 04_roaring_multiprocessing: processes 1
- 05_postings (roaring pattern): index roaring; fingerprint pattern; fp bits 2048; postings 128; runs 3
- 05_postings (roaring paths): index roaring; fingerprint paths; fp bits 4096; postings 128; runs 3
- 05_postings (numpy pattern): index numpy; fingerprint pattern; fp bits 2048; postings 128; runs 3
- 05_postings (numpy paths): index numpy; fingerprint paths; fp bits 4096; postings 128; runs 3
- 06_rust_postings (numpy): postings 128; runs 3
- 06_rust_postings (rust): postings 128; runs 3
- 07_rust_postings_process (rust): postings 128; runs 3; workers 1
- 08_rust_native: fp paths4 4096 bits; postings 128; order rare-root; threads 1; runs 3
- 10_chematic (naive): mode naive; threads 1; runs 3
- 10_chematic (screened): mode screened; threads 1; runs 3; screen paths4+branches 2048 bits, 128 postings
- 11_index_on_disk (numpy-pattern): language Python; postings 128; runs 3; threads 1; file cache dropped before search
- 11_index_on_disk (numpy-paths): language Python; postings 128; runs 3; threads 1; file cache dropped before search
- 11_index_on_disk (rust-paths4-branches): language Rust; postings 128; runs 3; threads 1; file cache dropped before search
- postgres (PostgreSQL 18): connections 1; sss_fp_size 2048

## Benchmark 09 sweep (50 configurations, fastest first)

| configuration | index prep | matching | exact matches |
| --- | ---: | ---: | ---: |
| paths4+branches 2048 p128 rare-root | 0.4304 | 0.0475 | 82,461 |
| paths4+branches+cycles 2048 p128 rare-root | 0.5035 | 0.0491 | 74,989 |
| paths4+branches 4096 p128 rare-root | 0.4297 | 0.0502 | 74,663 |
| paths4+branches+cycles 1024 p128 rare-root | 0.5107 | 0.0511 | 97,002 |
| paths4+cycles 2048 p128 rare-root | 0.4923 | 0.0514 | 87,373 |
| paths4 2048 p128 rare-root | 0.4074 | 0.0515 | 98,665 |
| paths4 1024 p128 rare-root | 0.4063 | 0.0518 | 123,539 |
| paths4+long6+branches+cycles 2048 p128 rare-root | 0.6175 | 0.0526 | 68,449 |
| paths4+branches 1024 p128 rare-root | 0.4222 | 0.0532 | 106,458 |
| paths6 2048 p128 rare-root | 0.5986 | 0.0542 | 83,543 |
| paths4+long6+branches+cycles 1024 p128 rare-root | 0.6143 | 0.0548 | 92,540 |
| paths4+branches+cycles 4096 p128 rare-root | 0.5123 | 0.0557 | 67,863 |
| paths4+cycles 1024 p128 rare-root | 0.4942 | 0.0575 | 110,323 |
| paths4+long6+branches+cycles 4096 p128 rare-root | 0.6266 | 0.0577 | 60,780 |
| paths6 1024 p128 rare-root | 0.5679 | 0.0584 | 129,625 |
| paths6 4096 p128 rare-root | 0.5900 | 0.0587 | 73,369 |
| paths4+long8+branches+cycles 2048 p128 rare-root | 0.7243 | 0.0599 | 70,929 |
| paths4 512 p128 rare-root | 0.4003 | 0.0617 | 206,487 |
| paths4+cycles 4096 p128 rare-root | 0.5088 | 0.0617 | 79,887 |
| paths4+long8+branches+cycles 1024 p128 rare-root | 0.7242 | 0.0627 | 102,808 |
| paths4+long8+branches+cycles 4096 p128 rare-root | 0.7383 | 0.0641 | 61,213 |
| paths4+branches 512 p128 rare-root | 0.4293 | 0.0646 | 194,355 |
| paths4+branches+cycles 8192 p128 rare-root | 0.5256 | 0.0648 | 64,836 |
| paths4+long6+branches+cycles 512 p128 rare-root | 0.6116 | 0.0654 | 212,281 |
| paths4-nocount 2048 p128 rare-root | 0.3889 | 0.0655 | 168,196 |
| paths4+branches+cycles 512 p128 rare-root | 0.5093 | 0.0661 | 181,710 |
| paths4-nocount 1024 p128 rare-root | 0.3887 | 0.0666 | 184,757 |
| paths4+cycles 512 p128 rare-root | 0.4876 | 0.0683 | 190,344 |
| paths4 4096 p128 rare-root | 0.4160 | 0.0685 | 90,417 |
| paths4-nocount 512 p128 rare-root | 0.3867 | 0.0687 | 223,145 |
| paths4-nocount 4096 p128 rare-root | 0.3928 | 0.0698 | 160,020 |
| paths4+long6+branches+cycles 8192 p128 rare-root | 0.6460 | 0.0711 | 57,629 |
| paths6 8192 p128 rare-root | 0.6084 | 0.0715 | 67,763 |
| paths4+branches 8192 p128 rare-root | 0.4572 | 0.0722 | 71,375 |
| paths4 8192 p128 rare-root | 0.4388 | 0.0734 | 86,729 |
| paths4+long8+branches+cycles 8192 p128 rare-root | 0.7532 | 0.0748 | 57,167 |
| paths4+cycles 8192 p128 rare-root | 0.5233 | 0.0751 | 76,546 |
| paths2 2048 p128 rare-root | 0.2709 | 0.0786 | 260,291 |
| paths2 1024 p128 rare-root | 0.2650 | 0.0830 | 318,754 |
| paths4+long8+branches+cycles 512 p128 rare-root | 0.7192 | 0.0843 | 335,367 |
| paths2 4096 p128 rare-root | 0.2766 | 0.0858 | 249,366 |
| paths2 512 p128 rare-root | 0.2639 | 0.0907 | 386,534 |
| paths4-nocount 8192 p128 rare-root | 0.4012 | 0.0911 | 155,320 |
| paths6 512 p128 rare-root | 0.5653 | 0.0963 | 527,390 |
| paths2 8192 p128 rare-root | 0.2886 | 0.1044 | 240,228 |
| atoms 512 p128 rare-root | 0.1840 | 0.3982 | 3,474,743 |
| atoms 1024 p128 rare-root | 0.1805 | 0.4114 | 3,352,688 |
| atoms 2048 p128 rare-root | 0.1854 | 0.5031 | 3,101,522 |
| atoms 4096 p128 rare-root | 0.1871 | 0.5192 | 3,031,375 |
| atoms 8192 p128 rare-root | 0.1927 | 0.7512 | 3,007,897 |

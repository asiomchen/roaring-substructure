# Single-thread benchmark run

- **Date:** 2026-10-11T15:36:57+02:00
- **Machine:** Apple M4, 10 cores, 24.0 GiB, Darwin 27.0 (arm64)
- **Software:** Python 3.13.14, RDKit 2026.3.6, NumPy 2.5.3, pyroaring 1.1.0, rustc 1.98.1 (48a229cea 2026-09-01)
- **Threads:** one thread of work per benchmark, run one after another; `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `MKL_NUM_THREADS=1`. No CPU pinning (macOS).
- **Data:** 50,000 SMARTS queries x 1,000 reactants = 50M pairs.
- **Match set:** every report returned the same matches, digest `72f6936d8ee5f05b`.

Times are seconds. Prep steps are outside the matching timer; matching is the median of the runs; end to end is prep plus median matching. "Exact matches" counts the pairs sent to the exact matcher.

| benchmark | variant | query prep | index prep | reactant prep | matching | end to end | matches | exact matches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01_naive |  | 0.7791 | - | 0.0866 | **76.3054** | 77.1711 | 27,723 | 50,000,000 |
| 02_pattern |  | 0.6653 | 2.7395 | 0.0876 | **22.4239** | 25.9163 | 27,723 | 350,393 |
| 03_roaring_approach |  | 0.6477 | 4.3252 | 0.0920 | **6.1549** | 11.2198 | 27,723 | 350,393 |
| 04_roaring_multiprocessing |  | 0.6570 | 4.2129 | 0.0864 | **6.1757** | 11.1321 | 27,723 | 350,393 |
| 05_postings | roaring pattern | 0.6569 | 5.0152 | 0.0910 | **1.6842** | 7.4473 | 27,723 | 350,393 |
| 05_postings | roaring paths | 0.6472 | 17.0042 | 0.0947 | **1.0418** | 18.7880 | 27,723 | 90,664 |
| 05_postings | numpy pattern | 0.6459 | 4.7862 | 0.0911 | **1.2846** | 6.8078 | 27,723 | 350,393 |
| 05_postings | numpy paths | 0.6726 | 16.8004 | 0.0960 | **1.0143** | 18.5834 | 27,723 | 90,664 |
| 06_rust_postings | numpy | 0.6871 | 4.8455 | 0.0907 | **1.2872** | 6.9106 | 27,723 | 350,393 |
| 06_rust_postings | rust | 0.6409 | 4.3659 | 0.0908 | **1.0649** | 6.1625 | 27,723 | 350,393 |
| 07_rust_postings_process | rust | 0.6898 | 4.3784 | 0.0963 | **1.0267** | 6.1912 | 27,723 | 350,393 |
| 08_rust_native |  | 0.1129 | 0.4287 | 0.0095 | **0.0712** | 0.6224 | 27,723 | 90,417 |
| 10_chematic | naive | 0.0983 | - | 0.0019 | **19.7397** | 19.8399 | 27,723 | 50,000,000 |
| 10_chematic | screened | 0.0936 | 0.5777 | 0.0120 | **0.1573** | 0.8406 | 27,723 | 82,461 |
| 11_index_on_disk | numpy-pattern | 0.7260 | 4.7472 | 0.3103 | **1.2435** | 7.0270 | 27,723 | 350,393 |
| 11_index_on_disk | numpy-paths | 0.6669 | 16.2941 | 0.3430 | **0.9766** | 18.2806 | 27,723 | 97,419 |
| 11_index_on_disk | rust-paths4-branches | 0.1164 | 0.4479 | 0.0104 | **0.0513** | 0.6259 | 27,723 | 82,461 |
| postgres | PostgreSQL 18 | 3.6075 | 4.6963 | 0.0013 | **7.9179** | 16.2230 | 27,723 | - |

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
| paths4+branches 2048 p128 rare-root | 0.4276 | 0.0497 | 82,461 |
| paths4+branches 1024 p128 rare-root | 0.4216 | 0.0501 | 106,458 |
| paths4+branches+cycles 2048 p128 rare-root | 0.5174 | 0.0503 | 74,989 |
| paths4+branches+cycles 1024 p128 rare-root | 0.5000 | 0.0515 | 97,002 |
| paths4+branches 4096 p128 rare-root | 0.4467 | 0.0519 | 74,663 |
| paths4+branches+cycles 4096 p128 rare-root | 0.5159 | 0.0530 | 67,863 |
| paths4+long6+branches+cycles 2048 p128 rare-root | 0.6272 | 0.0536 | 68,449 |
| paths4+cycles 2048 p128 rare-root | 0.4914 | 0.0538 | 87,373 |
| paths4+cycles 1024 p128 rare-root | 0.4989 | 0.0547 | 110,323 |
| paths6 2048 p128 rare-root | 0.5781 | 0.0550 | 83,543 |
| paths4+long6+branches+cycles 4096 p128 rare-root | 0.6427 | 0.0574 | 60,780 |
| paths4+cycles 4096 p128 rare-root | 0.4985 | 0.0576 | 79,887 |
| paths4+long6+branches+cycles 1024 p128 rare-root | 0.6263 | 0.0588 | 92,540 |
| paths4+long8+branches+cycles 2048 p128 rare-root | 0.7269 | 0.0590 | 70,929 |
| paths6 4096 p128 rare-root | 0.5876 | 0.0593 | 73,369 |
| paths4 2048 p128 rare-root | 0.4200 | 0.0594 | 98,665 |
| paths6 1024 p128 rare-root | 0.5857 | 0.0606 | 129,625 |
| paths4+branches 512 p128 rare-root | 0.4276 | 0.0607 | 194,355 |
| paths4 1024 p128 rare-root | 0.4170 | 0.0612 | 123,539 |
| paths4+branches+cycles 512 p128 rare-root | 0.5019 | 0.0614 | 181,710 |
| paths4+long8+branches+cycles 4096 p128 rare-root | 0.7388 | 0.0621 | 61,213 |
| paths4+long8+branches+cycles 1024 p128 rare-root | 0.7318 | 0.0624 | 102,808 |
| paths4+branches 8192 p128 rare-root | 0.4592 | 0.0639 | 71,375 |
| paths4+cycles 512 p128 rare-root | 0.4860 | 0.0653 | 190,344 |
| paths4+branches+cycles 8192 p128 rare-root | 0.5493 | 0.0659 | 64,836 |
| paths4-nocount 1024 p128 rare-root | 0.3891 | 0.0676 | 184,757 |
| paths4 4096 p128 rare-root | 0.4284 | 0.0679 | 90,417 |
| paths4+long6+branches+cycles 8192 p128 rare-root | 0.6604 | 0.0680 | 57,629 |
| paths4+cycles 8192 p128 rare-root | 0.5245 | 0.0681 | 76,546 |
| paths4 8192 p128 rare-root | 0.4524 | 0.0682 | 86,729 |
| paths6 8192 p128 rare-root | 0.6168 | 0.0708 | 67,763 |
| paths4-nocount 512 p128 rare-root | 0.3882 | 0.0711 | 223,145 |
| paths4+long6+branches+cycles 512 p128 rare-root | 0.6361 | 0.0711 | 212,281 |
| paths4+long8+branches+cycles 8192 p128 rare-root | 0.7760 | 0.0716 | 57,167 |
| paths4-nocount 2048 p128 rare-root | 0.3974 | 0.0719 | 168,196 |
| paths4 512 p128 rare-root | 0.4123 | 0.0726 | 206,487 |
| paths4-nocount 4096 p128 rare-root | 0.4026 | 0.0807 | 160,020 |
| paths4+long8+branches+cycles 512 p128 rare-root | 0.7182 | 0.0864 | 335,367 |
| paths2 2048 p128 rare-root | 0.2746 | 0.0884 | 260,291 |
| paths2 1024 p128 rare-root | 0.2730 | 0.0911 | 318,754 |
| paths6 512 p128 rare-root | 0.5827 | 0.0931 | 527,390 |
| paths4-nocount 8192 p128 rare-root | 0.4213 | 0.0991 | 155,320 |
| paths2 4096 p128 rare-root | 0.2879 | 0.1032 | 249,366 |
| paths2 8192 p128 rare-root | 0.3020 | 0.1112 | 240,228 |
| paths2 512 p128 rare-root | 0.2748 | 0.1113 | 386,534 |
| atoms 1024 p128 rare-root | 0.1916 | 0.5039 | 3,352,688 |
| atoms 512 p128 rare-root | 0.1986 | 0.5287 | 3,474,743 |
| atoms 2048 p128 rare-root | 0.1980 | 0.5576 | 3,101,522 |
| atoms 4096 p128 rare-root | 0.1972 | 0.7965 | 3,031,375 |
| atoms 8192 p128 rare-root | 0.1999 | 0.9336 | 3,007,897 |

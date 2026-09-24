#!/usr/bin/env bash
#
# Full pipeline, from the ORD dataset to the sample sets.
#
#   1. fetch_ord.py          download the uspto-grants config (~1.1 GB) and keep the
#                            reactions whose mapping holds up      -> data/ord_mapped.parquet
#   2. extract_templates.py  rxnutils templates, split to reactant-side substructures
#                            (the slow step: ~0.5 h at 9 workers)
#                                                      -> data/ord_reactant_substructures.parquet
#   3. list_reactants.py     the distinct reactant compounds  -> data/unique_reactants.csv
#   4. subset_compounds.py   random samples of both sets      -> data/reactant_sample.csv
#                                                                data/substructure_sample.parquet
#
# Steps 2 and 3 both read data/ord_mapped.parquet and are independent of each other. Both
# have to finish before step 4, which samples their outputs.
#
# Every step overwrites its own output, so re-running is safe but does not resume: step 2
# restarts from the beginning even if it ran before.
#
# Usage:
#   ./run_pipeline.sh
#   REACTANT_SAMPLE=5000 SUBSTRUCTURE_SAMPLE=5000 SEED=1 ./run_pipeline.sh
#
# To run only part of the flow, comment out the steps you do not need.

set -euo pipefail

# Work relative to the repository, not to wherever this was invoked from.
cd "$(dirname "$0")"

REACTANT_SAMPLE=${REACTANT_SAMPLE:-10000}
SUBSTRUCTURE_SAMPLE=${SUBSTRUCTURE_SAMPLE:-100000}
SEED=${SEED:-42}

step() {
    printf '\n=== %s ===\n\n' "$1"
}

step "1/4  fetch ORD uspto-grants"
uv run scripts/fetch_ord.py

step "2/4  extract reactant-side substructures (slow, parallel)"
uv run scripts/extract_templates.py

step "3/4  list the distinct reactants"
uv run scripts/list_reactants.py

step "4/4  draw ${REACTANT_SAMPLE} reactants and ${SUBSTRUCTURE_SAMPLE} substructures"
uv run scripts/subset_compounds.py \
    --reactants "${REACTANT_SAMPLE}" \
    --substructures "${SUBSTRUCTURE_SAMPLE}" \
    --seed "${SEED}"

printf '\n=== pipeline finished ===\n'

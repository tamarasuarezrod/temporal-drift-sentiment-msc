#!/usr/bin/env bash
#
# End-to-end reproduction of every table and figure in the paper. Needs a
# Python environment with requirements.txt installed, the LongEval data under
# data/, and the checkpoints unzipped under checkpoints/. See pipeline/README.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=".venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "No .venv found. Create one and install requirements first:"
  echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

echo "==> [01] check data"
"$PYTHON" "$HERE/01_check_data.py"

echo "==> [02] compute drift"
"$PYTHON" "$HERE/02_compute_drift.py"

echo "==> [03] compute polarity-mismatch flags"
"$PYTHON" "$HERE/03_compute_polarity_mismatch.py"

echo "==> [04] build per-model predictions (heavy)"
"$PYTHON" "$HERE/04_build_predictions.py"

echo "==> [05] build canonical parquet"
"$PYTHON" "$HERE/05_build_parquet.py"

echo "==> [06] regenerate tables"
"$PYTHON" "$HERE/06_regen_tables.py"

echo "==> [12] PretrainedTEA analysis (proposed method)"
"$PYTHON" "$HERE/12_evaluate_pretrainedtea.py"

echo "==> [13] T5 analysis"
"$PYTHON" "$HERE/13_evaluate_t5.py"

echo "==> [14] always-fail residual"
"$PYTHON" "$HERE/14_always_fail_residual.py"

# 08 runs after 12 so the best-strategy comparison can include PretrainedTEA.
echo "==> [08] threshold analysis"
"$PYTHON" "$HERE/08_threshold_analysis.py"

# 07 runs after 08 because the threshold figure consumes its CSVs.
echo "==> [07] regenerate figure"
"$PYTHON" "$HERE/07_regen_figures.py"

echo "==> [09] contamination check"
"$PYTHON" "$HERE/09_contamination_check.py"

echo "==> [10] paired bootstrap CIs"
"$PYTHON" "$HERE/10_paired_bootstrap.py"

echo "==> [11] practice-set selection"
"$PYTHON" "$HERE/11_practice_selection.py"

echo
echo "Done."
echo "  Tables : results/tables/"
echo "  Figures: results/figures/"
echo "  Parquet: results/per_tweet_analysis.parquet"

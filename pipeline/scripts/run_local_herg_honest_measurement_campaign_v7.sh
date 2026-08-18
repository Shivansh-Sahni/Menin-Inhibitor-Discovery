#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
CAMPAIGN="$SCRIPT_DIR/run_local_herg_honest_measurement_campaign_v7.py"
OUTPUT_ROOT="$REPO_ROOT/research/local_runs/herg_honest_measurement_campaign_v7_1"
LOG_PATH="$REPO_ROOT/research/local_runs/herg_honest_measurement_campaign_v7_1.log"

if (( $# != 0 )); then
  print -u2 "ERROR: this governed launcher accepts no arguments."
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$CAMPAIGN" ]]; then
  print -u2 "ERROR: V7 Python environment or campaign implementation is missing."
  exit 2
fi
if ! command -v caffeinate >/dev/null 2>&1; then
  print -u2 "ERROR: macOS caffeinate is required."
  exit 2
fi
POWER_STATUS="$(pmset -g batt 2>&1)"
if [[ "$POWER_STATUS" != *"AC Power"* ]]; then
  print -u2 "ERROR: connect the Mac to AC power before this multicore campaign."
  print -u2 "$POWER_STATUS"
  exit 2
fi
AVAILABLE_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
if (( AVAILABLE_KIB < 15 * 1024 * 1024 )); then
  print -u2 "ERROR: at least 15 GiB free disk space is required."
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

print "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting or resuming hERG V7"
print "Output: $OUTPUT_ROOT"
print "Workers: 6; accuracy and safety tracks; train-only nested scaffold evaluation"
print "Validation/test labels remain sealed. Identical command resumes completed units."

exec caffeinate -dimsu "$PYTHON_BIN" "$CAMPAIGN" run \
  --matrix-root "$REPO_ROOT/research/local_runs/herg_fundamental_optimization_v6" \
  --observations "$REPO_ROOT/research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/herg_training_observations.parquet" \
  --output-root "$OUTPUT_ROOT" \
  --workers 6 \
  > >(tee -a "$LOG_PATH") 2>&1

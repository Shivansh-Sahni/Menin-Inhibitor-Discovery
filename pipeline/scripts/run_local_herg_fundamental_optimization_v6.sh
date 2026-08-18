#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
OUTPUT_ROOT="$REPO_ROOT/research/local_runs/herg_fundamental_optimization_v6"
LOG_PATH="$REPO_ROOT/research/local_runs/herg_fundamental_optimization_v6.log"
PYTHON="$REPO_ROOT/.venv/bin/python"

if (( $# != 0 )); then
  print -u2 "ERROR: This governed launcher accepts no arguments."
  print -u2 "Run exactly: $0"
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  print -u2 "ERROR: Missing project Python: $PYTHON"
  exit 2
fi
if ! command -v caffeinate >/dev/null 2>&1; then
  print -u2 "ERROR: macOS caffeinate is required."
  exit 2
fi

FREE_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
if [[ -z "$FREE_KIB" ]] || (( FREE_KIB < 15 * 1024 * 1024 )); then
  print -u2 "ERROR: At least 15 GiB free disk space is required."
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

print "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting or resuming hERG V6 fundamental optimization" | tee -a "$LOG_PATH"
print "Output: $OUTPUT_ROOT" | tee -a "$LOG_PATH"
print "Six compute threads; identical launcher safely resumes completed units." | tee -a "$LOG_PATH"

cd "$REPO_ROOT"
exec caffeinate -dimsu "$PYTHON" \
  pipeline/scripts/run_local_herg_fundamental_optimization_v6.py run \
  --repo-root "$REPO_ROOT" \
  --feature-root "$REPO_ROOT/research/local_runs/herg_quantitative_24conformer_v4" \
  --base-root "$REPO_ROOT/research/local_runs/herg_discovery_campaign_v1" \
  --output-root "$OUTPUT_ROOT" \
  --workers 6 \
  > >(tee -a "$LOG_PATH") 2>&1

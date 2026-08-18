#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON="$REPO_ROOT/.venv/bin/python"
OUTPUT_ROOT="$REPO_ROOT/research/local_runs/herg_domain_mixture_campaign_v9"
LOG_PATH="$REPO_ROOT/research/local_runs/herg_domain_mixture_campaign_v9.log"

if [[ ! -x "$PYTHON" ]]; then
  print -u2 "ERROR: missing project Python: $PYTHON"
  exit 2
fi
if ! command -v caffeinate >/dev/null 2>&1; then
  print -u2 "ERROR: caffeinate is unavailable"
  exit 2
fi
if ! pmset -g batt | grep -q "AC Power"; then
  print -u2 "ERROR: connect the Mac to AC power before starting V9"
  pmset -g batt >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=20260817

print "[$(date -u +%FT%TZ)] Starting/resuming hERG V9"
print "Output: $OUTPUT_ROOT"
print "Six model threads; nested BLAS threads capped at one"
print "Validation/test labels remain sealed; identical command resumes at validated units"

exec caffeinate -dimsu "$PYTHON" \
  "$REPO_ROOT/pipeline/scripts/run_local_herg_domain_mixture_campaign_v9.py" run \
  --repo-root "$REPO_ROOT" \
  --v8-root "$REPO_ROOT/research/local_runs/herg_feature_lattice_campaign_v8" \
  --v81-root "$REPO_ROOT/research/local_runs/herg_feature_lattice_analysis_v81" \
  --mmp-root "$REPO_ROOT/research/data/platform/processed/herg_hierarchy/v1_5_mmp_analysis" \
  --output-root "$OUTPUT_ROOT" \
  --workers 6 \
  --bootstrap-replicates 10000 \
  > >(tee -a "$LOG_PATH") 2>&1

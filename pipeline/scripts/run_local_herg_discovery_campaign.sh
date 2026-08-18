#!/bin/zsh
# Launch or resume the bounded, train-only local hERG discovery campaign.

set -euo pipefail

readonly SCRIPT_DIR="${0:A:h}"
readonly REPO_ROOT="${SCRIPT_DIR:h:h}"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly CAMPAIGN_SCRIPT="${SCRIPT_DIR}/run_local_herg_discovery_campaign.py"
readonly WORKER_SCRIPT="${SCRIPT_DIR}/run_local_herg_discovery_worker.py"
readonly MATRIX_ROOT="${REPO_ROOT}/research/local_runs/herg_quantitative_feature_matrix_v1"
readonly OBSERVATIONS="${REPO_ROOT}/research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/herg_training_observations.parquet"
readonly OUTPUT_ROOT="${REPO_ROOT}/research/local_runs/herg_discovery_campaign_v1"
readonly LAUNCH_LOG="${OUTPUT_ROOT}/launcher.log"
readonly WORKERS=6
readonly MINIMUM_FREE_GIB=15

cd -- "${REPO_ROOT}"

if (( $# != 0 )); then
  print -u2 -- "ERROR: This reproducible launcher accepts no command-line overrides."
  print -u2 -- "Run it exactly as: ./pipeline/scripts/run_local_herg_discovery_campaign.sh"
  exit 2
fi

if ! command -v caffeinate >/dev/null 2>&1; then
  print -u2 -- "ERROR: macOS caffeinate is required but was not found."
  exit 2
fi
if ! command -v pmset >/dev/null 2>&1; then
  print -u2 -- "ERROR: pmset is required to verify AC power."
  exit 2
fi

power_status="$(pmset -g batt)"
if [[ "${power_status}" != *"AC Power"* ]]; then
  print -u2 -- "ERROR: Connect the Mac to AC power before starting this campaign."
  print -u2 -- "Power status: ${power_status}"
  exit 2
fi

required_files=(
  "${PYTHON_BIN}"
  "${CAMPAIGN_SCRIPT}"
  "${WORKER_SCRIPT}"
  "${MATRIX_ROOT}/combined_feature_matrix.parquet"
  "${MATRIX_ROOT}/manifest.json"
  "${MATRIX_ROOT}/validation.json"
  "${OBSERVATIONS}"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -s "${required_file}" ]]; then
    print -u2 -- "ERROR: Required nonempty input is missing: ${required_file}"
    exit 2
  fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
  print -u2 -- "ERROR: Project Python is not executable: ${PYTHON_BIN}"
  exit 2
fi

available_kib="$(df -Pk "${REPO_ROOT}" | awk 'NR == 2 {print $4}')"
if [[ "${available_kib}" != <-> ]]; then
  print -u2 -- "ERROR: Could not determine available disk space."
  exit 2
fi
readonly minimum_free_kib=$(( MINIMUM_FREE_GIB * 1024 * 1024 ))
if (( available_kib < minimum_free_kib )); then
  available_gib="$(( available_kib / 1024 / 1024 ))"
  print -u2 -- "ERROR: ${available_gib} GiB free; at least ${MINIMUM_FREE_GIB} GiB is required."
  exit 2
fi

mkdir -p -- "${OUTPUT_ROOT}"

# Six independent model workers may use native threads.  Keep nested BLAS
# runtimes at one thread to prevent 6x oversubscription and memory pressure.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${WORKERS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS="${WORKERS}"
export POLARS_MAX_THREADS="${WORKERS}"
export TOKENIZERS_PARALLELISM=false

print -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting or resuming hERG campaign" | tee -a "${LAUNCH_LOG}"
print -- "Output: ${OUTPUT_ROOT}" | tee -a "${LAUNCH_LOG}"
print -- "Workers: ${WORKERS}; target: 24 h; hard stop: 30 h" | tee -a "${LAUNCH_LOG}"

# Process substitution gives the terminal and log identical output without a
# pipeline masking the campaign's exit status.  exec leaves caffeinate/Python's
# real status as the status of this launcher.
exec caffeinate -dimsu \
  "${PYTHON_BIN}" "${CAMPAIGN_SCRIPT}" start \
  --repo-root "${REPO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --worker "${WORKER_SCRIPT}" \
  --python "${PYTHON_BIN}" \
  --matrix-root "${MATRIX_ROOT}" \
  --observations "${OBSERVATIONS}" \
  --workers "${WORKERS}" \
  --target-hours 24 \
  --hard-hours 30 \
  --finalization-reserve-minutes 60 \
  --minimum-free-disk-gib "${MINIMUM_FREE_GIB}" \
  --minimum-available-memory-gib 1.5 \
  > >(tee -a "${LAUNCH_LOG}") 2>&1

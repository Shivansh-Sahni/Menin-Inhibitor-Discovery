#!/bin/zsh
# Launch or resume the extended train-only hERG campaign without touching v2.

set -euo pipefail

readonly SCRIPT_DIR="${0:A:h}"
readonly REPO_ROOT="${SCRIPT_DIR:h:h}"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly CAMPAIGN_SCRIPT="${SCRIPT_DIR}/run_local_herg_discovery_campaign_v3.py"
readonly WORKER_SCRIPT="${SCRIPT_DIR}/run_local_herg_discovery_worker_v3.py"
readonly BASE_CAMPAIGN="${REPO_ROOT}/research/local_runs/herg_discovery_campaign_v1"
readonly BROAD_SURFACE="${REPO_ROOT}/research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/confirmed_wt_fixed_dose_structure_labels.parquet"
readonly OUTPUT_ROOT="${REPO_ROOT}/research/local_runs/herg_discovery_campaign_v3"
readonly LAUNCH_LOG="${OUTPUT_ROOT}/launcher.log"
readonly WORKERS=6
readonly MINIMUM_FREE_GIB=15

cd -- "${REPO_ROOT}"

if (( $# != 0 )); then
  print -u2 -- "ERROR: This governed launcher accepts no overrides."
  print -u2 -- "Run exactly: ./pipeline/scripts/run_local_herg_discovery_campaign_v3.sh"
  exit 2
fi

for command_name in caffeinate pmset; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    print -u2 -- "ERROR: Required macOS command not found: ${command_name}"
    exit 2
  fi
done
power_status="$(pmset -g batt)"
if [[ "${power_status}" != *"AC Power"* ]]; then
  print -u2 -- "ERROR: Connect the Mac to AC power before starting."
  print -u2 -- "Power status: ${power_status}"
  exit 2
fi

required_files=(
  "${PYTHON_BIN}"
  "${CAMPAIGN_SCRIPT}"
  "${WORKER_SCRIPT}"
  "${BASE_CAMPAIGN}/DONE.json"
  "${BASE_CAMPAIGN}/final_summary.json"
  "${BASE_CAMPAIGN}/prepared/exact_train_cache.parquet"
  "${BASE_CAMPAIGN}/prepared/nested_scaffold_splits.parquet"
  "${BASE_CAMPAIGN}/prepared/feature_registry.json"
  "${BASE_CAMPAIGN}/analysis/validation.json"
  "${BROAD_SURFACE}"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -s "${required_file}" ]]; then
    print -u2 -- "ERROR: Required nonempty input missing: ${required_file}"
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
  print -u2 -- "ERROR: $(( available_kib / 1024 / 1024 )) GiB free; ${MINIMUM_FREE_GIB} GiB required."
  exit 2
fi

mkdir -p -- "${OUTPUT_ROOT}"

# Six job workers are available to model libraries. Nested BLAS runtimes stay
# single-threaded so native libraries cannot multiply this into oversubscription.
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

print -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting or resuming hERG v3" | tee -a "${LAUNCH_LOG}"
print -- "Output: ${OUTPUT_ROOT}" | tee -a "${LAUNCH_LOG}"
print -- "Workers: ${WORKERS}; nominal schedule: 13.1 h; hard active-time ceiling: 30 h" | tee -a "${LAUNCH_LOG}"
print -- "It finishes early when the prespecified work completes; no time is padded." | tee -a "${LAUNCH_LOG}"
print -- "Exact pIC50 and broad fixed-dose binary tasks remain separate." | tee -a "${LAUNCH_LOG}"

exec caffeinate -dimsu \
  "${PYTHON_BIN}" "${CAMPAIGN_SCRIPT}" start \
  --repo-root "${REPO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --base-campaign-root "${BASE_CAMPAIGN}" \
  --broad-surface "${BROAD_SURFACE}" \
  --worker "${WORKER_SCRIPT}" \
  --python "${PYTHON_BIN}" \
  --workers "${WORKERS}" \
  --target-hours 24 \
  --hard-hours 30 \
  --finalization-reserve-minutes 60 \
  --minimum-free-disk-gib "${MINIMUM_FREE_GIB}" \
  --minimum-available-memory-gib 1.5 \
  > >(tee -a "${LAUNCH_LOG}") 2>&1

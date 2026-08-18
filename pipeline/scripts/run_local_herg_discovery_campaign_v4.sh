#!/bin/zsh
# Launch or exactly resume the governed train-only hERG v4 campaign.

set -euo pipefail

readonly SCRIPT_DIR="${0:A:h}"
readonly REPO_ROOT="${SCRIPT_DIR:h:h}"
readonly PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
readonly CAMPAIGN_SCRIPT="${SCRIPT_DIR}/run_local_herg_discovery_campaign_v4.py"
readonly WORKER_SCRIPT="${SCRIPT_DIR}/run_local_herg_discovery_worker_v4.py"
readonly CONFIG="${REPO_ROOT}/pipeline/config/herg_discovery_campaign_v4.yaml"
readonly PROTOCOL="${REPO_ROOT}/research/reports/platform/herg_paper/v4/HERG_V4_PREREGISTERED_PROTOCOL.md"
readonly BASE_V2="${REPO_ROOT}/research/local_runs/herg_discovery_campaign_v1"
readonly BASE_V3="${REPO_ROOT}/research/local_runs/herg_discovery_campaign_v3"
readonly OUTPUT_ROOT="${REPO_ROOT}/research/local_runs/herg_discovery_campaign_v4"
readonly LAUNCH_LOG="${OUTPUT_ROOT}/launcher.log"
readonly WORKERS=6

cd -- "${REPO_ROOT}"

if (( $# != 0 )); then
  print -u2 -- "ERROR: This governed launcher accepts no overrides."
  print -u2 -- "Run exactly: ./pipeline/scripts/run_local_herg_discovery_campaign_v4.sh"
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
  "${CONFIG}"
  "${PROTOCOL}"
  "${BASE_V2}/DONE.json"
  "${BASE_V2}/final_summary.json"
  "${BASE_V3}/DONE.json"
  "${BASE_V3}/final_summary.json"
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

mkdir -p -- "${OUTPUT_ROOT}"

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
export HERG_CAMPAIGN_FORBID_REPOSITORY_VALIDATION_TEST=1

"${PYTHON_BIN}" "${CAMPAIGN_SCRIPT}" validate-config \
  --repo-root "${REPO_ROOT}" \
  --config "${CONFIG}" \
  --protocol "${PROTOCOL}" \
  --base-v2-root "${BASE_V2}" \
  --base-v3-root "${BASE_V3}" \
  --worker "${WORKER_SCRIPT}" \
  --python "${PYTHON_BIN}" \
  --output-root "${OUTPUT_ROOT}"

print -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting or resuming hERG v4" | tee -a "${LAUNCH_LOG}"
print -- "Output: ${OUTPUT_ROOT}" | tee -a "${LAUNCH_LOG}"
print -- "Workers: ${WORKERS}; target 13.5 active hours, hard ceiling 15 hours, 60-minute reserve." | tee -a "${LAUNCH_LOG}"
print -- "The campaign finishes when prespecified science completes; runtime is never padded." | tee -a "${LAUNCH_LOG}"
print -- "This launch computes models, nested scaffold-heldout outputs, and fresh conformers; analysis is deferred." | tee -a "${LAUNCH_LOG}"
print -- "Exact pIC50 and broad fixed-dose binary endpoints remain separate; validation/test labels stay sealed." | tee -a "${LAUNCH_LOG}"

exec caffeinate -dimsu \
  "${PYTHON_BIN}" "${CAMPAIGN_SCRIPT}" start \
  --repo-root "${REPO_ROOT}" \
  --config "${CONFIG}" \
  --protocol "${PROTOCOL}" \
  --base-v2-root "${BASE_V2}" \
  --base-v3-root "${BASE_V3}" \
  --worker "${WORKER_SCRIPT}" \
  --python "${PYTHON_BIN}" \
  --output-root "${OUTPUT_ROOT}" \
  > >(tee -a "${LAUNCH_LOG}") 2>&1

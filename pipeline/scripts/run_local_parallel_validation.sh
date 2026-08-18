#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/pipeline/src"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/research/local_runs/${RUN_TAG}_parallel}"
export PYTHONDONTWRITEBYTECODE=1

# Prevent each independent process from internally oversubscribing the laptop.
# Parallelism is supplied by the job groups below.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/herg"
caffeinate -dimsu -w $$ &

typeset -a JOB_PIDS
typeset -a JOB_LABELS

start_job() {
  local label="$1"
  shift
  (
    echo "START $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$@"
    echo "PASS $label $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ) > >(tee "$RUN_ROOT/logs/${label}.log") 2>&1 &
  JOB_PIDS+=("$!")
  JOB_LABELS+=("$label")
}

wait_for_jobs() {
  local failed=0
  local index
  for index in {1..${#JOB_PIDS[@]}}; do
    if wait "${JOB_PIDS[$index]}"; then
      echo "COMPLETED ${JOB_LABELS[$index]}"
    else
      echo "FAILED ${JOB_LABELS[$index]} -- inspect $RUN_ROOT/logs/${JOB_LABELS[$index]}.log"
      failed=1
    fi
  done
  JOB_PIDS=()
  JOB_LABELS=()
  if (( failed != 0 )); then
    return 1
  fi
}

validate_registry() {
  .venv/bin/python -m menin_discovery.platform_training_surface_registry \
    --repo-root "$REPO_ROOT" \
    --output "$REPO_ROOT/research/data/platform/processed/training_surfaces/v1_0_platform_registry" \
    --validate-only
}

validate_affinity() {
  .venv/bin/python -m menin_discovery.platform_affinity_training_surfaces \
    --output-root "$REPO_ROOT/research/data/platform/processed/affinity_training/v1_0_chembl37_bindingdb202608" \
    --report-mirror-path "$REPO_ROOT/research/reports/platform/affinity_training/v1_0_chembl37_bindingdb202608/AFFINITY_TRAINING_SURFACES.md" \
    --validate-only
}

validate_pk_adme() {
  .venv/bin/python - <<'PY'
from pathlib import Path

from menin_discovery.platform_pk_adme_trainable_surfaces import validate_release

root = Path.cwd().resolve()
print(
    validate_release(
        root,
        root / "research/data/platform/processed/pk_adme/v1_0_trainable_surfaces",
        root / "research/reports/platform/pk_adme_trainable_surfaces_v1/PK_ADME_TRAINABLE_SURFACES.md",
    )
)
PY
}

validate_herg() {
  .venv/bin/python -m menin_discovery.platform_herg_training_surfaces \
    --output-root "$REPO_ROOT/research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces" \
    --report-root "$REPO_ROOT/research/reports/platform/herg_paper/training_surfaces" \
    --validate-only
}

validate_adjacent() {
  .venv/bin/python -m menin_discovery.platform_adjacent_training_surfaces \
    --output-root "$REPO_ROOT/research/data/platform/processed/multimodal/v1_0_adjacent_training_surfaces" \
    --validate-only
}

validate_hpc_preflight() {
  .venv/bin/python -m menin_discovery.platform_herg_hpc_preflight \
    --output-root "$REPO_ROOT/research/data/platform/processed/herg_hierarchy/v1_5_hpc_preflight" \
    --validate-only
}

run_broad_baseline() {
  .venv/bin/python -m menin_discovery.platform_herg_paper_baseline \
    --split "$REPO_ROOT/research/data/platform/processed/herg_hierarchy/v1_model_ready/structure_consensus_binary_scaffold_split.parquet" \
    --output-root "$RUN_ROOT/herg/broad_cpu_baseline"
}

run_quality_baselines() {
  .venv/bin/python -m menin_discovery.platform_herg_quality_baselines build \
    --q1 "$REPO_ROOT/research/data/platform/processed/herg_hierarchy/v1_2_quality_tasks/q1_quantitative_pic50.parquet" \
    --q2 "$REPO_ROOT/research/data/platform/processed/herg_hierarchy/v1_2_quality_tasks/q2_functional_assay_aware.parquet" \
    --output-root "$RUN_ROOT/herg/quality_baselines"
}

run_pipeline_tests() {
  .venv/bin/python -m pytest -q -p no:cacheprovider pipeline/tests
}

run_editor_tests() {
  .venv/bin/python -m pytest -q -p no:cacheprovider \
    -c packages/menin-edit/pyproject.toml packages/menin-edit/tests
}

run_ruff() {
  .venv/bin/python -m ruff check \
    pipeline/src pipeline/scripts pipeline/tests \
    packages/menin-edit/src packages/menin-edit/tests
}

run_format_check() {
  .venv/bin/python -m ruff format --check \
    pipeline/src pipeline/scripts pipeline/tests \
    packages/menin-edit/src packages/menin-edit/tests
}

run_mypy() {
  .venv/bin/python -m mypy \
    pipeline/src/menin_discovery packages/menin-edit/src/menin_edit
}

echo "Parallel local run: $RUN_ROOT"
.venv/bin/python -VV | tee "$RUN_ROOT/logs/python_version.log"
.venv/bin/python -m pip freeze > "$RUN_ROOT/logs/python_packages.txt"
df -h . | tee "$RUN_ROOT/logs/storage_before.log"
.venv/bin/python -m pip check | tee "$RUN_ROOT/logs/pip_check.log"

echo "PHASE 1: six independent release validators"
start_job platform_registry validate_registry
start_job affinity_validation validate_affinity
start_job pk_adme_validation validate_pk_adme
start_job herg_validation validate_herg
start_job adjacent_validation validate_adjacent
start_job hpc_preflight validate_hpc_preflight
wait_for_jobs

echo "PHASE 2: baselines, tests, and static checks in parallel"
start_job broad_herg_baseline run_broad_baseline
start_job quality_herg_baselines run_quality_baselines
start_job pipeline_tests run_pipeline_tests
start_job editor_tests run_editor_tests
start_job ruff run_ruff
start_job format_check run_format_check
start_job mypy run_mypy
wait_for_jobs

.venv/bin/python -m menin_discovery.platform_herg_paper_baseline \
  --split "$REPO_ROOT/research/data/platform/processed/herg_hierarchy/v1_model_ready/structure_consensus_binary_scaffold_split.parquet" \
  --output-root "$RUN_ROOT/herg/broad_cpu_baseline" \
  --validate-only

.venv/bin/python -m menin_discovery.platform_herg_quality_baselines verify \
  --output-root "$RUN_ROOT/herg/quality_baselines"

.venv/bin/python - <<'PY' | tee "$RUN_ROOT/logs/baseline_reproducibility.log"
import os
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

root = Path.cwd()
local = Path(os.environ["RUN_ROOT"])
comparisons = [
    (
        root / "research/data/platform/processed/herg_paper/cpu_baseline_v1/baseline_metrics.parquet",
        local / "herg/broad_cpu_baseline/baseline_metrics.parquet",
    ),
    (
        root / "research/data/platform/processed/herg_paper/cpu_baseline_v1/locked_evaluation_predictions.parquet",
        local / "herg/broad_cpu_baseline/locked_evaluation_predictions.parquet",
    ),
    (
        root / "research/data/platform/processed/herg_paper/quality_baselines_v1/q1_metrics.parquet",
        local / "herg/quality_baselines/q1_metrics.parquet",
    ),
    (
        root / "research/data/platform/processed/herg_paper/quality_baselines_v1/q2_metrics.parquet",
        local / "herg/quality_baselines/q2_metrics.parquet",
    ),
    (
        root / "research/data/platform/processed/herg_paper/quality_baselines_v1/q1_validation_test_predictions.parquet",
        local / "herg/quality_baselines/q1_validation_test_predictions.parquet",
    ),
    (
        root / "research/data/platform/processed/herg_paper/quality_baselines_v1/q2_validation_test_predictions.parquet",
        local / "herg/quality_baselines/q2_validation_test_predictions.parquet",
    ),
]

for canonical_path, reproduced_path in comparisons:
    canonical = pd.read_parquet(canonical_path)
    reproduced = pd.read_parquet(reproduced_path)
    assert_frame_equal(
        canonical,
        reproduced,
        check_dtype=True,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    print(f"REPRODUCED: {canonical_path.name} ({len(canonical):,} rows)")
print("ALL BASELINE METRICS AND PREDICTIONS REPRODUCED")
PY

git diff --check | tee "$RUN_ROOT/logs/git_diff_check.log"
find "$RUN_ROOT" -type f -print0 | sort -z | xargs -0 shasum -a 256 \
  > "$RUN_ROOT/SHA256SUMS.txt"
du -sh "$RUN_ROOT" | tee "$RUN_ROOT/logs/output_size.log"
df -h . | tee "$RUN_ROOT/logs/storage_after.log"

echo "PARALLEL LOCAL RUN COMPLETE"
echo "Results: $RUN_ROOT"

#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON="$REPO_ROOT/.venv/bin/python"
V8_ROOT="$REPO_ROOT/research/local_runs/herg_feature_lattice_campaign_v8"
V9_ROOT="$REPO_ROOT/research/local_runs/herg_domain_mixture_campaign_v9"
OUTPUT_ROOT="$REPO_ROOT/research/local_runs/herg_heavy_compound_sidecar_v9"
LOG_PATH="$REPO_ROOT/research/local_runs/herg_heavy_compound_sidecar_v9.log"

mkdir -p "$OUTPUT_ROOT"
print "[$(date -u +%FT%TZ)] Heavy-compound baseline frozen; waiting for completed V9" | tee -a "$LOG_PATH"
while [[ ! -f "$V9_ROOT/validation.json" ]]; do
  sleep 30
done
print "[$(date -u +%FT%TZ)] V9 complete; consolidating heavy-compound results" | tee -a "$LOG_PATH"
exec "$PYTHON" "$SCRIPT_DIR/analyze_local_herg_heavy_compounds_sidecar_v9.py" \
  --v8-root "$V8_ROOT" \
  --v9-root "$V9_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --bootstrap-replicates 10000 \
  > >(tee -a "$LOG_PATH") 2>&1

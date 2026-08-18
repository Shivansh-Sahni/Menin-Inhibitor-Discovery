#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
PYTHON="$REPO_ROOT/.venv/bin/python"
RUNNER="$SCRIPT_DIR/run_local_herg_feature_lattice_campaign_v8.py"
OUTPUT="$REPO_ROOT/research/local_runs/herg_feature_lattice_campaign_v8"
LOG="$REPO_ROOT/research/local_runs/herg_feature_lattice_campaign_v8.log"

if [[ $# -ne 0 ]]; then
  print -u2 "This launcher accepts no arguments. Rerun it identically to resume."
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$RUNNER" ]]; then
  print -u2 "Missing V8 runner or project virtual environment."
  exit 2
fi
if ! pmset -g batt | grep -q "AC Power"; then
  print -u2 "Connect the Mac to AC power before starting the 48-hour campaign."
  pmset -g batt >&2
  exit 2
fi
mkdir -p "$OUTPUT"
cd "$REPO_ROOT"
print "Starting/resuming V8: 2,048 coalitions x 5 scaffold contexts, six compute threads."
print "Hard active-time ceiling: 48 h; identical command resumes at validated unit boundaries."
exec caffeinate -dimsu "$PYTHON" "$RUNNER" run \
  --repo-root "$REPO_ROOT" \
  --matrix-root "$REPO_ROOT/research/local_runs/herg_fundamental_optimization_v6" \
  --base-root "$REPO_ROOT/research/local_runs/herg_discovery_campaign_v1" \
  --v7-root "$REPO_ROOT/research/local_runs/herg_honest_measurement_campaign_v7_1" \
  --output-root "$OUTPUT" \
  --workers 6 \
  > >(tee -a "$LOG") 2>&1

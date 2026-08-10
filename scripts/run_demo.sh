#!/usr/bin/env bash
# Start the API, drive it, print what comes back, then stop it.
#
#   bash scripts/run_demo.sh
#
# Assumes models exist. If they do not, run `make train` (or
# `bash scripts/train_all.sh`) first — this script says so and stops rather
# than silently demonstrating nothing.

set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
BASE="http://localhost:${PORT}"
ARTIFACTS="${TRIAGE_ARTIFACTS_DIR:-artifacts}"

if [ ! -d "$ARTIFACTS/models" ] || [ -z "$(ls -A "$ARTIFACTS/models" 2>/dev/null)" ]; then
    echo "No trained models in $ARTIFACTS/models." >&2
    echo "Run: make train" >&2
    exit 1
fi

echo "==> starting the API on port ${PORT}"
python -m uvicorn src.api.app:app --host 0.0.0.0 --port "$PORT" --log-level warning &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

python scripts/demo_client.py --base-url "$BASE"

echo
echo "==> stopping the server"

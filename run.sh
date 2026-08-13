#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Extract host/port from config.yaml safely
HOST=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['server']['host'])")
PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['server']['port'])")

LOG_FILE="app.log"

echo "Starting Dataset Builder on ${HOST}:${PORT}"
echo "All logs routed to: ${LOG_FILE}"
echo "Press Ctrl+C to stop gracefully"

# Run uvicorn
uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --loop uvloop \
    --log-level debug \
    --no-access-log

echo "Server stopped."


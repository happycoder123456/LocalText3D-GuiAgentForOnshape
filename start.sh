#!/usr/bin/env bash
# Start the Onshape GUI Agent sidecar on http://127.0.0.1:8767
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv-agent/bin/python
if [ ! -x "$PY" ]; then
  PY=python3
  echo "Note: .venv-agent not found, using $(command -v python3)"
fi

exec "$PY" -m agent serve "$@"

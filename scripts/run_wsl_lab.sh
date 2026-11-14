#!/usr/bin/env bash
# Start the Pong Learning Lab server inside WSL2.
#
# From Windows (PowerShell), in the DQN folder:
#   wsl --cd "$PWD" bash scripts/run_wsl_lab.sh
# then open http://localhost:8765 in your Windows browser (WSL2 forwards localhost).
#
# Extra arguments pass through to the server, e.g. --port 8800 or --data-dir PATH.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DQN_LAB_VENV:-$HOME/.venvs/dqn-lab}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "No lab environment at $VENV. Run scripts/setup_wsl_lab.sh first." >&2
  exit 1
fi
if [ ! -f "$PROJECT/web/dist/index.html" ]; then
  echo "The UI is not built. On Windows run: cd web; npm install; npm run build" >&2
  exit 1
fi

cd "$PROJECT"
echo "Pong Learning Lab: open http://localhost:8765 in your browser (Ctrl+C stops it)."
exec "$VENV/bin/python" -m dqn_project.pong_server "$@"

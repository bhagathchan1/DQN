#!/usr/bin/env bash
# Set up the Pong Learning Lab inside WSL2 (Ubuntu).
#
# Why WSL: two-player Atari Pong needs multi-agent-ale-py, which ships no Windows
# wheel. A locally compiled Windows DLL is blocked by Smart App Control, while the
# Linux build is ordinary and unrestricted.
#
# From Windows (PowerShell), in the DQN folder:
#   wsl --cd "$PWD" bash scripts/setup_wsl_lab.sh
#
# Creates a Linux virtual environment at ~/.venvs/dqn-lab (override with
# DQN_LAB_VENV). The Windows .venv is left untouched.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DQN_LAB_VENV:-$HOME/.venvs/dqn-lab}"

echo "Project: $PROJECT"
echo "Virtual environment: $VENV"

# Compiler, CMake and zlib build multi-agent-ale-py from source on recent Pythons.
PACKAGES=(python3 python3-venv python3-dev build-essential cmake zlib1g-dev)
MISSING=()
for package in "${PACKAGES[@]}"; do
  dpkg -s "$package" >/dev/null 2>&1 || MISSING+=("$package")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "Installing system packages: ${MISSING[*]}"
  sudo apt-get update
  sudo apt-get install -y "${MISSING[@]}"
fi

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch keeps the download small; the demo trains 128 updates per game.
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -m pip install -e "$PROJECT[env,lab,test,ui,analysis]" websockets

echo
echo "Checking the two-player Atari runtime..."
"$PY" - <<'EOF'
from pettingzoo.atari import pong_v3
import multi_agent_ale_py  # noqa: F401  fails here if the native build is broken
print('multi-agent-ale-py loaded; pong_v3 import OK')
EOF

echo
echo "Setup complete. Start the lab with:"
echo "  wsl --cd \"<DQN folder>\" bash scripts/run_wsl_lab.sh"

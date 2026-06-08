#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if command -v python >/dev/null 2>&1; then
  PYTHON="${PYTHON:-python}"
else
  PYTHON="${PYTHON:-python.exe}"
fi
if [[ "$PYTHON" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
  ROOT_FOR_PY="$(wslpath -w "$ROOT")"
  export PYTHONPATH="${ROOT_FOR_PY}\\src;${ROOT_FOR_PY}${PYTHONPATH:+;${PYTHONPATH}}"
else
  export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
fi

"$PYTHON" -m scripts.run_suite --smoke --seed-base 10
"$PYTHON" -m pytest -q

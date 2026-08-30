#!/usr/bin/env bash
set -Eeuo pipefail
RUNTIME_ROOT="${GTAIV_DUBBER_RUNTIME:-$HOME/.local/share/gtaiv-latam-ai-dubber}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$RUNTIME_ROOT/.venv/bin/activate"
export PATH="$RUNTIME_ROOT/bin:$PATH"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PROJECT_ROOT="${1:-$CODE_ROOT}"
shift || true
exec python -m gtaiv_dubber process "$PROJECT_ROOT" --resume "$@"

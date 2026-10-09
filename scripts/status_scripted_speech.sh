#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_ROOT="${GTAIV_DUBBER_RUNTIME:-$HOME/.local/share/gtaiv-latam-ai-dubber}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
JOB_ROOT="$RUNTIME_ROOT/jobs/scripted_speech"
source "$RUNTIME_ROOT/.venv/bin/activate"
export PATH="$RUNTIME_ROOT/bin:$PATH"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m gtaiv_dubber status "$JOB_ROOT"

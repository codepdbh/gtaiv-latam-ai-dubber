#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_ROOT="${GTAIV_DUBBER_RUNTIME:-$HOME/.local/share/gtaiv-latam-ai-dubber}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${1:-$CODE_ROOT}"
PID_FILE="$CODE_ROOT/work/mass.pid"
LOG_POINTER="$CODE_ROOT/work/mass.log.path"

mkdir -p "$CODE_ROOT/work" "$CODE_ROOT/logs"
source "$RUNTIME_ROOT/.venv/bin/activate"
export PATH="$RUNTIME_ROOT/bin:$PATH"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >>"$CODE_ROOT/logs/ollama.log" 2>&1 </dev/null &
  for _ in {1..60}; do
    curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
run_log="$CODE_ROOT/logs/mass-$timestamp.log"
printf '%s\n' "$$" >"$PID_FILE"
printf '%s\n' "$run_log" >"$LOG_POINTER"

# Stay in the foreground so Windows Task Scheduler owns the real workload.
exec "$SCRIPT_DIR/run.sh" "$PROJECT_ROOT" >>"$run_log" 2>&1

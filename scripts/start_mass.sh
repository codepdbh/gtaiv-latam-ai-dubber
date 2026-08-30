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

if [[ -s "$PID_FILE" ]]; then
  existing_pid="$(<"$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "El procesamiento masivo ya está activo (PID $existing_pid)."
    [[ -s "$LOG_POINTER" ]] && echo "Log: $(<"$LOG_POINTER")"
    exit 0
  fi
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >>"$CODE_ROOT/logs/ollama.log" 2>&1 </dev/null &
  ollama_pid=$!
  for _ in {1..60}; do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$ollama_pid" 2>/dev/null; then
      echo "ERROR: Ollama terminó durante el arranque. Revise logs/ollama.log."
      exit 1
    fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
run_log="$CODE_ROOT/logs/mass-$timestamp.log"
nohup "$SCRIPT_DIR/run.sh" "$PROJECT_ROOT" >"$run_log" 2>&1 </dev/null &
mass_pid=$!
printf '%s\n' "$mass_pid" >"$PID_FILE"
printf '%s\n' "$run_log" >"$LOG_POINTER"
sleep 2
if ! kill -0 "$mass_pid" 2>/dev/null; then
  echo "ERROR: el proceso terminó durante el arranque."
  tail -n 80 "$run_log"
  exit 1
fi

echo "Procesamiento masivo iniciado (PID $mass_pid)."
echo "Log: $run_log"


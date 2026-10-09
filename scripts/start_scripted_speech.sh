#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_ROOT="${GTAIV_DUBBER_RUNTIME:-$HOME/.local/share/gtaiv-latam-ai-dubber}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INPUT_ROOT="${1:-$CODE_ROOT/scripted_speech}"
OUTPUT_ROOT="${2:-$CODE_ROOT/scripted_speech_es}"
JOB_ROOT="$RUNTIME_ROOT/jobs/scripted_speech"
PID_FILE="$JOB_ROOT/work/process.pid"
LOG_POINTER="$JOB_ROOT/work/process.log.path"
LOG_DIR="$JOB_ROOT/logs"

[[ -d "$INPUT_ROOT" ]] || { echo "ERROR: no existe la carpeta de entrada: $INPUT_ROOT"; exit 2; }
[[ "$(realpath "$INPUT_ROOT")" != "$(realpath -m "$OUTPUT_ROOT")" ]] || {
  echo "ERROR: entrada y salida no pueden ser la misma carpeta."; exit 2;
}

mkdir -p "$JOB_ROOT/config" "$JOB_ROOT/work" "$JOB_ROOT/logs"
source "$RUNTIME_ROOT/.venv/bin/activate"
export PATH="$RUNTIME_ROOT/bin:$PATH"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python - "$JOB_ROOT/config/local.yaml" "$INPUT_ROOT" "$OUTPUT_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

config_path, input_root, output_root = map(Path, sys.argv[1:])
expected = {
    "project": {
        "speech_dir": str(input_root.resolve()),
        "speech_output": str(output_root.resolve()),
    }
}
if config_path.exists():
    current = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    current_project = current.get("project", {})
    for key, value in expected["project"].items():
        if current_project.get(key) != value:
            raise SystemExit(
                f"ERROR: {config_path} ya apunta a otra ruta ({key}); "
                "no la sobrescribí. Revísala manualmente."
            )
else:
    config_path.write_text(yaml.safe_dump(expected, sort_keys=False), encoding="utf-8")
PY

process_is_active() {
  local pid="$1" state
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')"
  [[ -n "$state" && "$state" != Z* ]]
}

if [[ -s "$PID_FILE" ]] && process_is_active "$(<"$PID_FILE")"; then
  echo "El doblaje de scripted_speech ya está activo (PID $(<"$PID_FILE"))."
  [[ -s "$LOG_POINTER" ]] && echo "Log: $(<"$LOG_POINTER")"
  exit 0
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >>"$LOG_DIR/ollama.log" 2>&1 </dev/null &
  ollama_pid=$!
  for _ in {1..60}; do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
    if ! process_is_active "$ollama_pid"; then
      echo "ERROR: Ollama terminó durante el arranque. Log: $LOG_DIR/ollama.log"
      exit 1
    fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null || {
    echo "ERROR: Ollama no respondió en 60 segundos. Log: $LOG_DIR/ollama.log"; exit 1;
  }
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
run_log="$LOG_DIR/run-$timestamp.log"
nohup "$SCRIPT_DIR/run.sh" "$JOB_ROOT" --speech >"$run_log" 2>&1 </dev/null &
run_pid=$!
printf '%s\n' "$run_pid" >"$PID_FILE"
printf '%s\n' "$run_log" >"$LOG_POINTER"
sleep 3
if ! process_is_active "$run_pid"; then
  echo "ERROR: el proceso terminó al iniciar. Últimas líneas:"
  tail -n 80 "$run_log"
  exit 1
fi

echo "Doblaje de scripted_speech iniciado (PID $run_pid)."
echo "Entrada: $INPUT_ROOT"
echo "Salida:  $OUTPUT_ROOT"
echo "Estado/checkpoint: $JOB_ROOT/work/state.sqlite3"
echo "Log: $run_log"

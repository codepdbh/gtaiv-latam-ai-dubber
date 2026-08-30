#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$CODE_ROOT/work/mass.pid"
LOG_POINTER="$CODE_ROOT/work/mass.log.path"

process_is_active() {
  local pid="$1" state
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')"
  [[ -n "$state" && "$state" != Z* ]]
}

if [[ ! -s "$PID_FILE" ]]; then
  echo "No hay PID de procesamiento masivo registrado."
  exit 1
fi

mass_pid="$(<"$PID_FILE")"
if process_is_active "$mass_pid"; then
  echo "ACTIVO (PID $mass_pid)"
else
  echo "DETENIDO (último PID $mass_pid)"
fi

if [[ -s "$LOG_POINTER" ]]; then
  run_log="$(<"$LOG_POINTER")"
  echo "Log: $run_log"
  [[ -f "$run_log" ]] && tail -n 30 "$run_log"
fi

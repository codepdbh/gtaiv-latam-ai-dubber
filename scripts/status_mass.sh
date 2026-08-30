#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$CODE_ROOT/work/mass.pid"
LOG_POINTER="$CODE_ROOT/work/mass.log.path"

if [[ ! -s "$PID_FILE" ]]; then
  echo "No hay PID de procesamiento masivo registrado."
  exit 1
fi

mass_pid="$(<"$PID_FILE")"
if kill -0 "$mass_pid" 2>/dev/null; then
  echo "ACTIVO (PID $mass_pid)"
else
  echo "DETENIDO (último PID $mass_pid)"
fi

if [[ -s "$LOG_POINTER" ]]; then
  run_log="$(<"$LOG_POINTER")"
  echo "Log: $run_log"
  [[ -f "$run_log" ]] && tail -n 30 "$run_log"
fi


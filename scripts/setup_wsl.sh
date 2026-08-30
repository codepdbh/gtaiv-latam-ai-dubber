#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_ROOT="${GTAIV_DUBBER_RUNTIME:-$HOME/.local/share/gtaiv-latam-ai-dubber}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$RUNTIME_ROOT/.venv"
REPO_DIR="$RUNTIME_ROOT/repos/CosyVoice"
MODEL_DIR="$RUNTIME_ROOT/models/Fun-CosyVoice3-0.5B"
COSYVOICE_COMMIT="074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
BIN_DIR="$RUNTIME_ROOT/bin"
export PATH="$BIN_DIR:$PATH"

echo "[ENV] Comprobando WSL2 y GPU antes de instalar dependencias pesadas"
grep -qi microsoft /proc/version || { echo "ERROR: este instalador debe ejecutarse en WSL2"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi no está expuesto en WSL2"; exit 1; }
nvidia-smi
python3 --version
uname -a
cat /proc/version
echo "[ENV] No se instalarán ni modificarán drivers NVIDIA de Linux."

HAS_SUDO=false
if command -v sudo >/dev/null 2>&1 && sudo -v; then
  HAS_SUDO=true
  sudo apt-get update
  sudo apt-get install -y ffmpeg git git-lfs curl build-essential cmake ninja-build \
    python3-pip python3-venv libsndfile1-dev sox libsox-dev libgl1 libglib2.0-0
else
  echo "[ENV] sudo no está disponible; se usará instalación local en $RUNTIME_ROOT"
  mkdir -p "$BIN_DIR" "$RUNTIME_ROOT/downloads"
  if [[ ! -x "$BIN_DIR/ffmpeg" || ! -x "$BIN_DIR/ffprobe" ]]; then
    FFMPEG_ARCHIVE="$RUNTIME_ROOT/downloads/ffmpeg-linux64.tar.xz"
    curl -fL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o "$FFMPEG_ARCHIVE"
    tar -xJf "$FFMPEG_ARCHIVE" -C "$RUNTIME_ROOT/downloads"
    FFMPEG_EXTRACTED="$(find "$RUNTIME_ROOT/downloads" -maxdepth 1 -type d -name 'ffmpeg-*-linux64-gpl' | head -n 1)"
    ln -sf "$FFMPEG_EXTRACTED/bin/ffmpeg" "$BIN_DIR/ffmpeg"
    ln -sf "$FFMPEG_EXTRACTED/bin/ffprobe" "$BIN_DIR/ffprobe"
  fi
fi

mkdir -p "$RUNTIME_ROOT" "$RUNTIME_ROOT/repos" "$RUNTIME_ROOT/models"
python3 -m pip --version >/dev/null || {
  echo "ERROR: Python pip no está disponible. Instale python3-pip o ejecute con un usuario que tenga sudo."
  exit 1
}
python3 -m pip install --user --break-system-packages uv
UV_BIN="$HOME/.local/bin/uv"
"$UV_BIN" python install 3.10
if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  "$UV_BIN" venv --python 3.10 --seed --clear "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip "setuptools<81" wheel

echo "[ENV] Instalando PyTorch CUDA 12.8 para GPU NVIDIA Blackwell"
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install "setuptools<81"
python -m pip install -e "${CODE_ROOT}[ai,gui,test]"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin "$COSYVOICE_COMMIT"
fi
git -C "$REPO_DIR" checkout --detach "$COSYVOICE_COMMIT"
git -C "$REPO_DIR" submodule update --init --recursive
python -m pip install --no-build-isolation -r "$CODE_ROOT/requirements-cosyvoice.txt"
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

if ! command -v ollama >/dev/null && [[ "$HAS_SUDO" == true ]]; then
  echo "[ENV] Instalando Ollama local"
  curl -fsSL https://ollama.com/install.sh | sh
elif ! command -v ollama >/dev/null; then
  echo "[ENV] Instalando Ollama dentro del home WSL"
  mkdir -p "$RUNTIME_ROOT/ollama"
  OLLAMA_ARCHIVE="$RUNTIME_ROOT/downloads/ollama-linux-amd64.tar.zst"
  curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o "$OLLAMA_ARCHIVE"
  python -m pip install zstandard
  OLLAMA_ARCHIVE="$OLLAMA_ARCHIVE" OLLAMA_DEST="$RUNTIME_ROOT/ollama" python - <<'PY'
import os
import tarfile
import zstandard
from pathlib import Path

destination = Path(os.environ["OLLAMA_DEST"]).resolve()
with open(os.environ["OLLAMA_ARCHIVE"], "rb") as compressed:
    with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                target = (destination / member.name).resolve()
                target.relative_to(destination)
                archive.extract(member, destination)
PY
  ln -sf "$RUNTIME_ROOT/ollama/bin/ollama" "$BIN_DIR/ollama"
fi

if [[ "${1:-}" == "--download-models" ]]; then
  echo "[ENV] Descargando Fun-CosyVoice3-0.5B-2512 (varios GB)"
  MODEL_DIR="$MODEL_DIR" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=os.environ["MODEL_DIR"])
PY
  if ! pgrep -x ollama >/dev/null; then
    nohup ollama serve >"$RUNTIME_ROOT/ollama.log" 2>&1 &
  fi
  for _ in {1..30}; do curl -fsS http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
  echo "[ENV] Descargando qwen3:8b (aprox. 5 GB)"
  ollama pull qwen3:8b
  echo "[ENV] Descargando faster-whisper large-v3-turbo"
  python - <<'PY'
from faster_whisper.utils import download_model
download_model("large-v3-turbo")
PY
fi

python - <<'PY'
import torch
print("torch.cuda.is_available() =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU =", torch.cuda.get_device_name(0))
    print("VRAM = %.2f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
PY

PYTHONPATH="$CODE_ROOT" python -m gtaiv_dubber env "$CODE_ROOT"
if [[ "${1:-}" == "--download-models" ]]; then
  echo "Modelos definitivos descargados y entorno listo."
else
  echo "Para descargar los tres modelos definitivos: ./scripts/setup_wsl.sh --download-models"
fi

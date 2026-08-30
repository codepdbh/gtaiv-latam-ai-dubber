from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.setWindowTitle("GTA IV LATAM AI Dubber")
        self.resize(980, 720)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.root_edit = QLineEdit(str(project_root))
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 9))
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.current_file = QLabel("Archivo: —")
        self.stage = QLabel("Etapa: LISTO")
        self.counts = QLabel("Speech: — / —     Cutscenes: — / —")
        self.env_labels: dict[str, QLabel] = {}
        self._build_ui()
        self._run("env")

    def _build_ui(self):
        body = QWidget()
        layout = QVBoxLayout(body)
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Directorio raíz:"))
        path_row.addWidget(self.root_edit, 1)
        browse = QPushButton("EXAMINAR")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        layout.addLayout(path_row)
        layout.addWidget(self.counts)

        environment = QGroupBox("Modelos y entorno")
        env_grid = QGridLayout(environment)
        for index, name in enumerate(("CUDA", "Whisper", "Qwen", "CosyVoice", "Demucs", "FFmpeg")):
            label = QLabel(f"{name}: COMPROBANDO")
            self.env_labels[name] = label
            env_grid.addWidget(label, index // 3, index % 3)
        layout.addWidget(environment)

        actions = QGridLayout()
        commands = (
            ("ESCANEAR AUDIOS", lambda: self._run("scan")),
            ("PROCESAR TODO", lambda: self._run("process")),
            ("SOLO CUTSCENES", lambda: self._run("process", "--cutscenes")),
            ("SOLO SPEECH", lambda: self._run("process", "--speech")),
            ("PAUSAR", self._pause),
            ("CONTINUAR", self._resume),
            ("DETENER DE FORMA SEGURA", self._stop),
            ("VERIFICAR SALIDA", lambda: self._run("verify")),
            ("ABRIR CUTSCENES_ES", lambda: self._open("Cutscenes_es")),
            ("ABRIR SPEECH_ES", lambda: self._open("Speech_es")),
            ("ABRIR LOGS", lambda: self._open("logs")),
        )
        for index, (text, callback) in enumerate(commands):
            button = QPushButton(text)
            button.clicked.connect(callback)
            actions.addWidget(button, index // 4, index % 4)
        layout.addLayout(actions)
        layout.addWidget(self.current_file)
        layout.addWidget(self.stage)
        layout.addWidget(self.progress)
        layout.addWidget(self.console, 1)
        self.setCentralWidget(body)

    def _browse(self):
        selected = QFileDialog.getExistingDirectory(self, "Seleccionar proyecto", self.root_edit.text())
        if selected:
            self.root_edit.setText(selected)

    def _run(self, command: str, extra: str | None = None):
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Proceso activo", "Hay un proceso en ejecución.")
            return
        arguments = ["-m", "gtaiv_dubber", command, self.root_edit.text()]
        if extra:
            arguments.append(extra)
        self.console.appendPlainText("\n$ python " + " ".join(arguments))
        self.stage.setText(f"Etapa: {command.upper()}")
        self.progress.show()
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        self.process.start(sys.executable, arguments)

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.console.appendPlainText(text.rstrip())
        for line in text.splitlines():
            if ".wav" in line:
                self.current_file.setText("Archivo: " + line[-160:])
            for stage in ("ASR", "TRANSLATE", "TTS", "SEPARATE", "ALIGN", "MIX", "EXPORT", "VERIFY"):
                if f"[{stage}]" in line:
                    self.stage.setText("Etapa: " + stage)
        if "CUDA:" in text:
            mappings = {
                "CUDA": "CUDA:", "Whisper": "faster-whisper:", "Qwen": "Qwen3/Ollama:",
                "CosyVoice": "CosyVoice 3:", "Demucs": "Demucs:", "FFmpeg": "FFmpeg:",
            }
            for name, marker in mappings.items():
                matching = next((line for line in text.splitlines() if line.startswith(marker)), "")
                self.env_labels[name].setText(f"{name}: " + ("OK" if "OK" in matching else "ERROR"))

    def _finished(self, exit_code: int):
        self.progress.hide()
        self.stage.setText("Etapa: LISTO" if exit_code == 0 else f"Etapa: ERROR ({exit_code})")

    def _pause(self):
        if self.process.state() == QProcess.NotRunning:
            return
        if os.name != "nt":
            os.kill(int(self.process.processId()), signal.SIGSTOP)
            self.stage.setText("Etapa: PAUSADO")
        else:
            QMessageBox.information(self, "WSL requerido", "Pausar está disponible al ejecutar la GUI en WSLg.")

    def _resume(self):
        if self.process.state() != QProcess.NotRunning and os.name != "nt":
            os.kill(int(self.process.processId()), signal.SIGCONT)
            self.stage.setText("Etapa: CONTINUANDO")

    def _stop(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            self.stage.setText("Etapa: DETENIENDO DE FORMA SEGURA")

    def _open(self, relative: str):
        path = Path(self.root_edit.text()) / relative
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def run_gui(project_root: Path) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(project_root)
    window.show()
    return app.exec()


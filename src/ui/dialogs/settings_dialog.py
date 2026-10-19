"""src/ui/dialogs/settings_dialog.py"""
from __future__ import annotations
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Chat / reasoning models that support tool-calling (suitable as the agent brain).
# nano tiers are intentionally excluded — too weak for multi-tool agent work.
_MODELS = [
    "gpt-5.2",        # recommended default — recent full model, strong tool use
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.1",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5.4-mini",   # cheaper option
    "gpt-4o-mini",
]

# Diffusion image-generation models (used for reference object images).
_IMAGE_MODELS = [
    "gpt-image-1",    # default — current behaviour
    "gpt-image-1.5",
    "gpt-image-2",
    "gpt-image-1-mini",
]

_DEFAULT_MODEL       = "gpt-5.2"
_DEFAULT_IMAGE_MODEL = "gpt-image-1"

_STYLE = """
QDialog   { background:#0e1117; }
QLabel    { color:#c9d1d9; font-family:'Consolas',monospace; font-size:13px; }
QLineEdit { background:#161b22; color:#c9d1d9; border:1px solid #30363d;
            border-radius:6px; padding:6px 10px;
            font-family:'Consolas',monospace; font-size:13px; }
QLineEdit:focus { border-color:#1f6feb; }
QComboBox { background:#161b22; color:#c9d1d9; border:1px solid #30363d;
            border-radius:6px; padding:6px 10px;
            font-family:'Consolas',monospace; font-size:13px; }
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView { background:#161b22; color:#c9d1d9;
                               selection-background-color:#1f6feb; }
QPushButton { background:#1f6feb; color:#fff; border:none; border-radius:6px;
              padding:7px 20px; font-family:'Consolas',monospace;
              font-size:13px; font-weight:bold; }
QPushButton:hover   { background:#388bfd; }
QPushButton:pressed { background:#1158c7; }
QPushButton[flat=true] { background:#21262d; color:#8b949e; }
QPushButton[flat=true]:hover { background:#30363d; color:#c9d1d9; }
"""


def _env_path() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / ".env").exists() or (p / "pyproject.toml").exists():
            return p / ".env"
    return here.parents[4] / ".env"


def load_settings() -> dict[str, str]:
    env = _env_path()
    s: dict[str, str] = {}
    try:
        from dotenv import dotenv_values
        s = dict(dotenv_values(env))
    except ImportError:
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    s[k.strip()] = v.strip().strip('"').strip("'")
    s.setdefault("OPENAI_API_KEY",    os.environ.get("OPENAI_API_KEY", ""))
    s.setdefault("OPENAI_MODEL",      os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL))
    s.setdefault("OPENAI_IMAGE_MODEL",
                 os.environ.get("OPENAI_IMAGE_MODEL", _DEFAULT_IMAGE_MODEL))
    return s


def save_settings(api_key: str, model: str, image_model: str = _DEFAULT_IMAGE_MODEL) -> None:
    env = _env_path()
    fields = {
        "OPENAI_API_KEY":     api_key,
        "OPENAI_MODEL":       model,
        "OPENAI_IMAGE_MODEL": image_model,
    }
    lines: list[str] = []
    written = set()
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key = line.strip().partition("=")[0].strip()
            if key in fields:
                lines.append(f"{key}={fields[key]}"); written.add(key)
            else:
                lines.append(line)
    for key, val in fields.items():
        if key not in written:
            lines.append(f"{key}={val}")
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for key, val in fields.items():
        os.environ[key] = val


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings — API Configuration")
        self.setFixedSize(480, 310)
        self.setStyleSheet(_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("⚙  API Configuration")
        title.setFont(QFont("Consolas", 14))
        title.setStyleSheet("color:#58a6ff; font-weight:bold;")
        layout.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#30363d;"); layout.addWidget(sep)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._key = QLineEdit()
        self._key.setPlaceholderText("sk-…")
        self._key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("OpenAI API Key:", self._key)
        self._model = QComboBox()
        self._model.addItems(_MODELS)
        self._model.setEditable(True)
        form.addRow("Agent model:", self._model)
        self._image_model = QComboBox()
        self._image_model.addItems(_IMAGE_MODELS)
        self._image_model.setEditable(True)
        form.addRow("Image model:", self._image_model)
        layout.addLayout(form)
        layout.addStretch()

        bl = QHBoxLayout(); bl.setSpacing(10)
        cancel = QPushButton("Cancel")
        cancel.setProperty("flat", True)
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.clicked.connect(self._save)
        save.setDefault(True)
        bl.addStretch(); bl.addWidget(cancel); bl.addWidget(save)
        layout.addLayout(bl)

        s = load_settings()
        self._key.setText(s.get("OPENAI_API_KEY", ""))

        model = s.get("OPENAI_MODEL", _DEFAULT_MODEL)
        idx = self._model.findText(model)
        if idx >= 0: self._model.setCurrentIndex(idx)
        else: self._model.setCurrentText(model)

        image_model = s.get("OPENAI_IMAGE_MODEL", _DEFAULT_IMAGE_MODEL)
        idx_img = self._image_model.findText(image_model)
        if idx_img >= 0: self._image_model.setCurrentIndex(idx_img)
        else: self._image_model.setCurrentText(image_model)

    def _save(self) -> None:
        key = self._key.text().strip()
        if key:
            save_settings(
                key,
                self._model.currentText().strip(),
                self._image_model.currentText().strip(),
            )
        self.accept()

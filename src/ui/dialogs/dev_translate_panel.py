"""
src/ui/dialogs/dev_translate_panel.py
─────────────────────────────────────
Developer-only floating panel for working in Spanish.

- Top box: type in Spanish + Enter → translated to English → typed (typewriter)
  into the chat input and sent, as if you wrote it there directly.
- Bottom box: every agent reply (English) is translated to Spanish and shown here.

Enabled via the environment variable SIMUSCAN_DEV_TRANSLATE=1 (see main_window),
or from the Settings menu. Translation uses the OpenAI API; the model can be
overridden with SIMUSCAN_TRANSLATE_MODEL (default: gpt-4o-mini).
"""
from __future__ import annotations

import os
import threading

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent


_TRANSLATE_DEFAULT_MODEL = "gpt-4o-mini"


def _translate(text: str, to_english: bool) -> str:
    """Translate text ES→EN or EN→ES. Preserves code, paths and markdown."""
    from openai import OpenAI
    from src.ui.dialogs.settings_dialog import load_settings

    s = load_settings()
    api_key = s.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("No OpenAI API key configured.")
    model = os.environ.get("SIMUSCAN_TRANSLATE_MODEL",
                           _TRANSLATE_DEFAULT_MODEL)

    target = "English" if to_english else "Spanish"
    system = (
        f"You are a translation engine. Translate the user's message to {target}. "
        "Output ONLY the translation, nothing else. "
        "Preserve verbatim: code blocks, inline code, file paths, commands, "
        "URLs, numbers, parameter names and markdown formatting. "
        "Do not add explanations or quotes."
    )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": text}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


class _EnterEdit(QTextEdit):
    enter_pressed = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.enter_pressed.emit()
        else:
            super().keyPressEvent(event)


class DevTranslatePanel(QDialog):
    """Floating ES↔EN bridge to the chat. Created on the Qt main thread."""

    # Marshal background-thread translation results back to the UI thread.
    _en_ready = pyqtSignal(str)        # ES→EN result (to send to chat)
    _es_ready = pyqtSignal(str)        # EN→ES result (agent reply, to display)
    _err      = pyqtSignal(str)

    def __init__(self, chat_panel, parent=None) -> None:
        super().__init__(parent)
        self._chat = chat_panel
        self.setWindowTitle("🌐 Dev — Español ↔ English")
        self.resize(460, 520)
        # Non-modal, stays on top so you can type while the main window works
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Tool)
        self.setStyleSheet("""
            QDialog { background:#0e1117; }
            QLabel  { color:#8b949e; font-family:'Consolas',monospace; font-size:11px; }
            QTextEdit { background:#0e1117; color:#c9d1d9; border:1px solid #30363d;
                        border-radius:8px; padding:8px 10px;
                        font-family:'Segoe UI',sans-serif; font-size:13px; }
            QTextEdit:focus { border-color:#1f6feb; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(6)

        root.addWidget(QLabel("Escribe en español y pulsa Enter "
                              "(Shift+Enter = salto de línea):"))
        self._in = _EnterEdit()
        self._in.setPlaceholderText("Tu mensaje en español…")
        self._in.setFixedHeight(90)
        self._in.enter_pressed.connect(self._on_submit)
        root.addWidget(self._in)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#6e7681; font-size:10px;")
        root.addWidget(self._status)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#30363d;"); root.addWidget(sep)

        root.addWidget(QLabel("Respuesta del agente (traducida al español):"))
        self._out = QTextEdit()
        self._out.setReadOnly(True)
        root.addWidget(self._out, stretch=1)

        # Wire background-result signals
        self._en_ready.connect(self._send_to_chat)
        self._es_ready.connect(self._append_reply)
        self._err.connect(self._show_error)

        # Receive every agent reply (English) from the bridge
        from src.bridge.ui_bridge import ui
        ui._signals.chat_message.connect(self._on_agent_message)

    # ── Input: ES → EN → chat ──────────────────────────────────────────────────

    def _on_submit(self) -> None:
        es = self._in.toPlainText().strip()
        if not es:
            return
        self._in.clear()
        self._status.setText("Traduciendo al inglés…")

        def work():
            try:
                en = _translate(es, to_english=True)
                self._en_ready.emit(en)
            except Exception as exc:
                self._err.emit(str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _send_to_chat(self, english: str) -> None:
        self._status.setText("")
        self._chat.type_and_send(english)

    # ── Output: agent reply EN → ES ────────────────────────────────────────────

    def _on_agent_message(self, markdown_en: str) -> None:
        # Skip translation while the panel is hidden/closed (saves tokens).
        if not self.isVisible():
            return
        txt = (markdown_en or "").strip()
        if not txt:
            return

        def work():
            try:
                es = _translate(txt, to_english=False)
                self._es_ready.emit(es)
            except Exception as exc:
                self._err.emit(str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _append_reply(self, spanish: str) -> None:
        self._out.append(spanish)
        self._out.append("")  # blank line between replies
        self._out.verticalScrollBar().setValue(
            self._out.verticalScrollBar().maximum())

    def _show_error(self, msg: str) -> None:
        self._status.setText(f"⚠ Error de traducción: {msg}")

"""
src/ui/log_panel.py
────────────────────
Bottom panel: terminal-style log output.
Receives (level, text) from ui.log() via Qt signal.
"""

from __future__ import annotations
from datetime import datetime
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QTextCursor

_COLORS = {
    "INFO": "#58a6ff",
    "OK":   "#3fb950",
    "WARN": "#d29922",
    "ERR":  "#f85149",
    "TOOL": "#bc8cff",
    "SYS":  "#8b949e",
}


class LogPanel(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setFixedHeight(32)
        header.setStyleSheet("QFrame { background:#161b22; border-top:1px solid #30363d; border-bottom:1px solid #30363d; }")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 0, 8, 0)
        title = QLabel("  🖥  Output / Logs")
        title.setStyleSheet("color:#c9d1d9; font-family:'Consolas',monospace; font-size:12px; font-weight:bold;")
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(56, 22)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet("""
            QPushButton { background:#21262d; color:#8b949e; border:1px solid #30363d;
                          border-radius:4px; font-family:'Consolas',monospace; font-size:11px; }
            QPushButton:hover { background:#30363d; color:#c9d1d9; }
        """)
        clear_btn.clicked.connect(self.clear)
        hl.addWidget(title, stretch=1)
        hl.addWidget(clear_btn)
        layout.addWidget(header)

        # Log area
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 11))
        self._log.setStyleSheet("""
            QTextEdit { background:#0a0d12; color:#c9d1d9; border:none; padding:6px 10px; }
            QScrollBar:vertical { background:#161b22; width:8px; border-radius:4px; }
            QScrollBar::handle:vertical { background:#30363d; border-radius:4px; }
        """)
        layout.addWidget(self._log)

    def append_line(self, level: str, text: str) -> None:
        """Called by ui.log() via Qt signal — always runs in main thread."""
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        ts    = datetime.now().strftime("%H:%M:%S")
        color = _COLORS.get(level, "#c9d1d9")

        fmt_ts = QTextCharFormat()
        fmt_ts.setForeground(QColor("#3d4449"))
        cursor.setCharFormat(fmt_ts)
        cursor.insertText(f"[{ts}] ")

        fmt_lvl = QTextCharFormat()
        fmt_lvl.setForeground(QColor(color))
        cursor.setCharFormat(fmt_lvl)
        cursor.insertText(f"{level:<4}  ")

        fmt_msg = QTextCharFormat()
        fmt_msg.setForeground(QColor("#c9d1d9"))
        cursor.setCharFormat(fmt_msg)
        cursor.insertText(text + "\n")

        self._log.setTextCursor(cursor)
        self._log.ensureCursorVisible()

    def clear(self) -> None:
        self._log.clear()

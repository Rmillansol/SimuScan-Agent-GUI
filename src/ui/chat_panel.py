"""
src/ui/chat_panel.py
────────────────────
Left panel: conversation display + input box.

Receives pre-converted HTML from ui.chat() via Qt signal.
User messages are sent via user_message_sent signal.
"""

from __future__ import annotations
import re
import html as _html_mod

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QScrollArea,
    QLabel, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QKeyEvent


# ── Markdown → HTML ───────────────────────────────────────────────────────────

def md_to_html(text: str, fg: str = "#c9d1d9", bold: str = "#7ee787") -> str:
    """
    Convert Markdown to HTML with inline colors suitable for QLabel RichText.
    Uses `markdown` library if installed, otherwise a simple fallback.
    """
    try:
        import markdown
        body = markdown.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
    except ImportError:
        body = _md_fallback(text)

    # Inject colors inline so QLabel respects them regardless of system theme
    body = (body
        .replace("<strong>", '<strong style="color:' + bold + '">')
        .replace("<b>",      '<b style="color:' + bold + '">')
        .replace("<h1>",     '<h1 style="color:' + bold + ';margin:4px 0">')
        .replace("<h2>",     '<h2 style="color:' + bold + ';margin:4px 0">')
        .replace("<h3>",     '<h3 style="color:' + bold + ';margin:4px 0">')
        .replace("<code>",
                 '<code style="background:#0d1117;color:#e6edf3;'
                 'font-family:Consolas,monospace;font-size:11px;padding:1px 4px;">')
        .replace("<pre>",
                 '<pre style="background:#0d1117;color:#e6edf3;'
                 'font-family:Consolas,monospace;font-size:11px;padding:8px;'
                 'border:1px solid #30363d;">')
        .replace("<li>",  '<li style="color:' + fg + '">')
        .replace("<p>",   '<p style="color:' + fg + ';margin:2px 0">')
        .replace("<em>",  '<em style="color:' + fg + '">')
    )
    return (
        '<span style="color:' + fg + ';'
        'font-family:Segoe UI,sans-serif;font-size:13px;">'
        + body + '</span>'
    )


def _md_fallback(text: str) -> str:
    t = _html_mod.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         t)
    t = re.sub(r"`([^`]+)`",     r"<code>\1</code>",     t)
    lines = t.split("\n")
    out, in_ul, in_ol = [], False, False
    _ol = re.compile(r"^[0-9]+[.] ")
    _ul = re.compile(r"^[-*] ")
    for line in lines:
        if _ol.match(line):
            if not in_ol:
                if in_ul: out.append("</ul>"); in_ul = False
                out.append("<ol>"); in_ol = True
            out.append("<li>" + _ol.sub("", line) + "</li>")
        elif _ul.match(line):
            if not in_ul:
                if in_ol: out.append("</ol>"); in_ol = False
                out.append("<ul>"); in_ul = True
            out.append("<li>" + line[2:] + "</li>")
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            if in_ol: out.append("</ol>"); in_ol = False
            if line.strip():
                out.append("<p>" + line + "</p>")
    if in_ul: out.append("</ul>")
    if in_ol: out.append("</ol>")
    return "\n".join(out)


# ── Message bubble ────────────────────────────────────────────────────────────

_ROLE_CFG = {
    "user": {
        "name": "You", "label": "#58a6ff", "fg": "#e6edf3", "bold": "#79c0ff",
        "bg": "#1a2840", "border": "#1f6feb55", "margin": "2px 48px 2px 4px",
    },
    "agent": {
        "name": "SimuScan", "label": "#3fb950", "fg": "#c9d1d9", "bold": "#7ee787",
        "bg": "#161b22", "border": "#30363d", "margin": "2px 4px 2px 48px",
    },
    "system": {
        "name": "System", "label": "#6e7681", "fg": "#8b949e", "bold": "#8b949e",
        "bg": "#21262d", "border": "#30363d", "margin": "2px 24px",
    },
}


class MessageBubble(QFrame):

    def __init__(self, role: str, html_content: str, parent=None) -> None:
        super().__init__(parent)
        cfg = _ROLE_CFG.get(role, _ROLE_CFG["system"])

        self.setStyleSheet(
            "QFrame {"
            "  background-color:" + cfg["bg"] + ";"
            "  border:1px solid " + cfg["border"] + ";"
            "  border-radius:10px;"
            "  margin:" + cfg["margin"] + ";"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(4)

        # Role label
        name_lbl = QLabel(cfg["name"])
        name_lbl.setStyleSheet(
            "color:" + cfg["label"] + ";"
            "font-size:11px; font-weight:bold;"
            "font-family:'Consolas',monospace;"
            "background:transparent; border:none;"
        )
        layout.addWidget(name_lbl)

        # Message — QLabel with RichText (auto-wraps, auto-sizes, no height hacks)
        msg_lbl = QLabel()
        msg_lbl.setTextFormat(Qt.TextFormat.RichText)
        msg_lbl.setWordWrap(True)
        msg_lbl.setOpenExternalLinks(True)
        msg_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        msg_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        msg_lbl.setStyleSheet("background:transparent; border:none; margin:0; padding:0;")
        msg_lbl.setText(html_content)
        layout.addWidget(msg_lbl)


# ── Input ─────────────────────────────────────────────────────────────────────

class _EnterTextEdit(QTextEdit):
    enter_pressed = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.enter_pressed.emit()
        else:
            super().keyPressEvent(event)


# ── Chat panel ────────────────────────────────────────────────────────────────

class ChatPanel(QWidget):
    user_message_sent = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("  💬  SimuScan Agent")
        header.setFixedHeight(36)
        header.setStyleSheet("""
            QLabel { background:#161b22; color:#c9d1d9;
                     font-family:'Consolas',monospace; font-size:13px; font-weight:bold;
                     border-bottom:1px solid #30363d; padding-left:8px; }
        """)
        layout.addWidget(header)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("""
            QScrollArea { background:#0e1117; border:none; }
            QScrollBar:vertical { background:#161b22; width:8px; border-radius:4px; }
            QScrollBar::handle:vertical { background:#30363d; border-radius:4px; }
        """)
        self._msgs_widget = QWidget()
        self._msgs_widget.setStyleSheet("background:#0e1117;")
        self._msgs_layout = QVBoxLayout(self._msgs_widget)
        self._msgs_layout.setContentsMargins(8, 12, 8, 12)
        self._msgs_layout.setSpacing(8)
        self._msgs_layout.addStretch()
        self._scroll.setWidget(self._msgs_widget)
        layout.addWidget(self._scroll, stretch=1)

        # Input
        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background:#161b22; border-top:1px solid #30363d; }")
        il = QHBoxLayout(input_frame)
        il.setContentsMargins(10, 10, 10, 10)
        il.setSpacing(8)

        self._input = _EnterTextEdit()
        self._input.setPlaceholderText("Type a message…  (Shift+Enter for newline)")
        self._input.setFixedHeight(72)
        self._input.setStyleSheet("""
            QTextEdit { background:#0e1117; color:#c9d1d9; border:1px solid #30363d;
                        border-radius:8px; padding:8px 12px;
                        font-family:'Segoe UI',sans-serif; font-size:13px; }
            QTextEdit:focus { border-color:#1f6feb; }
        """)
        self._input.enter_pressed.connect(self._send)

        send_btn = QPushButton("Send")
        send_btn.setFixedSize(72, 72)
        send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        send_btn.setStyleSheet("""
            QPushButton { background:#1f6feb; color:#fff; border:none;
                          border-radius:8px; font-family:'Consolas',monospace;
                          font-size:13px; font-weight:bold; }
            QPushButton:hover   { background:#388bfd; }
            QPushButton:pressed { background:#1158c7; }
        """)
        send_btn.clicked.connect(self._send)

        il.addWidget(self._input, stretch=1)
        il.addWidget(send_btn)
        layout.addWidget(input_frame)

        self._add_bubble("system",
            md_to_html("Welcome to **SimuScan Agent**. "
                       "Configure your API key in ⚙ Settings, then start chatting.",
                       "#8b949e", "#8b949e"))

    # ── Public API (called by main_window wiring) ─────────────────────────────

    def add_agent_message(self, markdown: str) -> None:
        cfg = _ROLE_CFG["agent"]
        self._add_bubble("agent", md_to_html(markdown, cfg["fg"], cfg["bold"]))

    def add_user_message(self, text: str) -> None:
        cfg = _ROLE_CFG["user"]
        self._add_bubble("user", md_to_html(text, cfg["fg"], cfg["bold"]))

    def add_system_message(self, text: str) -> None:
        self._add_bubble("system",
            md_to_html(text, "#8b949e", "#8b949e"))

    def clear_history(self) -> None:
        while self._msgs_layout.count() > 1:
            item = self._msgs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def type_and_send(self, text: str, interval_ms: int = 40) -> None:
        """
        Type `text` into the input box letter by letter (typewriter effect),
        then send it as if the user pressed Enter. Used by the dev translate panel.
        """
        text = (text or "").strip()
        if not text:
            return
        self._input.clear()
        self._tw_text = text
        self._tw_i    = 0
        self._tw_timer = QTimer(self)
        self._tw_timer.timeout.connect(self._tw_step)
        self._tw_timer.start(max(1, interval_ms))

    def _tw_step(self) -> None:
        if self._tw_i < len(self._tw_text):
            self._input.insertPlainText(self._tw_text[self._tw_i])
            self._tw_i += 1
        else:
            self._tw_timer.stop()
            self._send()

    # ── Private ───────────────────────────────────────────────────────────────

    def _send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self.add_user_message(text)
        self.user_message_sent.emit(text)

    def _add_bubble(self, role: str, html: str) -> None:
        bubble = MessageBubble(role, html)
        self._msgs_layout.insertWidget(self._msgs_layout.count() - 1, bubble)

        def _scroll_to_bubble():
            self._msgs_widget.adjustSize()
            pos = bubble.mapTo(self._msgs_widget, bubble.rect().topLeft())
            self._scroll.verticalScrollBar().setValue(pos.y())

        QTimer.singleShot(0,   lambda: self._msgs_widget.updateGeometry())
        QTimer.singleShot(100, _scroll_to_bubble)
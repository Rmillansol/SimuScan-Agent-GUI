"""
src/ui/plot_panel.py
────────────────────
Right panel with TWO views shown in the same area, switched by a button:

  • Gallery  — every generated image (previews, final training plots, GWY
               overlays, …). Unbounded; navigate with Prev/Next.
  • Live     — a single, in-place plot that updates while a training runs
               (the live loss curve). NOT part of the gallery carousel.

Receives PNG paths via Qt signals:
  ui.plot_path(path)      → show_image()        (gallery, appended)
  ui.plot_preview(fig)    → add_preview_image()  (gallery, appended, jumps here)
  ui.plot_live(fig, ...)  → update_live_plot()   (Live view, single slot)
"""

from __future__ import annotations
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QScrollArea, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont


class PlotPanel(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._images: list[str] = []     # gallery images (unbounded)
        self._idx = -1                   # current gallery index
        self._live_path: str | None = None   # current live plot image (single)
        self._mode = "gallery"           # "gallery" | "live"
        self._live_dirty = False         # live updated while viewing the gallery
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("  📊  Visualization")
        header.setFixedHeight(36)
        header.setStyleSheet("""
            QLabel { background:#161b22; color:#c9d1d9;
                     font-family:'Consolas',monospace; font-size:13px; font-weight:bold;
                     border-bottom:1px solid #30363d; padding-left:8px; }
        """)
        layout.addWidget(header)

        self._scroll = QScrollArea()
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("""
            QScrollArea { background:#0a0d12; border:none; }
            QScrollBar:vertical   { background:#161b22; width:8px; border-radius:4px; }
            QScrollBar:horizontal { background:#161b22; height:8px; border-radius:4px; }
            QScrollBar::handle { background:#30363d; border-radius:4px; }
        """)
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet("background:#0a0d12; color:#3d4449;")
        self._img_label.setText("No image yet.\nImages appear here\nwhen generated.")
        self._img_label.setFont(QFont("Consolas", 11))
        self._scroll.setWidget(self._img_label)
        layout.addWidget(self._scroll, stretch=1)

        # Navigation / mode bar
        nav = QFrame()
        nav.setFixedHeight(44)
        nav.setStyleSheet("QFrame { background:#161b22; border-top:1px solid #30363d; }")
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(10, 4, 10, 4)
        nl.setSpacing(8)

        btn_style = """
            QPushButton { background:#21262d; color:#c9d1d9; border:1px solid #30363d;
                          border-radius:6px; font-family:'Consolas',monospace;
                          font-size:13px; padding:4px 14px; }
            QPushButton:hover   { background:#30363d; }
            QPushButton:pressed { background:#1f6feb; border-color:#1f6feb; }
            QPushButton:disabled { color:#3d4449; border-color:#21262d; }
        """
        self._btn_prev = QPushButton("◀  Prev")
        self._btn_prev.setStyleSheet(btn_style)
        self._btn_prev.clicked.connect(self._prev)
        self._btn_prev.setEnabled(False)

        self._btn_next = QPushButton("Next  ▶")
        self._btn_next.setStyleSheet(btn_style)
        self._btn_next.clicked.connect(self._next)
        self._btn_next.setEnabled(False)

        self._btn_mode = QPushButton("⚡ Live")
        self._btn_mode.setStyleSheet(btn_style)
        self._btn_mode.clicked.connect(self._toggle_mode)
        self._btn_mode.setEnabled(False)

        self._counter = QLabel("—")
        self._counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._counter.setStyleSheet("color:#8b949e; font-family:'Consolas',monospace; font-size:11px;")

        self._name = QLabel("")
        self._name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name.setStyleSheet("color:#58a6ff; font-family:'Consolas',monospace; font-size:11px;")
        self._name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        nl.addWidget(self._btn_prev)
        nl.addWidget(self._counter)
        nl.addWidget(self._name, stretch=1)
        nl.addWidget(self._btn_mode)
        nl.addWidget(self._btn_next)
        layout.addWidget(nav)

    # ── Gallery ────────────────────────────────────────────────────────────────

    def show_image(self, path: str) -> None:
        """Append an image to the gallery (ui.plot_path / ui.plot). Jumps to it
        only if the gallery is already the active view, so it never yanks the user
        away from the live training curve."""
        self._images.append(path)
        if self._mode == "gallery":
            self._idx = len(self._images) - 1
        elif self._idx < 0:
            self._idx = len(self._images) - 1
        self._render()

    def add_preview_image(self, path: str) -> None:
        """Append a synthetic preview and show it (ui.plot_preview). Previews are
        user-requested, so this switches to the gallery and jumps to the new image.
        No cap — all previews are kept for comparison via Prev/Next."""
        self._images.append(path)
        self._idx = len(self._images) - 1
        self._mode = "gallery"
        self._render()

    # ── Live view ────────────────────────────────────────────────────────────

    def update_live_plot(self, path: str, key: str = "live", focus: bool = False) -> None:
        """Update the single Live plot (ui.plot_live). It is NOT added to the
        gallery. Auto-switches to the Live view on the first frame of a run or when
        focus=True; otherwise it refreshes in place only if Live is already showing,
        and flags the Live button so the user knows there is new activity."""
        first = self._live_path is None
        self._live_path = path
        if focus or first or self._mode == "live":
            self._mode = "live"
            self._live_dirty = False
            self._render()
        else:
            self._live_dirty = True   # updated in the background
            self._update_controls()

    def _toggle_mode(self) -> None:
        if self._mode == "gallery":
            if self._live_path is None:
                return
            self._mode = "live"
            self._live_dirty = False
        else:
            if not self._images:
                return
            self._mode = "gallery"
        self._render()

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._mode != "gallery":
            self._mode = "gallery"
        if self._idx > 0:
            self._idx -= 1
        self._render()

    def _next(self) -> None:
        if self._mode != "gallery":
            self._mode = "gallery"
        elif self._idx < len(self._images) - 1:
            self._idx += 1
        self._render()

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self) -> None:
        if self._mode == "live" and self._live_path:
            self._show_path(self._live_path)
            self._counter.setText("● live")
            self._name.setText("Live training")
        elif self._images and 0 <= self._idx < len(self._images):
            path = self._images[self._idx]
            self._show_path(path)
            self._counter.setText(f"{self._idx + 1} / {len(self._images)}")
            self._name.setText(Path(path).stem)
        self._update_controls()

    def _show_path(self, path: str) -> None:
        px = QPixmap(path)
        if px.isNull():
            self._img_label.setText(f"Cannot load:\n{path}")
            return
        available = self._scroll.viewport().size()
        scaled = px.scaled(available, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        self._img_label.setPixmap(scaled)

    def _update_controls(self) -> None:
        gallery_ready = len(self._images) > 0
        live_ready    = self._live_path is not None
        in_gallery    = self._mode == "gallery"

        # Prev/Next only make sense while browsing the gallery
        self._btn_prev.setEnabled(in_gallery and self._idx > 0)
        self._btn_next.setEnabled(in_gallery and self._idx < len(self._images) - 1)

        if in_gallery:
            self._btn_mode.setText("⚡ Live ●" if self._live_dirty else "⚡ Live")
            self._btn_mode.setEnabled(live_ready)
        else:
            self._btn_mode.setText("🖼 Gallery")
            self._btn_mode.setEnabled(gallery_ready)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

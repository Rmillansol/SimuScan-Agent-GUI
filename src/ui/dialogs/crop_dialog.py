"""
src/ui/dialogs/crop_dialog.py
──────────────────────────────
Interactive multi-crop dialog.
- Prev / Next buttons navigate between raw images of an article.
- The user draws a rectangle, adjusts it, then clicks "Save Crop".
  Multiple crops can be saved from the same or different images.
- "Done" closes the dialog and fires on_saved(crops) with all saved crops.
- Enter = Save Crop, Esc = Done (close).
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QImage
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QListWidget, QListWidgetItem,
    QSplitter, QWidget,
)


# ── image helpers ─────────────────────────────────────────────────────────────

def _cv2_to_qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def _process_crop(bgr_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale → square (black pad) → Otsu background zeroing. Returns (result_bgr, mask)."""
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY) if bgr_crop.ndim == 3 else bgr_crop
    h, w = gray.shape
    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.uint8)
    py, px = (side - h) // 2, (side - w) // 2
    sq[py:py + h, px:px + w] = gray
    _, mask = cv2.threshold(sq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5)
    result = sq.copy()
    result[mask == 0] = 0
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), mask


# ── drag mode ─────────────────────────────────────────────────────────────────

class _Mode(Enum):
    NONE = auto(); DRAW = auto(); MOVE = auto()
    RESIZE_TL = auto(); RESIZE_TR = auto(); RESIZE_BL = auto(); RESIZE_BR = auto()
    RESIZE_T  = auto(); RESIZE_B  = auto(); RESIZE_L  = auto(); RESIZE_R  = auto()

_HANDLE_R = 6

_MODE_CURSOR = {
    _Mode.NONE:      Qt.CursorShape.CrossCursor,
    _Mode.MOVE:      Qt.CursorShape.SizeAllCursor,
    _Mode.RESIZE_TL: Qt.CursorShape.SizeFDiagCursor,
    _Mode.RESIZE_BR: Qt.CursorShape.SizeFDiagCursor,
    _Mode.RESIZE_TR: Qt.CursorShape.SizeBDiagCursor,
    _Mode.RESIZE_BL: Qt.CursorShape.SizeBDiagCursor,
    _Mode.RESIZE_T:  Qt.CursorShape.SizeVerCursor,
    _Mode.RESIZE_B:  Qt.CursorShape.SizeVerCursor,
    _Mode.RESIZE_L:  Qt.CursorShape.SizeHorCursor,
    _Mode.RESIZE_R:  Qt.CursorShape.SizeHorCursor,
}


# ── canvas ────────────────────────────────────────────────────────────────────

class _CropCanvas(QLabel):
    def __init__(self, pixmap: QPixmap, scale: float, on_change: Callable) -> None:
        super().__init__()
        self._base_px = pixmap
        self._scale   = scale
        self._on_change = on_change
        self._rect: QRect | None = None
        self._mode  = _Mode.NONE
        self._drag_start: QPoint | None = None
        self._rect_at_drag: QRect | None = None
        self.setPixmap(pixmap)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def _handles(self) -> dict[_Mode, QRect]:
        if self._rect is None:
            return {}
        r = self._rect
        cx, cy = r.center().x(), r.center().y()
        def h(x, y): return QRect(x - _HANDLE_R, y - _HANDLE_R, _HANDLE_R*2, _HANDLE_R*2)
        return {
            _Mode.RESIZE_TL: h(r.left(),  r.top()),
            _Mode.RESIZE_TR: h(r.right(), r.top()),
            _Mode.RESIZE_BL: h(r.left(),  r.bottom()),
            _Mode.RESIZE_BR: h(r.right(), r.bottom()),
            _Mode.RESIZE_T:  h(cx,        r.top()),
            _Mode.RESIZE_B:  h(cx,        r.bottom()),
            _Mode.RESIZE_L:  h(r.left(),  cy),
            _Mode.RESIZE_R:  h(r.right(), cy),
        }

    def _hit_test(self, pos: QPoint) -> _Mode:
        if self._rect is None:
            return _Mode.NONE
        for mode, hrect in self._handles().items():
            if hrect.contains(pos):
                return mode
        if self._rect.contains(pos):
            return _Mode.MOVE
        return _Mode.NONE

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = ev.pos()
        hit = self._hit_test(pos)
        self._mode = _Mode.DRAW if hit == _Mode.NONE else hit
        if self._mode == _Mode.DRAW:
            self._rect = QRect(pos, pos)
        self._drag_start   = pos
        self._rect_at_drag = QRect(self._rect) if self._rect else None

    def mouseMoveEvent(self, ev) -> None:
        pos = ev.pos()
        if self._mode == _Mode.NONE:
            self.setCursor(_MODE_CURSOR.get(self._hit_test(pos), Qt.CursorShape.CrossCursor))
            return
        delta = pos - self._drag_start
        r = self._rect_at_drag or QRect()
        if   self._mode == _Mode.DRAW:      self._rect = QRect(self._drag_start, pos).normalized()
        elif self._mode == _Mode.MOVE:      self._rect = r.translated(delta)
        elif self._mode == _Mode.RESIZE_TL: self._rect = QRect(r.topLeft() + delta, r.bottomRight()).normalized()
        elif self._mode == _Mode.RESIZE_TR: self._rect = QRect(QPoint(r.left(), r.top() + delta.y()), QPoint(r.right() + delta.x(), r.bottom())).normalized()
        elif self._mode == _Mode.RESIZE_BL: self._rect = QRect(QPoint(r.left() + delta.x(), r.top()), QPoint(r.right(), r.bottom() + delta.y())).normalized()
        elif self._mode == _Mode.RESIZE_BR: self._rect = QRect(r.topLeft(), r.bottomRight() + delta).normalized()
        elif self._mode == _Mode.RESIZE_T:  self._rect = QRect(QPoint(r.left(), r.top() + delta.y()), r.bottomRight()).normalized()
        elif self._mode == _Mode.RESIZE_B:  self._rect = QRect(r.topLeft(), QPoint(r.right(), r.bottom() + delta.y())).normalized()
        elif self._mode == _Mode.RESIZE_L:  self._rect = QRect(QPoint(r.left() + delta.x(), r.top()), r.bottomRight()).normalized()
        elif self._mode == _Mode.RESIZE_R:  self._rect = QRect(r.topLeft(), QPoint(r.right() + delta.x(), r.bottom())).normalized()
        self._redraw()
        self._on_change()

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if self._mode == _Mode.DRAW and self._rect and \
                (self._rect.width() < 4 or self._rect.height() < 4):
            self._rect = None
        self._mode = _Mode.NONE
        self._drag_start = self._rect_at_drag = None
        self._redraw()
        self._on_change()

    def _redraw(self) -> None:
        px = self._base_px.copy()
        if self._rect and self._rect.width() > 1 and self._rect.height() > 1:
            p = QPainter(px)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            r = self._rect
            full = QRect(0, 0, px.width(), px.height())
            dim = QColor(0, 0, 0, 110)
            for strip in [
                QRect(0, 0, full.width(), r.top()),
                QRect(0, r.bottom(), full.width(), full.height() - r.bottom()),
                QRect(0, r.top(), r.left(), r.height()),
                QRect(r.right(), r.top(), full.width() - r.right(), r.height()),
            ]:
                p.fillRect(strip, dim)
            p.setPen(QPen(QColor(255, 80, 80), 2, Qt.PenStyle.SolidLine))
            p.drawRect(r)
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(QBrush(QColor(255, 80, 80)))
            for hrect in self._handles().values():
                p.drawEllipse(hrect)
            p.end()
        self.setPixmap(px)

    def selection_in_original(self) -> QRect | None:
        if self._rect is None or self._rect.width() < 4 or self._rect.height() < 4:
            return None
        s = self._scale
        return QRect(int(self._rect.x()/s), int(self._rect.y()/s),
                     int(self._rect.width()/s), int(self._rect.height()/s))

    def reset(self) -> None:
        self._rect = None
        self._mode = _Mode.NONE
        self.setPixmap(self._base_px)
        self._on_change()

    def load_image(self, bgr: np.ndarray, scale: float) -> None:
        h, w = bgr.shape[:2]
        display = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        px = QPixmap.fromImage(_cv2_to_qimage(display))
        self._base_px = px
        self._scale   = scale
        self._rect    = None
        self._mode    = _Mode.NONE
        self.setPixmap(px)
        self.resize(px.size())
        self._on_change()


# ── dialog ────────────────────────────────────────────────────────────────────

_BTN = """
    QPushButton {
        background:#21262d; color:#c9d1d9;
        border:1px solid #30363d; border-radius:4px;
        padding:4px 10px; font-size:12px;
    }
    QPushButton:hover   { background:#30363d; }
    QPushButton:disabled{ color:#484f58; }
    QPushButton[accent="true"] { border-color:#1f6feb; color:#79c0ff; }
"""


class CropDialog(QDialog):
    """
    Multi-image, multi-crop dialog.

    raw_images : list of absolute paths to the raw images (all images of the article).
    out_dir    : folder where processed crops are saved.
    prefix     : filename prefix for saved crops (e.g. "dna_origami_small").
    on_saved   : called on Done with list of dicts {image_path, crop_path, mask_path}.
    """

    MAX_DISPLAY = 620

    def __init__(
        self,
        raw_images: list[str],
        out_dir: str,
        prefix: str,
        on_saved: Callable[[list[dict]], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._raw_images = [Path(p) for p in raw_images]
        self._out_dir    = Path(out_dir)
        self._prefix     = prefix
        self._on_saved   = on_saved
        self._img_idx    = 0
        self._bgr: np.ndarray | None = None
        self._crops: list[dict] = []          # saved crops so far
        self._next_crop_idx = self._init_next_idx()

        self.setWindowTitle("Crop tool — article images")
        self.setModal(True)
        self._build_ui()
        self._load_current()

    def _init_next_idx(self) -> int:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        existing = [p for p in self._out_dir.iterdir()
                    if p.suffix == ".png" and p.name.startswith(self._prefix)
                    and not p.name.startswith("mask_")]
        return len(existing)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Top hint
        hint = QLabel(
            "Draw → adjust corners/edges → Save Crop.  "
            "Navigate images with ◀ ▶.  "
            "Enter = Save Crop,  Esc = Done."
        )
        hint.setStyleSheet("color:#8b949e; font-size:11px;")
        root.addWidget(hint)

        # Main area: canvas + saved list
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: image navigation + canvas
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)

        # Image nav bar
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_next = QPushButton("Next ▶")
        self._lbl_img  = QLabel()
        self._lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_img.setStyleSheet("color:#c9d1d9; font-size:11px;")
        for btn in (self._btn_prev, self._btn_next):
            btn.setFixedWidth(80)
            btn.setStyleSheet(_BTN)
        self._btn_prev.clicked.connect(self._prev_image)
        self._btn_next.clicked.connect(self._next_image)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._lbl_img, 1)
        nav.addWidget(self._btn_next)
        lv.addLayout(nav)

        # Canvas inside scroll
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setStyleSheet("background:#1a1f27; border:1px solid #30363d;")
        lv.addWidget(self._scroll)

        # Selection info
        self._lbl_sel = QLabel("No selection")
        self._lbl_sel.setStyleSheet("color:#8b949e; font-size:11px;")
        lv.addWidget(self._lbl_sel)

        splitter.addWidget(left)

        # Right: saved crops list
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)
        _lbl_crops = QLabel("Saved crops:")
        _lbl_crops.setStyleSheet("color:#8b949e; font-size:11px;")
        rv.addWidget(_lbl_crops)
        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget { background:#161b22; color:#c9d1d9; "
            "border:1px solid #30363d; font-size:11px; }"
            "QListWidget::item:selected { background:#1f6feb; }"
        )
        rv.addWidget(self._list)
        splitter.addWidget(right)
        splitter.setSizes([700, 220])

        root.addWidget(splitter, 1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._lbl_count = QLabel("0 crops saved")
        self._lbl_count.setStyleSheet("color:#8b949e; font-size:11px;")
        btn_row.addWidget(self._lbl_count)
        btn_row.addStretch()

        for label, slot, accent in [
            ("Reset",      self._reset,      False),
            ("Save Crop",  self._save_crop,  True),
            ("Done",       self._done,       False),
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(95 if label == "Save Crop" else 80)
            btn.setStyleSheet(_BTN)
            if accent:
                btn.setProperty("accent", "true")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        root.addLayout(btn_row)

        self.resize(1000, 680)

    # ── image loading ─────────────────────────────────────────────────────────

    def _load_current(self) -> None:
        path = self._raw_images[self._img_idx]
        bgr  = cv2.imread(str(path))
        if bgr is None:
            try:
                from PIL import Image as _PIL
                bgr = cv2.cvtColor(np.array(_PIL.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        self._bgr = bgr

        self._lbl_img.setText(
            f"{self._img_idx + 1} / {len(self._raw_images)}  —  {path.name}"
        )
        self._btn_prev.setEnabled(self._img_idx > 0)
        self._btn_next.setEnabled(self._img_idx < len(self._raw_images) - 1)

        if bgr is None:
            self._lbl_sel.setText("Could not load image.")
            return

        h, w = bgr.shape[:2]
        scale = min(1.0, self.MAX_DISPLAY / max(h, w))
        px    = QPixmap.fromImage(_cv2_to_qimage(
            cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        ))
        if hasattr(self, "_canvas"):
            self._canvas.load_image(bgr, scale)
        else:
            self._canvas = _CropCanvas(px, scale, self._update_sel_label)
            self._scroll.setWidget(self._canvas)

    def _prev_image(self) -> None:
        if self._img_idx > 0:
            self._img_idx -= 1
            self._load_current()

    def _next_image(self) -> None:
        if self._img_idx < len(self._raw_images) - 1:
            self._img_idx += 1
            self._load_current()

    # ── selection label ───────────────────────────────────────────────────────

    def _update_sel_label(self) -> None:
        r = self._canvas.selection_in_original()
        if r:
            self._lbl_sel.setText(f"Selection: ({r.x()}, {r.y()})  {r.width()} × {r.height()} px")
        else:
            self._lbl_sel.setText("No selection")

    # ── actions ───────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._canvas.reset()

    def _save_crop(self) -> None:
        if self._bgr is None:
            return
        r = self._canvas.selection_in_original()
        h, w = self._bgr.shape[:2]
        if r and r.width() >= 4 and r.height() >= 4:
            x1, y1 = max(0, r.x()), max(0, r.y())
            x2, y2 = min(w, r.x() + r.width()), min(h, r.y() + r.height())
            crop = self._bgr[y1:y2, x1:x2]
        else:
            crop = self._bgr

        result_bgr, mask = _process_crop(crop)

        img_name  = f"{self._prefix}_{self._next_crop_idx:03d}.png"
        mask_name = f"mask_{self._prefix}_{self._next_crop_idx:03d}.png"
        crop_path = self._out_dir / img_name
        mask_path = self._out_dir / mask_name

        cv2.imwrite(str(crop_path), result_bgr)
        cv2.imwrite(str(mask_path), mask)

        src_name = self._raw_images[self._img_idx].name
        self._crops.append({
            "image_path": str(self._raw_images[self._img_idx]),
            "crop_path":  str(crop_path),
            "mask_path":  str(mask_path),
            "crop_file":  img_name,
            "mask_file":  mask_name,
        })
        self._next_crop_idx += 1
        self._list.addItem(QListWidgetItem(f"[{self._next_crop_idx-1}] {img_name}  ← {src_name}"))
        self._lbl_count.setText(f"{len(self._crops)} crop(s) saved")
        self._canvas.reset()

    def _done(self) -> None:
        self._on_saved(self._crops)
        self.accept()

    def keyPressEvent(self, ev) -> None:
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._save_crop()
        elif ev.key() == Qt.Key.Key_Escape:
            self._done()
        elif ev.key() == Qt.Key.Key_Left:
            self._prev_image()
        elif ev.key() == Qt.Key.Key_Right:
            self._next_image()
        else:
            super().keyPressEvent(ev)

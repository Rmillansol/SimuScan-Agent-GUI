"""
PID Topography Tuner — interactive popup to tune PID simulation parameters.

Shows a synthetic pillar (512×512) processed through simulate_pid_topography,
with side-by-side images (original / PID) and a central horizontal profile.
"""
from __future__ import annotations

import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QVBoxLayout, QFormLayout,
    QDoubleSpinBox, QSpinBox, QLabel, QGroupBox, QComboBox, QWidget,
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ── Synthetic pillar ──────────────────────────────────────────────────────────

def _make_pillar(size: int = 512, radius: int = 60, height: float = 1.0) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    cx = cy = size // 2
    Y, X = np.ogrid[:size, :size]
    img[(X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2] = height
    return img


# ── Dialog ────────────────────────────────────────────────────────────────────

class PIDTunerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PID Topography Tuner")
        self.resize(1200, 680)
        self._build_ui()
        self._update()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── Parameter panel ───────────────────────────────────────────────
        param_widget = QWidget()
        param_widget.setFixedWidth(270)
        pv = QVBoxLayout(param_widget)
        pv.setSpacing(6)

        # Pillar geometry
        grp_pillar = QGroupBox("Pillar")
        fl = QFormLayout(grp_pillar)
        self.sb_radius = self._ispin(15, 1, 63)
        self.sb_height = self._fspin(1.0, 0.01, 999.0, 0.1)
        fl.addRow("Radius (px)", self.sb_radius)
        fl.addRow("Height", self.sb_height)

        # Tapping-mode physical params
        grp_phys = QGroupBox("Physical — tapping mode")
        fl2 = QFormLayout(grp_phys)
        self.sb_A0   = self._fspin(1.0,  0.01, 999.0, 0.1)
        self.sb_beta = self._fspin(1.0,  0.01, 999.0, 0.1)
        self.sb_z0   = self._fspin(2.0,  0.01, 999.0, 0.1)
        self.sb_ASP  = self._fspin(0.5,  0.001, 0.999, 0.01, dec=3)
        fl2.addRow("A0  (free amplitude)", self.sb_A0)
        fl2.addRow("β   (sensitivity)", self.sb_beta)
        fl2.addRow("z0  (decay dist.)", self.sb_z0)
        fl2.addRow("ASP (setpoint)", self.sb_ASP)

        # PID control params
        grp_pid = QGroupBox("PID Control")
        fl3 = QFormLayout(grp_pid)
        self.sb_Kp        = self._fspin(0.5,  0.0, 999.0, 0.05)
        self.sb_Ki        = self._fspin(0.1,  0.0, 999.0, 0.01)
        self.sb_Kd        = self._fspin(0.05, 0.0, 999.0, 0.01)
        self.sb_dt        = self._fspin(1.0,  0.01, 999.0, 0.1)
        self.sb_max_speed = self._fspin(1.0,  0.0, 999.0, 0.05)
        self.cb_dir       = QComboBox()
        self.cb_dir.addItems(["forward", "backward"])
        fl3.addRow("Kp", self.sb_Kp)
        fl3.addRow("Ki", self.sb_Ki)
        fl3.addRow("Kd", self.sb_Kd)
        fl3.addRow("dt", self.sb_dt)
        fl3.addRow("max_speed", self.sb_max_speed)
        fl3.addRow("Direction", self.cb_dir)

        # Artifact params
        grp_art = QGroupBox("Artifacts")
        fl4 = QFormLayout(grp_art)
        self.sb_sticky = self._fspin(0.0, 0.0, 1.0,   0.005, dec=3)
        self.sb_jprob  = self._fspin(0.0, 0.0, 1.0,   0.005, dec=3)
        self.sb_jmag   = self._fspin(1.0, 0.0, 999.0, 0.1)
        fl4.addRow("Stickiness prob", self.sb_sticky)
        fl4.addRow("Jump prob", self.sb_jprob)
        fl4.addRow("Jump magnitude", self.sb_jmag)

        pv.addWidget(grp_pillar)
        pv.addWidget(grp_phys)
        pv.addWidget(grp_pid)
        pv.addWidget(grp_art)
        pv.addStretch()

        # Auto-update on every change
        for w in (self.sb_radius, self.sb_height,
                  self.sb_A0, self.sb_beta, self.sb_z0, self.sb_ASP,
                  self.sb_Kp, self.sb_Ki, self.sb_Kd, self.sb_dt, self.sb_max_speed,
                  self.sb_sticky, self.sb_jprob, self.sb_jmag):
            w.valueChanged.connect(self._update)
        self.cb_dir.currentIndexChanged.connect(self._update)

        # ── Matplotlib canvas ─────────────────────────────────────────────
        self.fig = Figure(figsize=(9, 6))
        self.canvas = FigureCanvas(self.fig)

        root.addWidget(param_widget)
        root.addWidget(self.canvas, stretch=1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fspin(self, val, mn, mx, step, dec=2):
        sb = QDoubleSpinBox()
        sb.setRange(mn, mx)
        sb.setSingleStep(step)
        sb.setDecimals(dec)
        sb.setValue(val)
        return sb

    def _ispin(self, val, mn, mx):
        sb = QSpinBox()
        sb.setRange(mn, mx)
        sb.setValue(val)
        return sb

    # ── Render ────────────────────────────────────────────────────────────────

    def _update(self):
        from src.simuscan.noisegenerator import simulate_pid_topography  # lazy

        original = _make_pillar(128, self.sb_radius.value(), self.sb_height.value())
        pid_img  = simulate_pid_topography(
            original,
            Kp             = self.sb_Kp.value(),
            Ki             = self.sb_Ki.value(),
            Kd             = self.sb_Kd.value(),
            dt             = self.sb_dt.value(),
            max_speed      = self.sb_max_speed.value(),
            scan_direction = self.cb_dir.currentText(),
            stickiness_prob= self.sb_sticky.value(),
            jump_prob      = self.sb_jprob.value(),
            jump_magnitude = self.sb_jmag.value(),
            A0             = self.sb_A0.value(),
            beta           = self.sb_beta.value(),
            z0             = self.sb_z0.value(),
            ASP            = self.sb_ASP.value(),
        )

        self.fig.clear()
        gs  = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.15,
                                    left=0.06, right=0.97, top=0.95, bottom=0.08)
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[1, :])

        vmin, vmax = 0.0, float(self.sb_height.value())

        ax1.imshow(original, cmap="afmhot", vmin=vmin, vmax=vmax, origin="upper")
        ax1.set_title("Original", fontsize=9)
        ax1.axis("off")

        ax2.imshow(pid_img, cmap="afmhot", vmin=vmin, vmax=vmax, origin="upper")
        ax2.set_title("PID output", fontsize=9)
        ax2.axis("off")

        size = original.shape[0]
        cx  = size // 2
        xs  = np.arange(size)
        ax3.plot(xs, original[cx, :], lw=1.5, label="Ideal")
        ax3.plot(xs, pid_img[cx, :],  lw=1.5, alpha=0.85, label="PID")
        ax3.set_xlabel("Pixel X", fontsize=8)
        ax3.set_ylabel("Height", fontsize=8)
        ax3.set_title("Central horizontal profile  (row 256)", fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, lw=0.5)
        ax3.tick_params(labelsize=7)

        self.canvas.draw()


# ── Public entry point ────────────────────────────────────────────────────────

# Keep a module-level reference so the dialog is not garbage-collected
_active_tuner: PIDTunerDialog | None = None


def open_pid_tuner_on_main_thread():
    """Called on the Qt main thread via ui signal. Creates and shows the dialog."""
    global _active_tuner
    _active_tuner = PIDTunerDialog()
    _active_tuner.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    _active_tuner.show()


def launch_pid_tuner():
    """Request the main thread to open the PID tuner via the ui bridge signal."""
    from src.bridge.ui_bridge import ui
    ui.open_pid_tuner()

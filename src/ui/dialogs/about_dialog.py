"""src/ui/dialogs/about_dialog.py"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QFrame, QHBoxLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

_STYLE = """
QDialog { background:#0e1117; }
QLabel  { color:#c9d1d9; font-family:'Segoe UI',sans-serif; font-size:13px; }
QPushButton { background:#1f6feb; color:#fff; border:none; border-radius:6px;
              padding:7px 28px; font-family:'Consolas',monospace;
              font-size:13px; font-weight:bold; }
QPushButton:hover { background:#388bfd; }
"""

class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About SimuScan")
        self.setFixedSize(460, 340)
        self.setStyleSheet(_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(12)

        name = QLabel("SimuScan Agent")
        name.setFont(QFont("Consolas", 20))
        name.setStyleSheet("color:#58a6ff; font-weight:bold;")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name)

        ver = QLabel("v1.0.0  —  Synthetic AFM Dataset Generator & AI Agent")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ver.setStyleSheet("color:#8b949e; font-size:12px; font-family:'Consolas',monospace;")
        layout.addWidget(ver)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#30363d;"); layout.addWidget(sep)

        desc = QLabel(
            "SimuScan generates fully labeled synthetic Atomic Force Microscopy "
            "(AFM) datasets for training deep learning models. The integrated AI "
            "agent orchestrates configuration, dataset generation, YOLO training, "
            "triplet encoder training, and real AFM microscope automation via "
            "natural language."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignJustify)
        desc.setStyleSheet("color:#8b949e; font-size:12px;")
        layout.addWidget(desc)
        layout.addStretch()

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color:#30363d;"); layout.addWidget(sep2)

        footer = QLabel(
            "Built with Python · PyQt6 · FastMCP · OpenAI · Ultralytics YOLO\n"
            "Oak Ridge National Laboratory  ·  2025"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color:#3d4449; font-size:11px; font-family:'Consolas',monospace;")
        layout.addWidget(footer)

        bl = QHBoxLayout()
        ok = QPushButton("Close")
        ok.clicked.connect(self.accept)
        bl.addStretch(); bl.addWidget(ok); bl.addStretch()
        layout.addLayout(bl)

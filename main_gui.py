"""
main_gui.py
───────────
SimuScan Agent — PyQt6 GUI entry point.

Author: Rubén Millán-Solsona

Usage:
    python main_gui.py
    uv run python main_gui.py

Build exe:
    pyinstaller simuscan.spec
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")  # make .env vars available to os.environ before any import

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.ui.main_window import MainWindow


def main() -> None:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("SimuScan Agent")
    app.setOrganizationName("ORNL")
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

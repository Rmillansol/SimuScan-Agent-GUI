from __future__ import annotations

from pathlib import Path
import os
import platform
import sys


def _exe_dir() -> Path:
    # PyInstaller frozen: sys.executable points to the binary
    return Path(sys.executable).resolve().parent


def resource_root() -> Path:
    """
    Returns the folder where the resources live (config, STL_files).
    In PyInstaller onedir it can be:
      - next to the exe
      - or inside _internal (depending on how you package)
    In onefile (if you ever use it) it is usually sys._MEIPASS.
    In dev: repo root (the folder containing /config and /STL_files).
    """
    # 1) PyInstaller onefile (or some onedir) case: _MEIPASS exists
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(meipass).resolve()
        # if the resources are there, great
        if (base / "data" / "config").exists() or (base / "data" / "STL_files").exists():
            return base

    # 2) Frozen onedir case: next to the exe or in _internal
    if getattr(sys, "frozen", False):
        exe_dir = _exe_dir()
        candidates = [
            exe_dir,
            exe_dir / "_internal",  # typical on Windows
            exe_dir.parent,         # in case the exe is in a subfolder
        ]
        for c in candidates:
            if (c / "data" / "config").exists() or (c / "data" / "STL_files").exists():
                return c
        # reasonable fallback
        return exe_dir

    # 3) Dev case: this file is in /synthetic/paths.py
    # repo_root = .../SimuScan (where config/, STL_files/ exist)
    return Path(__file__).resolve().parent.parent.parent


def config_dir() -> Path:
    return resource_root() / "data" / "config"


def stl_dir() -> Path:
    return resource_root()  / "data" / "STL_files"


def default_yaml() -> Path:
    return config_dir() / "default.yaml"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_data_root() -> Path:
    """
    Base folder where datasets are saved (ALWAYS outside the bundle).
    """
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        # USERPROFILE\Documents\SimuScan
        return home / "Documents" / "SimuScan"
    elif system == "Darwin":
        # ~/Documents/SimuScan
        return home / "Documents" / "SimuScan"
    else:
        return home / "SimuScan"


def datasets_root() -> Path:
    return user_data_root() / "Datasets"


def ref_objects_dir() -> Path:
    """ref_objects/ folder at the repo root. Class image folders live directly here."""
    return resource_root() / "ref_objects"

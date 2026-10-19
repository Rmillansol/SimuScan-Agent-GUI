"""
src/trainer/triplet_explorer.py
────────────────────────────────
Interactive UMAP explorer for a triplet embedding library.

Opens a PyQt6 window with three tabs (combined descriptor / shape features /
embedding UMAPs).  Clicking a point shows the corresponding crop image and
metadata on the right panel.

Public API
----------
show_triplet_library_explorer(library_run_name=None) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# ── Path helpers ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _out_lib_root() -> Path:
    return _repo_root() / "data" / "out_triplet_lib"


def _out_triplet_root() -> Path:
    return _repo_root() / "data" / "out_triplet"


def _ok(data=None, message: str = "OK") -> dict:
    r: dict = {"status": "ok", "message": message}
    if data is not None:
        r["data"] = data
    return r


def _err(message: str) -> dict:
    return {"status": "error", "message": message}


def _latest_library_dir() -> Optional[Path]:
    """Most recent dir with a library.json, across both library and training roots."""
    candidates: list[Path] = []
    for root in (_out_lib_root(), _out_triplet_root()):
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.is_dir() and (p / "library.json").is_file():
                candidates.append(p)
    if not candidates:
        return None
    # Sort by folder name (time_<stamp>) newest-first
    return sorted(candidates, key=lambda p: p.name, reverse=True)[0]


def _resolve_explorer_dir(run_name: Optional[str]) -> Optional[Path]:
    """
    Resolve a run name to a directory containing library.json.
    Searches data/out_triplet_lib/ first, then data/out_triplet/ (training runs).
    If run_name is None, returns the most recent across both roots.
    """
    if not run_name:
        return _latest_library_dir()
    for root in (_out_lib_root(), _out_triplet_root()):
        cand = root / run_name
        if cand.is_dir() and (cand / "library.json").is_file():
            return cand
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Explorer window
# ══════════════════════════════════════════════════════════════════════════════

def _run_explorer(lib_dir: Path) -> None:
    """Build and exec the Qt window — must be called from the main thread."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

    from PyQt6.QtWidgets import (
        QDialog, QHBoxLayout, QLabel,
        QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap, QImage

    # ── Load library data ──────────────────────────────────────────────────────
    with open(lib_dir / "library.json") as f:
        records = json.load(f)

    n = len(records)
    # embeddings and features
    emb_dim = len([k for k in records[0] if k.startswith("emb_")]) if records else 0
    embs  = np.array([[r["embedding"][i]  for i in range(len(r["embedding"]))]  for r in records], dtype=np.float32)
    feats = np.array([[r["shape_features"].get(k, 0.0)
                       for k in ["area","eccentricity","major_axis","minor_axis",
                                 "solidity","circularity","mean_intensity","std_intensity"]]
                      for r in records], dtype=np.float32)
    descs = np.array([r["descriptor"] for r in records], dtype=np.float32)

    classes  = np.array([r["yolo_class"]      for r in records])
    cnames   = [r.get("yolo_class_name", f"cls{r['yolo_class']}") for r in records]
    clusters = np.array([r["cluster"]          for r in records])

    # ── Compute UMAPs ──────────────────────────────────────────────────────────
    try:
        import umap as umap_lib
    except ImportError:
        raise RuntimeError("umap-learn is not installed. Run: pip install umap-learn")

    def _clean(data: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with column mean (or 0 if all NaN)."""
        out = data.copy()
        for col in range(out.shape[1]):
            bad = ~np.isfinite(out[:, col])
            if bad.any():
                good_vals = out[~bad, col]
                out[bad, col] = good_vals.mean() if len(good_vals) > 0 else 0.0
        return out

    def _umap(data: np.ndarray) -> np.ndarray:
        reducer = umap_lib.UMAP(n_neighbors=min(15, n - 1),
                                min_dist=0.1, random_state=42)
        return reducer.fit_transform(_clean(data))

    proj_desc  = _umap(descs)
    proj_feats = _umap(feats)
    proj_embs  = _umap(embs)

    # ── Color maps ────────────────────────────────────────────────────────────
    unique_classes  = sorted(set(int(c) for c in classes))
    unique_clusters = sorted(set(int(c) for c in clusters))
    cmap_cls = plt.get_cmap("tab20", max(len(unique_classes), 1))
    cmap_clu = plt.get_cmap("tab10", max(len(unique_clusters), 1))
    cls_color_map = {c: cmap_cls(i) for i, c in enumerate(unique_classes)}
    clu_color_map = {c: cmap_clu(i) for i, c in enumerate(unique_clusters)}

    # ── Qt dialog (QApplication already running on main thread) ──────────────
    dlg = QDialog()
    dlg.setWindowTitle(f"Triplet Library Explorer — {lib_dir.name}")
    dlg.resize(1300, 700)
    dlg.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")

    main_layout = QHBoxLayout(dlg)

    # Left: tab widget with 3 UMAP tabs
    tabs = QTabWidget()
    tabs.setMinimumWidth(750)
    tabs.setStyleSheet("""
        QTabWidget::pane { border: 1px solid #45475a; background: #1e1e2e; }
        QTabBar::tab { background: #313244; color: #cdd6f4; padding: 6px 14px; }
        QTabBar::tab:selected { background: #89b4fa; color: #1e1e2e; }
    """)

    # Right: crop + metadata panel
    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_panel.setFixedWidth(480)
    right_panel.setStyleSheet("background-color: #181825;")

    crop_label = QLabel()
    crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    crop_label.setFixedSize(300, 300)
    crop_label.setStyleSheet("border: 1px solid #45475a; background: #11111b;")
    crop_label.setText("Click a point")

    meta_label = QLabel()
    meta_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    meta_label.setWordWrap(True)
    meta_label.setStyleSheet("color: #cdd6f4; font-size: 12px; padding: 8px;")

    right_layout.addWidget(crop_label, alignment=Qt.AlignmentFlag.AlignHCenter)
    right_layout.addWidget(meta_label)
    right_layout.addStretch()

    # ── Build one UMAP tab ────────────────────────────────────────────────────
    def _make_tab(proj: np.ndarray, color_by: str, title: str):
        """color_by: 'class' or 'cluster'"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        ax.xaxis.label.set_color("#cdd6f4")
        ax.yaxis.label.set_color("#cdd6f4")
        ax.title.set_color("#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        scatter_collections = []
        point_indices = []   # parallel list: collection_idx → list of original record indices

        if color_by == "class":
            for cls_id in unique_classes:
                mask = classes == cls_id
                idxs = np.where(mask)[0].tolist()
                sc = ax.scatter(
                    proj[mask, 0], proj[mask, 1],
                    s=18, alpha=0.8, picker=6,
                    color=cls_color_map[cls_id],
                    label=f"{cnames[idxs[0]]} (n={mask.sum()})",
                )
                scatter_collections.append((sc, idxs, proj[mask]))
        else:
            for clu_id in unique_clusters:
                mask = clusters == clu_id
                idxs = np.where(mask)[0].tolist()
                sc = ax.scatter(
                    proj[mask, 0], proj[mask, 1],
                    s=18, alpha=0.8, picker=6,
                    color=clu_color_map[clu_id],
                    label=f"cluster {clu_id} (n={mask.sum()})",
                )
                scatter_collections.append((sc, idxs, proj[mask]))

        ax.legend(fontsize=7, markerscale=1.5, facecolor="#313244",
                  labelcolor="#cdd6f4", edgecolor="#45475a")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(canvas)

        # ── Click handler ──────────────────────────────────────────────────────
        def _on_pick(event):
            if not hasattr(event, "ind") or len(event.ind) == 0:
                return
            # Find which collection was picked
            for sc, idxs, sub_proj in scatter_collections:
                if event.artist is sc:
                    local_idx = event.ind[0]
                    record_idx = idxs[local_idx]
                    _show_record(record_idx)
                    break

        canvas.mpl_connect("pick_event", _on_pick)
        return widget

    # ── Show record in right panel ─────────────────────────────────────────────
    def _show_record(idx: int):
        r = records[idx]
        crop_path = Path(r.get("crop_path", ""))
        if crop_path.is_file():
            img = QImage(str(crop_path))
            pix = QPixmap.fromImage(img).scaled(
                300, 300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            crop_label.setPixmap(pix)
        else:
            crop_label.setText("No crop image")

        bbox = r.get("bbox", ["-", "-", "-", "-"])
        sf   = r.get("shape_features", {})
        meta_label.setText(
            f"<b>obj_id</b>: {r.get('obj_id', idx)}<br>"
            f"<b>class</b>: {r.get('yolo_class_name', r.get('yolo_class'))}<br>"
            f"<b>cluster</b>: {r.get('cluster')}<br>"
            f"<b>novelty</b>: {r.get('novelty_score', ''):.4f}<br>"
            f"<b>conf</b>: {r.get('yolo_conf', ''):.3f}<br>"
            f"<b>bbox</b>: {bbox}<br>"
            f"<b>area</b>: {sf.get('area',''):.1f} px²<br>"
            f"<b>eccentricity</b>: {sf.get('eccentricity',''):.3f}<br>"
            f"<b>solidity</b>: {sf.get('solidity',''):.3f}<br>"
            f"<b>circularity</b>: {sf.get('circularity',''):.3f}<br>"
            f"<b>img</b>: {Path(r.get('img_path','')).name}"
        )

    # Build tabs
    tabs.addTab(_make_tab(proj_desc,  "class",   "UMAP — combined descriptor (by class)"),  "Descriptor")
    tabs.addTab(_make_tab(proj_feats, "class",   "UMAP — shape features (by class)"),        "Features")
    tabs.addTab(_make_tab(proj_embs,  "cluster", "UMAP — embedding (by cluster)"),           "Embedding")

    main_layout.addWidget(tabs)
    main_layout.addWidget(right_panel)

    dlg.exec()


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def show_triplet_library_explorer(library_run_name: Optional[str] = None) -> dict:
    """
    Open an interactive UMAP explorer for a triplet embedding library.

    Parameters
    ----------
    library_run_name : str | None
        Folder name inside data/out_triplet_lib/ (experimental library) OR
        data/out_triplet/ (training run), e.g. 'time_20260618_120842'.
        If None, the most recent run with a library.json across both is used.
    """
    lib_dir = _resolve_explorer_dir(library_run_name)

    if lib_dir is None or not lib_dir.is_dir():
        return _err(f"Library not found: '{library_run_name or 'latest'}'")
    if not (lib_dir / "library.json").is_file():
        return _err(f"library.json missing in '{lib_dir}'")

    from src.bridge.ui_bridge import ui
    ui.log("INFO", f"Opening UMAP explorer — {lib_dir.name}")

    try:
        _run_explorer(lib_dir)
    except Exception as exc:
        return _err(f"Explorer error: {exc}")

    return _ok(message=f"Explorer closed — {lib_dir.name}")

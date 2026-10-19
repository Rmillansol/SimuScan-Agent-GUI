"""
src/trainer/triplet_visualizer.py
──────────────────────────────────
Visualisation and inspection utilities for trained TripletEncoder models.

Public API
----------
show_crop_grid(run_dir, class_idx, max_items)  → shows grid of crops
show_embedding_umap(run_dir)                   → interactive UMAP plot
get_triplet_model_info(model_path)             → dict with model metadata
find_latest_triplet_pt()                       → str | None
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.bridge.ui_bridge import ui


# ── Path helpers ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _out_triplet_root() -> Path:
    return _repo_root() / "data" / "out_triplet"


def _ok(data=None, message: str = "OK") -> dict:
    r: dict = {"status": "ok", "message": message}
    if data is not None:
        r["data"] = data
    return r


def _err(message: str, detail=None) -> dict:
    r: dict = {"status": "error", "message": message}
    if detail is not None:
        r["detail"] = detail
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Auto-select latest run
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_triplet_pt() -> Optional[str]:
    """Return path to best triplet_encoder.pt in most recent run, or None."""
    root = _out_triplet_root()
    if not root.exists():
        return None
    for p in sorted(root.iterdir(), reverse=True):
        if p.is_dir() and p.name.startswith("time_"):
            pt = p / "triplet_encoder.pt"
            if pt.is_file():
                return str(pt)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Model info
# ══════════════════════════════════════════════════════════════════════════════

def get_triplet_model_info(model_path: Optional[str] = None) -> dict:
    """
    Return metadata about a saved TripletEncoder .pt file.

    Parameters
    ----------
    model_path : str | None   Path to a .pt file. Auto-selects latest if None.
    """
    resolved = model_path or find_latest_triplet_pt()
    if resolved is None:
        return _err("No trained triplet model found. Train one first.")
    pt_path = Path(resolved)
    if not pt_path.is_file():
        return _err(f"Model file not found: '{resolved}'")

    # Try to infer latent dim from state dict (last Linear layer of fc Sequential)
    try:
        import torch
        state = torch.load(pt_path, map_location="cpu")
        latent_dim = None
        # fc is now a Sequential; last weight tensor has shape [latent_dim, prev_dim]
        fc_weights = [(k, v) for k, v in state.items()
                      if k.startswith("fc.") and k.endswith(".weight")]
        if fc_weights:
            # sort by layer index so we pick the last one
            fc_weights.sort(key=lambda kv: kv[0])
            latent_dim = int(fc_weights[-1][1].shape[0])
    except Exception:
        latent_dim = None

    # Check for companion metrics
    run_dir     = pt_path.parent
    metrics     = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception:
            pass

    size_mb = round(pt_path.stat().st_size / 1e6, 2)
    return _ok(
        data={
            "model_path"       : str(pt_path),
            "run_dir"          : str(run_dir),
            "size_mb"          : size_mb,
            "latent_dim"       : latent_dim,
            "n_crops"          : metrics.get("n_crops"),
            "n_classes"        : metrics.get("n_classes"),
            "class_names"      : metrics.get("class_names"),
            "final_val_acc"    : metrics.get("final_val_acc"),
            "final_val_loss"   : metrics.get("final_val_loss"),
            "silhouette_global": metrics.get("silhouette_global"),
            "knn_accuracy"     : metrics.get("knn_accuracy"),
        },
        message=f"Model: {pt_path.name}  ({size_mb} MB)",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Crop grid
# ══════════════════════════════════════════════════════════════════════════════

def show_crop_grid(
    run_dir     : str,
    class_idx   : Optional[int] = None,
    max_items   : int           = 64,
    cols        : int           = 8,
    save_path   : Optional[str] = None,
) -> dict:
    """
    Display a grid of crop images from a triplet training run.

    Parameters
    ----------
    run_dir   : Path to a time_<stamp> triplet run directory.
    class_idx : Show only this class (None = show all classes).
    max_items : Maximum number of crops to display.
    cols      : Number of columns in the grid.
    save_path : If given, save figure here instead of plt.show().
    """
    crop_dir = Path(run_dir) / "crops"
    if not crop_dir.is_dir():
        return _err(f"Crops folder not found: '{crop_dir}'")

    patterns = (
        [f"*_cls{class_idx}.png"] if class_idx is not None
        else ["*.png"]
    )
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(crop_dir.glob(pat)))
    if not files:
        return _err(f"No crop images found in '{crop_dir}'"
                    + (f" for class {class_idx}" if class_idx is not None else ""))

    files = files[:max_items]
    rows  = max(1, int(np.ceil(len(files) / cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes      = np.array(axes).reshape(rows, cols)

    for i, fp in enumerate(files):
        r, c = divmod(i, cols)
        img  = plt.imread(str(fp))
        axes[r, c].imshow(img, cmap="afmhot", vmin=0, vmax=1)
        axes[r, c].set_title(fp.stem[-12:], fontsize=4)
        axes[r, c].axis("off")

    for i in range(len(files), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    title = f"Crop Grid — {Path(run_dir).name}"
    if class_idx is not None:
        title += f"  class {class_idx}"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        ui.log("OK", f"Saved: {save_path}")
        return _ok(data={"saved_to": save_path, "n_crops": len(files)})
    else:
        ui.plot(fig)
        return _ok(data={"n_crops": len(files)})


# ══════════════════════════════════════════════════════════════════════════════
# UMAP from saved embeddings CSV
# ══════════════════════════════════════════════════════════════════════════════

def show_embedding_umap(
    run_dir    : str,
    save_path  : Optional[str] = None,
) -> dict:
    """
    Generate (or re-generate) a UMAP plot from the saved embeddings.csv.

    Parameters
    ----------
    run_dir   : Path to a time_<stamp> triplet run directory.
    save_path : If given, save figure here instead of plt.show().
    """
    import pandas as pd

    emb_csv = Path(run_dir) / "results" / "embeddings.csv"
    if not emb_csv.is_file():
        return _err(f"embeddings.csv not found in '{run_dir}/results/'")

    try:
        import umap
    except ImportError:
        return _err("umap-learn is not installed. "
                    "Run: pip install umap-learn")

    df      = pd.read_csv(emb_csv)
    emb_col = [c for c in df.columns if c.startswith("emb_")]
    if not emb_col:
        return _err("No embedding columns (emb_*) found in embeddings.csv")

    embs    = df[emb_col].values.astype(np.float32)
    labels  = df["label"].values
    names   = df["class_name"].values if "class_name" in df.columns else labels.astype(str)

    reducer = umap.UMAP(n_neighbors=min(15, len(embs) - 1),
                        min_dist=0.1, random_state=42)
    proj    = reducer.fit_transform(embs)

    unique  = np.unique(labels)
    cmap    = plt.get_cmap("tab20", len(unique))
    label_to_name = {int(l): n for l, n in zip(labels, names)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(unique):
        idx  = labels == cls
        name = label_to_name.get(int(cls), f"cls{cls}")
        ax.scatter(proj[idx, 0], proj[idx, 1], s=12, alpha=0.7,
                   color=cmap(i), label=f"{name} (n={idx.sum()})")

    ax.legend(markerscale=2, fontsize=7, loc="best")
    ax.set_title(f"UMAP — {Path(run_dir).name}")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        ui.log("OK", f"Saved: {save_path}")
        return _ok(data={"saved_to": save_path, "n_points": int(len(embs))})
    else:
        ui.plot(fig)
        return _ok(data={"n_points": int(len(embs))})
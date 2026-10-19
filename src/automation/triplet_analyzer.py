"""
src/automation/triplet_analyzer.py
────────────────────────────────────
Post-session Triplet embedding analysis for AFM exploration results.

Given an exploration output directory (produced by ExplorationSession.run())
and a path to a trained TripletEncoder .pt file, this module:

  1. Reads all *_flat.png crops from library/crops/.
  2. Prepares each crop the same way as _crop_and_normalise() in triplet_trainer:
       - grayscale float32
       - p2/p98 normalisation over object pixels (non-zero)
       - background = 0
       - resize to 64×64
  3. Loads TripletEncoder and computes embeddings.
  4. Computes per-class centroids and Euclidean novelty scores.
  5. Generates:
       library/grid_<class>_typical.png   (sorted by distance to centroid ASC)
       library/grid_<class>_novel.png     (sorted by distance to centroid DESC)
       library/umap.png
       library/novelty.png
       library/library_objects.csv
       library/library_summary.csv

Public API
──────────
TripletAnalyzer(out_dir, triplet_model_path, class_names)
  .run() → dict {"status": "ok"|"error", ...}
"""

from __future__ import annotations

import csv
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Encoder input size — must match what was used during training
_CROP_SIZE = 64


# ══════════════════════════════════════════════════════════════════════════════
# Crop preparation  (mirrors _crop_and_normalise from triplet_trainer.py)
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_crop(flat_png_path: str) -> Optional[np.ndarray]:
    """
    Load a *_flat.png crop and prepare it for the TripletEncoder:
      - convert to float32 grayscale
      - normalise p2/p98 over object pixels (non-zero)
      - background (zero pixels) stays 0
      - resize to _CROP_SIZE × _CROP_SIZE

    Returns float32 array (_CROP_SIZE, _CROP_SIZE) or None on failure.
    Logs the reason for rejection via warnings.warn so it appears in the
    exploration output and session_summary.
    """
    name = Path(flat_png_path).name
    if not _HAS_CV2:
        warnings.warn(f"[TripletAnalyzer] Skipping '{name}': cv2 not available.")
        return None
    try:
        img = cv2.imread(flat_png_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            warnings.warn(f"[TripletAnalyzer] Skipping '{name}': cv2.imread returned None (file unreadable or missing).")
            return None
        patch = img.astype(np.float32) / 255.0

        obj_pix = patch[patch > 0]
        if len(obj_pix) < 10:
            warnings.warn(
                f"[TripletAnalyzer] Skipping '{name}': "
                f"only {len(obj_pix)} non-zero pixels (object too small or mask empty)."
            )
            return None

        p2, p98 = np.percentile(obj_pix, 2), np.percentile(obj_pix, 98)
        if p98 > p2:
            patch = np.clip((patch - p2) / (p98 - p2), 0.0, 1.0)
            patch[img == 0] = 0.0
        else:
            warnings.warn(
                f"[TripletAnalyzer] '{name}': p2==p98 ({p2:.4f}), "
                "using binary mask instead of intensity normalisation."
            )
            patch = (patch > 0).astype(np.float32)

        patch = cv2.resize(patch, (_CROP_SIZE, _CROP_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        return patch.astype(np.float32)
    except Exception as exc:
        warnings.warn(f"[TripletAnalyzer] Skipping '{name}': exception — {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Encoder loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_encoder(model_path: str):
    """
    Load a TripletEncoder from a .pt state dict.
    Infers latent_dim from the fc layer weight shape.
    Returns (encoder, device_str).
    """
    import torch
    import torch.nn as nn

    state = torch.load(model_path, map_location="cpu")

    # Infer the fc-head architecture + latent_dim from the checkpoint. The
    # trainer's head is a Sequential (keys 'fc.0/1/4/6.*'); older checkpoints
    # may have a single Linear ('fc.weight'). Support both so the encoder loads
    # regardless of which trainer produced it.
    seq_fc = any(k.startswith("fc.") and k.split(".")[1].isdigit() for k in state)
    if "fc.weight" in state:
        latent_dim = int(state["fc.weight"].shape[0])
    else:
        fc_w = [k for k in state if k.startswith("fc.") and k.endswith(".weight")]
        last = max(fc_w, key=lambda k: int(k.split(".")[1]))   # output layer
        latent_dim = int(state[last].shape[0])

    class TripletEncoder(nn.Module):
        def __init__(self, ld: int = 32):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1,  16, 3, stride=2, padding=1), nn.BatchNorm2d(16),  nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            )
            if seq_fc:
                # Must mirror trainer_triplet's head exactly (indices 0,1,4,6
                # carry weights: Linear, BatchNorm1d, Linear, Linear).
                self.fc = nn.Sequential(
                    nn.Linear(128 * 4 * 4, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, ld),
                )
            else:
                self.fc = nn.Linear(128 * 4 * 4, ld)

        def forward(self, x):
            z = self.fc(self.conv(x).flatten(1))
            return nn.functional.normalize(z, p=2, dim=1)

    encoder = TripletEncoder(latent_dim)
    encoder.load_state_dict(state)
    encoder.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    return encoder, device, latent_dim


# ══════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ══════════════════════════════════════════════════════════════════════════════

def _embed(crops: List[np.ndarray], encoder, device: str) -> np.ndarray:
    """Return (N, latent_dim) float32 embeddings."""
    import torch
    embs = []
    with torch.no_grad():
        for c in crops:
            t = torch.from_numpy(c).unsqueeze(0).unsqueeze(0).to(device)
            embs.append(encoder(t).cpu().numpy()[0])
    return np.array(embs, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Novelty: Euclidean distance to nearest class centroid
# ══════════════════════════════════════════════════════════════════════════════

def _compute_novelty(
    embs      : np.ndarray,
    labels    : List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    For each embedding compute the Euclidean distance to the centroid of its
    own class.  Also returns the centroid dict for later use.

    Returns
    -------
    novelty_scores : float32 array (N,)
    centroids      : {class_name: centroid_vector}
    """
    unique_classes = list(dict.fromkeys(labels))   # preserve order
    centroids: Dict[str, np.ndarray] = {}
    for cls in unique_classes:
        idx = [i for i, l in enumerate(labels) if l == cls]
        centroids[cls] = embs[idx].mean(axis=0)

    scores = np.array([
        float(np.linalg.norm(embs[i] - centroids[labels[i]]))
        for i in range(len(labels))
    ], dtype=np.float32)

    return scores, centroids


# ══════════════════════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════════════════════

_PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
]

def _class_color(cls: str, class_list: List[str]) -> str:
    idx = class_list.index(cls) if cls in class_list else 0
    return _PALETTE[idx % len(_PALETTE)]


def _plot_umap(
    embs        : np.ndarray,
    labels      : List[str],
    class_list  : List[str],
    out_path    : str,
) -> None:
    try:
        import umap as umap_lib
    except ImportError:
        warnings.warn("umap-learn not installed — skipping UMAP plot.")
        return
    try:
        n_neighbors = min(15, len(embs) - 1)
        reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                                random_state=42)
        proj = reducer.fit_transform(embs)

        fig, ax = plt.subplots(figsize=(7, 6))
        for cls in class_list:
            idx = [i for i, l in enumerate(labels) if l == cls]
            if not idx:
                continue
            ax.scatter(proj[idx, 0], proj[idx, 1], s=18, alpha=0.8,
                       color=_class_color(cls, class_list),
                       label=f"{cls} (n={len(idx)})")
        ax.legend(markerscale=2, fontsize=8)
        ax.set_title("UMAP — Triplet embeddings")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        warnings.warn(f"UMAP plot failed: {exc}")


def _plot_novelty(
    novelty     : np.ndarray,
    labels      : List[str],
    class_list  : List[str],
    out_path    : str,
) -> None:
    try:
        fig, ax = plt.subplots(figsize=(max(6, len(class_list) * 1.5), 5))
        for i, cls in enumerate(class_list):
            idx = [j for j, l in enumerate(labels) if l == cls]
            if not idx:
                continue
            scores = novelty[idx]
            # Violin or strip plot depending on n
            x = np.full(len(scores), i) + np.random.default_rng(42).uniform(
                -0.15, 0.15, len(scores)
            )
            ax.scatter(x, scores, s=20, alpha=0.7,
                       color=_class_color(cls, class_list), zorder=3)
            ax.hlines(scores.mean(), i - 0.3, i + 0.3,
                      colors=_class_color(cls, class_list),
                      linewidth=2, label=f"{cls} μ={scores.mean():.3f}")
            # Mark mean + 2σ threshold
            thresh = scores.mean() + 2 * scores.std()
            ax.hlines(thresh, i - 0.25, i + 0.25,
                      colors=_class_color(cls, class_list),
                      linewidth=1.2, linestyles="dashed")

        ax.set_xticks(range(len(class_list)))
        ax.set_xticklabels(class_list, rotation=15)
        ax.set_ylabel("Novelty score (Euclidean distance to centroid)")
        ax.set_title("Novelty scores per class\n(dashed = mean + 2σ threshold)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        warnings.warn(f"Novelty plot failed: {exc}")


def _plot_grid(
    crop_paths  : List[str],
    title       : str,
    out_path    : str,
    cols        : int = 8,
    max_items   : int = 64,
) -> None:
    """Save a grid of crop images sorted in the order provided."""
    if not crop_paths:
        return
    paths = crop_paths[:max_items]
    rows  = max(1, math.ceil(len(paths) / cols))

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 1.4, rows * 1.6))
    axes = np.array(axes).reshape(rows, cols)

    for i, p in enumerate(paths):
        r, c = divmod(i, cols)
        try:
            img = plt.imread(p)
            axes[r, c].imshow(img, cmap="afmhot", vmin=0, vmax=1)
        except Exception:
            axes[r, c].imshow(np.zeros((_CROP_SIZE, _CROP_SIZE)), cmap="gray")
        axes[r, c].axis("off")
        axes[r, c].set_title(Path(p).stem[-10:], fontsize=4)

    for i in range(len(paths), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# CSV writers
# ══════════════════════════════════════════════════════════════════════════════

def _write_objects_csv(
    records     : List[dict],
    embs        : np.ndarray,
    novelty     : np.ndarray,
    latent_dim  : int,
    out_path    : str,
) -> None:
    """
    Write library_objects.csv.
    crop_file is a path relative to the exploration output directory.
    """
    base_fields = ["zoom_index", "crop_file", "class", "novelty_score"]
    emb_fields  = [f"emb_{i:03d}" for i in range(latent_dim)]
    fieldnames  = base_fields + emb_fields

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, rec in enumerate(records):
            row = {
                "zoom_index"   : rec["zoom_index"],
                "crop_file"    : rec["flat_path"],   # relative path
                "class"        : rec["label"],
                "novelty_score": round(float(novelty[i]), 6),
            }
            for j, ef in enumerate(emb_fields):
                row[ef] = round(float(embs[i, j]), 6)
            writer.writerow(row)


def _write_summary_csv(
    records     : List[dict],
    embs        : np.ndarray,
    novelty     : np.ndarray,
    centroids   : Dict[str, np.ndarray],
    class_list  : List[str],
    out_path    : str,
) -> None:
    fieldnames = [
        "class", "n_objects",
        "novelty_mean", "novelty_std", "novelty_thresh_2sigma",
        "centroid_norm",
        "inter_class_dist_mean",
    ]

    rows = []
    for cls in class_list:
        idx = [i for i, r in enumerate(records) if r["label"] == cls]
        if not idx:
            continue
        nov_cls = novelty[idx]

        # Mean distance from this class centroid to all other centroids
        other_cents = [v for k, v in centroids.items() if k != cls]
        if other_cents:
            inter = float(np.mean([
                np.linalg.norm(centroids[cls] - oc) for oc in other_cents
            ]))
        else:
            inter = float("nan")

        rows.append({
            "class"                    : cls,
            "n_objects"                : len(idx),
            "novelty_mean"             : round(float(nov_cls.mean()), 6),
            "novelty_std"              : round(float(nov_cls.std()), 6),
            "novelty_thresh_2sigma"    : round(
                float(nov_cls.mean() + 2 * nov_cls.std()), 6
            ),
            "centroid_norm"            : round(float(np.linalg.norm(centroids[cls])), 6),
            "inter_class_dist_mean"    : round(inter, 6),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main analyser class
# ══════════════════════════════════════════════════════════════════════════════

class TripletAnalyzer:
    """
    Post-session Triplet analysis.

    Usage
    -----
    TripletAnalyzer(out_dir, triplet_model_path, class_names).run()
    """

    def __init__(
        self,
        out_dir            : str,
        triplet_model_path : str,
        class_names        : Optional[List[str]] = None,
    ) -> None:
        self.out_dir             = Path(out_dir)
        self.triplet_model_path  = triplet_model_path
        self.class_names         = class_names or []

    def run(self) -> dict:
        lib_dir = self.out_dir / "library"
        lib_dir.mkdir(exist_ok=True)

        # ── Find all _flat.png crops (organised in per-class subdirs) ──────
        crops_root = lib_dir / "crops"
        if not crops_root.is_dir():
            return {"status": "error",
                    "message": f"Crops directory not found: {crops_root}"}

        # Search recursively: library/crops/<class>/*_flat.png
        flat_files = sorted(crops_root.rglob("*_flat.png"))
        if not flat_files:
            return {"status": "error",
                    "message": "No *_flat.png crops found in library/crops/"}

        # ── Prepare crops ─────────────────────────────────────────────────────
        records    : List[dict]      = []
        crop_arrays: List[np.ndarray] = []

        n_skipped = 0
        for fp in flat_files:
            arr = _prepare_crop(str(fp))
            if arr is None:
                n_skipped += 1
                continue
            # Class is the name of the immediate parent folder
            label  = fp.parent.name
            stem   = fp.stem.replace("_flat", "")
            raw_fp = fp.parent / f"{stem}_raw.png"

            # Store paths relative to out_dir for portability
            try:
                rel_flat = fp.relative_to(self.out_dir)
                rel_raw  = raw_fp.relative_to(self.out_dir) if raw_fp.exists() else None
            except ValueError:
                rel_flat = fp
                rel_raw  = raw_fp if raw_fp.exists() else None

            records.append({
                "zoom_index" : len(records),
                "label"      : label,
                "flat_path"  : str(rel_flat).replace(os.sep, "/"),
                "raw_path"   : str(rel_raw).replace(os.sep, "/") if rel_raw else "",
                "_abs_flat"  : str(fp),   # internal use only, not written to CSV
            })
            crop_arrays.append(arr)

        if not crop_arrays:
            return {"status": "error",
                    "message": (
                        f"No valid crops could be prepared "
                        f"({n_skipped} skipped — check warnings for details)."
                    )}

        if n_skipped > 0:
            warnings.warn(
                f"[TripletAnalyzer] {n_skipped}/{len(flat_files)} crops skipped "
                f"during preparation. Check warnings above for details."
            )

        # ── Load encoder and embed ────────────────────────────────────────────
        try:
            encoder, device, latent_dim = _load_encoder(self.triplet_model_path)
        except Exception as exc:
            return {"status": "error",
                    "message": f"Failed to load TripletEncoder: {exc}"}

        embs = _embed(crop_arrays, encoder, device)

        # ── Novelty scores ────────────────────────────────────────────────────
        labels     = [r["label"] for r in records]
        class_list = self.class_names or sorted(set(labels))

        novelty, centroids = _compute_novelty(embs, labels)

        # ── Per-class grids ───────────────────────────────────────────────────
        for cls in class_list:
            idx = [i for i, l in enumerate(labels) if l == cls]
            if not idx:
                continue

            flat_paths = [records[i]["_abs_flat"] for i in idx]
            nov_scores = novelty[idx]

            # typical: ascending distance (most typical first)
            order_typ = np.argsort(nov_scores)
            _plot_grid(
                [flat_paths[j] for j in order_typ],
                title    = f"{cls} — typical (sorted by similarity to centroid)",
                out_path = str(lib_dir / f"grid_{cls}_typical.png"),
            )

            # novel: descending distance (most novel first)
            order_nov = np.argsort(nov_scores)[::-1]
            _plot_grid(
                [flat_paths[j] for j in order_nov],
                title    = f"{cls} — novel (sorted by novelty score DESC)",
                out_path = str(lib_dir / f"grid_{cls}_novel.png"),
            )

        # ── UMAP ─────────────────────────────────────────────────────────────
        if len(embs) >= 4:
            _plot_umap(embs, labels, class_list,
                       out_path=str(lib_dir / "umap.png"))

        # ── Novelty plot ──────────────────────────────────────────────────────
        _plot_novelty(novelty, labels, class_list,
                      out_path=str(lib_dir / "novelty.png"))

        # ── CSVs ──────────────────────────────────────────────────────────────
        _write_objects_csv(
            records, embs, novelty, latent_dim,
            out_path=str(lib_dir / "library_objects.csv"),
        )
        _write_summary_csv(
            records, embs, novelty, centroids, class_list,
            out_path=str(lib_dir / "library_summary.csv"),
        )

        return {
            "status"       : "ok",
            "message"      : (
                f"Triplet analysis complete. "
                f"{len(records)} objects, {len(class_list)} classes. "
                f"Output: {lib_dir}"
            ),
            "data"         : {
                "n_objects"  : len(records),
                "n_skipped"  : n_skipped,
                "n_classes"  : len(class_list),
                "library_dir": str(lib_dir),
                "latent_dim" : latent_dim,
            },
        }
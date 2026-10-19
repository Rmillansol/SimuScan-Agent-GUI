"""
src/trainer/triplet_inference.py
─────────────────────────────────
Build a triplet embedding library from a folder of experimental AFM images.

Pipeline per object
-------------------
1.  YOLO detects objects → bbox + segmentation mask
2.  Crop with margin (15 %)
3.  Normalise intensity (percentile 2–98 on non-zero pixels)
4.  Otsu threshold on non-zero pixels → binary clean mask
5.  Largest connected component → final clean mask
6.  Extract classical shape/intensity features at original crop size
7.  Align by major axis (SVD) + resolve 180° ambiguity
8.  Re-apply clean rotated mask (removes substrate noise after rotation)
9.  Centre on object bbox + letterbox resize to 64×64
10. TripletEncoder forward pass → 32-d L2-normalised embedding

Library outputs (saved to data/out_triplet_lib/<timestamp>/)
-------------------------------------------------------------
library.json            All descriptors + metadata per object
embeddings.csv          One row per object: embedding + shape features
crops/                  Processed crop PNGs (64×64)
umap_library.png        UMAP of combined descriptor (embedding + features)
umap_features.png       UMAP of shape features only
umap_embedding.png      UMAP of embedding only
distance_matrix.npy     Symmetric pairwise distance matrix
cluster_assignments.csv Cluster label per object (KMeans)
novelty_scores.csv      Distance to nearest neighbour per object
summary.json            Global statistics

Public API
----------
run_triplet_library(image_folder, triplet_run_name, yolo_run_name, ...) → dict
list_triplet_libraries() → dict
get_triplet_library_results(library_dir) → dict
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reuse preprocessing helpers from triplet_trainer ─────────────────────────
from trainer.triplet_trainer import (
    _otsu_on_patch,
    _largest_cc,
    _compute_angle_svd,
    _rotate_patch_and_mask,
    _resolve_180,
    _center_and_pad,
)

# ── Path helpers ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def _out_lib_root() -> Path:
    return _repo_root() / "data" / "out_triplet_lib"

def _out_triplet_root() -> Path:
    return _repo_root() / "data" / "out_triplet"

def _out_yolo_root() -> Path:
    return _repo_root() / "data" / "out_yolo"

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
# Classical shape features
# ══════════════════════════════════════════════════════════════════════════════

_FEATURE_KEYS = [
    "area", "eccentricity", "major_axis", "minor_axis",
    "solidity", "circularity", "mean_intensity", "std_intensity",
]

def _extract_shape_features(patch: np.ndarray, clean_mask: np.ndarray,
                             pixel_size_nm: float) -> dict:
    """
    Extract 8 classical features from the clean crop at original (pre-resize) size.
    Lengths/areas are in nm (or nm²) when pixel_size_nm > 0, else in pixels.
    """
    nan = float("nan")
    empty = {k: nan for k in _FEATURE_KEYS}

    area_px = float(clean_mask.sum())
    if area_px < 5:
        return empty

    scale = pixel_size_nm if pixel_size_nm > 0 else 1.0

    # Area
    area = area_px * (scale ** 2)

    # Contour → perimeter, circularity, solidity
    contours, _ = cv2.findContours(clean_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return empty
    cnt        = max(contours, key=cv2.contourArea)
    perim_px   = float(cv2.arcLength(cnt, True))
    circ       = (4.0 * np.pi * area_px / perim_px ** 2) if perim_px > 0 else nan
    hull_area  = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity   = area_px / hull_area if hull_area > 0 else nan

    # Ellipse → major/minor axis, eccentricity
    if len(cnt) >= 5:
        _, (d1, d2), _ = cv2.fitEllipse(cnt)
        major_px = max(d1, d2)
        minor_px = min(d1, d2)
        major    = major_px * scale
        minor    = minor_px * scale
        ecc      = float(np.sqrt(max(0.0, 1.0 - (minor_px / major_px) ** 2))) \
                   if major_px > 0 else nan
    else:
        major = minor = ecc = nan

    # Intensity on non-zero pixels of the normalised patch
    obj_pix    = patch[clean_mask > 0]
    mean_int   = float(obj_pix.mean()) if len(obj_pix) else nan
    std_int    = float(obj_pix.std())  if len(obj_pix) else nan

    return {
        "area"           : round(area,      4),
        "eccentricity"   : round(ecc,       4),
        "major_axis"     : round(major,     2),
        "minor_axis"     : round(minor,     2),
        "solidity"       : round(solidity,  4),
        "circularity"    : round(circ,      4),
        "mean_intensity" : round(mean_int,  4),
        "std_intensity"  : round(std_int,   4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-image processing
# ══════════════════════════════════════════════════════════════════════════════

def _process_image(img_path: Path, yolo_model, encoder, device: str,
                   cfg: dict) -> list[dict]:
    """
    Run full pipeline on one image. Returns list of object records.
    Each record contains embedding, shape_features, crop array, and metadata.
    """
    import torch

    margin_pct       = cfg["margin_pct"]
    otsu_min_frac    = cfg["otsu_min_frac"]
    crop_size        = cfg["crop_size"]
    pixel_size_nm    = cfg["pixel_size_nm"]
    yolo_conf        = cfg["yolo_conf"]
    yolo_iou         = cfg["yolo_iou"]
    border_margin_pct= cfg.get("border_margin_pct", 0.0)
    min_separation_px= cfg.get("min_separation_px", 0)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W    = gray.shape

    preds  = yolo_model.predict(source=img_bgr, conf=yolo_conf,
                                iou=yolo_iou, verbose=False)[0]
    boxes  = preds.boxes.xyxy.cpu().numpy().astype(int)
    labels = preds.boxes.cls.cpu().numpy().astype(int)
    confs  = preds.boxes.conf.cpu().numpy()

    masks: list[np.ndarray] = []
    if preds.masks is not None:
        for m in preds.masks.data.cpu().numpy():
            mr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append((mr > 0.5).astype(np.uint8))
    else:
        for box in boxes:
            m = np.zeros((H, W), dtype=np.uint8)
            x1, y1, x2, y2 = box
            m[y1:y2, x1:x2] = 1
            masks.append(m)

    # ── Optional: border filter ───────────────────────────────────────────────
    if border_margin_pct > 0 and len(boxes):
        bx = int(W * border_margin_pct)
        by = int(H * border_margin_pct)
        keep = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if bx <= cx <= W - bx and by <= cy <= H - by:
                keep.append(i)
        boxes  = boxes[keep]
        labels = labels[keep]
        confs  = confs[keep]
        masks  = [masks[i] for i in keep]

    # ── Optional: min separation filter ──────────────────────────────────────
    if min_separation_px > 0 and len(boxes) > 1:
        centers = np.array([[(b[0]+b[2])//2, (b[1]+b[3])//2] for b in boxes])
        keep = []
        for i in range(len(boxes)):
            others = [j for j in range(len(boxes)) if j != i]
            if not others:
                keep.append(i); continue
            dists = np.linalg.norm(centers[others] - centers[i], axis=1)
            if dists.min() >= min_separation_px:
                keep.append(i)
        boxes  = boxes[keep]
        labels = labels[keep]
        confs  = confs[keep]
        masks  = [masks[i] for i in keep]

    records = []
    for k, (box, lbl, conf, mask) in enumerate(zip(boxes, labels, confs, masks)):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin_pct), int(bh * margin_pct)
        x1c = max(0, x1 - mx); y1c = max(0, y1 - my)
        x2c = min(W, x2 + mx); y2c = min(H, y2 + my)

        # Step 2: crop + apply YOLO mask
        patch = (gray * mask.astype(np.float32))[y1c:y2c, x1c:x2c]
        pmask = mask[y1c:y2c, x1c:x2c]
        if patch.size == 0:
            continue

        # Step 3: normalise intensity
        obj_pix = patch[patch > 0]
        if len(obj_pix) < 10:
            continue
        p2, p98 = np.percentile(obj_pix, 2), np.percentile(obj_pix, 98)
        if p98 > p2:
            patch = np.clip((patch - p2) / (p98 - p2), 0.0, 1.0)
        else:
            patch = (patch > 0).astype(np.float32)

        # Step 4+5: Otsu + largest CC → clean mask
        thresh = _otsu_on_patch(patch, otsu_min_frac)
        binary = ((patch >= thresh) & (pmask > 0)).astype(np.uint8)
        clean  = _largest_cc(binary)
        if int(clean.sum()) < 10:
            continue

        # Step 6: classical features (original crop size)
        feats = _extract_shape_features(patch, clean, pixel_size_nm)

        # Step 7: align + 180° resolve
        angle                  = _compute_angle_svd(clean)
        rot_gray, rot_mask     = _rotate_patch_and_mask(patch, clean, angle)
        rot_gray, rot_mask     = _resolve_180(rot_gray, rot_mask)

        # Step 8: re-apply clean rotated mask
        rot_gray = rot_gray * rot_mask.astype(np.float32)

        # Step 9: centre + letterbox
        crop_64 = _center_and_pad(rot_gray, rot_mask, crop_size)

        # Step 10: encoder → embedding
        with torch.no_grad():
            t   = torch.from_numpy(crop_64).unsqueeze(0).unsqueeze(0).to(device)
            emb = encoder(t).cpu().numpy()[0]

        records.append({
            "crop"          : crop_64,
            "embedding"     : emb,
            "shape_features": feats,
            "img_path"      : str(img_path),
            "bbox"          : [int(x1), int(y1), int(x2), int(y2)],
            "yolo_class"    : int(lbl),
            "yolo_conf"     : round(float(conf), 4),
            "obj_idx"       : k,
        })

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Descriptor normalisation + distance
# ══════════════════════════════════════════════════════════════════════════════

def _build_descriptors(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (embeddings [N,D], feat_matrix [N,8], descriptors [N, D+8]).
    Both parts are StandardScaler-normalised before concatenation.
    NaN features are replaced with 0 before scaling.
    """
    from sklearn.preprocessing import StandardScaler

    embs  = np.array([r["embedding"]                              for r in records], np.float32)
    feats = np.array([[r["shape_features"].get(k, 0.0) or 0.0
                       for k in _FEATURE_KEYS]                    for r in records], np.float32)

    # Replace NaN with column mean
    for col in range(feats.shape[1]):
        col_vals = feats[:, col]
        nan_mask = ~np.isfinite(col_vals)
        if nan_mask.any():
            col_mean = col_vals[~nan_mask].mean() if (~nan_mask).any() else 0.0
            feats[nan_mask, col] = col_mean

    embs_sc  = StandardScaler().fit_transform(embs)  if len(embs)  > 1 else embs
    feats_sc = StandardScaler().fit_transform(feats) if len(feats) > 1 else feats
    descs    = np.concatenate([embs_sc, feats_sc], axis=1).astype(np.float32)
    return embs, feats, descs


# ══════════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════════

def _auto_cluster(descs: np.ndarray, n_clusters: Optional[int]) -> np.ndarray:
    """KMeans clustering. Auto-selects k via silhouette if n_clusters is None."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    N = len(descs)
    if N < 4:
        return np.zeros(N, dtype=int)

    if n_clusters is not None:
        k = max(2, min(n_clusters, N - 1))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        return km.fit_predict(descs)

    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, min(9, N)):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(descs)
        try:
            sc = silhouette_score(descs, labels)
        except Exception:
            sc = -1.0
        if sc > best_score:
            best_score, best_k, best_labels = sc, k, labels
    return best_labels if best_labels is not None else np.zeros(N, dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

_BG   = "#0d1117"
_AX   = "#161b22"
_TXT  = "#c9d1d9"
_TICK = "#8b949e"
_GRID = "#30363d"

def _umap_plot(data: np.ndarray, labels: np.ndarray, class_names: dict,
               title: str, out_path: Path) -> None:
    try:
        import umap as umap_lib
    except ImportError:
        print(f"  [UMAP] umap-learn not installed; skipping {out_path.name}")
        return
    N = len(data)
    reducer = umap_lib.UMAP(n_neighbors=min(15, N - 1), min_dist=0.1, random_state=42)
    proj    = reducer.fit_transform(data)
    unique  = np.unique(labels)
    cmap    = plt.get_cmap("tab20", len(unique))

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_AX)
    ax.tick_params(colors=_TICK)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)

    for i, cls in enumerate(unique):
        idx  = labels == cls
        name = class_names.get(int(cls), f"cls{cls}")
        ax.scatter(proj[idx, 0], proj[idx, 1], s=14, alpha=0.75,
                   color=cmap(i), label=f"{name} (n={idx.sum()})")
    ax.legend(markerscale=2, fontsize=7, loc="best",
              facecolor=_AX, edgecolor=_GRID, labelcolor=_TXT)
    ax.set_title(title,    color=_TXT)
    ax.set_xlabel("UMAP-1", color=_TICK)
    ax.set_ylabel("UMAP-2", color=_TICK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def _show_png(path: Path, title: str = "") -> None:
    from src.bridge.ui_bridge import ui
    if not path.is_file():
        return
    img = plt.imread(str(path))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img); ax.axis("off")
    ax.set_title(title, color=_TXT, fontsize=9)
    fig.patch.set_facecolor(_BG)
    fig.tight_layout()
    ui.plot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Medoids & greedy coverage
# ══════════════════════════════════════════════════════════════════════════════

def _compute_medoids(descs: np.ndarray, cluster_labels: np.ndarray,
                     all_records: list, embs: np.ndarray,
                     class_names: dict) -> list[dict]:
    """Return one medoid dict per cluster (closest object to cluster centroid)."""
    medoids = []
    for clu_id in sorted(set(cluster_labels)):
        mask = cluster_labels == clu_id
        idxs = np.where(mask)[0]
        centroid = descs[idxs].mean(axis=0)
        dists = np.linalg.norm(descs[idxs] - centroid, axis=1)
        local_best = int(dists.argmin())
        best_idx = int(idxs[local_best])
        r = all_records[best_idx]
        medoids.append({
            "cluster"        : int(clu_id),
            "obj_id"         : r["obj_id"],
            "crop_path"      : r["crop_path"],
            "img_path"       : r["img_path"],
            "yolo_class"     : r["yolo_class"],
            "yolo_class_name": class_names.get(r["yolo_class"], f"cls{r['yolo_class']}"),
            "yolo_conf"      : r["yolo_conf"],
            "dist_to_centroid": round(float(dists[local_best]), 6),
            "shape_features" : r["shape_features"],
            "embedding"      : [round(float(v), 6) for v in embs[best_idx]],
        })
    return medoids


def _greedy_coverage(descs: np.ndarray, all_records: list, embs: np.ndarray,
                     cluster_labels: np.ndarray, class_names: dict,
                     coverage_target: float = 0.90) -> list[dict]:
    """
    Greedy set-cover: find minimal subset of objects that covers >= coverage_target
    fraction of all objects within radius r (median nearest-neighbour distance).
    """
    n = len(descs)
    # radius = 90th percentile of nearest-neighbour distances
    from sklearn.metrics.pairwise import euclidean_distances
    dist_mat = euclidean_distances(descs).astype(np.float32)
    np.fill_diagonal(dist_mat, np.inf)
    nn_dists = dist_mat.min(axis=1)
    r = float(np.percentile(nn_dists, 90))
    np.fill_diagonal(dist_mat, 0)

    covered   = np.zeros(n, dtype=bool)
    selected  = []
    in_radius = dist_mat <= r  # (n, n) bool — precomputed

    while covered.mean() < coverage_target:
        # count how many uncovered points each candidate would newly cover
        new_counts = (in_radius & ~covered).sum(axis=1)  # vectorised
        best_idx   = int(new_counts.argmax())
        best_count = int(new_counts[best_idx])
        if best_count == 0:
            break  # can't improve further
        covered |= (dist_mat[best_idx] <= r)
        r_val = all_records[best_idx]
        selected.append({
            "rep_rank"       : len(selected),
            "obj_id"         : r_val["obj_id"],
            "crop_path"      : r_val["crop_path"],
            "img_path"       : r_val["img_path"],
            "cluster"        : int(cluster_labels[best_idx]),
            "yolo_class"     : r_val["yolo_class"],
            "yolo_class_name": class_names.get(r_val["yolo_class"],
                                               f"cls{r_val['yolo_class']}"),
            "yolo_conf"      : r_val["yolo_conf"],
            "n_covered"      : int(covered.sum()),
            "coverage_pct"   : round(float(covered.mean()) * 100, 2),
            "shape_features" : r_val["shape_features"],
            "embedding"      : [round(float(v), 6) for v in embs[best_idx]],
        })

    return selected


def _crop_grid_png(entries: list[dict], title: str, out_path: Path,
                   cols: int = 8, cell: int = 80) -> None:
    """Save a grid of crop images for a list of representative entries."""
    n    = len(entries)
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * cell / 72, rows * (cell + 24) / 72),
                             dpi=100)
    axes = np.array(axes).reshape(rows, cols)
    fig.patch.set_facecolor(_BG)

    for i, entry in enumerate(entries):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        crop_p = Path(entry["crop_path"])
        if crop_p.is_file():
            img = plt.imread(str(crop_p))
            ax.imshow(img, cmap="afmhot", vmin=0, vmax=1)
        else:
            ax.set_facecolor(_AX)
        label = (f"{entry.get('yolo_class_name', '')} | "
                 f"cl{entry.get('cluster', '?')}")
        if "coverage_pct" in entry:
            label += f"\n{entry['coverage_pct']:.0f}%"
        ax.set_title(label, fontsize=5, color=_TXT, pad=2)
        ax.axis("off")

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=8, color=_TXT)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=100, facecolor=_BG)
    plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main public function
# ══════════════════════════════════════════════════════════════════════════════

def run_triplet_library(
    image_folder      : str,
    triplet_run_name  : Optional[str] = None,
    yolo_run_name     : Optional[str] = None,
    *,
    pixel_size_nm     : float = 1.0,
    margin_pct        : float = 0.15,
    otsu_min_frac     : float = 0.10,
    crop_size         : int   = 64,
    yolo_conf         : float = 0.50,
    yolo_iou          : float = 0.45,
    n_clusters        : Optional[int] = None,
    border_margin_pct : float = 0.0,
    min_separation_px : int   = 0,
) -> dict:
    """
    Process a folder of AFM images with a trained YOLO + TripletEncoder pair
    and build an embedding library.

    Parameters
    ----------
    image_folder     : Folder containing .png / .jpg images to process.
    triplet_run_name : Name of a run in data/out_triplet/ (auto = latest).
    yolo_run_name    : Name of a run in data/out_yolo/ (auto = latest segment).
    pixel_size_nm    : Physical pixel size in nm (0 = use pixels).
    margin_pct       : Crop margin around bbox (default 0.15).
    otsu_min_frac    : Otsu threshold floor (default 0.10).
    crop_size        : Encoder input size (default 64).
    yolo_conf        : YOLO confidence threshold (default 0.25).
    yolo_iou         : YOLO NMS IoU threshold (default 0.45).
    n_clusters       : KMeans clusters (None = auto via silhouette, max 8).

    Returns
    -------
    dict  {"status": "ok"|"error", "data": {...}, ...}
    """
    import torch
    from ultralytics import YOLO
    from tqdm import tqdm
    from src.bridge.ui_bridge import ui

    # ── Resolve paths ─────────────────────────────────────────────────────────
    img_folder = Path(image_folder).resolve()
    if not img_folder.is_dir():
        return _err(f"Image folder not found: '{image_folder}'")

    # Auto-detect GWY vs PNG/JPG
    gwy_paths = sorted(img_folder.glob("*.gwy"))
    img_paths  = sorted(img_folder.glob("*.png")) + sorted(img_folder.glob("*.jpg"))

    if gwy_paths and not img_paths:
        # GWY folder — flatten and convert to PNG first, then process those PNGs
        ui.log("INFO", f"Detected {len(gwy_paths)} GWY files — converting to PNG…")
        from src.automation.automation import load_gwy_channel, flatten_line_by_line, array_to_png
        import tempfile, shutil
        tmp_dir = Path(tempfile.mkdtemp(prefix="simuscan_gwy_"))
        try:
            for gwy_path in gwy_paths:
                try:
                    data, _ = load_gwy_channel(str(gwy_path), scan_direction="forward")
                    flat, _ = flatten_line_by_line(data)
                    out_png = tmp_dir / (gwy_path.stem + ".png")
                    array_to_png(flat, str(out_png))
                except Exception as e:
                    ui.log("WARN", f"Skipping {gwy_path.name}: {e}")
            img_paths = sorted(tmp_dir.glob("*.png"))
            if not img_paths:
                return _err("GWY conversion produced no valid PNG images.")
            ui.log("OK", f"Converted {len(img_paths)} GWY → PNG.")
            # Use tmp_dir as working folder; caller's folder kept for reference
            _gwy_tmp_dir = tmp_dir
            img_folder   = tmp_dir
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return _err(f"GWY conversion failed: {e}")
    elif not img_paths:
        return _err(f"No PNG/JPG or GWY images found in '{image_folder}'")

    # Triplet model
    triplet_root = _out_triplet_root()
    if triplet_run_name:
        pt_path = triplet_root / triplet_run_name / "triplet_encoder.pt"
    else:
        pt_path = None
        for p in sorted(triplet_root.iterdir(), reverse=True):
            candidate = p / "triplet_encoder.pt"
            if p.is_dir() and candidate.is_file():
                pt_path = candidate
                break
    if pt_path is None or not pt_path.is_file():
        return _err("No trained TripletEncoder found. Train one first.")

    # YOLO model
    yolo_root = _out_yolo_root()
    if yolo_run_name:
        yolo_pt = yolo_root / yolo_run_name / "weights" / "best.pt"
    else:
        yolo_pt = None
        for p in sorted(yolo_root.iterdir(), reverse=True):
            candidate = p / "weights" / "best.pt"
            if p.is_dir() and candidate.is_file():
                yolo_pt = candidate
                break
    if yolo_pt is None or not yolo_pt.is_file():
        return _err("No trained YOLO model found. Train one first.")

    # Load class names from triplet run metrics
    class_names: dict[int, str] = {}
    metrics_path = pt_path.parent / "metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            raw = m.get("class_names", {})
            class_names = {int(k): v for k, v in raw.items()}
        except Exception:
            pass

    # ── Output directory ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = _out_lib_root() / f"time_{timestamp}"
    crop_dir  = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder (rebuild architecture, load weights)
    from trainer.triplet_trainer import _build_encoder_and_dataset, TripletConfig
    cfg_dummy    = TripletConfig(latent_dim=32)  # latent_dim overwritten below
    state        = torch.load(str(pt_path), map_location=device)
    # Infer latent_dim from last fc weight
    fc_weights   = sorted([(k, v) for k, v in state.items()
                            if k.startswith("fc.") and k.endswith(".weight")],
                           key=lambda kv: kv[0])
    latent_dim   = int(fc_weights[-1][1].shape[0]) if fc_weights else 32
    cfg_dummy.latent_dim = latent_dim
    TripletEncoderCls, _ = _build_encoder_and_dataset(cfg_dummy)
    encoder = TripletEncoderCls(latent_dim).to(device)
    encoder.load_state_dict(state)
    encoder.eval()

    yolo_model = YOLO(str(yolo_pt))

    proc_cfg = dict(
        margin_pct        = margin_pct,
        otsu_min_frac     = otsu_min_frac,
        crop_size         = crop_size,
        pixel_size_nm     = pixel_size_nm,
        yolo_conf         = yolo_conf,
        yolo_iou          = yolo_iou,
        border_margin_pct = border_margin_pct,
        min_separation_px = min_separation_px,
    )

    # ── Process images ────────────────────────────────────────────────────────
    ui.log("INFO", f"Processing {len(img_paths)} images → {out_dir.name}")
    all_records: list[dict] = []

    for img_path in tqdm(img_paths, desc="Building library"):
        recs = _process_image(img_path, yolo_model, encoder, device, proc_cfg)
        for r in recs:
            obj_id   = f"{img_path.stem}_obj{r['obj_idx']:03d}"
            crop_png = crop_dir / f"{obj_id}_cls{r['yolo_class']}.png"
            cv2.imwrite(str(crop_png), (r["crop"] * 255).astype(np.uint8))
            r["crop_path"] = str(crop_png)
            r["obj_id"]    = obj_id
            r["timestamp"] = timestamp
        all_records.extend(recs)

    if not all_records:
        return _err("No objects detected in any image. "
                    "Check YOLO confidence threshold or image contents.")

    ui.log("INFO", f"Detected {len(all_records)} objects across {len(img_paths)} images")

    # ── Descriptors ───────────────────────────────────────────────────────────
    embs, feats, descs = _build_descriptors(all_records)
    yolo_labels = np.array([r["yolo_class"] for r in all_records])

    # ── Distance matrix ───────────────────────────────────────────────────────
    from sklearn.metrics.pairwise import euclidean_distances
    dist_mat = euclidean_distances(descs).astype(np.float32)
    np.save(str(out_dir / "distance_matrix.npy"), dist_mat)

    # ── Novelty scores (distance to nearest neighbour) ────────────────────────
    np.fill_diagonal(dist_mat, np.inf)
    novelty = dist_mat.min(axis=1)
    np.fill_diagonal(dist_mat, 0)

    # ── Clustering ────────────────────────────────────────────────────────────
    cluster_labels = _auto_cluster(descs, n_clusters)

    # ── UMAP plots ────────────────────────────────────────────────────────────
    ui.log("INFO", "Generating UMAP plots…")
    _umap_plot(descs, yolo_labels, class_names,
               f"UMAP — combined descriptor ({out_dir.name})",
               out_dir / "umap_library.png")
    _umap_plot(feats, yolo_labels, class_names,
               "UMAP — shape features only",
               out_dir / "umap_features.png")
    _umap_plot(embs, yolo_labels, class_names,
               "UMAP — embedding only",
               out_dir / "umap_embedding.png")

    # ── Show plots in GUI ─────────────────────────────────────────────────────
    for png, title in [
        ("umap_library.png",  "Library UMAP — combined descriptor"),
        ("umap_features.png", "Library UMAP — shape features"),
        ("umap_embedding.png","Library UMAP — embedding"),
    ]:
        _show_png(out_dir / png, f"{out_dir.name} — {title}")

    # ── Build sorted list of unique class names (for pivot columns) ───────────
    sorted_class_ids   = sorted({r["yolo_class"] for r in all_records})
    sorted_class_names = [class_names.get(c, f"cls{c}") for c in sorted_class_ids]

    # ── CSV: embeddings + features ────────────────────────────────────────────
    emb_cols  = [f"emb_{i}" for i in range(embs.shape[1])]
    feat_cols = _FEATURE_KEYS
    csv_path  = out_dir / "embeddings.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "obj_id", "img_path",
            "x0", "y0", "width", "height",
            "yolo_class", "yolo_class_name", "yolo_conf",
            "cluster", "novelty_score", "timestamp",
        ] + feat_cols + emb_cols)
        writer.writeheader()
        for i, r in enumerate(all_records):
            bbox = r["bbox"]  # (x0, y0, w, h)
            row = {
                "obj_id"         : r["obj_id"],
                "img_path"       : r["img_path"],
                "x0"             : bbox[0],
                "y0"             : bbox[1],
                "width"          : bbox[2],
                "height"         : bbox[3],
                "yolo_class"     : r["yolo_class"],
                "yolo_class_name": class_names.get(r["yolo_class"], f"cls{r['yolo_class']}"),
                "yolo_conf"      : r["yolo_conf"],
                "cluster"        : int(cluster_labels[i]),
                "novelty_score"  : round(float(novelty[i]), 6),
                "timestamp"      : r["timestamp"],
            }
            for k in feat_cols:
                row[k] = r["shape_features"].get(k, "")
            for j, ec in enumerate(emb_cols):
                row[ec] = round(float(embs[i, j]), 6)
            writer.writerow(row)

    # ── CSV: cluster assignments ──────────────────────────────────────────────
    ca_fields = ["obj_id", "cluster", "yolo_class", "yolo_class_name"] + sorted_class_names
    with open(out_dir / "cluster_assignments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ca_fields)
        writer.writeheader()
        for i, r in enumerate(all_records):
            cls_id   = r["yolo_class"]
            cls_name = class_names.get(cls_id, f"cls{cls_id}")
            cluster  = int(cluster_labels[i])
            row = {
                "obj_id"         : r["obj_id"],
                "cluster"        : cluster,
                "yolo_class"     : cls_id,
                "yolo_class_name": cls_name,
            }
            for name in sorted_class_names:
                row[name] = cluster if name == cls_name else ""
            writer.writerow(row)

    # ── CSV: novelty scores ───────────────────────────────────────────────────
    ns_fields = ["obj_id", "novelty_score", "yolo_class", "yolo_class_name"] + sorted_class_names
    with open(out_dir / "novelty_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ns_fields)
        writer.writeheader()
        for i, r in enumerate(all_records):
            cls_id   = r["yolo_class"]
            cls_name = class_names.get(cls_id, f"cls{cls_id}")
            score    = round(float(novelty[i]), 6)
            row = {
                "obj_id"         : r["obj_id"],
                "novelty_score"  : score,
                "yolo_class"     : cls_id,
                "yolo_class_name": cls_name,
            }
            for name in sorted_class_names:
                row[name] = score if name == cls_name else ""
            writer.writerow(row)

    # ── library.json ─────────────────────────────────────────────────────────
    lib_entries = []
    for i, r in enumerate(all_records):
        lib_entries.append({
            "obj_id"         : r["obj_id"],
            "embedding"      : [round(float(v), 6) for v in embs[i]],
            "shape_features" : r["shape_features"],
            "descriptor"     : [round(float(v), 6) for v in descs[i]],
            "img_path"       : r["img_path"],
            "bbox"           : r["bbox"],
            "yolo_class"     : r["yolo_class"],
            "yolo_class_name": class_names.get(r["yolo_class"], f"cls{r['yolo_class']}"),
            "yolo_conf"      : r["yolo_conf"],
            "cluster"        : int(cluster_labels[i]),
            "novelty_score"  : round(float(novelty[i]), 6),
            "crop_path"      : r["crop_path"],
            "timestamp"      : r["timestamp"],
        })
    with open(out_dir / "library.json", "w") as f:
        json.dump(lib_entries, f, indent=2)

    # ── Medoids ───────────────────────────────────────────────────────────────
    ui.log("INFO", "Computing cluster medoids…")
    medoids = _compute_medoids(descs, cluster_labels, all_records, embs, class_names)
    with open(out_dir / "medoids.json", "w") as f:
        json.dump(medoids, f, indent=2)
    _crop_grid_png(medoids,
                   f"Cluster medoids — {out_dir.name}",
                   out_dir / "medoids_grid.png")
    _show_png(out_dir / "medoids_grid.png", f"{out_dir.name} — Cluster medoids")

    # ── Greedy 90% coverage ───────────────────────────────────────────────────
    ui.log("INFO", "Computing greedy 90% coverage representatives…")
    coverage_reps = _greedy_coverage(descs, all_records, embs,
                                     cluster_labels, class_names,
                                     coverage_target=0.90)
    with open(out_dir / "coverage_representatives.json", "w") as f:
        json.dump(coverage_reps, f, indent=2)
    _crop_grid_png(coverage_reps,
                   f"Greedy 90% coverage — {out_dir.name}  "
                   f"({len(coverage_reps)} representatives)",
                   out_dir / "coverage_grid.png")
    _show_png(out_dir / "coverage_grid.png",
              f"{out_dir.name} — Greedy 90% coverage ({len(coverage_reps)} reps)")
    ui.log("OK", f"Coverage: {len(coverage_reps)} representatives cover ≥90% "
                 f"of {len(all_records)} objects")

    # ── Greedy 90% coverage per class ─────────────────────────────────────────
    sorted_class_ids_cov = sorted({r["yolo_class"] for r in all_records})
    for cls_id in sorted_class_ids_cov:
        cls_name = class_names.get(cls_id, f"cls{cls_id}")
        cls_mask = np.array([r["yolo_class"] == cls_id for r in all_records])
        cls_idxs = np.where(cls_mask)[0]
        if len(cls_idxs) < 2:
            continue
        cls_recs   = [all_records[i] for i in cls_idxs]
        cls_descs  = descs[cls_idxs]
        cls_embs   = embs[cls_idxs]
        cls_labels = cluster_labels[cls_idxs]
        cls_reps   = _greedy_coverage(cls_descs, cls_recs, cls_embs,
                                      cls_labels, class_names, coverage_target=0.90)
        png_name = f"coverage_grid_{cls_name}.png"
        _crop_grid_png(cls_reps,
                       f"Greedy 90% coverage — {cls_name}  ({len(cls_reps)} reps)",
                       out_dir / png_name)
        _show_png(out_dir / png_name,
                  f"{out_dir.name} — {cls_name}: 90% coverage ({len(cls_reps)} reps)")
        ui.log("OK", f"{cls_name}: {len(cls_reps)} reps cover ≥90% "
                     f"of {int(cls_mask.sum())} objects")

    # ── summary.json ─────────────────────────────────────────────────────────
    class_counts  = Counter(int(r["yolo_class"]) for r in all_records)
    cluster_counts= Counter(int(cluster_labels[i]) for i in range(len(all_records)))
    summary = {
        "run_dir"           : str(out_dir),
        "image_folder"      : str(img_folder),
        "triplet_model"     : str(pt_path),
        "yolo_model"        : str(yolo_pt),
        "n_images"          : len(img_paths),
        "n_objects"         : len(all_records),
        "n_clusters"        : int(len(set(cluster_labels))),
        "latent_dim"        : latent_dim,
        "pixel_size_nm"     : pixel_size_nm,
        "class_names"       : {str(k): v for k, v in class_names.items()},
        "class_counts"      : {str(k): v for k, v in class_counts.items()},
        "cluster_counts"    : {str(k): v for k, v in cluster_counts.items()},
        "novelty_mean"      : round(float(novelty.mean()), 4),
        "novelty_max"       : round(float(novelty.max()), 4),
        "n_medoids"         : len(medoids),
        "n_coverage_reps"   : len(coverage_reps),
        "coverage_radius"   : None,  # computed inside _greedy_coverage
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {out_dir / 'summary.json'}")

    done_msg = (f"Library built: {len(all_records)} objects from {len(img_paths)} images, "
                f"{len(set(cluster_labels))} clusters → {out_dir.name}")
    ui.log("INFO", done_msg)

    return _ok(data=summary, message=done_msg)


# ══════════════════════════════════════════════════════════════════════════════
# Post-build inspection
# ══════════════════════════════════════════════════════════════════════════════

def list_triplet_libraries() -> dict:
    """List all library runs in data/out_triplet_lib/ sorted newest-first."""
    root = _out_lib_root()
    if not root.exists():
        return _ok(data=[], message="No triplet libraries found.")
    runs = []
    for p in sorted(root.iterdir(), reverse=True):
        if not p.is_dir() or not p.name.startswith("time_"):
            continue
        summary_path = p / "summary.json"
        summary = {}
        if summary_path.is_file():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except Exception:
                pass
        runs.append({
            "name"      : p.name,
            "path"      : str(p),
            "n_objects" : summary.get("n_objects"),
            "n_images"  : summary.get("n_images"),
            "n_clusters": summary.get("n_clusters"),
        })
    return _ok(data=runs, message=f"{len(runs)} triplet librar{'y' if len(runs)==1 else 'ies'} found.")


def get_triplet_library_results(library_dir: str) -> dict:
    """Return summary and show saved plots for a specific library run."""
    from src.bridge.ui_bridge import ui

    p = Path(library_dir).resolve()
    if not p.is_dir():
        return _err(f"Library directory not found: '{library_dir}'")

    summary = {}
    summary_path = p / "summary.json"
    if summary_path.is_file():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            pass

    for png_name, title in [
        ("umap_library.png",   "UMAP — combined descriptor"),
        ("umap_features.png",  "UMAP — shape features"),
        ("umap_embedding.png", "UMAP — embedding"),
    ]:
        _show_png(p / png_name, f"{p.name} — {title}")

    return _ok(data={
        "library_dir"         : str(p),
        "summary"             : summary,
        "library_json"        : str(p / "library.json"),
        "embeddings_csv"      : str(p / "embeddings.csv"),
        "cluster_assignments" : str(p / "cluster_assignments.csv"),
        "novelty_scores"      : str(p / "novelty_scores.csv"),
        "distance_matrix"     : str(p / "distance_matrix.npy"),
    }, message=f"Library results loaded from '{p.name}'.")

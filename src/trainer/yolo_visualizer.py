"""
src/trainer/yolo_visualizer.py
──────────────────────────────
Visualization and prediction utilities for SimuScan YOLO datasets.

Public API
----------
visualize_yolo_dataset(yolo_folder, ...)      → dict   (matplotlib, blocking)
predict_folder(image_folder, model_path, ...) → dict   (saves to disk, no display)
find_latest_best_pt()                         → str | None
get_model_info(model_path)                    → dict
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from src.bridge.ui_bridge import ui

# ── Path helpers ───────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def _out_yolo_root() -> Path:
    return _repo_root() / "data" / "out_yolo"

# ── Result helpers ─────────────────────────────────────────────────────────────

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

# ── Color palette (per class index, visually distinct) ────────────────────────

_PALETTE = [
    (255,  60,  60), ( 60, 180, 255), ( 60, 220,  60), (255, 180,  30),
    (180,  60, 255), ( 30, 220, 200), (255, 100, 180), (120, 120, 255),
    (255, 160,  80), ( 80, 255, 160), (200,  80,  80), ( 80, 200, 200),
]

def _color(cls_idx: int) -> tuple[int, int, int]:
    return _PALETTE[cls_idx % len(_PALETTE)]


# ── CSV column definitions ─────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    # Identity
    "index", "filename", "label", "class_id", "confidence",
    # Bounding box
    "x1", "y1", "x2", "y2",
    "img_width", "img_height",
    # Geometry — basic
    "area_px", "perimeter",
    "cx", "cy",
    # Geometry — shape
    "aspect_ratio", "extent", "solidity", "equivalent_diameter",
    # Geometry — ellipse fit
    "compactness", "orientation_deg",
    "major_axis", "minor_axis",
    # Source
    "features_from",   # "mask" | "contour_bbox"
]


# ══════════════════════════════════════════════════════════════════════════════
# Shape feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def _features_from_mask(mask_bin: np.ndarray) -> dict:
    """
    Extract shape features from a binary mask (uint8, values 0/1 or 0/255).
    Uses the largest external contour for ellipse fitting.
    Returns a flat dict of feature values (None where not computable).
    """
    import cv2

    mask_u8 = (mask_bin > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_features("mask")

    cnt = max(contours, key=cv2.contourArea)
    return _features_from_contour(cnt, source="mask")


def _features_from_bbox(x1: int, y1: int, x2: int, y2: int,
                         img: np.ndarray) -> dict:
    """
    Fallback when no segmentation mask exists (detection mode).
    Crops the bbox region from the image, converts to grayscale,
    thresholds to find the dominant contour, and extracts features.
    Falls back to bbox geometry if no contour is found.
    """
    import cv2

    roi = img[max(y1, 0):y2, max(x1, 0):x2]
    if roi.size == 0:
        return _bbox_geometry_features(x1, y1, x2, y2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    # Otsu threshold to find object
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        # Offset contour back to full-image coordinates
        cnt_offset = cnt + np.array([[[x1, y1]]], dtype=np.int32)
        return _features_from_contour(cnt_offset, source="contour_bbox")

    return _bbox_geometry_features(x1, y1, x2, y2)


def _features_from_contour(cnt, source: str) -> dict:
    """Compute all shape features from an OpenCV contour."""
    import cv2

    area       = float(cv2.contourArea(cnt))
    perimeter  = float(cv2.arcLength(cnt, True))
    M          = cv2.moments(cnt)

    cx = float(M["m10"] / M["m00"]) if M["m00"] else 0.0
    cy = float(M["m01"] / M["m00"]) if M["m00"] else 0.0

    x, y, w, h    = cv2.boundingRect(cnt)
    hull          = cv2.convexHull(cnt)
    hull_area     = float(cv2.contourArea(hull))
    bbox_area     = float(w * h) if w * h > 0 else 1.0

    aspect_ratio        = round(w / h, 4)          if h > 0 else None
    extent              = round(area / bbox_area, 4)
    solidity            = round(area / hull_area, 4) if hull_area > 0 else None
    equivalent_diameter = round(float(np.sqrt(4 * area / np.pi)), 4) if area > 0 else None
    compactness         = round((perimeter ** 2) / (4 * np.pi * area), 4) if area > 0 else None

    # Ellipse fit (needs ≥ 5 points)
    orientation_deg = major_axis = minor_axis = None
    if len(cnt) >= 5:
        try:
            (_, _), (ma, Mi), angle = cv2.fitEllipse(cnt)
            major_axis      = round(float(max(ma, Mi)), 4)
            minor_axis      = round(float(min(ma, Mi)), 4)
            orientation_deg = round(float(angle), 4)
        except cv2.error:
            pass

    return {
        "area_px":            round(area, 2),
        "perimeter":          round(perimeter, 4),
        "cx":                 round(cx, 2),
        "cy":                 round(cy, 2),
        "aspect_ratio":       aspect_ratio,
        "extent":             extent,
        "solidity":           solidity,
        "equivalent_diameter": equivalent_diameter,
        "compactness":        compactness,
        "orientation_deg":    orientation_deg,
        "major_axis":         major_axis,
        "minor_axis":         minor_axis,
        "features_from":      source,
    }


def _bbox_geometry_features(x1: int, y1: int, x2: int, y2: int) -> dict:
    """Pure bbox fallback — no image data needed."""
    w, h = x2 - x1, y2 - y1
    area = float(w * h)
    return {
        "area_px":             round(area, 2),
        "perimeter":           round(2 * (w + h), 4),
        "cx":                  round(x1 + w / 2, 2),
        "cy":                  round(y1 + h / 2, 2),
        "aspect_ratio":        round(w / h, 4) if h > 0 else None,
        "extent":              1.0,
        "solidity":            None,
        "equivalent_diameter": round(float(np.sqrt(4 * area / np.pi)), 4) if area > 0 else None,
        "compactness":         None,
        "orientation_deg":     None,
        "major_axis":          round(float(max(w, h)), 4),
        "minor_axis":          round(float(min(w, h)), 4),
        "features_from":       "bbox_fallback",
    }


def _empty_features(source: str = "mask") -> dict:
    keys = [
        "area_px", "perimeter", "cx", "cy", "aspect_ratio", "extent",
        "solidity", "equivalent_diameter", "compactness",
        "orientation_deg", "major_axis", "minor_axis",
    ]
    return {k: None for k in keys} | {"features_from": source}


# ══════════════════════════════════════════════════════════════════════════════
# Model helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_best_pt() -> str | None:
    """
    Scan data/out_yolo/ and return the path of the most recent best.pt.
    Returns None if no trained model is found.
    """
    out_root = _out_yolo_root()
    if not out_root.exists():
        return None
    candidates = sorted(
        (p / "weights" / "best.pt" for p in out_root.iterdir() if p.is_dir()),
        key=lambda p: p.parent.parent.name,
    )
    for pt in reversed(candidates):
        if pt.is_file():
            return str(pt)
    return None


def get_model_info(model_path: str) -> dict:
    """
    Return basic metadata about a YOLO .pt model file.

    Parameters
    ----------
    model_path : str   Path to a .pt file.
    """
    pt = Path(model_path)
    if not pt.is_file():
        return _err(f"Model file not found: '{model_path}'")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        task  = getattr(model, "task", "unknown")
        names = getattr(model, "names", {})
        info  = {
            "model_path":   str(pt),
            "file_size_mb": round(pt.stat().st_size / 1e6, 2),
            "task":         task,
            "n_classes":    len(names),
            "class_names":  list(names.values()) if isinstance(names, dict) else names,
            "run_dir":      str(pt.parent.parent),
        }
        return _ok(data=info, message=f"Model loaded: task='{task}', {len(names)} classes.")
    except Exception as exc:
        return _err(f"Could not load model '{model_path}': {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Dataset visualization  (matplotlib — blocking, call from agent process)
# ══════════════════════════════════════════════════════════════════════════════

def visualize_yolo_dataset(
    yolo_folder:  str,
    splits:       list[str] | None = None,
    n_per_split:  int  = 3,
    show_boxes:   bool = True,
    show_masks:   bool = True,
    show_labels:  bool = True,
) -> dict:
    """
    Display random images from a YOLO dataset with ground-truth annotations.
    One separate matplotlib figure per image (blocking until all windows closed).

    Parameters
    ----------
    yolo_folder  : str        Path to a _YOLO_SEG or _YOLO_DET folder.
    splits       : list[str]  Splits to show. Default: ['train'].
    n_per_split  : int        Images per split. Default: 3.
    show_boxes   : bool       Draw bounding boxes.
    show_masks   : bool       Draw segmentation polygons (SEG datasets only).
    show_labels  : bool       Draw class name text.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon as MplPolygon
    import yaml

    yolo_folder = str(Path(yolo_folder).resolve())
    splits      = splits or ["train"]

    yaml_path = os.path.join(yolo_folder, "dataset.yaml")
    if not os.path.isfile(yaml_path):
        return _err(f"dataset.yaml not found in '{yolo_folder}'.")

    with open(yaml_path, encoding="utf-8") as f:
        ds_yaml = yaml.safe_load(f)
    class_names: list = ds_yaml.get("names", [])
    task: str         = ds_yaml.get("task", "detect")

    shown_total = 0
    summary_per_split: dict[str, int] = {}

    for split in splits:
        img_dir = Path(yolo_folder) / "images" / split
        lbl_dir = Path(yolo_folder) / "labels" / split

        if not img_dir.is_dir():
            ui.log('WARN', f"Split '{split}' not found, skipping.")
            continue

        img_files = [
            f for f in img_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        ]
        if not img_files:
            ui.log('WARN', f"No images found in split '{split}'.")
            continue

        sample     = random.sample(img_files, min(n_per_split, len(img_files)))
        shown_split = 0

        for img_path in sample:
            img_bgr = _imread(str(img_path))
            if img_bgr is None:
                continue
            H, W    = img_bgr.shape[:2]
            img_rgb = img_bgr[:, :, ::-1].copy()

            lbl_path    = lbl_dir / (img_path.stem + ".txt")
            annotations = _parse_yolo_label(lbl_path, W, H, task)

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(img_rgb)
            ax.set_title(f"{split} — {img_path.name}", fontsize=10)
            ax.axis("off")

            for ann in annotations:
                cls  = ann["cls"]
                col  = tuple(c / 255 for c in _color(cls))
                name = class_names[cls] if cls < len(class_names) else str(cls)

                if show_masks and ann.get("poly") is not None:
                    patch = MplPolygon(
                        ann["poly"], closed=True,
                        facecolor=(*col, 0.25), edgecolor=col, linewidth=1.5,
                    )
                    ax.add_patch(patch)

                if show_boxes and ann.get("box") is not None:
                    x1, y1, x2, y2 = ann["box"]
                    rect = mpatches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=1.5, edgecolor=col, facecolor="none",
                    )
                    ax.add_patch(rect)
                    if show_labels:
                        ax.text(
                            x1, max(y1 - 4, 0), name,
                            fontsize=7, color="white",
                            bbox=dict(facecolor=col, alpha=0.75, pad=1, edgecolor="none"),
                        )

            plt.tight_layout()
            ui.plot(fig)
            shown_split += 1
            shown_total += 1

        summary_per_split[split] = shown_split

    if shown_total == 0:
        return _err("No images could be displayed.")

    return _ok(
        data={
            "shown_per_split": summary_per_split,
            "total_shown":     shown_total,
            "task":            task,
            "dataset":         yolo_folder,
        },
        message=f"Displayed {shown_total} image(s) from {list(summary_per_split.keys())}.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Folder prediction  (saves to disk — no display)
# ══════════════════════════════════════════════════════════════════════════════

def predict_folder(
    image_folder:   str,
    model_path:     str | None = None,
    show_boxes:     bool  = True,
    show_masks:     bool  = True,
    show_labels:    bool  = True,
    conf_threshold: float = 0.25,
) -> dict:
    """
    Run YOLO inference on every image in a folder.

    For each image saves:
      - A side-by-side PNG  (original | overlay with boxes / masks / labels)
      - A single CSV        (all detections from all images, one row per object)

    Output → <image_folder>/out_prediction_<timestamp>/

    CSV columns
    -----------
    index, filename, label, class_id, confidence,
    x1, y1, x2, y2, img_width, img_height,
    area_px, perimeter, cx, cy,
    aspect_ratio, extent, solidity, equivalent_diameter,
    compactness, orientation_deg, major_axis, minor_axis,
    features_from

    Parameters
    ----------
    image_folder   : str        Folder containing input images.
    model_path     : str|None   .pt file; uses latest best.pt if None.
    show_boxes     : bool       Draw bounding boxes in overlay.
    show_masks     : bool       Draw segmentation masks in overlay.
    show_labels    : bool       Draw class name + confidence text.
    conf_threshold : float      Minimum confidence (default 0.25).
    """
    import cv2

    # ── Resolve model ──────────────────────────────────────────────────────────
    if model_path is None:
        model_path = find_latest_best_pt()
    if model_path is None:
        return _err("No model found. Train a model first or provide a model_path.")
    if not Path(model_path).is_file():
        return _err(f"Model file not found: '{model_path}'")

    # ── Resolve images ─────────────────────────────────────────────────────────
    img_dir = Path(image_folder).resolve()
    if not img_dir.is_dir():
        return _err(f"Image folder not found: '{image_folder}'")

    img_files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    )
    if not img_files:
        return _err(f"No images found in '{image_folder}'.")

    # ── Output dir & CSV ───────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = img_dir / f"out_prediction_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "detections.csv"

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model       = YOLO(model_path)
        task        = getattr(model, "task", "detect")
        names_dict  = getattr(model, "names", {})
        class_names = (
            list(names_dict.values()) if isinstance(names_dict, dict) else names_dict
        )
    except Exception as exc:
        return _err(f"Could not load model: {exc}")

    # ── Inference loop ─────────────────────────────────────────────────────────
    csv_rows: list[dict]  = []
    det_index  = 0
    n_processed = 0

    for img_path in img_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            ui.log('WARN', f"Could not read '{img_path.name}', skipping.")
            continue

        H, W    = img_bgr.shape[:2]
        overlay = img_bgr.copy()

        try:
            preds = model.predict(str(img_path), conf=conf_threshold, verbose=False)
        except Exception as exc:
            ui.log('WARN', f"Inference failed for '{img_path.name}': {exc}")
            continue

        pred = preds[0]

        # ── Build per-detection masks dict (index → binary mask H×W) ──────────
        mask_by_det: dict[int, np.ndarray] = {}
        if hasattr(pred, "masks") and pred.masks is not None:
            for i, seg_mask in enumerate(pred.masks.data.cpu().numpy()):
                mask_resized = cv2.resize(
                    seg_mask.astype(np.uint8),
                    (W, H), interpolation=cv2.INTER_NEAREST,
                )
                mask_by_det[i] = mask_resized

        # ── Draw masks onto overlay ────────────────────────────────────────────
        if show_masks and mask_by_det:
            mask_layer = overlay.copy()
            for i, m in mask_by_det.items():
                cls = int(pred.boxes.cls[i].item()) if pred.boxes is not None else 0
                col = _color(cls)
                mask_layer[m > 0] = col
            overlay = cv2.addWeighted(overlay, 0.6, mask_layer, 0.4, 0)

        # ── Per-detection: boxes, labels, feature extraction, CSV ─────────────
        if pred.boxes is not None:
            for i, box in enumerate(pred.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0].item())
                cls   = int(box.cls[0].item())
                col   = _color(cls)
                label = class_names[cls] if cls < len(class_names) else str(cls)

                # ── Feature extraction ─────────────────────────────────────────
                if i in mask_by_det:
                    feats = _features_from_mask(mask_by_det[i])
                else:
                    # Detection mode: use contour on bbox crop
                    feats = _features_from_bbox(x1, y1, x2, y2, img_bgr)

                # ── Draw box ───────────────────────────────────────────────────
                if show_boxes:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)

                if show_labels:
                    txt = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        overlay,
                        (x1, max(y1 - th - 6, 0)),
                        (x1 + tw + 4, y1),
                        col, -1,
                    )
                    cv2.putText(
                        overlay, txt, (x1 + 2, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )

                # ── Accumulate CSV row ─────────────────────────────────────────
                row: dict = {
                    "index":      det_index,
                    "filename":   img_path.name,
                    "label":      label,
                    "class_id":   cls,
                    "confidence": round(conf, 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "img_width":  W,
                    "img_height": H,
                }
                row.update(feats)
                csv_rows.append(row)
                det_index += 1

        # ── Save side-by-side PNG ──────────────────────────────────────────────
        combined = np.concatenate([img_bgr, overlay], axis=1)
        cv2.imwrite(str(out_dir / img_path.name), combined)
        n_img_dets = sum(1 for r in csv_rows if r["filename"] == img_path.name)
        n_processed += 1
        ui.log("INFO", f"[{n_processed}/{len(img_files)}] {img_path.name} — {n_img_dets} detection(s)")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    return _ok(
        data={
            "output_dir":         str(out_dir),
            "n_images_processed": n_processed,
            "total_detections":   det_index,
            "csv_path":           str(csv_path),
            "model_path":         model_path,
            "task":               task,
            "class_names":        class_names,
        },
        message=(
            f"Prediction complete: {n_processed} image(s) processed, "
            f"{det_index} detection(s). Output → '{out_dir}'."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GWY experimental file prediction pipeline
# ══════════════════════════════════════════════════════════════════════════════

GWY_CSV_FIELDNAMES = [
    # Identity
    "index", "gwy_source", "filename", "label", "class_id", "confidence",
    # Bounding box (pixels)
    "x1", "y1", "x2", "y2",
    "img_width", "img_height",
    # Physical dimensions (µm)
    "x_real_um", "y_real_um", "pixel_size_um",
    "bbox_w_um", "bbox_h_um",
    # Geometry — basic (pixels)
    "area_px", "perimeter",
    "cx", "cy",
    # Geometry — shape
    "aspect_ratio", "extent", "solidity", "equivalent_diameter",
    # Geometry — ellipse fit
    "compactness", "orientation_deg",
    "major_axis", "minor_axis",
    # Physical area
    "area_um2",
    # Source
    "features_from",
    # GWY metadata
    "scan_direction",
]


def predict_gwy_folder(
    gwy_folder:     str,
    model_path:     str | None = None,
    scan_direction: str   = "forward",
    flatten_perc:   float = 40.0,
    show_boxes:     bool  = True,
    show_masks:     bool  = True,
    show_labels:    bool  = True,
    conf_threshold: float = 0.25,
) -> dict:
    """
    Load .gwy experimental files, flatten topography, run YOLO inference,
    and save all outputs in a timestamped subfolder next to the .gwy files.

    Pipeline per file
    -----------------
    1. load_gwy_channel()      — reads the topographic channel
    2. flatten_line_by_line()  — AFM line-by-line flattening (same as automation)
    3. array_to_png()          — saves to extracted_png/
    4. YOLO inference          — overlays saved to overlays/
    5. CSV row with pixel + physical dimensions

    Output structure
    ----------------
    gwy_folder/
      gwy_analysis_<timestamp>/
        extracted_png/   — flattened PNG images
        overlays/        — side-by-side (raw | prediction) PNGs
        detections.csv   — all detections with pixel + physical columns
        analysis_config.json

    Parameters
    ----------
    gwy_folder     : str        Folder containing .gwy files.
    model_path     : str|None   YOLO .pt file; auto-selects latest best.pt if None.
    scan_direction : str        'forward' | 'backward' — selects the single topo
                                channel to read (default 'forward').
    flatten_perc   : float      Percentile of lowest pixels used for line flattening.
    show_boxes     : bool       Draw bounding boxes on overlays.
    show_masks     : bool       Draw segmentation masks on overlays.
    show_labels    : bool       Draw class + confidence labels.
    conf_threshold : float      Minimum YOLO confidence (default 0.25).
    """
    import cv2
    from src.automation.automation import load_gwy_channel, flatten_line_by_line, array_to_png

    # ── Resolve model ──────────────────────────────────────────────────────────
    if model_path is None:
        model_path = find_latest_best_pt()
    if model_path is None:
        return _err("No YOLO model found. Train one first or provide model_path.")
    if not Path(model_path).is_file():
        return _err(f"Model file not found: '{model_path}'")

    # ── Resolve GWY files ──────────────────────────────────────────────────────
    gwy_dir = Path(gwy_folder).resolve()
    if not gwy_dir.is_dir():
        return _err(f"Folder not found: '{gwy_folder}'")

    gwy_files = sorted(gwy_dir.glob("*.gwy"))
    if not gwy_files:
        return _err(f"No .gwy files found in '{gwy_folder}'.")

    # ── Output directories ─────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir     = gwy_dir / f"gwy_analysis_{ts}"
    png_dir     = out_dir / "extracted_png"
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir()
    overlay_dir.mkdir()

    # ── Load YOLO model ────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model       = YOLO(model_path)
        task        = getattr(model, "task", "detect")
        names_dict  = getattr(model, "names", {})
        class_names = list(names_dict.values()) if isinstance(names_dict, dict) else list(names_dict)
    except Exception as exc:
        return _err(f"Could not load YOLO model: {exc}")

    # ── Per-file pipeline ──────────────────────────────────────────────────────
    csv_rows: list[dict] = []
    det_index   = 0
    n_processed = 0
    n_failed    = 0

    for gwy_path in gwy_files:
        # 1. Load GWY channel
        try:
            data, meta = load_gwy_channel(str(gwy_path), scan_direction=scan_direction)
        except Exception as exc:
            ui.log("WARN", f"Cannot load '{gwy_path.name}': {exc}  — skipping.")
            n_failed += 1
            continue

        x_real_m   = meta.get("xreal", 0.0)
        y_real_m   = meta.get("yreal", 0.0)
        x_real_um  = x_real_m * 1e6
        y_real_um  = y_real_m * 1e6
        H, W       = data.shape
        px_size_um = x_real_um / W if W > 0 else 0.0

        # 2. Flatten
        flat, _ = flatten_line_by_line(data, perc=flatten_perc)

        # 3. Save PNG
        stem     = gwy_path.stem
        png_name = f"{stem}.png"
        png_path = png_dir / png_name
        array_to_png(flat, str(png_path))

        # 4. YOLO inference
        img_bgr = cv2.imread(str(png_path))
        if img_bgr is None:
            ui.log("WARN", f"PNG unreadable after save for '{gwy_path.name}' — skipping.")
            n_failed += 1
            continue

        img_h, img_w = img_bgr.shape[:2]
        overlay = img_bgr.copy()

        try:
            preds = model.predict(str(png_path), conf=conf_threshold, verbose=False)
        except Exception as exc:
            ui.log("WARN", f"Inference failed for '{gwy_path.name}': {exc}")
            n_failed += 1
            continue

        pred = preds[0]

        # Build masks dict
        mask_by_det: dict[int, np.ndarray] = {}
        if hasattr(pred, "masks") and pred.masks is not None:
            for i, seg_mask in enumerate(pred.masks.data.cpu().numpy()):
                mask_by_det[i] = cv2.resize(
                    seg_mask.astype(np.uint8), (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                )

        # Draw masks
        if show_masks and mask_by_det:
            mask_layer = overlay.copy()
            for i, m in mask_by_det.items():
                cls = int(pred.boxes.cls[i].item()) if pred.boxes is not None else 0
                mask_layer[m > 0] = _color(cls)
            overlay = cv2.addWeighted(overlay, 0.6, mask_layer, 0.4, 0)

        # Per-detection processing
        if pred.boxes is not None:
            for i, box in enumerate(pred.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0].item())
                cls   = int(box.cls[0].item())
                col   = _color(cls)
                label = class_names[cls] if cls < len(class_names) else str(cls)

                feats = (_features_from_mask(mask_by_det[i])
                         if i in mask_by_det
                         else _features_from_bbox(x1, y1, x2, y2, img_bgr))

                if show_boxes:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
                if show_labels:
                    txt = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(overlay, (x1, max(y1 - th - 6, 0)),
                                  (x1 + tw + 4, y1), col, -1)
                    cv2.putText(overlay, txt, (x1 + 2, max(y1 - 4, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                bbox_w_um = (x2 - x1) * px_size_um
                bbox_h_um = (y2 - y1) * px_size_um
                area_um2  = round(feats.get("area_px", 0) * px_size_um ** 2, 4)

                row: dict = {
                    "index":      det_index,
                    "gwy_source": gwy_path.name,
                    "filename":   png_name,
                    "label":      label,
                    "class_id":   cls,
                    "confidence": round(conf, 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "img_width":  img_w,
                    "img_height": img_h,
                    "x_real_um":  round(x_real_um, 4),
                    "y_real_um":  round(y_real_um, 4),
                    "pixel_size_um": round(px_size_um, 6),
                    "bbox_w_um":  round(bbox_w_um, 4),
                    "bbox_h_um":  round(bbox_h_um, 4),
                    "area_um2":   area_um2,
                    "scan_direction": scan_direction,
                }
                row.update(feats)
                csv_rows.append(row)
                det_index += 1

        # Save side-by-side overlay
        combined = np.concatenate([img_bgr, overlay], axis=1)
        cv2.imwrite(str(overlay_dir / png_name), combined)

        n_processed += 1
        n_dets = sum(1 for r in csv_rows if r["gwy_source"] == gwy_path.name)
        ui.log("INFO", f"[{n_processed}/{len(gwy_files)}] {gwy_path.name} "
                       f"({x_real_um:.1f}×{y_real_um:.1f} µm) — {n_dets} detection(s)")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "detections.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GWY_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Write analysis config JSON ─────────────────────────────────────────────
    import json as _json
    config_path = out_dir / "analysis_config.json"
    config = {
        "model_path":     model_path,
        "model_task":     task,
        "class_names":    class_names,
        "scan_direction": scan_direction,
        "flatten_perc":   flatten_perc,
        "conf_threshold": conf_threshold,
        "show_boxes":     show_boxes,
        "show_masks":     show_masks,
        "show_labels":    show_labels,
        "timestamp":      ts,
        "n_gwy_files":    len(gwy_files),
        "n_processed":    n_processed,
        "n_failed":       n_failed,
        "total_detections": det_index,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        _json.dump(config, f, indent=2)

    return _ok(
        data={
            "output_dir":         str(out_dir),
            "extracted_png_dir":  str(png_dir),
            "overlays_dir":       str(overlay_dir),
            "csv_path":           str(csv_path),
            "config_path":        str(config_path),
            "n_gwy_files":        len(gwy_files),
            "n_processed":        n_processed,
            "n_failed":           n_failed,
            "total_detections":   det_index,
            "model_path":         model_path,
            "class_names":        class_names,
        },
        message=(
            f"GWY analysis complete: {n_processed}/{len(gwy_files)} files, "
            f"{det_index} detection(s). Output → '{out_dir}'."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _imread(path: str):
    import cv2
    return cv2.imread(path)


def _parse_yolo_label(
    lbl_path: Path, W: int, H: int, task: str
) -> list[dict[str, Any]]:
    """
    Parse a YOLO label .txt file into a list of annotation dicts.
    Keys: cls (int), box (x1,y1,x2,y2 px), poly (Nx2 array px | None).
    """
    if not lbl_path.is_file():
        return []
    annotations = []
    with open(lbl_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls  = int(parts[0])
            nums = [float(v) for v in parts[1:]]

            if task == "segment" and len(nums) >= 6:
                poly_px = np.array(
                    [[nums[i] * W, nums[i+1] * H] for i in range(0, len(nums)-1, 2)]
                )
                xs, ys = poly_px[:, 0], poly_px[:, 1]
                box    = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                annotations.append({"cls": cls, "box": box, "poly": poly_px})

            elif len(nums) == 4:
                xc, yc, w, h = nums
                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)
                annotations.append({"cls": cls, "box": (x1, y1, x2, y2), "poly": None})

    return annotations
"""
src/trainer/triplet_trainer.py
──────────────────────────────
Triplet Encoder training utilities for SimuScan AFM datasets.

Pipeline
--------
1. Load images from a YOLO_SEG dataset (train / val / test splits).
2. Run YOLO segmentation inference → bboxes + masks per object.
3. Filter isolated objects by pairwise IoU between YOLO bounding boxes.
   Objects whose bbox overlaps with any neighbour above max_iou are dropped,
   keeping only the higher-confidence detection of each conflicting pair.
4. Crop + normalise each object patch (masked background = 0).
5. Train a TripletEncoder (anchor / positive / negative by class).
6. Evaluate embeddings:
   - Triplet loss curve (train + val) + triplet accuracy
   - UMAP scatter plot (all objects, coloured by class)
   - Silhouette score — global and per class
   - kNN leave-one-out accuracy
7. Save model, metrics CSV and plots to data/out_triplet/time_<stamp>/.

Public API
----------
run_triplet_training(yolo_seg_folder, yolo_model_path, ...)  → dict
list_triplet_runs()                                          → dict
get_triplet_results(run_dir)                                 → dict
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path helpers ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _out_triplet_root() -> Path:
    return _repo_root() / "data" / "out_triplet"


def _datasets_root() -> Path:
    return _repo_root() / "data" / "Datasets"


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


# ══════════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TripletConfig:
    # ── Paths (resolved at runtime) ──────────────────────────────────────────
    yolo_seg_folder : str = ""
    yolo_model_path : str = ""
    output_dir      : str = ""

    # ── YOLO inference ────────────────────────────────────────────────────────
    splits       : tuple = ("train", "val", "test")
    yolo_conf    : float = 0.50
    yolo_iou     : float = 0.45

    # ── Overlap / isolation filtering ─────────────────────────────────────────
    # Pairwise bbox IoU above max_iou → drop lower-confidence object.
    max_iou          : float = 0.10
    # Minimum pixel distance between any two objects' bboxes.
    min_separation_px: int   = 10
    # Minimum distance from image border (fraction of min image dimension).
    border_margin_pct: float = 0.05

    # ── Crop normalisation ────────────────────────────────────────────────────
    crop_size     : int   = 64
    margin_pct    : float = 0.15
    # Otsu threshold floor: if Otsu result < this fraction, clip up to it.
    otsu_min_frac : float = 0.10
    save_crops    : bool  = True

    # ── TripletEncoder ────────────────────────────────────────────────────────
    latent_dim       : int   = 32
    triplet_epochs   : int   = 40
    triplet_batch    : int   = 64
    triplet_lr       : float = 1e-3
    triplet_margin   : float = 0.5
    val_fraction     : float = 0.2

    # ── Evaluation ────────────────────────────────────────────────────────────
    knn_k            : int   = 5
    umap_neighbors   : int   = 15
    umap_min_dist    : float = 0.1

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed   : int = 42
    device : str = ""   # auto-detected if empty


# ══════════════════════════════════════════════════════════════════════════════
# 1. Image loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_image_paths(cfg: TripletConfig) -> list[tuple[Path, str]]:
    """Return list of (image_path, split_name) for all configured splits."""
    root  = Path(cfg.yolo_seg_folder) / "images"
    paths: list[tuple[Path, str]] = []
    for split in cfg.splits:
        d = root / split
        if not d.exists():
            continue
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            paths.append((p, split))
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 2. YOLO inference
# ══════════════════════════════════════════════════════════════════════════════

def _detect_with_yolo(img_paths: list[tuple[Path, str]], cfg: TripletConfig) -> list[dict]:
    """Run YOLO segmentation; return list of per-image dicts (includes split name)."""
    from ultralytics import YOLO
    from tqdm import tqdm

    model   = YOLO(cfg.yolo_model_path)
    results = []

    for p, split in tqdm(img_paths, desc="YOLO seg inference"):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_bgr.shape[:2]

        preds   = model.predict(source=img_bgr, conf=cfg.yolo_conf,
                                iou=cfg.yolo_iou, verbose=False)[0]
        boxes   = preds.boxes.xyxy.cpu().numpy().astype(int)
        labels  = preds.boxes.cls.cpu().numpy().astype(int)
        confs   = preds.boxes.conf.cpu().numpy()

        masks: list[np.ndarray] = []
        if preds.masks is not None:
            raw_masks = preds.masks.data.cpu().numpy()
            for m in raw_masks:
                m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                masks.append((m_resized > 0.5).astype(np.uint8))
        else:
            for box in boxes:
                m = np.zeros((H, W), dtype=np.uint8)
                x1, y1, x2, y2 = box
                m[y1:y2, x1:x2] = 1
                masks.append(m)

        results.append(dict(
            img_path=p, img=img_rgb, split=split,
            boxes=boxes, labels=labels, confs=confs, masks=masks,
        ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. Overlap filtering
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def _bbox_min_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum Euclidean distance between two non-overlapping bboxes [x1,y1,x2,y2]."""
    dx = max(0, int(max(a[0], b[0])) - int(min(a[2], b[2])))
    dy = max(0, int(max(a[1], b[1])) - int(min(a[3], b[3])))
    return float((dx * dx + dy * dy) ** 0.5)


def _filter_isolated(detections: list[dict], cfg: TripletConfig) -> list[dict]:
    """
    Keep only objects that are:
      - Not overlapping with neighbours (pairwise bbox IoU > max_iou → drop lower-conf).
      - Not too close to other objects (bbox distance < min_separation_px → drop lower-conf).
      - Not too close to the image border (bbox within border_margin_pct * min(H,W) → drop).
    """
    filtered = []
    for det in detections:
        boxes  = det["boxes"]
        labels = det["labels"]
        confs  = det["confs"]
        masks  = det["masks"]
        img    = det["img"]
        H, W   = img.shape[:2]
        N      = len(boxes)

        if N == 0:
            filtered.append(det)
            continue

        border = int(cfg.border_margin_pct * min(H, W))
        keep   = [True] * N

        # Border filter (independent — drop any box that touches the margin)
        for i in range(N):
            x1, y1, x2, y2 = boxes[i]
            if x1 < border or y1 < border or x2 > W - border or y2 > H - border:
                keep[i] = False

        # Pairwise: IoU overlap + minimum separation distance
        for i in range(N):
            for j in range(i + 1, N):
                if not keep[i] or not keep[j]:
                    continue
                too_close = (
                    _bbox_iou(boxes[i], boxes[j]) > cfg.max_iou
                    or _bbox_min_dist(boxes[i], boxes[j]) < cfg.min_separation_px
                )
                if too_close:
                    if confs[i] >= confs[j]:
                        keep[j] = False
                    else:
                        keep[i] = False

        idx = [k for k, v in enumerate(keep) if v]
        filtered.append(dict(
            img_path = det["img_path"],
            img      = img,
            split    = det["split"],
            boxes    = boxes[idx]         if idx else np.empty((0, 4), int),
            labels   = labels[idx]        if idx else np.empty((0,), int),
            confs    = confs[idx]         if idx else np.empty((0,), float),
            masks    = [masks[k] for k in idx],
        ))

    n_before = sum(len(d["boxes"]) for d in detections)
    n_after  = sum(len(d["boxes"]) for d in filtered)
    print(f"[Filter] Objects before: {n_before}  after: {n_after}  "
          f"(dropped {n_before - n_after}; "
          f"max_iou={cfg.max_iou}, min_sep={cfg.min_separation_px}px, "
          f"border={cfg.border_margin_pct*100:.0f}%)")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# 4. Crop + normalise
# ══════════════════════════════════════════════════════════════════════════════

def _otsu_on_patch(patch: np.ndarray, min_frac: float = 0.10) -> float:
    """
    Otsu threshold computed only on non-zero pixels of a float32 [0,1] patch.
    Returns threshold in [0,1], clamped to at least min_frac.
    Uses cv2's Otsu on a 1-row uint8 image built from non-zero pixels.
    """
    obj_pix = patch[patch > 0]
    if len(obj_pix) < 10:
        return min_frac
    pix_u8 = (obj_pix * 255).clip(0, 255).astype(np.uint8).reshape(1, -1)
    t, _ = cv2.threshold(pix_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(t) / 255.0, min_frac)


def _largest_cc(binary: np.ndarray) -> np.ndarray:
    """Return binary mask containing only the largest 8-connected component."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    if num <= 1:
        return binary.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return (labels == (int(np.argmax(areas)) + 1)).astype(np.uint8)


def _compute_angle_svd(mask: np.ndarray) -> float:
    """SVD on non-zero mask pixel coordinates → major-axis angle (degrees)."""
    coords = np.column_stack(np.where(mask > 0))  # (row, col)
    if len(coords) < 5:
        return 0.0
    mean = coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords - mean)
    return float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])))


def _rotate_patch_and_mask(gray: np.ndarray, mask: np.ndarray,
                            angle: float) -> tuple:
    """Rotate both the normalised gray patch and the binary mask by `angle` degrees."""
    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    rot_gray = cv2.warpAffine(gray, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    rot_mask_f = cv2.warpAffine(mask.astype(np.float32), M, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return rot_gray, (rot_mask_f > 0.5).astype(np.uint8)


def _resolve_180(gray: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Resolve 180° ambiguity: ensure the center of mass (computed on a
    Gaussian-blurred masked patch to reduce noise influence) is in the right half.
    Flips both gray and mask horizontally if not.
    """
    blurred = cv2.GaussianBlur(gray * mask.astype(np.float32), (0, 0), sigmaX=2.0)
    total   = blurred.sum()
    if total == 0:
        return gray, mask
    col_cm = float((blurred * np.arange(gray.shape[1])[None, :]).sum() / total)
    if col_cm < gray.shape[1] / 2.0:
        gray = cv2.flip(gray, 1)
        mask = cv2.flip(mask, 1)
    return gray, mask


def _center_and_pad(gray: np.ndarray, mask: np.ndarray, sz: int) -> np.ndarray:
    """Crop to the tight bounding box of the mask, then letterbox-pad to sz×sz."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros((sz, sz), dtype=np.float32)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    cropped = gray[y1:y2, x1:x2]
    # Letterbox resize preserving aspect ratio
    h, w    = cropped.shape
    scale   = sz / max(h, w)
    nh, nw  = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((sz, sz), dtype=np.float32)
    y0      = (sz - nh) // 2
    x0      = (sz - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _crop_and_normalise(detections: list[dict], cfg: TripletConfig,
                        crop_dir: Path) -> list[dict]:
    from tqdm import tqdm

    crops: list[dict] = []
    sz   = cfg.crop_size
    marg = cfg.margin_pct

    for det in tqdm(detections, desc="Cropping"):
        img   = det["img"]
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        H, W  = gray.shape
        stem  = Path(det["img_path"]).stem

        for k, (box, lbl, conf, mask) in enumerate(
                zip(det["boxes"], det["labels"], det["confs"], det["masks"])):

            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            mx, my = int(bw * marg), int(bh * marg)
            x1c = max(0, x1 - mx); y1c = max(0, y1 - my)
            x2c = min(W, x2 + mx); y2c = min(H, y2 + my)

            # ── Step 2: apply YOLO mask → zero background ─────────────────────
            patch = (gray * mask.astype(np.float32))[y1c:y2c, x1c:x2c]
            pmask = mask[y1c:y2c, x1c:x2c]
            if patch.size == 0:
                continue

            # ── Step 3: normalise intensity (percentile 2-98 on non-zero px) ──
            obj_pix = patch[patch > 0]
            if len(obj_pix) < 10:
                continue
            p2, p98 = np.percentile(obj_pix, 2), np.percentile(obj_pix, 98)
            if p98 > p2:
                patch = np.clip((patch - p2) / (p98 - p2), 0, 1)
            else:
                patch = (patch > 0).astype(np.float32)

            # ── Step 4: Otsu on non-zero px → separate object from substrate ──
            thresh = _otsu_on_patch(patch, cfg.otsu_min_frac)
            binary = ((patch >= thresh) & (pmask > 0)).astype(np.uint8)

            # ── Step 5: largest connected component ───────────────────────────
            clean = _largest_cc(binary)
            if int(clean.sum()) < 10:
                continue

            # ── Step 6: SVD → principal axis angle ───────────────────────────
            angle = _compute_angle_svd(clean)

            # ── Step 7: rotate normalised gray + clean mask ───────────────────
            rot_gray, rot_mask = _rotate_patch_and_mask(patch, clean, angle)

            # ── Step 8: resolve 180° ambiguity (Gaussian-blurred mass) ────────
            rot_gray, rot_mask = _resolve_180(rot_gray, rot_mask)

            # ── Step 9: re-apply clean rotated mask (remove substrate noise) ──
            rot_gray = rot_gray * rot_mask.astype(np.float32)

            # ── Steps 10+11: center on object bbox + letterbox pad to sz×sz ───
            result = _center_and_pad(rot_gray, rot_mask, sz)

            crop_path = ""
            if cfg.save_crops:
                out_path = crop_dir / f"{stem}_obj{k:03d}_cls{lbl}.png"
                cv2.imwrite(str(out_path), (result * 255).astype(np.uint8))
                crop_path = str(out_path)

            crops.append(dict(
                crop     = result,
                label    = int(lbl),
                conf     = float(conf),
                split    = det["split"],
                img_path = str(det["img_path"]),
                bbox_orig= [int(x1), int(y1), int(x2), int(y2)],
                crop_id  = f"{stem}_obj{k:03d}",
                crop_path= crop_path,
            ))

    cnt = Counter(c["label"] for c in crops)
    print(f"[Crops] Total: {len(crops)}")
    for cls, n in sorted(cnt.items()):
        print(f"  class {cls}: {n}")
    return crops


# ══════════════════════════════════════════════════════════════════════════════
# 5. TripletEncoder + training
# ══════════════════════════════════════════════════════════════════════════════

def _build_encoder_and_dataset(cfg: TripletConfig):
    """Import torch lazily and return (TripletEncoder class, TripletDataset class)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from torchvision import transforms

    class TripletDataset(Dataset):
        def __init__(self, crops: list[dict], split: str = "train"):
            """
            Use YOLO dataset splits directly: 'train' crops for training,
            'val' crops for validation.  Falls back to random 80/20 split
            per class if a split has fewer than 2 classes after filtering.
            """
            assert split in ("train", "val")

            # Primary: filter by YOLO split label
            selected = [c for c in crops if c.get("split", "train") == split]

            # Fallback: if fewer than 2 classes survive, use random split
            classes_found = {c["label"] for c in selected}
            if len(classes_found) < 2:
                rng = np.random.default_rng(cfg.seed)
                by_class_all: dict[int, list[int]] = {}
                for i, c in enumerate(crops):
                    by_class_all.setdefault(c["label"], []).append(i)
                sel_idx: list[int] = []
                for cls, idxs in by_class_all.items():
                    arr = np.array(idxs); rng.shuffle(arr)
                    n_val = max(1, int(len(arr) * cfg.val_fraction))
                    sel_idx.extend(arr[:n_val].tolist() if split == "val"
                                   else arr[n_val:].tolist())
                selected = [crops[i] for i in sel_idx]

            self.crops  = selected
            self.labels = [c["label"] for c in self.crops]
            self.by_class: dict[int, list[int]] = {}
            for i, c in enumerate(self.crops):
                self.by_class.setdefault(c["label"], []).append(i)
            self.classes = list(self.by_class.keys())
            if len(self.classes) < 2:
                raise ValueError(f"Need ≥2 classes in '{split}' split, found: {self.classes}")

            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(360),
                transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
            ])

        def __len__(self): return len(self.crops)

        def __getitem__(self, idx):
            anchor_lbl = self.labels[idx]
            pos_pool   = [i for i in self.by_class[anchor_lbl] if i != idx]
            pos_idx    = int(np.random.choice(pos_pool)) if pos_pool else idx

            neg_classes = [l for l in self.classes if l != anchor_lbl]
            weights     = np.array([len(self.by_class[l]) for l in neg_classes],
                                    dtype=float)
            weights    /= weights.sum()
            neg_lbl     = neg_classes[np.random.choice(len(neg_classes), p=weights)]
            neg_idx     = int(np.random.choice(self.by_class[neg_lbl]))

            def to_t(i, strong=False):
                t = torch.from_numpy(self.crops[i]["crop"]).unsqueeze(0)
                return self.aug(t)

            return to_t(idx), to_t(pos_idx, strong=(pos_idx == idx)), to_t(neg_idx)

    class TripletEncoder(nn.Module):
        def __init__(self, ld: int = 32):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1,  16, 3, stride=2, padding=1), nn.BatchNorm2d(16),  nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
                nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, ld),
            )

        def forward(self, x):
            z = self.fc(self.conv(x).flatten(1))
            return nn.functional.normalize(z, p=2, dim=1)

    return TripletEncoder, TripletDataset


def _train_triplet_encoder(crops: list[dict], cfg: TripletConfig,
                            out_dir: Path) -> tuple:
    """
    Train TripletEncoder. Returns (encoder, history_dict).
    Saves best weights to out_dir/triplet_encoder.pt.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    TripletEncoder, TripletDataset = _build_encoder_and_dataset(cfg)

    ds_train = TripletDataset(crops, split="train")
    ds_val   = TripletDataset(crops, split="val")

    loader_tr = DataLoader(ds_train, batch_size=cfg.triplet_batch,
                           shuffle=True,  num_workers=0, drop_last=True)
    loader_va = DataLoader(ds_val,   batch_size=cfg.triplet_batch,
                           shuffle=False, num_workers=0, drop_last=False)

    encoder = TripletEncoder(cfg.latent_dim).to(device)
    opt     = optim.Adam(encoder.parameters(), lr=cfg.triplet_lr, weight_decay=1e-5)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.triplet_epochs)
    loss_fn = nn.TripletMarginLoss(margin=cfg.triplet_margin, p=2)

    best_val_loss = float("inf")
    pt_path       = out_dir / "triplet_encoder.pt"
    history       = {"train_loss": [], "val_loss": [], "val_acc": []}

    from src.bridge.ui_bridge import ui

    header = (f"Triplet training — {cfg.triplet_epochs} epochs | "
              f"latent={cfg.latent_dim} | margin={cfg.triplet_margin} | device={device}")
    print(f"[Triplet] {header}")
    ui.log("INFO", header)

    def _update_live_curve() -> None:
        """Redraw the live triplet-loss curve (one point per epoch), single slot.

        Sent via ui.plot_live() so it is NOT recorded to the session gallery.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            tl = history["train_loss"]
            vl = history["val_loss"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(1, len(tl) + 1), tl, label="Train", color="#f85149",
                    linewidth=1.5, marker="o", markersize=3)
            if vl:
                ax.plot(range(1, len(vl) + 1), vl, label="Val", color="#3fb950",
                        linewidth=1.5, marker="o", markersize=3)
            ax.set_title("Triplet — training loss (live)", fontsize=10, color="#c9d1d9")
            ax.set_xlabel("Epoch", color="#8b949e", fontsize=8)
            ax.set_ylabel("loss", color="#8b949e", fontsize=8)
            ax.set_xlim(1, max(int(cfg.triplet_epochs), len(tl)))
            ax.legend(labelcolor="#c9d1d9", facecolor="#161b22",
                      edgecolor="#30363d", fontsize=8)
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="#8b949e", labelsize=7)
            ax.grid(True, color="#21262d", linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            fig.patch.set_facecolor("#161b22")
            fig.tight_layout()
            ui.plot_live(fig, focus=(len(tl) <= 1))  # Live view; auto-shows on 1st epoch
        except Exception:
            pass  # non-critical

    for epoch in range(1, cfg.triplet_epochs + 1):
        encoder.train()
        tr_loss = 0.0
        for a, p, n in loader_tr:
            ea = encoder(a.to(device))
            ep = encoder(p.to(device))
            en = encoder(n.to(device))
            loss = loss_fn(ea, ep, en)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * a.size(0)
        tr_loss /= max(len(ds_train), 1)

        encoder.eval()
        va_loss = n_corr = n_tot = 0
        with torch.no_grad():
            for a, p, n in loader_va:
                ea = encoder(a.to(device))
                ep = encoder(p.to(device))
                en = encoder(n.to(device))
                va_loss += loss_fn(ea, ep, en).item() * a.size(0)
                d_ap = (ea - ep).pow(2).sum(1).sqrt()
                d_an = (ea - en).pow(2).sum(1).sqrt()
                n_corr += (d_ap < d_an).sum().item()
                n_tot  += a.size(0)
        va_loss /= max(len(ds_val), 1)
        va_acc   = n_corr / max(n_tot, 1)
        sched.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # Live loss curve — single updating plot, NOT saved to the gallery
        _update_live_curve()

        is_best = va_loss < best_val_loss
        if is_best:
            best_val_loss = va_loss
            torch.save(encoder.state_dict(), pt_path)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            msg = (f"Epoch {epoch:3d}/{cfg.triplet_epochs}  "
                   f"train={tr_loss:.4f}  val={va_loss:.4f}  "
                   f"acc={va_acc*100:.1f}%"
                   + ("  ★ best" if is_best else ""))
            print(f"  {msg}")
            ui.log("TRAIN", msg)

    encoder.load_state_dict(torch.load(pt_path, map_location=device))
    encoder.eval()
    done_msg = (f"Done. Best val_loss={best_val_loss:.4f}  "
                f"final acc={history['val_acc'][-1]*100:.1f}%")
    print(f"[Triplet] {done_msg}")
    ui.log("INFO", done_msg)
    return encoder, history


# ══════════════════════════════════════════════════════════════════════════════
# 6. Evaluation: embeddings + metrics
# ══════════════════════════════════════════════════════════════════════════════

def _extract_embeddings(crops: list[dict], encoder, cfg: TripletConfig,
                        device: str) -> np.ndarray:
    """Return (N, latent_dim) float32 array."""
    import torch
    encoder.eval()
    embs = []
    with torch.no_grad():
        for c in crops:
            t = torch.from_numpy(c["crop"]).unsqueeze(0).unsqueeze(0).to(device)
            embs.append(encoder(t).cpu().numpy()[0])
    return np.array(embs, dtype=np.float32)


def _silhouette(embs: np.ndarray, labels: list[int]) -> dict:
    """
    Per-class and global silhouette score.
    Returns {"global": float, "per_class": {cls: float}}.
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    lbl_arr = np.array(labels)
    unique  = np.unique(lbl_arr)
    if len(unique) < 2 or len(embs) < 4:
        return {"global": float("nan"), "per_class": {}}
    try:
        scores  = silhouette_samples(embs, lbl_arr)
        global_ = float(silhouette_score(embs, lbl_arr))
        per_cls = {int(c): float(scores[lbl_arr == c].mean()) for c in unique}
        return {"global": global_, "per_class": per_cls}
    except Exception:
        return {"global": float("nan"), "per_class": {}}


def _knn_accuracy(embs: np.ndarray, labels: list[int], k: int = 5) -> float:
    """Leave-one-out kNN accuracy."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import LeaveOneOut
    if len(embs) < k + 1:
        return float("nan")
    try:
        loo   = LeaveOneOut()
        y     = np.array(labels)
        preds = np.empty(len(y), dtype=y.dtype)
        for train_idx, test_idx in loo.split(embs):
            knn = KNeighborsClassifier(n_neighbors=min(k, len(train_idx)))
            knn.fit(embs[train_idx], y[train_idx])
            preds[test_idx] = knn.predict(embs[test_idx])
        return float((preds == y).mean())
    except Exception:
        return float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Visualisations
# ══════════════════════════════════════════════════════════════════════════════

def _plot_loss(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Triplet Loss", color="#c9d1d9")
    axes[0].set_xlabel("Epoch", color="#8b949e")
    axes[0].set_ylabel("Loss",  color="#8b949e")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    axes[0].grid(True, color="#30363d")

    axes[1].plot(epochs, [v * 100 for v in history["val_acc"]], color="#3fb950")
    axes[1].set_title("Val Triplet Accuracy (%)", color="#c9d1d9")
    axes[1].set_xlabel("Epoch",      color="#8b949e")
    axes[1].set_ylabel("Accuracy %", color="#8b949e")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, color="#30363d")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_umap(embs: np.ndarray, labels: list[int],
               class_names: dict[int, str], out_path: Path) -> None:
    try:
        import umap
    except ImportError:
        print("  [UMAP] umap-learn not installed; skipping UMAP plot.")
        return
    lbl_arr = np.array(labels)
    reducer = umap.UMAP(n_neighbors=min(15, len(embs) - 1),
                        min_dist=0.1, random_state=42)
    proj = reducer.fit_transform(embs)
    unique = np.unique(lbl_arr)
    cmap   = plt.get_cmap("tab20", len(unique))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    for i, cls in enumerate(unique):
        idx = lbl_arr == cls
        name = class_names.get(int(cls), f"cls{cls}")
        ax.scatter(proj[idx, 0], proj[idx, 1], s=12, alpha=0.7,
                   color=cmap(i), label=name)
    ax.legend(markerscale=2, fontsize=7, loc="best",
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.set_title("UMAP — Triplet Embeddings", color="#c9d1d9")
    ax.set_xlabel("UMAP-1", color="#8b949e")
    ax.set_ylabel("UMAP-2", color="#8b949e")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_silhouette(sil: dict, class_names: dict[int, str], out_path: Path) -> None:
    per_cls = sil.get("per_class", {})
    if not per_cls:
        return
    classes = sorted(per_cls.keys())
    scores  = [per_cls[c] for c in classes]
    names   = [class_names.get(int(c), f"cls{c}") for c in classes]
    colors  = ["steelblue" if s >= 0 else "salmon" for s in scores]

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8 + 2), 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    ax.bar(names, scores, color=colors)
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.axhline(sil.get("global", 0), color="#f85149", linewidth=1.2,
               linestyle="--", label=f"Global={sil['global']:.3f}")
    ax.set_title("Silhouette Score per Class", color="#c9d1d9")
    ax.set_ylabel("Silhouette", color="#8b949e")
    ax.set_ylim(-1, 1)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    plt.xticks(rotation=30, ha="right", color="#8b949e")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7b. Build library.json (same format as the experimental triplet library)
# ══════════════════════════════════════════════════════════════════════════════

def _build_library_json(crops: list[dict], embs: np.ndarray,
                        class_names: dict[int, str], out_dir: Path) -> int:
    """
    Build a library.json from the synthetic training crops so the interactive
    explorer (tool_show_triplet_library_explorer) works on training runs too.

    Reuses the same descriptor / clustering / novelty / shape-feature logic as
    the experimental library (triplet_inference) to keep the format identical.
    Uses ground-truth class labels (no YOLO inference error on synthetic data).
    Returns the number of records written.
    """
    from src.trainer.triplet_inference import (
        _extract_shape_features, _build_descriptors, _auto_cluster,
    )
    from sklearn.metrics.pairwise import euclidean_distances

    # Build per-object records (shape features from the masked crop)
    records: list[dict] = []
    for i, c in enumerate(crops):
        crop      = c["crop"]
        clean_msk = (crop > 0).astype(np.uint8)
        feats     = _extract_shape_features(crop, clean_msk, pixel_size_nm=0.0)
        records.append({
            "embedding"      : embs[i].tolist(),
            "shape_features" : feats,
            "obj_id"         : c["crop_id"],
            "img_path"       : c["img_path"],
            "bbox"           : c["bbox_orig"],
            "yolo_class"     : int(c["label"]),
            "yolo_conf"      : float(c["conf"]),
            "crop_path"      : c.get("crop_path", ""),
            "split"          : c.get("split", "train"),
        })

    # Descriptor (embedding + scaled features), clustering and novelty
    embs_arr, _feats, descs = _build_descriptors(records)
    cluster_labels = _auto_cluster(descs, None)

    dist_mat = euclidean_distances(descs).astype(np.float32)
    np.fill_diagonal(dist_mat, np.inf)
    novelty = dist_mat.min(axis=1)

    lib_entries = []
    for i, r in enumerate(records):
        lib_entries.append({
            "obj_id"         : r["obj_id"],
            "embedding"      : [round(float(v), 6) for v in embs_arr[i]],
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
            "split"          : r["split"],
            "timestamp"      : out_dir.name,
        })
    with open(out_dir / "library.json", "w") as f:
        json.dump(lib_entries, f, indent=2)
    print(f"  → {out_dir / 'library.json'}  ({len(lib_entries)} objects)")
    return len(lib_entries)


# ══════════════════════════════════════════════════════════════════════════════
# 8. CSV saving
# ══════════════════════════════════════════════════════════════════════════════

def _save_training_csv(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
        w.writeheader()
        for e, tl, vl, va in zip(
                epochs, history["train_loss"],
                history["val_loss"], history["val_acc"]):
            w.writerow({"epoch": e, "train_loss": round(tl, 6),
                        "val_loss": round(vl, 6), "val_acc": round(va, 6)})
    print(f"  → {out_path}")


def _save_eval_csv(crops: list[dict], embs: np.ndarray,
                   class_names: dict[int, str], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "index", "crop_id", "label", "class_name", "conf"] +
            [f"emb_{i}" for i in range(embs.shape[1])])
        w.writeheader()
        for i, (c, e) in enumerate(zip(crops, embs)):
            row = {"index": i, "crop_id": c["crop_id"],
                   "label": c["label"],
                   "class_name": class_names.get(c["label"], f"cls{c['label']}"),
                   "conf": round(c["conf"], 4)}
            for j, v in enumerate(e):
                row[f"emb_{j}"] = round(float(v), 6)
            w.writerow(row)
    print(f"  → {out_path}")


def _save_metrics_json(metrics: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Class names helper
# ══════════════════════════════════════════════════════════════════════════════

def _read_class_names(yolo_seg_folder: str) -> dict[int, str]:
    """Read class names from dataset.yaml in the YOLO_SEG folder."""
    import yaml
    yaml_path = Path(yolo_seg_folder) / "dataset.yaml"
    if yaml_path.is_file():
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}
            names = data.get("names", [])
            if isinstance(names, list):
                return {i: n for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
        except Exception:
            pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Main public function
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TripletResult:
    status         : str
    message        : str
    run_dir        : str
    model_path     : str
    epochs_done    : int
    n_crops        : int
    n_classes      : int
    final_val_loss : float
    final_val_acc  : float
    silhouette_global : float
    knn_accuracy      : float
    class_counts   : dict = field(default_factory=dict)


def run_triplet_training(
    yolo_seg_folder   : str,
    yolo_model_path   : str,
    *,
    splits            : list[str]  = None,
    max_iou           : float = 0.10,
    min_separation_px : int   = 10,
    border_margin_pct : float = 0.05,
    crop_size         : int   = 64,
    margin_pct        : float = 0.15,
    otsu_min_frac     : float = 0.10,
    latent_dim        : int   = 32,
    triplet_epochs    : int   = 40,
    triplet_batch     : int   = 64,
    triplet_lr        : float = 1e-3,
    triplet_margin    : float = 0.5,
    knn_k             : int   = 5,
    seed              : int   = 42,
    save_crops        : bool  = True,
) -> dict:
    """
    Run the full triplet encoder training pipeline.

    Parameters
    ----------
    yolo_seg_folder   : Path to the _yolo_seg dataset folder.
    yolo_model_path   : Path to a trained YOLO segmentation .pt file.
    splits            : Dataset splits to use (default: ['train','val','test']).
    max_iou           : Max pairwise bbox IoU; above this the lower-conf object
                        is dropped (default 0.10).
    min_separation_px : Minimum pixel distance between any two objects' bboxes;
                        closer pairs drop the lower-conf object (default 10).
    border_margin_pct : Objects whose bbox is within this fraction of min(H,W)
                        from any image edge are dropped (default 0.05 = 5%).
    crop_size         : Pixel size of square crop (default 64).
    margin_pct        : Fractional margin around bbox (default 0.15).
    otsu_min_frac     : Floor for Otsu threshold on normalised patch (default 0.10).
    latent_dim        : Embedding dimension (default 32).
    triplet_epochs    : Training epochs (default 40).
    triplet_batch     : Batch size (default 64).
    triplet_lr        : Learning rate (default 1e-3).
    triplet_margin    : Triplet loss margin (default 0.5).
    knn_k             : k for kNN leave-one-out evaluation (default 5).
    seed              : Random seed (default 42).
    save_crops        : Save crop images to disk (default True).

    Returns
    -------
    dict  {"status": "ok"|"error", "data": TripletResult, ...}
    """
    import torch

    # ── Validate inputs ───────────────────────────────────────────────────────
    yolo_seg_folder = str(Path(yolo_seg_folder).resolve())
    yolo_model_path = str(Path(yolo_model_path).resolve())

    if not Path(yolo_seg_folder).is_dir():
        return _err(f"YOLO_SEG folder not found: '{yolo_seg_folder}'")
    if not (Path(yolo_seg_folder) / "dataset.yaml").is_file():
        return _err(f"dataset.yaml not found in '{yolo_seg_folder}'")
    if not Path(yolo_model_path).is_file():
        return _err(f"YOLO model not found: '{yolo_model_path}'")

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = TripletConfig(
        yolo_seg_folder   = yolo_seg_folder,
        yolo_model_path   = yolo_model_path,
        splits            = tuple(splits or ["train", "val", "test"]),
        max_iou           = max_iou,
        min_separation_px = min_separation_px,
        border_margin_pct = border_margin_pct,
        crop_size         = crop_size,
        margin_pct        = margin_pct,
        otsu_min_frac     = otsu_min_frac,
        latent_dim        = latent_dim,
        triplet_epochs    = triplet_epochs,
        triplet_batch     = triplet_batch,
        triplet_lr        = triplet_lr,
        triplet_margin    = triplet_margin,
        knn_k             = knn_k,
        seed              = seed,
        save_crops        = save_crops,
    )

    # ── Create output directory ───────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = _out_triplet_root() / f"time_{timestamp}"
    crop_dir  = out_dir / "crops"
    res_dir   = out_dir / "results"
    crop_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    class_names = _read_class_names(yolo_seg_folder)
    print(f"[Triplet] Output dir : {out_dir}")
    print(f"[Triplet] Classes    : {class_names}")

    # ── Pre-flight notice (before the slow YOLO detection + crop extraction) ──
    from src.bridge.ui_bridge import ui as _ui
    _ui.log("INFO",
        f"🚀 Launching triplet training — dataset={Path(yolo_seg_folder).parent.name}  "
        f"epochs={triplet_epochs}  latent={latent_dim}  batch={triplet_batch}  device={device}. "
        "This may take a while (YOLO detection + crop extraction run first); "
        "a live loss curve will appear once epochs begin.")

    try:
        # 1. Load images
        img_paths  = _load_image_paths(cfg)
        split_counts = Counter(s for _, s in img_paths)
        print(f"[Triplet] Images found: {len(img_paths)}  "
              + "  ".join(f"{s}={n}" for s, n in sorted(split_counts.items())))

        # 2. YOLO detection
        detections = _detect_with_yolo(img_paths, cfg)

        # 3. Filter isolated objects
        detections = _filter_isolated(detections, cfg)

        # 4. Crop + normalise
        crops = _crop_and_normalise(detections, cfg, crop_dir)
        if len(crops) < 4:
            return _err(f"Too few crops ({len(crops)}) after filtering. "
                        "Relax overlap thresholds or use a larger dataset.")

        labels = [c["label"] for c in crops]
        n_classes = len(set(labels))
        if n_classes < 2:
            return _err(f"Need ≥2 classes in crops. Found: {set(labels)}")

        # 5. Train
        encoder, history = _train_triplet_encoder(crops, cfg, out_dir)

        # 6. Extract embeddings for evaluation
        embs = _extract_embeddings(crops, encoder, cfg, device)

        # 7. Metrics
        sil   = _silhouette(embs, labels)
        knn_a = _knn_accuracy(embs, labels, k=cfg.knn_k)
        print(f"[Eval] Silhouette global : {sil['global']:.4f}")
        print(f"[Eval] kNN accuracy (k={cfg.knn_k}): {knn_a*100:.1f}%")

        # 8. Visualisations
        print("\n[Plots] Generating...")
        _plot_loss(history, res_dir / "triplet_loss.png")
        _plot_umap(embs, labels, class_names, res_dir / "umap_embeddings.png")
        _plot_silhouette(sil, class_names, res_dir / "silhouette_scores.png")

        # 9. CSVs
        print("[CSV] Saving...")
        _save_training_csv(history, res_dir / "training_history.csv")
        _save_eval_csv(crops, embs, class_names, res_dir / "embeddings.csv")

        # 9b. Library (so the interactive explorer works on this run too)
        print("[Library] Building library.json from synthetic crops...")
        try:
            _build_library_json(crops, embs, class_names, out_dir)
        except Exception as lib_exc:
            print(f"  [Library] Skipped: {lib_exc}")

        final_metrics = {
            "run_dir"          : str(out_dir),
            "yolo_seg_folder"  : yolo_seg_folder,
            "yolo_model_path"  : yolo_model_path,
            "n_crops"          : len(crops),
            "n_classes"        : n_classes,
            "class_names"      : class_names,
            "class_counts"     : {str(k): v for k, v in Counter(labels).items()},
            "max_iou"          : max_iou,
            "latent_dim"       : latent_dim,
            "triplet_epochs"   : triplet_epochs,
            "final_train_loss" : round(history["train_loss"][-1], 6),
            "final_val_loss"   : round(history["val_loss"][-1],   6),
            "best_val_loss"    : round(min(history["val_loss"]),   6),
            "final_val_acc"    : round(history["val_acc"][-1],     6),
            "silhouette_global": round(sil["global"], 6)
                                  if not (sil["global"] != sil["global"]) else None,
            "silhouette_per_class": {str(k): round(v, 6)
                                      for k, v in sil["per_class"].items()},
            "knn_accuracy"     : round(knn_a, 6) if knn_a == knn_a else None,
        }
        _save_metrics_json(final_metrics, out_dir / "metrics.json")

        result = TripletResult(
            status            = "ok",
            message           = f"Triplet training complete. Output: {out_dir}",
            run_dir           = str(out_dir),
            model_path        = str(out_dir / "triplet_encoder.pt"),
            epochs_done       = triplet_epochs,
            n_crops           = len(crops),
            n_classes         = n_classes,
            final_val_loss    = round(history["val_loss"][-1], 6),
            final_val_acc     = round(history["val_acc"][-1],  6),
            silhouette_global = round(sil["global"], 4)
                                 if sil["global"] == sil["global"] else float("nan"),
            knn_accuracy      = round(knn_a, 4) if knn_a == knn_a else float("nan"),
            class_counts      = {str(k): v for k, v in Counter(labels).items()},
        )
        return _ok(data=asdict(result), message=result.message)

    except Exception as exc:
        import traceback
        return _err(f"Training failed: {exc}", detail=traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# Post-training inspection
# ══════════════════════════════════════════════════════════════════════════════

def list_triplet_runs() -> dict:
    """List all training runs in data/out_triplet/ sorted newest-first."""
    root = _out_triplet_root()
    if not root.exists():
        return _ok(data=[], message="No triplet runs found.")
    runs = []
    for p in sorted(root.iterdir(), reverse=True):
        if not p.is_dir() or not p.name.startswith("time_"):
            continue
        has_model   = (p / "triplet_encoder.pt").is_file()
        has_metrics = (p / "metrics.json").is_file()
        has_umap    = (p / "results" / "umap_embeddings.png").is_file()
        runs.append({
            "name"       : p.name,
            "path"       : str(p),
            "has_model"  : has_model,
            "has_metrics": has_metrics,
            "has_umap"   : has_umap,
        })
    return _ok(data=runs, message=f"{len(runs)} triplet run(s) found.")


def get_triplet_results(run_dir: str) -> dict:
    """Return metrics.json, model path and show saved plots in the GUI."""
    from src.bridge.ui_bridge import ui

    p = Path(run_dir).resolve()
    if not p.is_dir():
        return _err(f"Run directory not found: '{run_dir}'")

    metrics: dict = {}
    metrics_path  = p / "metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception:
            pass

    res_dir    = p / "results"
    model_path = p / "triplet_encoder.pt"

    # Show saved plots in GUI plot panel
    for png_name, title in [
        ("triplet_loss.png",       "Triplet loss & accuracy curves"),
        ("umap_embeddings.png",    "UMAP embeddings"),
        ("silhouette_scores.png",  "Silhouette scores"),
    ]:
        png = res_dir / png_name
        if png.is_file():
            img = plt.imread(str(png))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{p.name} — {title}", color="#c9d1d9", fontsize=10)
            fig.patch.set_facecolor("#0d1117")
            fig.tight_layout()
            ui.plot(fig)

    data = {
        "run_dir"          : str(p),
        "model_path"       : str(model_path) if model_path.is_file() else None,
        "metrics"          : metrics,
        "results_csv"      : str(res_dir / "training_history.csv"),
        "embeddings_csv"   : str(res_dir / "embeddings.csv"),
        "umap_plot"        : str(res_dir / "umap_embeddings.png"),
        "loss_plot"        : str(res_dir / "triplet_loss.png"),
        "silhouette_plot"  : str(res_dir / "silhouette_scores.png"),
    }
    return _ok(data=data, message=f"Results loaded from '{p.name}'.")
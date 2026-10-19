"""
afm_pipeline_classical.py
=========================
Same as afm_pipeline.py but the novelty criterion uses
ONLY the 5 geometric contour descriptors:

    area_norm, perimeter_norm, circularity, elongation, solidity

No autoencoder, no embedding, no PCA.
The KDE operates directly in that 5-dimensional space.

Useful for:
  - Interpretable baseline
  - Validate that the classical descriptor discriminates shapes
  - Compare against the hybrid version

Outputs in output_classical/:
    crops/
    results/
        novelty_curve.png
        library_grid.png
        umap_library.png
        umap_all.png
        contour_descriptors.csv
        novelty_history.csv
        library_table.csv
        contour_scatter.png       ← scatter matrix of the 5 descriptors
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

import torch
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    project_root  : Path  = Path(__file__).resolve().parent.parent
    dataset_root  : str   = ""          # built in __post_init__
    yolo_model    : str   = r"C:\Users\z78\Documents\Projects\SIMUSCAN assisted by LLM\code\SimuScanAgent\data\out_yolo\20260408_162101_segment\weights\best.pt"          # built in __post_init__
    output_dir    : str   = ""          # built in __post_init__
    # Name of the dataset folder inside data/Datasets/
    dataset_name  : str   = "20260408_142058_YOLO_SEG"

    def __post_init__(self):
        if not self.dataset_root:
            self.dataset_root = str(
                self.project_root / "data" / "Datasets" / self.dataset_name)
        if not self.yolo_model:
            self.yolo_model = str(
                self.project_root / "data" / "models" / "best.pt")
        if not self.output_dir:
            self.output_dir = str(
                self.project_root / "data" / "output_classical")

    # ── Splits ────────────────────────────────────────────────────────────────
    splits        : tuple = ("train", "val", "test")

    # ── Crop ──────────────────────────────────────────────────────────────────
    crop_size     : int   = 64
    margin_pct    : float = 0.15
    yolo_conf     : float = 0.25
    yolo_iou      : float = 0.45
    save_crops    : bool  = True

    # ── KDE (operates in 5D, no PCA) ────────────────────────────────────────────
    kde_bandwidth : float = 0.5      # or "scott" for automatic

    # ── Novelty criterion ───────────────────────────────────────────────────
    novelty_thr   : float = 0.35     # score > thr → enters the library
    kl_thr        : float = 0.08     # convergence when moving average < kl_thr
    window_size   : int   = 10

    # ── Visualization ─────────────────────────────────────────────────────────
    umap_neighbors: int   = 15
    umap_min_dist : float = 0.1
    grid_max_items: int   = 64

    # ── Misc ──────────────────────────────────────────────────────────────────
    device        : str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed          : int   = 42


CFG = Config()
np.random.seed(CFG.seed)
for sub in ["crops", "results"]:
    (Path(CFG.output_dir) / sub).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  IMAGE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_image_paths() -> list[Path]:
    root = Path(CFG.dataset_root) / "images"
    print(f"  [Dataset] Searching in: {root.resolve()}")
    if root.exists():
        print(f"  [Dataset] Subfolders: "
              f"{[p.name for p in root.iterdir() if p.is_dir()]}")
    paths = []
    for split in CFG.splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [Dataset] Skipping: {split_dir}")
            continue
        found = sorted(split_dir.glob("*.png")) + sorted(split_dir.glob("*.jpg"))
        print(f"  [Dataset] {split:5s}: {len(found):4d} images")
        paths.extend(found)
    if not paths:
        raise FileNotFoundError(f"No images found in {root}.")
    print(f"  [Dataset] Total: {len(paths)} images\n")
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  YOLO DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_with_yolo(img_paths: list[Path]) -> list[dict]:
    model   = YOLO(CFG.yolo_model)
    results = []
    for p in tqdm(img_paths, desc="YOLO inference"):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preds   = model.predict(source=img_bgr, conf=CFG.yolo_conf,
                                iou=CFG.yolo_iou, verbose=False)[0]
        results.append(dict(
            img_path = p,
            img      = img_rgb,
            boxes    = preds.boxes.xyxy.cpu().numpy().astype(int),
            labels   = preds.boxes.cls.cpu().numpy().astype(int),
            confs    = preds.boxes.conf.cpu().numpy(),
        ))
    total = sum(len(r["boxes"]) for r in results)
    print(f"[YOLO] {total} objects detected in {len(results)} images.\n")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CROPPING AND NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def crop_and_normalize(detections: list[dict]) -> list[dict]:
    crops    = []
    sz       = CFG.crop_size
    marg     = CFG.margin_pct
    crop_dir = Path(CFG.output_dir) / "crops"

    for det in tqdm(detections, desc="Cropping objects"):
        img  = det["img"]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        H, W = gray.shape
        stem = Path(det["img_path"]).stem

        for k, (box, lbl, conf) in enumerate(
                zip(det["boxes"], det["labels"], det["confs"])):
            x1, y1, x2, y2 = box
            bw, bh = x2-x1, y2-y1
            mx, my = int(bw*marg), int(bh*marg)
            x1c = max(0, x1-mx);  y1c = max(0, y1-my)
            x2c = min(W, x2+mx);  y2c = min(H, y2+my)

            patch = gray[y1c:y2c, x1c:x2c]
            if patch.size == 0:
                continue

            p2, p98 = np.percentile(patch, 2), np.percentile(patch, 98)
            if p98 > p2:
                patch = np.clip((patch - p2) / (p98 - p2), 0, 1)
            else:
                patch = np.zeros_like(patch)

            patch = _align_orientation(patch)
            patch = cv2.resize(patch, (sz, sz),
                               interpolation=cv2.INTER_LINEAR).astype(np.float32)

            if CFG.save_crops:
                cv2.imwrite(
                    str(crop_dir / f"{stem}_obj{k:03d}_cls{lbl}.png"),
                    (patch * 255).astype(np.uint8))

            crops.append(dict(
                crop      = patch,
                label     = int(lbl),
                conf      = float(conf),
                img_path  = str(det["img_path"]),
                bbox_orig = [int(x1), int(y1), int(x2), int(y2)],
                crop_id   = f"{stem}_obj{k:03d}",
            ))

    print(f"[Crops] {len(crops)} crops generated.\n")
    return crops


def _align_orientation(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    if h < 8 or w < 8:
        return patch
    ys, xs  = np.mgrid[0:h, 0:w]
    weights = patch.flatten()
    total_w = weights.sum()
    if total_w < 1e-6:
        return patch
    xs_f = xs.flatten().astype(np.float32)
    ys_f = ys.flatten().astype(np.float32)
    cx   = (weights * xs_f).sum() / total_w
    cy   = (weights * ys_f).sum() / total_w
    cov  = np.cov(np.vstack([xs_f-cx, ys_f-cy]), aweights=weights+1e-8)
    _, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1,-1], vecs[0,-1]))
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(patch, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CONTOUR DESCRIPTOR  (5 features)
# ═══════════════════════════════════════════════════════════════════════════════

# Names of the 5 features (for CSV and scatter)
CONTOUR_FEATURE_NAMES = [
    "area_norm",       # contour area / total crop area
    "perimeter_norm",  # perimeter / maximum possible perimeter
    "circularity",     # 4π·A / P²  →  1.0=circle, <1=irregular
    "elongation",      # short_side / long_side of the minimum rect.
    "solidity",        # area / convex_area  →  1.0=convex, <1=concave
]


def extract_contour(crop: np.ndarray) -> np.ndarray:
    """
    Extracts the 5 geometric contour descriptors.
    Input  : crop (64×64) float32 [0,1]
    Output : array (5,) float32
    """
    sz     = CFG.crop_size
    thresh = (crop > 0.1).astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt   = max(contours, key=cv2.contourArea)
        area  = cv2.contourArea(cnt)
        peri  = cv2.arcLength(cnt, True)
        circ  = 4 * np.pi * area / (peri**2 + 1e-8)
        hull  = cv2.convexHull(cnt)
        soli  = area / (cv2.contourArea(hull) + 1e-8)
        rect  = cv2.minAreaRect(cnt)
        bw_r, bh_r = rect[1]
        elong = min(bw_r, bh_r) / (max(bw_r, bh_r) + 1e-8)
    else:
        area = peri = circ = soli = elong = 0.0

    return np.array([
        area  / sz**2,
        peri  / (4 * sz),
        float(circ),
        float(elong),
        float(soli),
    ], dtype=np.float32)


def build_contour_descriptors(crops: list[dict]) -> np.ndarray:
    """
    Extracts and normalises the contour descriptors of all crops.
    Returns (N, 5) normalised with StandardScaler.
    Keeps the scaler so the original values can be interpreted in the CSV.
    """
    print("[Descriptors] Extracting contours...")
    raw = np.stack([
        extract_contour(c["crop"])
        for c in tqdm(crops, desc="  contours")
    ])                                       # (N, 5) real values

    sc      = StandardScaler().fit(raw)
    scaled  = sc.transform(raw)             # (N, 5) normalised for the KDE

    print(f"[Descriptors] {raw.shape[1]} contour features  "
          f"({raw.shape[0]} objects)\n")
    return scaled, raw   # scaled for the criterion, raw for the CSV


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MORPHOLOGICAL LIBRARY — CONTOUR ONLY
# ═══════════════════════════════════════════════════════════════════════════════

class ContourLibrary:
    """
    Novelty criterion based solely on the 5 contour descriptors.

    No PCA (5D is manageable directly by the KDE).
    No autoencoder.

    Flujo:
        1. calibrate(all_descriptors)   fits a global KDE on all the data
        2. process(descriptor, meta)    evaluates novelty object by object
    """

    def __init__(self):
        self._descriptors : list[np.ndarray] = []
        self._metadata    : list[dict]        = []
        self._scores      : list[float]       = []
        self._kde_global  : Optional[KernelDensity] = None
        self._lp_p5       : float = 0.0
        self._lp_p95      : float = 1.0
        self._global_scale: float = 1.0
        self._calibrated  : bool  = False

    @property
    def size(self): return len(self._descriptors)
    @property
    def all_metadata(self): return self._metadata
    @property
    def novelty_history(self): return self._scores

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, all_descriptors: np.ndarray):
        """
        Fits the global KDE on ALL the normalised descriptors.
        all_descriptors : (N, 5) already normalised with StandardScaler.
        """
        print(f"[Library] Calibrating with {len(all_descriptors)} descriptors "
              f"of 5 dimensions...")

        # Bandwidth: Scott's rule adapted to 5D
        n, d  = all_descriptors.shape
        bw    = max(CFG.kde_bandwidth, n**(-1/(d+4)))
        self._kde_global = KernelDensity(
            bandwidth=bw, kernel="gaussian").fit(all_descriptors)

        lp_all       = self._kde_global.score_samples(all_descriptors)
        self._lp_p5  = float(np.percentile(lp_all, 5))
        self._lp_p95 = float(np.percentile(lp_all, 95))

        # Global scale = mean standard deviation in the normalised space
        self._global_scale = float(all_descriptors.std(axis=0).mean())

        print(f"  KDE global: bandwidth={bw:.3f}")
        print(f"  Log-prob range: [{self._lp_p5:.2f}, {self._lp_p95:.2f}]")
        print(f"  Global scale: {self._global_scale:.3f}")
        print(f"  Calibration complete.\n")
        self._calibrated = True

    # ── Evaluation ────────────────────────────────────────────────────────────

    def process(self, descriptor: np.ndarray, metadata: dict) -> dict:
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before process().")

        desc = np.asarray(descriptor, dtype=float).flatten()

        surprise  = self._surprise_global(desc)
        lib_score = self._library_score(desc)

        score    = float(np.clip(0.4*surprise + 0.6*lib_score, 0, 1))
        is_novel = score > CFG.novelty_thr

        if is_novel:
            self._descriptors.append(desc)
            self._metadata.append(metadata)

        self._scores.append(score)
        recent    = float(np.mean(self._scores[-CFG.window_size:]))
        converged = (len(self._scores) >= CFG.window_size
                     and recent < CFG.kl_thr)

        return dict(
            surprise      = round(surprise, 4),
            lib_score     = round(lib_score, 4),
            novelty_score = round(score, 4),
            is_novel      = is_novel,
            converged     = converged,
            lib_size      = self.size,
            recent_novelty= round(recent, 4),
        )

    def get_descriptors(self) -> np.ndarray:
        return np.stack(self._descriptors)

    def save(self, path: str):
        if not self._descriptors:
            return
        np.savez_compressed(
            path,
            descriptors = self.get_descriptors(),
            metadata    = np.array(
                [json.dumps(m) for m in self._metadata], dtype=object),
        )
        print(f"[Library] Saved → {path}  ({self.size} objects)")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _surprise_global(self, desc: np.ndarray) -> float:
        lp = self._kde_global.score_samples(desc.reshape(1,-1))[0]
        s  = (self._lp_p95 - lp) / (self._lp_p95 - self._lp_p5 + 1e-10)
        return float(np.clip(s, 0, 1))

    def _library_score(self, desc: np.ndarray) -> float:
        if self.size == 0:
            return 1.0
        lib = np.stack(self._descriptors)
        d_min = float(np.linalg.norm(lib - desc, axis=1).min())
        score = 1 / (1 + np.exp(
            -(d_min / (self._global_scale + 1e-8) - 1.5)))
        return float(np.clip(score, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _class_colors(labels):
    cmap = plt.get_cmap("tab20")
    uniq = sorted(set(labels))
    lut  = {l: cmap(i/max(len(uniq)-1,1)) for i,l in enumerate(uniq)}
    return [lut[l] for l in labels]


def plot_novelty_curve(history, out):
    xs  = np.arange(len(history))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(xs, history, lw=1.2, color="#3a86ff", alpha=0.8,
            label="novelty score")
    ax.axhline(CFG.novelty_thr, ls="--", color="#ff595e", lw=1,
               label=f"novelty_thr={CFG.novelty_thr}")
    ax.axhline(CFG.kl_thr, ls=":", color="#8ac926", lw=1,
               label=f"kl_thr={CFG.kl_thr}")
    if len(history) >= CFG.window_size:
        win = np.convolve(history,
                          np.ones(CFG.window_size)/CFG.window_size, "valid")
        ax.plot(np.arange(CFG.window_size-1, len(history)), win,
                color="#ffbe0b", lw=2,
                label=f"moving average (w={CFG.window_size})")
    ax.set_xlabel("Object processed"); ax.set_ylabel("Novelty score")
    ax.set_title("Novelty evolution — contour only")
    ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.05); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


def plot_library_gallery(lib_crops, lib_meta, out):
    n    = min(CFG.grid_max_items, len(lib_crops))
    cols = 8
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.9))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(n):
        r, c = divmod(i, cols)
        axes[r,c].imshow(lib_crops[i], cmap="afmhot", vmin=0, vmax=1)
        axes[r,c].set_title(f"cls{lib_meta[i]['label']}\n"
                            f"#{lib_meta[i]['index']}", fontsize=5)
        axes[r,c].axis("off")
    for i in range(n, rows*cols):
        r, c = divmod(i, cols); axes[r,c].axis("off")
    fig.suptitle(f"Morphological library (contour) — {n} objects", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


def plot_contour_scatter(raw: np.ndarray, labels: list[int],
                          in_library: list[bool], out: Path):
    """
    Scatter matrix of the 5 contour descriptors.
    Each pair of features in a subplot.
    Points coloured by class; those in the library have a black edge.
    Very useful to see which features separate the classes best.
    """
    df = pd.DataFrame(raw, columns=CONTOUR_FEATURE_NAMES)
    df["label"]      = labels
    df["in_library"] = in_library

    n     = len(CONTOUR_FEATURE_NAMES)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    colors = _class_colors(labels)
    uniq   = sorted(set(labels))
    cmap   = plt.get_cmap("tab20")

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram per class
                for k, lbl in enumerate(uniq):
                    vals = df.loc[df["label"]==lbl,
                                  CONTOUR_FEATURE_NAMES[i]].values
                    ax.hist(vals, bins=20, alpha=0.5,
                            color=cmap(k/max(len(uniq)-1,1)),
                            label=f"cls{lbl}")
            else:
                # Off-diagonal: scatter
                not_lib = [idx for idx, v in enumerate(in_library) if not v]
                in_lib  = [idx for idx, v in enumerate(in_library) if v]
                xf, yf  = CONTOUR_FEATURE_NAMES[j], CONTOUR_FEATURE_NAMES[i]

                if not_lib:
                    ax.scatter(raw[not_lib, j], raw[not_lib, i],
                               c="lightgray", s=6, alpha=0.3, zorder=1)
                if in_lib:
                    ax.scatter(raw[in_lib, j], raw[in_lib, i],
                               c=[colors[idx] for idx in in_lib],
                               s=20, alpha=0.8, edgecolors="k",
                               lw=0.3, zorder=2)

            if i == n-1:
                ax.set_xlabel(CONTOUR_FEATURE_NAMES[j], fontsize=6)
            if j == 0:
                ax.set_ylabel(CONTOUR_FEATURE_NAMES[i], fontsize=6)
            ax.tick_params(labelsize=5)

    # Class legend in the corner
    for k, lbl in enumerate(uniq):
        axes[0, n-1].scatter([], [],
                             c=[cmap(k/max(len(uniq)-1,1))],
                             s=20, label=f"cls {lbl}")
    axes[0, n-1].legend(fontsize=6, loc="center")

    fig.suptitle("Scatter matrix — contour descriptors\n"
                 "(color=class, gray=not in library, black edge=in library)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


def plot_umap(descriptors, labels, in_library, title, out):
    try:
        import umap
    except ImportError:
        print("  [UMAP] umap-learn not installed. Skipping.")
        return
    print(f"  Computing UMAP ({len(descriptors)} points)...")
    # With only 5 dims no prior PCA is needed
    proj = umap.UMAP(n_neighbors=CFG.umap_neighbors,
                     min_dist=CFG.umap_min_dist,
                     random_state=CFG.seed).fit_transform(descriptors)

    fig, ax = plt.subplots(figsize=(9,7))
    colors  = _class_colors(labels)
    not_lib = [i for i,v in enumerate(in_library) if not v]
    in_lib  = [i for i,v in enumerate(in_library) if v]
    if not_lib:
        ax.scatter(proj[not_lib,0], proj[not_lib,1],
                   c="lightgray", s=12, alpha=0.4,
                   label="not in library", zorder=1)
    if in_lib:
        ax.scatter(proj[in_lib,0], proj[in_lib,1],
                   c=[colors[i] for i in in_lib],
                   s=40, alpha=0.9, edgecolors="k", lw=0.4,
                   label="in library", zorder=2)
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    for i,l in enumerate(uniq):
        ax.scatter([],[],c=[cmap(i/max(len(uniq)-1,1))],s=40,label=f"cls {l}")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=7, ncol=min(4, len(uniq)+2))
    ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  CSV TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def save_tables(crops, all_results, raw, library, out_dir):

    # contour_descriptors.csv — real (non-normalised) values per object
    rows = []
    for i, (c, feat) in enumerate(zip(crops, raw)):
        row = {"index": i, "crop_id": c["crop_id"],
               "label": c["label"], "conf": round(c["conf"],3)}
        for name, val in zip(CONTOUR_FEATURE_NAMES, feat):
            row[name] = round(float(val), 6)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        out_dir / "contour_descriptors.csv", index=False)
    print(f"  → {out_dir}/contour_descriptors.csv")

    # novelty_history.csv
    rows_nov = []
    for i, (c, r) in enumerate(zip(crops, all_results)):
        rows_nov.append({
            "index"         : i,
            "crop_id"       : c["crop_id"],
            "label"         : c["label"],
            "conf"          : round(c["conf"],3),
            "surprise"      : r["surprise"],
            "lib_score"     : r["lib_score"],
            "novelty_score" : r["novelty_score"],
            "is_novel"      : r["is_novel"],
            "lib_size"      : r["lib_size"],
            "recent_novelty": r["recent_novelty"],
            "converged"     : r["converged"],
        })
    pd.DataFrame(rows_nov).to_csv(
        out_dir / "novelty_history.csv", index=False)
    print(f"  → {out_dir}/novelty_history.csv")

    # library_table.csv — with the contour values of each unique object
    rows_lib = []
    for i, (m, desc) in enumerate(
            zip(library.all_metadata, library.get_descriptors())):
        row = {"lib_index": i, "crop_id": m.get("crop_id",""),
               "label": m["label"], "conf": round(m["conf"],3),
               "img_path": Path(m["img_path"]).name,
               "bbox": str(m["bbox"])}
        # Add the normalised values of the contour descriptor
        for name, val in zip(CONTOUR_FEATURE_NAMES, desc):
            row[f"norm_{name}"] = round(float(val), 4)
        rows_lib.append(row)
    pd.DataFrame(rows_lib).to_csv(
        out_dir / "library_table.csv", index=False)
    print(f"  → {out_dir}/library_table.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  AFM Pipeline — Contour-only criterion (5 features)")
    print("=" * 65, "\n")

    rdir = Path(CFG.output_dir) / "results"

    # 1. Load images
    img_paths  = load_image_paths()

    # 2. YOLO detection
    detections = detect_with_yolo(img_paths)

    # 3. Normalised crops
    crops = crop_and_normalize(detections)
    if not crops:
        print("ERROR: No crops were generated."); return

    # 4. Contour descriptors
    scaled, raw = build_contour_descriptors(crops)

    # 5. Calibrate library
    library = ContourLibrary()
    library.calibrate(scaled)

    # 6. Evaluate object by object
    all_results  = []
    in_library   = []
    converged_at = None

    print("[Library] Processing objects...\n")
    for i, (crop, desc) in enumerate(zip(crops, scaled)):
        meta = dict(index=i, label=crop["label"], conf=crop["conf"],
                    img_path=crop["img_path"], bbox=crop["bbox_orig"],
                    crop_id=crop["crop_id"])
        r = library.process(desc, meta)
        all_results.append(r)
        in_library.append(r["is_novel"])

        status = "NEW     " if r["is_novel"] else "known   "
        print(f"  [{i:4d}] {status} | score={r['novelty_score']:.3f} "
              f"(S={r['surprise']:.2f} lib_d={r['lib_score']:.2f}) "
              f"| lib={r['lib_size']} | recent={r['recent_novelty']:.3f}")

        if r["converged"] and converged_at is None:
            converged_at = i
            print(f"\n  *** CONVERGENCE at object {i} ***\n")

    # 7. Save library
    library.save(str(Path(CFG.output_dir) / "morpho_library_contour.npz"))

    # 8. Visualizations
    print("\n[Visualizations] Generating plots...")
    plot_novelty_curve(library.novelty_history,
                       rdir / "novelty_curve.png")

    lib_crops = [crops[m["index"]]["crop"] for m in library.all_metadata]
    plot_library_gallery(lib_crops, library.all_metadata,
                         rdir / "library_grid.png")

    plot_contour_scatter(raw, [c["label"] for c in crops], in_library,
                         rdir / "contour_scatter.png")

    plot_umap(scaled, [c["label"] for c in crops], in_library,
              "UMAP — all objects (contour)", rdir / "umap_all.png")

    if library.size > 5:
        plot_umap(library.get_descriptors(),
                  [m["label"] for m in library.all_metadata],
                  [True]*library.size,
                  "UMAP — morphological library (contour)",
                  rdir / "umap_library.png")

    # 9. Tables
    print("\n[Tables] Saving CSV...")
    save_tables(crops, all_results, raw, library, rdir)

    # 10. Summary
    print("\n" + "=" * 65)
    print(f"  Objects processed  : {len(crops)}")
    print(f"  Final library      : {library.size} unique objects")
    print(f"  Convergence        : "
          + (f"YES (object {converged_at})"
             if converged_at is not None else "NOT reached"))
    print(f"\n  Outputs in: {CFG.output_dir}/")
    print(f"    morpho_library_contour.npz")
    print(f"    crops/                          {len(crops)} PNGs")
    print(f"    results/novelty_curve.png")
    print(f"    results/library_grid.png")
    print(f"    results/contour_scatter.png     ← scatter matrix 5 features")
    print(f"    results/umap_all.png")
    print(f"    results/umap_library.png")
    print(f"    results/contour_descriptors.csv ← real values per object")
    print(f"    results/novelty_history.csv")
    print(f"    results/library_table.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
   
"""
afm_pipeline.py
===============
Full pipeline (single script, no agent):

  1. Load PNG/JPG images from the standard YOLO structure
  2. Use YOLO (.pt) to detect objects → bounding boxes
  3. Crop and normalise each object (canonical 64×64 crop)
  4. Train a CNN autoencoder on all crops
  5. Extract hybrid descriptor (26 classical + 32 embedding)
  6. Calibrate the morphological space with all descriptors
  7. Apply the C4 information criterion → morphological library
  8. Save results and generate visualizations

Expected dataset structure:
    dataset_root/
        images/  train/ val/ test/   *.png | *.jpg
        labels/  train/ val/ test/   *.txt  (YOLO format)

Outputs in output_pipeline/:
    encoder.pt
    autoencoder.pt
    morpho_library.npz
    crops/
    results/
        ae_loss.png
        reconstruction_grid.png
        classical_boxplots.png
        novelty_curve.png
        library_grid.png
        umap_all.png
        umap_library.png
        classical_descriptors.csv
        novelty_history.csv
        library_table.csv

Dependencies:
    pip install torch torchvision ultralytics scikit-learn scipy
                opencv-python tqdm umap-learn matplotlib pandas
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
                self.project_root / "data" / "output_pipeline")

    # ── Splits ────────────────────────────────────────────────────────────────
    splits        : tuple = ("train", "val", "test")

    # ── Crop ──────────────────────────────────────────────────────────────────
    crop_size     : int   = 64
    margin_pct    : float = 0.15
    yolo_conf     : float = 0.25
    yolo_iou      : float = 0.45
    save_crops    : bool  = True

    # ── Autoencoder ───────────────────────────────────────────────────────────
    latent_dim    : int   = 32
    ae_epochs     : int   = 40
    ae_batch      : int   = 32
    ae_lr         : float = 5e-4

    # ── Classical descriptor ────────────────────────────────────────────────────
    glcm_distances: list  = field(default_factory=lambda: [1, 2])

    # ── PCA for the KDE ───────────────────────────────────────────────────────
    pca_components: int   = 8

    # ── C4 criterion ───────────────────────────────────────────────────────────
    kde_bandwidth : float = 0.5
    novelty_thr   : float = 0.35
    kl_thr        : float = 0.08
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
torch.manual_seed(CFG.seed)
for sub in ["crops", "results"]:
    (Path(CFG.output_dir) / sub).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  IMAGE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_image_paths() -> list[Path]:
    root = Path(CFG.dataset_root) / "images"
    print(f"  [Dataset] Searching in: {root.resolve()}")
    if root.exists():
        found_dirs = [p.name for p in root.iterdir() if p.is_dir()]
        print(f"  [Dataset] Subfolders found: {found_dirs}")

    paths = []
    for split in CFG.splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [Dataset] Folder not found, skipping: {split_dir}")
            continue
        found = sorted(split_dir.glob("*.png")) + sorted(split_dir.glob("*.jpg"))
        print(f"  [Dataset] {split:5s}: {len(found):4d} images")
        paths.extend(found)

    if not paths:
        raise FileNotFoundError(
            f"No images found in {root}. "
            f"Splits searched: {CFG.splits}")
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

            # Robust normalisation with percentiles
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
# 4.  AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class CropDataset(Dataset):
    def __init__(self, crops):
        self.data = [c["crop"] for c in crops]
        self.aug  = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).unsqueeze(0)
        return self.aug(img), self.aug(img)


class ConvEncoder(nn.Module):
    def __init__(self, ld=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc = nn.Linear(128*4*4, ld)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class ConvDecoder(nn.Module):
    def __init__(self, ld=32):
        super().__init__()
        self.fc = nn.Linear(ld, 128*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32,  16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16,   1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 128, 4, 4))


class ConvAutoencoder(nn.Module):
    def __init__(self, ld=32):
        super().__init__()
        self.encoder = ConvEncoder(ld)
        self.decoder = ConvDecoder(ld)
    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(crops: list[dict]) -> tuple[ConvAutoencoder, list[float]]:
    device = CFG.device
    loader = DataLoader(CropDataset(crops), batch_size=CFG.ae_batch,
                        shuffle=True, num_workers=0)
    model  = ConvAutoencoder(CFG.latent_dim).to(device)
    opt    = optim.Adam(model.parameters(), lr=CFG.ae_lr, weight_decay=1e-5)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, CFG.ae_epochs)
    mse_fn = nn.MSELoss()

    def grad_loss(pred, target):
        gx_p = pred[:,:,:,1:] - pred[:,:,:,:-1]
        gx_t = target[:,:,:,1:] - target[:,:,:,:-1]
        gy_p = pred[:,:,1:,:] - pred[:,:,:-1,:]
        gy_t = target[:,:,1:,:] - target[:,:,:-1,:]
        return mse_fn(gx_p, gx_t) + mse_fn(gy_p, gy_t)

    best_loss, losses = float("inf"), []
    print(f"[Autoencoder] Training {CFG.ae_epochs} epochs on {device}...")

    for epoch in range(1, CFG.ae_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            out   = model(x)
            loss  = 0.8 * mse_fn(out, y) + 0.2 * grad_loss(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * x.size(0)
        epoch_loss /= len(crops)
        sched.step()
        losses.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       Path(CFG.output_dir) / "autoencoder.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CFG.ae_epochs}  "
                  f"loss={epoch_loss:.5f}  best={best_loss:.5f}")

    model.load_state_dict(
        torch.load(Path(CFG.output_dir) / "autoencoder.pt", map_location=device))
    model.eval()
    torch.save(model.encoder.state_dict(), Path(CFG.output_dir) / "encoder.pt")
    print(f"[Autoencoder] Done. Best loss={best_loss:.5f}\n")
    return model, losses


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CLASSICAL DESCRIPTOR
# ═══════════════════════════════════════════════════════════════════════════════

CLASSICAL_FEATURE_NAMES = [
    "height_mean", "height_std", "height_max",
    "height_skewness", "height_kurtosis", "height_rms",
    "area_norm", "perimeter_norm", "circularity",
    "elongation", "solidity",
    "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7",
    "glcm1_contrast", "glcm1_dissimilarity",
    "glcm1_homogeneity", "glcm1_energy",
    "glcm2_contrast", "glcm2_dissimilarity",
    "glcm2_homogeneity", "glcm2_energy",
]


def extract_classical(crop: np.ndarray) -> np.ndarray:
    feats = []

    # Height statistics
    flat = crop.flatten()
    feats += [
        float(flat.mean()), float(flat.std()), float(flat.max()),
        float(skew(flat)), float(kurtosis(flat)),
        float(np.sqrt(np.mean(flat**2))),
    ]

    # Contour geometry
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
    feats += [
        area / CFG.crop_size**2, peri / (4*CFG.crop_size),
        float(circ), float(elong), float(soli),
    ]

    # Hu moments (log-transformed)
    hu = cv2.HuMoments(cv2.moments(crop)).flatten()
    feats += (-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).tolist()

    # GLCM
    img8 = (crop * 255).astype(np.uint8)
    for dist in CFG.glcm_distances:
        shifted = np.roll(img8, -dist, axis=1)
        joint   = np.zeros((256, 256), dtype=np.float32)
        np.add.at(joint, (img8.flatten(), shifted.flatten()), 1)
        joint  /= joint.sum() + 1e-8
        ii, jj  = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
        diff    = (ii - jj).astype(np.float32)
        feats  += [
            float((joint * diff**2).sum()) / 1e4,
            float((joint * np.abs(diff)).sum()) / 255,
            float((joint / (1 + diff**2)).sum()),
            float((joint**2).sum()) * 1e3,
        ]

    return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  HYBRID DESCRIPTOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_hybrid_descriptors(
    crops: list[dict], model: ConvAutoencoder
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = CFG.device
    model.eval()

    print("[Descriptors] Extracting classical descriptors...")
    classical = np.stack([
        extract_classical(c["crop"])
        for c in tqdm(crops, desc="  classical")
    ])

    print("[Descriptors] Extracting CNN embeddings...")
    embeddings = []
    for i in range(0, len(crops), 128):
        batch = crops[i:i+128]
        t = torch.stack([
            torch.from_numpy(c["crop"]).unsqueeze(0) for c in batch
        ]).to(device)
        with torch.no_grad():
            embeddings.append(model.encoder(t).cpu().numpy())
    embeddings = np.vstack(embeddings)

    sc_cl  = StandardScaler().fit(classical)
    sc_em  = StandardScaler().fit(embeddings)
    hybrid = np.hstack([sc_cl.transform(classical),
                        sc_em.transform(embeddings)])

    print(f"[Descriptors] {classical.shape[1]} classical + "
          f"{embeddings.shape[1]} embedding = {hybrid.shape[1]} dims  "
          f"({hybrid.shape[0]} objects)\n")
    return hybrid, classical, embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MORPHOLOGICAL LIBRARY + C4 CRITERION
# ═══════════════════════════════════════════════════════════════════════════════

class MorphoLibrary:
    """
    C4 criterion with a prior global calibration.

    Flujo:
        1. calibrate(all_descriptors)  — fits PCA+KDE on ALL the data
        2. process(descriptor, meta)   — evaluates novelty object by object
    """

    def __init__(self):
        self._descriptors : list[np.ndarray] = []
        self._metadata    : list[dict]        = []
        self._scores      : list[float]       = []
        self._kde_global  : Optional[KernelDensity]  = None
        self._kde_lib     : Optional[KernelDensity]  = None
        self._pca         : Optional[PCA]            = None
        self._sc          : Optional[StandardScaler] = None
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
        Fits PCA and a global KDE on ALL the dataset descriptors.
        Must be called exactly once before process().
        """
        print(f"[Library] Calibrating with {len(all_descriptors)} descriptors...")

        self._sc  = StandardScaler().fit(all_descriptors)
        scaled    = self._sc.transform(all_descriptors)

        nc = min(CFG.pca_components, len(all_descriptors)-1,
                 all_descriptors.shape[1])
        self._pca = PCA(n_components=nc, random_state=CFG.seed).fit(scaled)
        reduced   = self._pca.transform(scaled)

        var_exp = self._pca.explained_variance_ratio_.sum()
        print(f"  PCA: {nc} components → {var_exp*100:.1f}% variance explained")

        # Automatic bandwidth (Scott's rule)
        bw = max(CFG.kde_bandwidth, len(all_descriptors)**(-1/(nc+4)))
        self._kde_global = KernelDensity(
            bandwidth=bw, kernel="gaussian").fit(reduced)

        # Percentiles to normalise the surprise
        lp_all       = self._kde_global.score_samples(reduced)
        self._lp_p5  = float(np.percentile(lp_all, 5))
        self._lp_p95 = float(np.percentile(lp_all, 95))

        # Global scale to normalise distances
        self._global_scale = float(
            np.sqrt(self._pca.explained_variance_.mean()))

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
        xr   = self._reduce(desc)

        surprise  = self._surprise_global(xr)
        lib_score = self._library_score(xr)

        # 40% global surprise + 60% distance to library
        score    = float(np.clip(0.4*surprise + 0.6*lib_score, 0, 1))
        is_novel = score > CFG.novelty_thr

        if is_novel:
            self._add(desc, metadata)

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

    def _add(self, d: np.ndarray, m: dict):
        self._descriptors.append(d)
        self._metadata.append(m)
        if self.size >= 2:
            lib_r = np.stack([self._reduce(x) for x in self._descriptors])
            self._kde_lib = KernelDensity(
                bandwidth=CFG.kde_bandwidth,
                kernel="gaussian").fit(lib_r)

    def _reduce(self, desc: np.ndarray) -> np.ndarray:
        return self._pca.transform(
                   self._sc.transform(desc.reshape(1,-1)))[0]

    def _surprise_global(self, xr: np.ndarray) -> float:
        """Surprise relative to the dataset's global distribution."""
        lp = self._kde_global.score_samples(xr.reshape(1,-1))[0]
        s  = (self._lp_p95 - lp) / (self._lp_p95 - self._lp_p5 + 1e-10)
        return float(np.clip(s, 0, 1))

    def _library_score(self, xr: np.ndarray) -> float:
        """Normalised distance to the nearest neighbour in the library."""
        if self.size == 0:
            return 1.0
        lib_r = np.stack([self._reduce(d) for d in self._descriptors])
        d_min = float(np.linalg.norm(lib_r - xr, axis=1).min())
        # Sigmoid centred at 1.5 × global scale
        score = 1 / (1 + np.exp(
            -(d_min / (self._global_scale + 1e-8) - 1.5)))
        return float(np.clip(score, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _class_colors(labels):
    cmap = plt.get_cmap("tab20")
    uniq = sorted(set(labels))
    lut  = {l: cmap(i/max(len(uniq)-1,1)) for i,l in enumerate(uniq)}
    return [lut[l] for l in labels]


def plot_ae_loss(losses, out):
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(losses, color="#3a86ff", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Autoencoder loss (MSE + gradient)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


def plot_reconstruction_grid(crops, model, out):
    device = CFG.device
    n      = min(CFG.grid_max_items, len(crops))
    sel    = crops[:n]
    model.eval()

    recons, errors = [], []
    for c in sel:
        t = torch.from_numpy(c["crop"]).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            r = model(t).squeeze().cpu().numpy()
        recons.append(r)
        errors.append(float(np.mean((c["crop"] - r)**2)))

    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows*2, cols, figsize=(cols*1.6, rows*3.5))
    axes = np.array(axes).reshape(rows*2, cols)

    for i in range(n):
        r, c = divmod(i, cols)
        vmin = min(sel[i]["crop"].min(), recons[i].min())
        vmax = max(sel[i]["crop"].max(), recons[i].max())
        axes[r*2,   c].imshow(sel[i]["crop"], cmap="afmhot", vmin=vmin, vmax=vmax)
        axes[r*2+1, c].imshow(recons[i],      cmap="afmhot", vmin=vmin, vmax=vmax)
        axes[r*2,   c].set_title(f"cls{sel[i]['label']}", fontsize=6)
        axes[r*2+1, c].set_title(f"mse={errors[i]:.4f}", fontsize=5,
                                  color="red" if errors[i]>0.05 else "green")
        if c == 0:
            axes[r*2,   c].set_ylabel("orig",  fontsize=6)
            axes[r*2+1, c].set_ylabel("recon", fontsize=6)

    for ax in axes.flatten(): ax.axis("off")
    fig.suptitle(f"Original vs Reconstructed — mean MSE={np.mean(errors):.4f}", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}  (mean MSE={np.mean(errors):.4f})")


def plot_novelty_curve(history, out):
    xs  = np.arange(len(history))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(xs, history, lw=1.2, color="#3a86ff", alpha=0.8, label="novelty score")
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
    ax.set_title("Novelty evolution"); ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.05); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


def plot_classical_boxplots(classical, labels, out):
    df = pd.DataFrame(classical, columns=CLASSICAL_FEATURE_NAMES)
    df["label"] = labels
    grupos = {
        "Height statistics" : CLASSICAL_FEATURE_NAMES[:6],
        "Contour geometry" : CLASSICAL_FEATURE_NAMES[6:11],
        "Hu moments"             : CLASSICAL_FEATURE_NAMES[11:18],
        "GLCM texture"           : CLASSICAL_FEATURE_NAMES[18:],
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    uniq_labels = sorted(df["label"].unique())
    for ax, (titulo, feats) in zip(axes, grupos.items()):
        data_by_feat, tick_labels = [], []
        for feat in feats:
            for lbl in uniq_labels:
                data_by_feat.append(df.loc[df["label"]==lbl, feat].values)
                tick_labels.append(f"{feat}\ncls{lbl}")
        bp = ax.boxplot(data_by_feat, patch_artist=True)
        cmap = plt.get_cmap("tab20")
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(cmap((i % len(uniq_labels)) /
                                     max(len(uniq_labels)-1,1)))
            patch.set_alpha(0.7)
        ax.set_title(titulo, fontsize=9)
        ax.set_xticks(range(1, len(tick_labels)+1))
        ax.set_xticklabels(tick_labels, fontsize=5, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Distribution of classical descriptors per class", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


def plot_library_gallery(lib_crops, lib_meta, out):
    n    = min(CFG.grid_max_items, len(lib_crops))
    cols = 8
    rows = int(np.ceil(n / cols))
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
    fig.suptitle(f"Morphological library — {n} unique objects", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  → {out}")


def plot_umap(descriptors, labels, in_library, title, out):
    try:
        import umap
    except ImportError:
        print("  [UMAP] umap-learn not installed. Skipping.")
        return
    print(f"  Computing UMAP ({len(descriptors)} points)...")
    nc   = min(50, descriptors.shape[1], descriptors.shape[0]-1)
    proj = umap.UMAP(n_neighbors=CFG.umap_neighbors,
                     min_dist=CFG.umap_min_dist,
                     random_state=CFG.seed).fit_transform(
            PCA(n_components=nc).fit_transform(
                StandardScaler().fit_transform(descriptors)))

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
# 9.  CSV TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def save_tables(crops, all_results, classical, library, out_dir):

    # classical_descriptors.csv
    rows_cl = []
    for i, (c, feat) in enumerate(zip(crops, classical)):
        row = {"index": i, "crop_id": c["crop_id"],
               "label": c["label"], "conf": round(c["conf"],3)}
        for name, val in zip(CLASSICAL_FEATURE_NAMES, feat):
            row[name] = round(float(val), 6)
        rows_cl.append(row)
    pd.DataFrame(rows_cl).to_csv(
        out_dir / "classical_descriptors.csv", index=False)
    print(f"  → {out_dir}/classical_descriptors.csv")

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

    # library_table.csv
    rows_lib = []
    for i, m in enumerate(library.all_metadata):
        rows_lib.append({
            "lib_index": i,
            "crop_id"  : m.get("crop_id",""),
            "label"    : m["label"],
            "conf"     : round(m["conf"],3),
            "img_path" : Path(m["img_path"]).name,
            "bbox"     : str(m["bbox"]),
        })
    pd.DataFrame(rows_lib).to_csv(
        out_dir / "library_table.csv", index=False)
    print(f"  → {out_dir}/library_table.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  AFM Pipeline: Dataset → Encoder → Hybrid → C4 Library")
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

    # 4. Train autoencoder
    model, ae_losses = train_autoencoder(crops)

    # 5. Hybrid descriptors
    hybrid, classical, embeddings = build_hybrid_descriptors(crops, model)

    # 6. Calibrate library with ALL descriptors
    library = MorphoLibrary()
    library.calibrate(hybrid)

    # 7. Evaluate object by object
    all_results  = []
    in_library   = []
    converged_at = None

    print("[Library] Processing objects with C4 criterion...\n")
    for i, (crop, desc) in enumerate(zip(crops, hybrid)):
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

    # 8. Save library
    library.save(str(Path(CFG.output_dir) / "morpho_library.npz"))

    # 9. Visualizations
    print("\n[Visualizations] Generating plots...")
    plot_ae_loss(ae_losses, rdir / "ae_loss.png")
    plot_novelty_curve(library.novelty_history, rdir / "novelty_curve.png")
    plot_reconstruction_grid(crops, model, rdir / "reconstruction_grid.png")
    plot_classical_boxplots(classical, [c["label"] for c in crops],
                            rdir / "classical_boxplots.png")

    lib_crops = [crops[m["index"]]["crop"] for m in library.all_metadata]
    plot_library_gallery(lib_crops, library.all_metadata,
                         rdir / "library_grid.png")

    plot_umap(hybrid, [c["label"] for c in crops], in_library,
              "UMAP — all objects", rdir / "umap_all.png")
    if library.size > 5:
        plot_umap(library.get_descriptors(),
                  [m["label"] for m in library.all_metadata],
                  [True]*library.size,
                  "UMAP — morphological library", rdir / "umap_library.png")

    # 10. CSV tables
    print("\n[Tables] Saving CSV...")
    save_tables(crops, all_results, classical, library, rdir)

    # 11. Final summary
    print("\n" + "=" * 65)
    print(f"  Objects processed  : {len(crops)}")
    print(f"  Final library      : {library.size} unique objects")
    print(f"  Convergence        : "
          + (f"YES (object {converged_at})"
             if converged_at is not None else "NOT reached"))
    print(f"\n  Outputs in: {CFG.output_dir}/")
    print(f"    encoder.pt / autoencoder.pt")
    print(f"    morpho_library.npz")
    print(f"    crops/                           {len(crops)} PNGs")
    print(f"    results/ae_loss.png")
    print(f"    results/reconstruction_grid.png")
    print(f"    results/classical_boxplots.png")
    print(f"    results/novelty_curve.png")
    print(f"    results/library_grid.png")
    print(f"    results/umap_all.png / umap_library.png")
    print(f"    results/classical_descriptors.csv")
    print(f"    results/novelty_history.csv")
    print(f"    results/library_table.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
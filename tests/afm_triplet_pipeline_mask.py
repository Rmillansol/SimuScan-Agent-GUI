"""
afm_triplet_pipeline.py
=======================
Same as the previous version with a single change:
the PNGs saved in crops/ show the YOLO mask contour
overlaid in green on the grayscale AFM image.

The crop["crop"] array that goes to the encoder does NOT change — only
what is saved to disk for visualization changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

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
from sklearn.decomposition import PCA

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
    project_root  : Path  = Path(__file__).resolve().parent.parent
    dataset_root  : str   = ""          # built in __post_init__
    yolo_model    : str   = r"data\out_yolo\20260409_103608_segment\weights\best.pt"          # built in __post_init__
    output_dir    : str   = ""          # built in __post_init__
    # Name of the dataset folder inside data/Datasets/
    dataset_name  : str   = "20260409_102513_YOLO_SEG"

    def __post_init__(self):
        if not self.dataset_root:
            self.dataset_root = str(
                self.project_root / "data" / "Datasets" / self.dataset_name)
        if not self.yolo_model:
            self.yolo_model = str(
                self.project_root / "data" / "models" / "best.pt")
        if not self.output_dir:
            self.output_dir = str(
                self.project_root / "data" / "output_triplet_embedding")
# ── Descriptor mode ────────────────────────────────────────────────────
    descriptor_mode : Literal["embedding", "hybrid"] = "embedding"
    splits          : tuple = ("train", "val", "test")
    crop_size       : int   = 64
    margin_pct      : float = 0.15
    yolo_conf       : float = 0.25
    yolo_iou        : float = 0.45
    save_crops      : bool  = True

    # Contour colour in the saved PNGs (BGR for cv2)
    # Green by default; change to (0,0,255) for red, (255,128,0) for blue
    contour_color   : tuple = (0, 255, 0)
    contour_thickness: int  = 1   # pixels — 1 works well on 64px crops

    latent_dim      : int   = 32
    triplet_epochs  : int   = 40
    triplet_batch   : int   = 64
    triplet_lr      : float = 1e-3
    triplet_margin  : float = 0.5

    glcm_distances  : list  = field(default_factory=lambda: [1, 2])

    kde_bandwidth   : float = 0.5
    novelty_thr     : float = 0.55
    kl_thr          : float = 0.10
    window_size     : int   = 10

    umap_neighbors  : int   = 15
    umap_min_dist   : float = 0.1
    grid_max_items  : int   = 64

    device          : str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed            : int   = 42


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
    print(f"  [Dataset] Buscando en: {root.resolve()}")
    if root.exists():
        print(f"  [Dataset] Subcarpetas: "
              f"{[p.name for p in root.iterdir() if p.is_dir()]}")
    paths = []
    for split in CFG.splits:
        d = root / split
        if not d.exists():
            print(f"  [Dataset] Saltando: {d}"); continue
        found = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        print(f"  [Dataset] {split:5s}: {len(found):4d} images")
        paths.extend(found)
    if not paths:
        raise FileNotFoundError(f"No images found in {root}.")
    print(f"  [Dataset] Total: {len(paths)} images\n")
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  YOLO DETECTION WITH SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_with_yolo(img_paths: list[Path]) -> list[dict]:
    model   = YOLO(CFG.yolo_model)
    results = []

    for p in tqdm(img_paths, desc="YOLO seg inference"):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_bgr.shape[:2]

        preds  = model.predict(source=img_bgr, conf=CFG.yolo_conf,
                               iou=CFG.yolo_iou, verbose=False)[0]
        boxes  = preds.boxes.xyxy.cpu().numpy().astype(int)
        labels = preds.boxes.cls.cpu().numpy().astype(int)
        confs  = preds.boxes.conf.cpu().numpy()

        masks = []
        if preds.masks is not None:
            for m in preds.masks.data.cpu().numpy():
                m_r = cv2.resize(m, (W, H),interpolation=cv2.INTER_CUBIC)
                masks.append((m_r > 0.5).astype(np.uint8))
        else:
            for box in boxes:
                m = np.zeros((H, W), dtype=np.uint8)
                x1, y1, x2, y2 = box
                m[y1:y2, x1:x2] = 1
                masks.append(m)

        results.append(dict(
            img_path=p, img=img_rgb,
            boxes=boxes, labels=labels, confs=confs, masks=masks,
        ))

    total = sum(len(r["boxes"]) for r in results)
    print(f"[YOLO] {total} objects in {len(results)} images.\n")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MASK CROPPING  ← only modified block
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_mask_contour(patch_float: np.ndarray,
                       mask_crop: np.ndarray) -> np.ndarray:
    """
    Draws the contour of mask_crop over patch_float and returns
    a BGR uint8 image ready to save with cv2.imwrite.

    patch_float : (H, W) float32 [0, 1]  — the normalised crop
    mask_crop   : (H, W) uint8  {0,1}    — mask cropped to the same bbox

    The intensity channel is converted to 3 gray channels and the contour
    is overlaid in CFG.contour_color (green by default).
    """
    # Convert the crop to uint8 BGR (gray → 3 channels)
    gray_u8 = (patch_float * 255).astype(np.uint8)
    vis     = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    # Find contours of the cropped mask
    contours, _ = cv2.findContours(
        mask_crop.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        vis, contours, -1,          # -1 = all contours
        CFG.contour_color,
        CFG.contour_thickness,
    )
    return vis


def crop_and_normalize(detections: list[dict]) -> list[dict]:
    """
    For each detected object:
      1. Apply the segmentation mask → background = 0
      2. Crop the bbox with margin
      3. Normalise height to the [0,1] range using the 2-98 percentile
      4. Resize to crop_size × crop_size
      5. Save PNG with the segmentation contour overlaid (if save_crops)
         The crop["crop"] array sent to the encoder is the clean patch without drawing.
    """
    crops    = []
    sz       = CFG.crop_size
    marg     = CFG.margin_pct
    crop_dir = Path(CFG.output_dir) / "crops"

    for det in tqdm(detections, desc="Cropping objects"):
        img  = det["img"]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        H, W = gray.shape
        stem = Path(det["img_path"]).stem

        for k, (box, lbl, conf, mask) in enumerate(
                zip(det["boxes"], det["labels"],
                    det["confs"], det["masks"])):

            x1, y1, x2, y2 = box
            bw, bh = x2-x1, y2-y1
            mx, my = int(bw*marg), int(bh*marg)
            x1c = max(0, x1-mx);  y1c = max(0, y1-my)
            x2c = min(W, x2+mx);  y2c = min(H, y2+my)

            # ── Crop the mask to the same bbox ───────────────────────────
            mask_crop = mask[y1c:y2c, x1c:x2c]

            # ── Masked patch ─────────────────────────────────────────────
            masked = gray * mask.astype(np.float32)
            patch  = masked[y1c:y2c, x1c:x2c]
            if patch.size == 0:
                continue

            obj_pixels = patch[patch > 0]
            if len(obj_pixels) < 10:
                continue

            p2, p98 = np.percentile(obj_pixels, 2), np.percentile(obj_pixels, 98)
            if p98 > p2:
                patch = np.clip((patch - p2) / (p98 - p2), 0, 1)
            else:
                patch = (patch > 0).astype(np.float32)

            # ── Resize patch AND mask to the same canonical size ────────
            patch_resized = cv2.resize(
                patch, (sz, sz), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

            mask_resized = cv2.resize(
                mask_crop.astype(np.float32), (sz, sz),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)

            # ── Save PNG with overlaid contour ──────────────────────────
            if CFG.save_crops:
                vis = _draw_mask_contour(patch_resized, mask_resized)
                cv2.imwrite(
                    str(crop_dir / f"{stem}_obj{k:03d}_cls{lbl}.png"),
                    vis,
                )

            # ── The crop that goes to the encoder is the clean patch (no drawing) ─────
            crops.append(dict(
                crop      = patch_resized,
                label     = int(lbl),
                conf      = float(conf),
                img_path  = str(det["img_path"]),
                bbox_orig = [int(x1), int(y1), int(x2), int(y2)],
                crop_id   = f"{stem}_obj{k:03d}",
            ))

    from collections import Counter
    cnt = Counter(c["label"] for c in crops)
    print(f"[Crops] {len(crops)} crops generated:")
    for cls, n in sorted(cnt.items()):
        print(f"  Class {cls}: {n} objects")
    print()
    return crops


def _align_orientation(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    if h < 8 or w < 8: return patch
    ys, xs  = np.mgrid[0:h, 0:w]
    weights = patch.flatten()
    total_w = weights.sum()
    if total_w < 1e-6: return patch
    xs_f = xs.flatten().astype(np.float32)
    ys_f = ys.flatten().astype(np.float32)
    cx = (weights*xs_f).sum()/total_w
    cy = (weights*ys_f).sum()/total_w
    cov = np.cov(np.vstack([xs_f-cx, ys_f-cy]), aweights=weights+1e-8)
    _, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1,-1], vecs[0,-1]))
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(patch, M, (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TRIPLET ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class TripletDataset(Dataset):
    def __init__(self, crops, split="train", val_fraction=0.2):
        assert split in ("train", "val")
        rng     = np.random.default_rng(CFG.seed)
        sel_idx = []
        by_class_all: dict[int, list[int]] = {}
        for i, c in enumerate(crops):
            by_class_all.setdefault(c["label"], []).append(i)
        for cls, idxs in by_class_all.items():
            idxs_arr = np.array(idxs); rng.shuffle(idxs_arr)
            n_val = max(1, int(len(idxs_arr)*val_fraction))
            sel_idx.extend((idxs_arr[:n_val] if split=="val"
                            else idxs_arr[n_val:]).tolist())

        self.crops  = [crops[i] for i in sel_idx]
        self.labels = [c["label"] for c in self.crops]
        self.by_class: dict[int, list[int]] = {}
        for i, c in enumerate(self.crops):
            self.by_class.setdefault(c["label"], []).append(i)
        self.classes = list(self.by_class.keys())
        if len(self.classes) < 2:
            raise ValueError(f"At least 2 classes are needed. Found: {self.classes}")
        self.aug_strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(360),
            transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
        ])
        self.aug_light = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(360),
        ])
        self.split = split

    def __len__(self): return len(self.crops)

    def __getitem__(self, idx):
        anchor_lbl = self.labels[idx]
        pos_pool = [i for i in self.by_class[anchor_lbl] if i != idx]
        pos_idx  = int(np.random.choice(pos_pool)) if pos_pool else idx
        neg_classes = [l for l in self.classes if l != anchor_lbl]
        weights = np.array([len(self.by_class[l]) for l in neg_classes], float)
        weights /= weights.sum()
        neg_lbl = neg_classes[np.random.choice(len(neg_classes), p=weights)]
        neg_idx = int(np.random.choice(self.by_class[neg_lbl]))
        def to_t(i, strong=False):
            t = torch.from_numpy(self.crops[i]["crop"]).unsqueeze(0)
            return (self.aug_strong if strong else self.aug_light)(t)
        return to_t(idx), to_t(pos_idx, strong=(pos_idx==idx)), to_t(neg_idx)


class TripletEncoder(nn.Module):
    def __init__(self, ld=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  16, 3, stride=2, padding=1), nn.BatchNorm2d(16),  nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc = nn.Linear(128*4*4, ld)
    def forward(self, x):
        return nn.functional.normalize(self.fc(self.conv(x).flatten(1)), p=2, dim=1)


def train_triplet_encoder(crops):
    device   = CFG.device
    ds_train = TripletDataset(crops, split="train")
    ds_val   = TripletDataset(crops, split="val")
    print(f"  Split train: {len(ds_train)}  |  val: {len(ds_val)}")
    loader_tr = DataLoader(ds_train, batch_size=CFG.triplet_batch,
                           shuffle=True,  num_workers=0, drop_last=True)
    loader_va = DataLoader(ds_val,   batch_size=CFG.triplet_batch,
                           shuffle=False, num_workers=0, drop_last=False)
    encoder  = TripletEncoder(CFG.latent_dim).to(device)
    opt      = optim.Adam(encoder.parameters(), lr=CFG.triplet_lr, weight_decay=1e-5)
    sched    = optim.lr_scheduler.CosineAnnealingLR(opt, CFG.triplet_epochs)
    loss_fn  = nn.TripletMarginLoss(margin=CFG.triplet_margin, p=2)
    history  = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")
    print(f"[TripletEncoder] {CFG.triplet_epochs} epochs on {device} | margin={CFG.triplet_margin}")

    for epoch in range(1, CFG.triplet_epochs+1):
        encoder.train(); tl = 0.0
        for a, p, n in loader_tr:
            a,p,n = a.to(device), p.to(device), n.to(device)
            loss  = loss_fn(encoder(a), encoder(p), encoder(n))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * a.size(0)
        tl /= len(ds_train)

        encoder.eval(); vl = nc = nt = 0
        with torch.no_grad():
            for a, p, n in loader_va:
                a,p,n = a.to(device), p.to(device), n.to(device)
                ea,ep,en = encoder(a), encoder(p), encoder(n)
                vl += loss_fn(ea,ep,en).item() * a.size(0)
                nc += ((ea-ep).pow(2).sum(1).sqrt() <
                       (ea-en).pow(2).sum(1).sqrt()).sum().item()
                nt += a.size(0)
        vl /= len(ds_val); va = nc/max(nt,1)
        sched.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        if vl < best_val:
            best_val = vl
            torch.save(encoder.state_dict(), Path(CFG.output_dir)/"triplet_encoder.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CFG.triplet_epochs}  "
                  f"train={tl:.4f}  val={vl:.4f}  acc={va*100:.1f}%"
                  + ("  ← best" if vl==best_val else ""))

    encoder.load_state_dict(torch.load(Path(CFG.output_dir)/"triplet_encoder.pt",
                                        map_location=device))
    encoder.eval()
    print(f"[TripletEncoder] Completo. best_val={best_val:.4f}  "
          f"acc_final={history['val_acc'][-1]*100:.1f}%\n")
    return encoder, history


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DESCRIPTORS
# ═══════════════════════════════════════════════════════════════════════════════

CONTOUR_FEATURE_NAMES = ["area_norm","perimeter_norm","circularity","elongation","solidity"]
HYBRID_FEATURE_NAMES  = CONTOUR_FEATURE_NAMES + [f"emb_{i}" for i in range(CFG.latent_dim)]


def extract_contour(crop):
    sz = CFG.crop_size
    thresh = (crop > 0.05).astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt  = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt); peri = cv2.arcLength(cnt, True)
        circ = 4*np.pi*area/(peri**2+1e-8)
        soli = area/(cv2.contourArea(cv2.convexHull(cnt))+1e-8)
        bw,bh = cv2.minAreaRect(cnt)[1]
        elong = min(bw,bh)/(max(bw,bh)+1e-8)
    else:
        area=peri=circ=soli=elong=0.0
    return np.array([area/sz**2, peri/(4*sz), float(circ),
                     float(elong), float(soli)], dtype=np.float32)


def extract_embeddings(crops, encoder):
    device = CFG.device; encoder.eval(); embs = []
    for i in range(0, len(crops), 128):
        t = torch.stack([torch.from_numpy(c["crop"]).unsqueeze(0)
                         for c in crops[i:i+128]]).to(device)
        with torch.no_grad(): embs.append(encoder(t).cpu().numpy())
    return np.vstack(embs)


def build_descriptors(crops, encoder):
    print(f"[Descriptores] Modo: {CFG.descriptor_mode}")
    print("  Extrayendo embeddings...")
    embeddings = extract_embeddings(crops, encoder)
    if CFG.descriptor_mode == "embedding":
        print(f"  Descriptor: {embeddings.shape[1]} dims\n")
        return embeddings, None
    print("  Extracting classical contours...")
    contour_raw = np.stack([extract_contour(c["crop"])
                            for c in tqdm(crops, desc="  contornos")])
    sc_c = StandardScaler().fit(contour_raw)
    sc_e = StandardScaler(with_std=False).fit(embeddings)
    hybrid = np.hstack([sc_c.transform(contour_raw), sc_e.transform(embeddings)])
    print(f"  Descriptor: {contour_raw.shape[1]} classical + "
          f"{embeddings.shape[1]} embedding = {hybrid.shape[1]} dims\n")
    return hybrid, contour_raw


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MULTICLASS MORPHOLOGICAL LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

class _SingleClassLib:
    def __init__(self, cls_label):
        self.cls_label = cls_label
        self._descriptors: list[np.ndarray] = []
        self._metadata   : list[dict]        = []
        self._scores     : list[float]       = []
        self._kde_global  = None
        self._lp_p5 = self._lp_p95 = 0.0
        self._global_scale = 1.0
        self.total_seen = 0

    @property
    def size(self): return len(self._descriptors)
    @property
    def converged(self):
        return (len(self._scores) >= CFG.window_size and
                np.mean(self._scores[-CFG.window_size:]) < CFG.kl_thr)
    @property
    def recent_novelty(self):
        return float(np.mean(self._scores[-CFG.window_size:])) if self._scores else 1.0

    def calibrate(self, descriptors):
        n, d = descriptors.shape
        bw   = max(CFG.kde_bandwidth, n**(-1/(d+4)))
        self._kde_global = KernelDensity(bandwidth=bw, kernel="gaussian").fit(descriptors)
        lp = self._kde_global.score_samples(descriptors)
        self._lp_p5  = float(np.percentile(lp, 5))
        self._lp_p95 = float(np.percentile(lp, 95))
        sample = descriptors[np.random.choice(n, min(200,n), replace=False)]
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(sample)
        dists, _ = nn.kneighbors(sample)
        self._global_scale = float(np.median(dists[:,1]) + 1e-8)

    def process(self, descriptor, metadata):
        self.total_seen += 1
        desc = np.asarray(descriptor, dtype=float).flatten()
        surprise  = self._surprise(desc)
        lib_score = self._lib_score(desc)
        score     = float(np.clip(0.4*surprise + 0.6*lib_score, 0, 1))
        is_novel  = score > CFG.novelty_thr
        if is_novel:
            self._descriptors.append(desc)
            self._metadata.append(metadata)
        self._scores.append(score)
        return dict(surprise=round(surprise,4), lib_score=round(lib_score,4),
                    novelty_score=round(score,4), is_novel=is_novel,
                    converged=self.converged, lib_size=self.size,
                    recent_novelty=round(self.recent_novelty,4))

    def get_descriptors(self):
        return np.stack(self._descriptors) if self._descriptors else None

    def _surprise(self, desc):
        lp = self._kde_global.score_samples(desc.reshape(1,-1))[0]
        return float(np.clip((self._lp_p95-lp)/(self._lp_p95-self._lp_p5+1e-10), 0, 1))

    def _lib_score(self, desc):
        if self.size == 0: return 1.0
        d_min = float(np.linalg.norm(np.stack(self._descriptors)-desc, axis=1).min())
        return float(np.clip(1/(1+np.exp(-(d_min/self._global_scale-1.0))), 0, 1))


class MultiClassMorphoLibrary:
    def __init__(self):
        self._sub_libs  : dict[int, _SingleClassLib] = {}
        self._all_meta  : list[dict]  = []
        self._all_scores: list[float] = []

    @property
    def size(self): return sum(s.size for s in self._sub_libs.values())
    @property
    def all_metadata(self): return self._all_meta
    @property
    def novelty_history(self): return self._all_scores

    def calibrate(self, all_descriptors, labels):
        unique = sorted(set(labels))
        print(f"[Library] Calibrating {len(unique)} clases: {unique}")
        for cls in unique:
            idx = [i for i,l in enumerate(labels) if l==cls]
            sub = _SingleClassLib(cls)
            sub.calibrate(all_descriptors[idx])
            self._sub_libs[cls] = sub
            print(f"  Class {cls}: {len(idx)} objects")
        print()

    def process(self, descriptor, metadata):
        cls = metadata["label"]
        if cls not in self._sub_libs:
            sub = _SingleClassLib(cls)
            sub.calibrate(descriptor.reshape(1,-1))
            self._sub_libs[cls] = sub
        r = self._sub_libs[cls].process(descriptor, metadata)
        if r["is_novel"]: self._all_meta.append(metadata)
        self._all_scores.append(r["novelty_score"])
        r["converged_global"] = all(s.converged for s in self._sub_libs.values())
        r["lib_size_class"] = self._sub_libs[cls].size
        r["lib_size_total"] = self.size
        return r

    def get_descriptors(self):
        parts = [s.get_descriptors() for s in self._sub_libs.values()
                 if s.get_descriptors() is not None]
        return np.vstack(parts) if parts else np.array([])

    def save(self, path):
        descs = self.get_descriptors()
        if descs.size == 0: return
        np.savez_compressed(path,
            descriptors=descs,
            metadata=np.array([json.dumps(m) for m in self._all_meta], dtype=object))
        print(f"[Library] Saved → {path}  ({self.size} objects)")
        for cls, sub in sorted(self._sub_libs.items()):
            print(f"  Class {cls}: {sub.size} unique / {sub.total_seen} seen")

    def summary_by_class(self):
        rows = []
        for cls, sub in sorted(self._sub_libs.items()):
            rows.append({"class":cls, "total_seen":sub.total_seen,
                         "in_library":sub.size,
                         "pct_novel":round(sub.size/max(sub.total_seen,1)*100,1),
                         "converged":sub.converged,
                         "recent_novelty":round(sub.recent_novelty,4)})
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _cmap_colors(labels):
    cmap = plt.get_cmap("tab20"); uniq = sorted(set(labels))
    lut  = {l: cmap(i/max(len(uniq)-1,1)) for i,l in enumerate(uniq)}
    return [lut[l] for l in labels]


def plot_triplet_loss(history, out):
    train,val,acc = history["train_loss"],history["val_loss"],history["val_acc"]
    xs = np.arange(1, len(train)+1)
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].plot(xs,train,label="train",color="#3a86ff",lw=1.5)
    axes[0].plot(xs,val,  label="val",  color="#ff595e",lw=1.5)
    axes[0].set_title("Triplet Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8); axes[0].grid(True,alpha=0.3)
    axes[1].plot(xs,[a*100 for a in acc],color="#8ac926",lw=1.5)
    axes[1].axhline(50,ls=":",color="red",lw=0.8,alpha=0.5,label="chance (50%)")
    axes[1].set_title("Val Accuracy\n(% triplets d(a,p)<d(a,n))")
    axes[1].set_ylim(0,105); axes[1].legend(fontsize=8); axes[1].grid(True,alpha=0.3)
    gap=[v-t for t,v in zip(train,val)]
    axes[2].plot(xs,gap,color="#ffbe0b",lw=1.5)
    axes[2].axhline(0,ls="--",color="gray",lw=0.8)
    axes[2].fill_between(xs,0,gap,where=[g>0 for g in gap],alpha=0.25,color="red",label="overfitting")
    axes[2].fill_between(xs,0,gap,where=[g<=0 for g in gap],alpha=0.25,color="green",label="underfitting")
    axes[2].set_title("Gap val−train"); axes[2].legend(fontsize=7); axes[2].grid(True,alpha=0.3)
    fig.suptitle(f"TripletEncoder — margin={CFG.triplet_margin}  latent_dim={CFG.latent_dim}",fontsize=10)
    fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig); print(f"  → {out}")


def plot_umap(descriptors, labels, in_library, title, out, class_names=None):
    try: import umap
    except ImportError: print("  [UMAP] no instalado."); return
    print(f"  Computing UMAP ({len(descriptors)} points)...")
    nc = min(50, descriptors.shape[1], descriptors.shape[0]-1)
    proj = umap.UMAP(n_neighbors=CFG.umap_neighbors, min_dist=CFG.umap_min_dist,
                     random_state=CFG.seed).fit_transform(
           PCA(n_components=nc).fit_transform(
               StandardScaler().fit_transform(descriptors)))
    fig, ax = plt.subplots(figsize=(9,7))
    colors = _cmap_colors(labels); uniq = sorted(set(labels)); cmap = plt.get_cmap("tab20")
    not_lib=[i for i,v in enumerate(in_library) if not v]
    in_lib =[i for i,v in enumerate(in_library) if v]
    if not_lib: ax.scatter(proj[not_lib,0],proj[not_lib,1],c="lightgray",s=10,alpha=0.35,label="not in library",zorder=1)
    if in_lib:  ax.scatter(proj[in_lib,0], proj[in_lib,1], c=[colors[i] for i in in_lib],s=45,alpha=0.9,edgecolors="k",lw=0.4,label="in library",zorder=2)
    for i,l in enumerate(uniq):
        name = class_names.get(l,f"cls {l}") if class_names else f"cls {l}"
        ax.scatter([],[],c=[cmap(i/max(len(uniq)-1,1))],s=40,label=name)
    ax.set_title(f"{title}\nmode={CFG.descriptor_mode}")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=7,ncol=min(4,len(uniq)+2)); ax.grid(True,alpha=0.2)
    fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig); print(f"  → {out}")


def plot_novelty_per_class(library, out):
    uniq=sorted(library._sub_libs.keys()); cmap=plt.get_cmap("tab10")
    fig,ax=plt.subplots(figsize=(12,5))
    for i,cls in enumerate(uniq):
        sub=library._sub_libs[cls]; scores=sub._scores
        if not scores: continue
        col=cmap(i/max(len(uniq)-1,1))
        ax.plot(np.arange(len(scores)),scores,lw=1.0,alpha=0.6,color=col)
        if len(scores)>=CFG.window_size:
            win=np.convolve(scores,np.ones(CFG.window_size)/CFG.window_size,"valid")
            ax.plot(np.arange(CFG.window_size-1,len(scores)),win,lw=2.2,color=col,label=f"cls {cls}")
    ax.axhline(CFG.novelty_thr,ls="--",color="black",lw=1,alpha=0.5,label=f"novelty_thr={CFG.novelty_thr}")
    ax.axhline(CFG.kl_thr,ls=":",color="gray",lw=1,alpha=0.5,label=f"kl_thr={CFG.kl_thr}")
    ax.set_xlabel("Object N of that class"); ax.set_ylabel("Novelty score")
    ax.set_title("Novelty evolution per class\n(solid=raw score, thick=moving average)")
    ax.legend(fontsize=7,ncol=min(5,len(uniq)+1)); ax.set_ylim(-0.02,1.05); ax.grid(True,alpha=0.3)
    fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig); print(f"  → {out}")


def plot_library_summary(library, out):
    df=library.summary_by_class()
    if df.empty: return
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x,w=np.arange(len(df)),0.35
    axes[0].bar(x-w/2,df["total_seen"],w,label="total seen",color="#3a86ff",alpha=0.75)
    axes[0].bar(x+w/2,df["in_library"],w,label="in library",color="#ff595e",alpha=0.75)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"cls{c}" for c in df["class"]],fontsize=8)
    axes[0].set_ylabel("N objects"); axes[0].set_title("Total detected vs. in library")
    axes[0].legend(fontsize=8); axes[0].grid(True,alpha=0.3,axis="y")
    colors=["#8ac926" if c else "#ff595e" for c in df["converged"]]
    axes[1].bar(x,df["pct_novel"],color=colors,alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"cls{c}" for c in df["class"]],fontsize=8)
    axes[1].set_ylabel("% unique objects"); axes[1].set_title("% novelty per class\n(green=converged)")
    axes[1].set_ylim(0,115)
    for xi,pct in enumerate(df["pct_novel"]): axes[1].text(xi,pct+2,f"{pct:.0f}%",ha="center",fontsize=7)
    axes[1].grid(True,alpha=0.3,axis="y")
    fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig); print(f"  → {out}")


def plot_library_grid(crops, library, out_dir):
    """
    One gallery per class.
    Each thumbnail shows the PNG saved on disk (with green contour),
    not the clean array — so the segmentation can be seen in the gallery.
    """
    meta_all = library.all_metadata
    if not meta_all: return
    crop_dir = Path(CFG.output_dir) / "crops"

    by_class: dict[int, list[dict]] = {}
    for m in meta_all:
        by_class.setdefault(m["label"], []).append(m)

    for cls, meta_list in sorted(by_class.items()):
        n    = min(CFG.grid_max_items, len(meta_list))
        cols = 8
        rows = max(1, int(np.ceil(n/cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*2.0))
        axes = np.array(axes).reshape(rows, cols)

        for i in range(n):
            r, c = divmod(i, cols)
            m    = meta_list[i]
            # Try to load the saved PNG (it has the contour)
            png_path = crop_dir / f"{Path(m['img_path']).stem}_obj{int(m['crop_id'].split('obj')[1][:3]):03d}_cls{m['label']}.png"
            if png_path.exists():
                img_vis = cv2.cvtColor(cv2.imread(str(png_path)), cv2.COLOR_BGR2RGB)
                axes[r,c].imshow(img_vis)
            else:
                # Fallback: use the clean crop array
                axes[r,c].imshow(crops[m["index"]]["crop"], cmap="afmhot", vmin=0, vmax=1)
            axes[r,c].set_title(f"#{m['index']}", fontsize=5)
            axes[r,c].axis("off")

        for i in range(n, rows*cols):
            r,c=divmod(i,cols); axes[r,c].axis("off")

        total_cls = library._sub_libs[cls].total_seen
        fig.suptitle(f"Library — class {cls}  "
                     f"({n} shown of {len(meta_list)} / {total_cls} seen)  "
                     f"mode={CFG.descriptor_mode}", fontsize=9)
        fig.tight_layout()
        out = out_dir / f"library_grid_cls{cls}.png"
        fig.savefig(out, dpi=150); plt.close(fig); print(f"  → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CSV TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def save_tables(crops, all_results, contour_raw, library, out_dir):
    if contour_raw is not None:
        rows=[]
        for i,(c,feat) in enumerate(zip(crops,contour_raw)):
            row={"index":i,"crop_id":c["crop_id"],"label":c["label"],"conf":round(c["conf"],3)}
            for name,val in zip(CONTOUR_FEATURE_NAMES,feat): row[name]=round(float(val),6)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir/"contour_descriptors.csv",index=False)
        print(f"  → {out_dir}/contour_descriptors.csv")

    rows_nov=[]
    for i,(c,r) in enumerate(zip(crops,all_results)):
        rows_nov.append({"index":i,"crop_id":c["crop_id"],"label":c["label"],
                         "conf":round(c["conf"],3),"surprise":r["surprise"],
                         "lib_score":r["lib_score"],"novelty_score":r["novelty_score"],
                         "is_novel":r["is_novel"],"lib_size_class":r["lib_size_class"],
                         "lib_size_total":r["lib_size_total"],"recent_novelty":r["recent_novelty"],
                         "converged":r["converged"],"converged_global":r["converged_global"]})
    pd.DataFrame(rows_nov).to_csv(out_dir/"novelty_history.csv",index=False)
    print(f"  → {out_dir}/novelty_history.csv")

    rows_lib=[]
    for i,m in enumerate(library.all_metadata):
        rows_lib.append({"lib_index":i,"crop_id":m.get("crop_id",""),"label":m["label"],
                         "conf":round(m["conf"],3),"img_path":Path(m["img_path"]).name,
                         "bbox":str(m["bbox"])})
    pd.DataFrame(rows_lib).to_csv(out_dir/"library_table.csv",index=False)
    print(f"  → {out_dir}/library_table.csv")

    library.summary_by_class().to_csv(out_dir/"library_summary.csv",index=False)
    print(f"  → {out_dir}/library_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print(f"  AFM Triplet Pipeline  |  mode={CFG.descriptor_mode}")
    print("="*65, "\n")
    rdir = Path(CFG.output_dir) / "results"

    img_paths  = load_image_paths()
    detections = detect_with_yolo(img_paths)
    crops      = crop_and_normalize(detections)
    if not crops: print("ERROR: No crops were generated."); return

    labels = [c["label"] for c in crops]
    encoder, history = train_triplet_encoder(crops)
    descriptors, contour_raw = build_descriptors(crops, encoder)

    library = MultiClassMorphoLibrary()
    library.calibrate(descriptors, labels)

    all_results=[]; in_library=[]; converged_at=None
    print("[Library] Processing objects...\n")
    for i,(crop,desc) in enumerate(zip(crops,descriptors)):
        meta=dict(index=i,label=crop["label"],conf=crop["conf"],
                  img_path=crop["img_path"],bbox=crop["bbox_orig"],crop_id=crop["crop_id"])
        r=library.process(desc,meta)
        all_results.append(r); in_library.append(r["is_novel"])
        status="NEW     " if r["is_novel"] else "known   "
        print(f"  [{i:4d}] cls{crop['label']} {status} | "
              f"score={r['novelty_score']:.3f} "
              f"(S={r['surprise']:.2f} lib_d={r['lib_score']:.2f}) | "
              f"lib_cls={r['lib_size_class']} lib_tot={r['lib_size_total']} | "
              f"recent={r['recent_novelty']:.3f}")
        if r["converged_global"] and converged_at is None:
            converged_at=i; print(f"\n  *** GLOBAL CONVERGENCE at object {i} ***\n")

    library.save(str(Path(CFG.output_dir)/"morpho_library.npz"))

    print("\n[Visualizations] Generating plots...")
    plot_triplet_loss(history, rdir/"triplet_loss.png")
    plot_novelty_per_class(library, rdir/"novelty_per_class.png")
    plot_library_summary(library, rdir/"library_summary.png")
    plot_library_grid(crops, library, rdir)
    plot_umap(descriptors, labels, in_library, "UMAP — all objects", rdir/"umap_all.png")
    if library.size > 5:
        plot_umap(library.get_descriptors(),
                  [m["label"] for m in library.all_metadata],
                  [True]*library.size,
                  "UMAP — morphological library", rdir/"umap_library.png")

    print("\n[Tables] Saving CSV...")
    save_tables(crops, all_results, contour_raw, library, rdir)

    print("\n"+"="*65)
    print(f"  Descriptor mode   : {CFG.descriptor_mode}")
    print(f"  Objects processed  : {len(crops)}")
    print(f"  Final library      : {library.size} unique objects")
    print(f"  Global convergence: "
          +(f"YES (object {converged_at})" if converged_at is not None else "NOT reached"))
    print(); print(library.summary_by_class().to_string(index=False))
    print(f"\n  Outputs in: {CFG.output_dir}/")
    print(f"    triplet_encoder.pt  |  morpho_library.npz")
    print(f"    crops/              ← PNGs with green segmentation contour")
    print(f"    results/triplet_loss.png  novelty_per_class.png")
    print(f"    results/library_summary.png  library_grid_clsN.png")
    print(f"    results/umap_all.png  umap_library.png")
    print(f"    results/novelty_history.csv  library_table.csv  library_summary.csv")
    if contour_raw is not None: print(f"    results/contour_descriptors.csv")
    print("="*65)


if __name__ == "__main__":
    main()
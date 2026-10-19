"""
src/trainer/unet_trainer.py
────────────────────────────
UNet training, evaluation, and COCO→UNet conversion for SimuScan datasets.

Converted data lives INSIDE the COCO folder:
  <timestamp>_COCO/_unet/
    images/{train,val,test}/
    masks/{train,val,test}/
    dataset_info.json   ← n_classes, class_names (written here)

Training output goes to:
  data/out_unet/<timestamp>_unet/
    unet_best.pth
    unet_last.pth
    results.json        ← loss history + final metrics
    loss_curve.png      ← saved figure (sent to UI)

Public API
----------
convert_coco_to_unet(coco_folder)              → unet_folder (str)
train_unet_model(coco_folder, ...)             → dict
get_unet_training_results(run_dir)             → dict
list_unet_training_runs()                      → dict
predict_folder_unet(image_folder, weights, …)  → dict
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.bridge.ui_bridge import ui

# ── Path helpers ───────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _datasets_root() -> Path:
    return _repo_root() / "data" / "Datasets"


def _out_unet_root() -> Path:
    d = _repo_root() / "data" / "out_unet"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Result helpers ─────────────────────────────────────────────────────────────

def _ok(data=None, message: str = "OK") -> dict:
    r: dict = {"status": "ok", "message": message}
    if data is not None:
        r["data"] = data
    return r


def _err(message: str) -> dict:
    return {"status": "error", "message": message}


# ══════════════════════════════════════════════════════════════════════════════
# COCO → UNet conversion
# ══════════════════════════════════════════════════════════════════════════════

def convert_coco_to_unet(coco_folder: str) -> str:
    """
    Convert a COCO dataset to UNet format stored inside the COCO folder:
      <coco_folder>/_unet/
        images/{train,val,test}/
        masks/{train,val,test}/
        dataset_info.json

    Masks are single-channel PNGs with pixel values = class index:
      0 = background, 1 = first class, 2 = second class, …

    Skips conversion if _unet/dataset_info.json already exists.

    Returns the path to the _unet folder.
    """
    from PIL import Image, ImageDraw

    coco_folder = str(Path(coco_folder).resolve())
    if not coco_folder.endswith("_COCO"):
        raise ValueError("coco_folder must end with '_COCO'.")

    unet_folder = os.path.join(coco_folder, "_unet")
    info_path   = os.path.join(unet_folder, "dataset_info.json")

    # Already converted — skip
    if os.path.isfile(info_path):
        return unet_folder

    coco_images_dir = os.path.join(coco_folder, "images")
    coco_ann_dir    = os.path.join(coco_folder, "annotations")

    if not os.path.isdir(coco_ann_dir):
        raise FileNotFoundError(f"Annotations folder not found: {coco_ann_dir}")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(unet_folder, "images", split), exist_ok=True)
        os.makedirs(os.path.join(unet_folder, "masks",  split), exist_ok=True)

    class_names: List[str] = []
    catid_to_idx: Dict[int, int] = {}

    for split in ("train", "val", "test"):
        ann_file   = os.path.join(coco_ann_dir, f"instances_{split}.json")
        img_folder = os.path.join(coco_images_dir, split)

        if not (os.path.isfile(ann_file) and os.path.isdir(img_folder)):
            continue

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        if not class_names:
            cats_sorted  = sorted(coco["categories"], key=lambda c: c["id"])
            class_names  = [c["name"] for c in cats_sorted]
            # index 0 = background; classes start at 1
            catid_to_idx = {c["id"]: i + 1 for i, c in enumerate(cats_sorted)}

        images_info = {im["id"]: im for im in coco["images"]}

        anns_by_img: Dict[int, list] = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        n_imgs = len(images_info)
        ui.log("INFO", f"UNet conversion [{split}]: {n_imgs} images...")
        for done, (img_id, img_info) in enumerate(images_info.items(), 1):
            fname  = img_info["file_name"]
            W, H   = img_info["width"], img_info["height"]
            src    = os.path.join(img_folder, fname)
            if not os.path.isfile(src):
                continue

            shutil.copyfile(src, os.path.join(unet_folder, "images", split, fname))

            mask = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask)

            for ann in anns_by_img.get(img_id, []):
                cls_idx = catid_to_idx.get(ann["category_id"], 0)
                for seg in ann.get("segmentation", []):
                    if isinstance(seg, list) and len(seg) >= 6:
                        polygon = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                        draw.polygon(polygon, fill=cls_idx)

            mask_name = os.path.splitext(fname)[0] + ".png"
            mask.save(os.path.join(unet_folder, "masks", split, mask_name))

            if done % 100 == 0 or done == n_imgs:
                ui.log("INFO", f"  [{split}] {done}/{n_imgs} masks written")

    if not class_names:
        raise RuntimeError("No valid splits found — class names could not be determined.")

    # Save dataset_info.json (also used as the "already converted" marker)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "n_classes":   len(class_names),
            "n_out_channels": len(class_names) + 1,  # +1 for background
        }, f)

    return unet_folder


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def _build_unet(in_channels: int, out_channels: int,
                features: list | None = None):
    """Build and return a UNet model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    features = features or [64, 128, 256, 512]

    class _DoubleConv(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.downs      = nn.ModuleList()
            self.ups        = nn.ModuleList()
            self.pool       = nn.MaxPool2d(2, 2)

            in_c = in_channels
            for feat in features:
                self.downs.append(_DoubleConv(in_c, feat))
                in_c = feat

            self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

            for feat in reversed(features):
                self.ups.append(nn.ConvTranspose2d(feat * 2, feat, 2, 2))
                self.ups.append(_DoubleConv(feat * 2, feat))

            self.final = nn.Conv2d(features[0], out_channels, 1)

        def forward(self, x):
            skips = []
            for down in self.downs:
                x = down(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            skips = skips[::-1]
            for i in range(0, len(self.ups), 2):
                x    = self.ups[i](x)
                skip = skips[i // 2]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:],
                                      mode="bilinear", align_corners=False)
                x = torch.cat((skip, x), dim=1)
                x = self.ups[i + 1](x)
            return self.final(x)

    return UNet()


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helper
# ══════════════════════════════════════════════════════════════════════════════

def _make_dataset(unet_folder: str, split: str, image_size: int):
    """Return a torch Dataset for the given split."""
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from PIL import Image as PILImage

    img_dir  = Path(unet_folder) / "images" / split
    mask_dir = Path(unet_folder) / "masks"  / split

    if not img_dir.is_dir():
        return None

    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),   # → [1, H, W] uint8
    ])

    class _DS(Dataset):
        def __init__(self):
            self.files = sorted(
                f for f in img_dir.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            )

        def __len__(self): return len(self.files)

        def __getitem__(self, idx):
            img_path  = self.files[idx]
            stem      = img_path.stem
            mask_path = None
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                p = mask_dir / (stem + ext)
                if p.is_file():
                    mask_path = p
                    break
            if mask_path is None:
                raise FileNotFoundError(f"No mask for {img_path.name}")

            img  = PILImage.open(img_path).convert("RGB")
            mask = PILImage.open(mask_path)   # keep palette/index intact

            img_t  = img_transform(img)
            mask_t = mask_transform(mask)     # [1, H, W] uint8
            if mask_t.ndim == 3 and mask_t.shape[0] == 1:
                mask_t = mask_t.squeeze(0)    # [H, W]
            return img_t, mask_t.to(dtype=torch.long)

    return _DS()


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, path: Path) -> None:
    """Save a matplotlib figure to disk and close it."""
    import matplotlib.pyplot as plt
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _show_png(path: Path, title: str = "") -> None:
    """Load a PNG from disk and send it to the plot panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(mpimg.imread(str(path)))
        if title:
            ax.set_title(title, fontsize=9, color="#c9d1d9")
        ax.axis("off")
        fig.patch.set_facecolor("#161b22")
        plt.tight_layout()
        ui.plot(fig)
        plt.close(fig)
    except Exception:
        pass


def train_unet_model(
    coco_folder: str,
    epochs:      int   = 25,
    batch_size:  int   = 8,
    lr:          float = 1e-3,
    image_size:  int   = 256,
    features:    list | None = None,
) -> dict:
    """
    Train a UNet on data derived from a COCO dataset.

    Converts COCO→UNet format automatically if not already done.
    Detects n_classes from dataset_info.json (out_channels = n_classes + 1).
    Saves best and last checkpoints + results.json + loss curve to
    data/out_unet/<timestamp>_unet/.

    Parameters
    ----------
    coco_folder : str   Path to the *_COCO folder.
    epochs      : int   Training epochs (default 25).
    batch_size  : int   Batch size (default 8).
    lr          : float Learning rate (default 1e-3).
    image_size  : int   Resize images to this square size (default 512).
    features    : list  UNet encoder channel widths (default [64,128,256,512]).
    """
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader

    coco_folder = str(Path(coco_folder).resolve())

    # ── Pre-flight notice (shown BEFORE the slow conversion / dataset load) ────
    ui.log("INFO",
        f"🚀 Launching UNet training — dataset={Path(coco_folder).name}  epochs={epochs}  "
        f"batch={batch_size}  lr={lr}  imgsz={image_size}. This may take a while; "
        "per-epoch progress and a live loss curve will appear below.")

    # ── Convert if needed ────────────────────────────────────────────────────
    try:
        unet_folder = convert_coco_to_unet(coco_folder)
    except Exception as e:
        return _err(f"COCO→UNet conversion failed: {e}")

    # ── Read class info ──────────────────────────────────────────────────────
    info_path = os.path.join(unet_folder, "dataset_info.json")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    out_channels = info["n_out_channels"]   # n_classes + 1 (background)
    class_names  = info["class_names"]

    ui.log("INFO", f"UNet: {len(class_names)} class(es) → {out_channels} output channels "
                   f"(including background)")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = _make_dataset(unet_folder, "train", image_size)
    val_ds   = _make_dataset(unet_folder, "val",   image_size)

    if train_ds is None or len(train_ds) == 0:
        return _err("No training images found in _unet/images/train/.")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=pin)
                    if val_ds and len(val_ds) > 0 else None)

    # ── Model ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _build_unet(in_channels=3, out_channels=out_channels,
                         features=features).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Output dir ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{timestamp}_unet"
    out_dir   = _out_unet_root() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ui.log("INFO", f"Training UNet — epochs={epochs}  batch={batch_size}  "
                   f"lr={lr}  imgsz={image_size}  device={device}  "
                   f"(tip: reduce image_size or epochs for faster CPU runs)")

    train_losses: list = []
    val_losses:   list = []
    best_val_loss = float("inf")

    def _update_live_curve() -> None:
        """Redraw the live loss curve (one point per epoch), single in-place slot.

        Sent via ui.plot_live() so it is NOT recorded to the session gallery.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train",
                    color="#f85149", linewidth=1.5, marker="o", markersize=3)
            if val_losses:
                ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val",
                        color="#3fb950", linewidth=1.5, marker="o", markersize=3)
            ax.set_title("UNet — training loss (live)", fontsize=10, color="#c9d1d9")
            ax.set_xlabel("Epoch", color="#8b949e", fontsize=8)
            ax.set_ylabel("loss", color="#8b949e", fontsize=8)
            ax.set_xlim(1, max(int(epochs), len(train_losses)))
            ax.legend(labelcolor="#c9d1d9", facecolor="#161b22",
                      edgecolor="#30363d", fontsize=8)
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="#8b949e", labelsize=7)
            ax.grid(True, color="#21262d", linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            fig.patch.set_facecolor("#161b22")
            fig.tight_layout()
            ui.plot_live(fig, focus=(len(train_losses) <= 1))  # Live view; auto-shows on 1st epoch
        except Exception:
            pass  # non-critical

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs  = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    vl   += criterion(model(imgs), masks).item() * imgs.size(0)
            val_loss = vl / len(val_loader.dataset)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), str(out_dir / "unet_best.pth"))

        scheduler.step()
        ui.log("INFO", f"Epoch {epoch:>4}/{epochs}  "
                       f"train_loss={train_loss:.4f}  "
                       f"val_loss={val_loss:.4f}  "
                       f"lr={scheduler.get_last_lr()[0]:.2e}")
        # Live loss curve — single updating plot, NOT saved to the gallery
        _update_live_curve()

    # Save last checkpoint
    torch.save(model.state_dict(), str(out_dir / "unet_last.pth"))

    # ── Loss curve ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, epochs + 1), train_losses, label="Train", color="#f85149")
        if val_losses:
            ax.plot(range(1, epochs + 1), val_losses, label="Val", color="#3fb950")
        ax.set_xlabel("Epoch", color="#c9d1d9")
        ax.set_ylabel("Loss", color="#c9d1d9")
        ax.set_title("UNet Training Loss", color="#c9d1d9")
        ax.legend(labelcolor="#c9d1d9", facecolor="#161b22", edgecolor="#30363d")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        fig.patch.set_facecolor("#161b22")
        plt.tight_layout()
        _save_fig(fig, out_dir / "loss_curve.png")
        # Re-open and display
        _show_png(out_dir / "loss_curve.png", "Loss curve")
    except Exception:
        pass

    # ── Save results.json (evaluate_unet_model will update test_metrics) ────────
    actual_features = features or [64, 128, 256, 512]
    results = {
        "run_dir":          str(out_dir),
        "coco_folder":      coco_folder,
        "class_names":      class_names,
        "n_classes":        len(class_names),
        "out_channels":     out_channels,
        "epochs":           epochs,
        "batch_size":       batch_size,
        "lr":               lr,
        "image_size":       image_size,
        "features":         actual_features,
        "best_val_loss":    best_val_loss if val_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "test_metrics":     {},
        "best_pth":         str(out_dir / "unet_best.pth") if val_losses else None,
        "last_pth":         str(out_dir / "unet_last.pth"),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ui.log("OK", f"UNet training done. best_val_loss={best_val_loss:.4f}  → {out_dir}")

    # ── Evaluate (generates metrics + confusion matrix + predictions) ────────
    eval_result = evaluate_unet_model(str(out_dir))
    if eval_result.get("status") == "ok":
        results["test_metrics"] = eval_result.get("data", {})

    return _ok(data=results,
               message=f"UNet training completed. Run dir: {out_dir}")


def _show_unet_test_predictions(model, dataset, device, image_size,
                                out_channels, class_names, n_samples=4,
                                save_dir: Path | None = None,
                                split_name: str = "test"):
    """Show and save side-by-side input / GT / predicted mask panels."""
    import random
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as _F
    import numpy as np

    _COLORS = [
        (31,  119, 180), (255, 127,  14), (44,  160,  44),
        (214,  39,  40), (148, 103, 189), (140,  86,  75),
    ]

    def _colorize(mask_arr):
        rgb = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
        for c in range(1, out_channels):
            rgb[mask_arr == c] = _COLORS[(c - 1) % len(_COLORS)]
        return rgb

    def _blend(img, overlay, alpha=0.5):
        mask_any = overlay.any(axis=2, keepdims=True)
        return np.where(mask_any,
                        (img * (1 - alpha) + overlay * alpha).astype(np.uint8),
                        img)

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    ui.log("INFO", f"Generating {len(indices)} UNet prediction previews ({split_name})...")

    model.eval()
    with torch.no_grad():
        for sample_i, idx in enumerate(indices):
            try:
                img_t, mask_t = dataset[idx]
                # Recover filename from dataset if available
                fname = ""
                if hasattr(dataset, "files"):
                    fname = dataset.files[idx].name

                logits = model(img_t.unsqueeze(0).to(device))
                pred   = _F.softmax(logits, dim=1).argmax(1)[0].cpu().numpy()

                img_np     = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gt_np      = mask_t.numpy().astype(np.uint8)
                gt_blend   = _blend(img_np, _colorize(gt_np))
                pred_blend = _blend(img_np, _colorize(pred))

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_np)
                axes[0].set_title(f"[{split_name}] {fname}", fontsize=8, color="#c9d1d9")
                axes[0].axis("off")
                axes[1].imshow(gt_blend)
                axes[1].set_title("GT mask", fontsize=9, color="#c9d1d9")
                axes[1].axis("off")
                axes[2].imshow(pred_blend)
                axes[2].set_title("Prediction", fontsize=9, color="#c9d1d9")
                axes[2].axis("off")

                handles = [plt.Rectangle((0, 0), 1, 1,
                           color=[v/255 for v in _COLORS[(c-1) % len(_COLORS)]])
                           for c in range(1, out_channels)]
                labels  = [class_names[c-1] if c-1 < len(class_names) else f"cls {c}"
                           for c in range(1, out_channels)]
                if handles:
                    axes[2].legend(handles, labels, loc="lower right",
                                   fontsize=7, framealpha=0.6,
                                   labelcolor="#c9d1d9", facecolor="#161b22")

                fig.patch.set_facecolor("#161b22")
                plt.tight_layout()

                if save_dir is not None:
                    save_path = Path(save_dir) / f"test_pred_{sample_i}.png"
                    fig.savefig(str(save_path), dpi=100, bbox_inches="tight")
                    ui.log("INFO", f"  Saved: {save_path.name}")

                ui.plot(fig)
                plt.close(fig)

            except Exception as e:
                ui.log("WARN", f"  Prediction {sample_i} failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_unet_model(run_dir: str) -> dict:
    """
    Load saved UNet weights and run full evaluation on test (or val) split.

    Reads run metadata from results.json, reconstructs the model, evaluates
    on test (fallback: val), and saves/shows:
      metrics_bar.png, confusion_matrix.png, test_pred_0..3.png

    Also updates results.json with fresh test_metrics.
    """
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as _F
    import numpy as np

    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        return _err(f"Run directory not found: '{run_dir}'")

    json_path = run_path / "results.json"
    if not json_path.is_file():
        return _err(f"results.json not found in '{run_dir}'")

    with open(json_path, encoding="utf-8") as f:
        saved = json.load(f)

    coco_folder  = saved.get("coco_folder", "")
    unet_folder  = os.path.join(coco_folder, "_unet")
    class_names  = saved["class_names"]
    out_channels = saved["out_channels"]
    image_size   = saved.get("image_size", 256)
    batch_size   = saved.get("batch_size", 8)
    features     = saved.get("features")

    if not os.path.isdir(unet_folder):
        return _err(f"UNet data folder not found: '{unet_folder}'")

    weights_path = run_path / "unet_best.pth"
    if not weights_path.is_file():
        weights_path = run_path / "unet_last.pth"
    if not weights_path.is_file():
        return _err("No weights file found (unet_best.pth / unet_last.pth).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _build_unet(3, out_channels, features).to(device)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.eval()
    ui.log("INFO", f"Evaluating UNet — {weights_path.name}  device={device}")

    # Load dataset
    test_ds    = _make_dataset(unet_folder, "test", image_size)
    split_used = "test"
    if test_ds is None or len(test_ds) == 0:
        test_ds    = _make_dataset(unet_folder, "val", image_size)
        split_used = "val"
    if test_ds is None or len(test_ds) == 0:
        return _err("No test/val images found in UNet folder.")

    pin = torch.cuda.is_available()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)

    # Metrics
    tp       = torch.zeros(out_channels)
    fp       = torch.zeros(out_channels)
    fn       = torch.zeros(out_channels)
    conf_mat = np.zeros((out_channels, out_channels), dtype=np.int64)

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            preds = _F.softmax(model(imgs), dim=1).argmax(1)
            for c in range(out_channels):
                pred_c = (preds == c); true_c = (masks == c)
                tp[c] += (pred_c & true_c).sum().float()
                fp[c] += (pred_c & ~true_c).sum().float()
                fn[c] += (~pred_c & true_c).sum().float()
            np.add.at(conf_mat,
                      (masks.cpu().numpy().ravel(), preds.cpu().numpy().ravel()), 1)

    iou_per_class  = {}
    dice_per_class = {}
    for c in range(1, out_channels):
        cname = class_names[c - 1] if c - 1 < len(class_names) else f"class_{c}"
        iou  = (tp[c] / (tp[c] + fp[c] + fn[c] + 1e-8)).item()
        dice = (2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-8)).item()
        iou_per_class[cname]  = round(iou,  4)
        dice_per_class[cname] = round(dice, 4)

    mean_iou  = round(sum(iou_per_class.values())  / max(len(iou_per_class), 1), 4)
    mean_dice = round(sum(dice_per_class.values()) / max(len(dice_per_class), 1), 4)
    metrics = {
        "mean_iou": mean_iou, "mean_dice": mean_dice,
        "iou_per_class": iou_per_class, "dice_per_class": dice_per_class,
        "split_used": split_used,
    }
    ui.log("OK", f"Metrics ({split_used}) — mIoU={mean_iou:.4f}  mDice={mean_dice:.4f}")
    for cname, iou in iou_per_class.items():
        ui.log("INFO", f"  {cname}: IoU={iou:.4f}  Dice={dice_per_class[cname]:.4f}")

    # ── Metrics bar chart ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cnames = list(iou_per_class.keys())
        x = np.arange(len(cnames))
        fig, ax = plt.subplots(figsize=(max(5, len(cnames) * 1.5), 4))
        ax.bar(x - 0.2, [iou_per_class[c]  for c in cnames], 0.35, label="IoU",  color="#3fb950")
        ax.bar(x + 0.2, [dice_per_class[c] for c in cnames], 0.35, label="Dice", color="#58a6ff")
        ax.set_xticks(x); ax.set_xticklabels(cnames, color="#c9d1d9", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score", color="#c9d1d9")
        ax.set_title(f"Metrics ({split_used}) — mIoU={mean_iou:.3f}  mDice={mean_dice:.3f}",
                     color="#c9d1d9")
        ax.legend(labelcolor="#c9d1d9", facecolor="#161b22", edgecolor="#30363d")
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
        fig.patch.set_facecolor("#161b22")
        plt.tight_layout()
        _save_fig(fig, run_path / "metrics_bar.png")
        _show_png(run_path / "metrics_bar.png", "IoU / Dice per class")
    except Exception as e:
        ui.log("WARN", f"metrics_bar failed: {e}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        tick_labels = ["background"] + class_names
        conf_norm   = conf_mat / conf_mat.sum(axis=1, keepdims=True).clip(min=1)
        n = out_channels
        fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n * 1.0)))
        im = ax.imshow(conf_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", color="#c9d1d9", fontsize=8)
        ax.set_yticklabels(tick_labels, color="#c9d1d9", fontsize=8)
        ax.set_xlabel("Predicted", color="#c9d1d9"); ax.set_ylabel("True", color="#c9d1d9")
        ax.set_title(f"Confusion matrix ({split_used}, normalised)", color="#c9d1d9")
        for i in range(n):
            for j in range(n):
                val = conf_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.5 else "#c9d1d9")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#8b949e")
        plt.tight_layout()
        _save_fig(fig, run_path / "confusion_matrix.png")
        _show_png(run_path / "confusion_matrix.png", "Confusion matrix")
    except Exception as e:
        ui.log("WARN", f"confusion_matrix failed: {e}")

    # ── Test predictions ─────────────────────────────────────────────────────
    _show_unet_test_predictions(model, test_ds, device, image_size,
                                out_channels, class_names, n_samples=4,
                                save_dir=run_path, split_name=split_used)

    # ── Show loss curve if it exists ─────────────────────────────────────────
    loss_curve = run_path / "loss_curve.png"
    if loss_curve.is_file():
        _show_png(loss_curve, "Loss curve")

    # ── Update results.json with fresh metrics ───────────────────────────────
    saved["test_metrics"] = metrics
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)

    return _ok(data=metrics,
               message=f"UNet evaluation complete ({split_used}) — mIoU={mean_iou:.4f}  mDice={mean_dice:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Post-training inspection
# ══════════════════════════════════════════════════════════════════════════════

def get_unet_training_results(run_dir: str) -> dict:
    """Return results.json and show all evaluation plots.
    If key plots are missing (older run or interrupted training),
    re-runs evaluate_unet_model to regenerate them."""
    run_dir = str(Path(run_dir).resolve())
    if not os.path.isdir(run_dir):
        return _err(f"Run directory not found: '{run_dir}'")

    json_path = os.path.join(run_dir, "results.json")
    if not os.path.isfile(json_path):
        return _err(f"results.json not found in '{run_dir}'.")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    run_path  = Path(run_dir)
    key_plots = ["metrics_bar.png", "confusion_matrix.png"]
    missing   = [p for p in key_plots if not (run_path / p).is_file()]
    pred_pngs = sorted(run_path.glob("test_pred_*.png"))

    if missing or not pred_pngs:
        ui.log("INFO", "Evaluation plots missing — re-running evaluation...")
        return evaluate_unet_model(run_dir)

    # All plots present — just show them
    for name in ["loss_curve.png"] + key_plots:
        p = run_path / name
        if p.is_file():
            _show_png(p, p.stem.replace("_", " "))
    for p in pred_pngs:
        _show_png(p, f"Test prediction — {p.stem}")

    return _ok(data=data, message=f"UNet results loaded from '{run_dir}'.")


def list_unet_training_runs() -> dict:
    """List all UNet run folders inside data/out_unet/ sorted newest-first."""
    out_root = _out_unet_root()
    runs = []
    for p in sorted(out_root.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        json_path = p / "results.json"
        data: dict = {}
        if json_path.is_file():
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        runs.append({
            "name":          p.name,
            "path":          str(p),
            "has_best_pth":  (p / "unet_best.pth").is_file(),
            "has_last_pth":  (p / "unet_last.pth").is_file(),
            "has_results":   json_path.is_file(),
            "best_val_loss": data.get("best_val_loss"),
            "class_names":   data.get("class_names", []),
            "epochs":        data.get("epochs"),
        })
    return _ok(data=runs, message=f"{len(runs)} UNet run(s) found.")


# ══════════════════════════════════════════════════════════════════════════════
# Inference / prediction
# ══════════════════════════════════════════════════════════════════════════════

def predict_folder_unet(
    image_folder:  str,
    weights_path:  str,
    out_channels:  int,
    class_names:   list | None = None,
    image_size:    int   = 512,
    features:      list | None = None,
    alpha:         float = 0.5,
    save_masks:    bool  = False,
) -> dict:
    """
    Run UNet inference on all images in a folder.
    Saves side-by-side PNGs (original | overlay) and detections.csv to
    <image_folder>/out_unet_prediction_<timestamp>/.

    Parameters
    ----------
    image_folder : str    Folder with PNG/JPG images.
    weights_path : str    Path to a .pth state-dict file.
    out_channels : int    Must match the trained model (n_classes + 1).
    class_names  : list   Class labels for CSV (optional).
    image_size   : int    Resize to this square (default 512).
    features     : list   UNet channel widths (default [64,128,256,512]).
    alpha        : float  Overlay transparency (default 0.5).
    save_masks   : bool   Also save raw mask PNGs (default False).
    """
    import torch
    import torch.nn.functional as F
    import cv2
    import numpy as np
    import csv

    image_folder = str(Path(image_folder).resolve())
    if not os.path.isdir(image_folder):
        return _err(f"Image folder not found: '{image_folder}'")
    if not os.path.isfile(weights_path):
        return _err(f"Weights file not found: '{weights_path}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _build_unet(3, out_channels, features).to(device)
    sd     = torch.load(weights_path, map_location=device)
    sd     = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Default colors: class 1=red, 2=green, 3=blue, 4=yellow, …
    _COLORS = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),
               (0,255,255),(255,0,255),(128,0,255),(255,128,0)]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir   = Path(image_folder) / f"out_unet_prediction_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    exts  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = sorted(p for p in Path(image_folder).iterdir()
                   if p.suffix.lower() in exts)
    if not files:
        return _err(f"No images found in '{image_folder}'.")

    csv_path = out_dir / "detections.csv"
    fieldnames = ["index", "filename", "class_id", "class_name",
                  "area_px", "cx", "cy", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

    rows = []
    for idx, p in enumerate(files):
        img_bgr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            continue
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        H0, W0 = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_in  = cv2.resize(img_rgb, (image_size, image_size),
                             interpolation=cv2.INTER_AREA)

        x = (torch.from_numpy(img_in).float() / 255.0
             ).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(x)
            mask_s = F.softmax(logits, dim=1).argmax(1)[0].cpu().numpy().astype(np.uint8)

        # Resize mask back to original
        mask = cv2.resize(mask_s, (W0, H0), interpolation=cv2.INTER_NEAREST)

        # Overlay
        vis     = img_bgr.copy()
        overlay = vis.copy()
        for cls in range(1, out_channels):
            color = _COLORS[(cls - 1) % len(_COLORS)]
            overlay[mask == cls] = color
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # Contours
        for cls in range(1, out_channels):
            color = _COLORS[(cls - 1) % len(_COLORS)]
            cnt_mask = ((mask == cls).astype(np.uint8) * 255)
            if cnt_mask.max() == 0:
                continue
            contours, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)

            # CSV: one row per connected component
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 4:
                    continue
                M   = cv2.moments(cnt)
                cx  = int(M["m10"] / M["m00"]) if M["m00"] else 0
                cy  = int(M["m01"] / M["m00"]) if M["m00"] else 0
                x1, y1, w, h = cv2.boundingRect(cnt)
                cname = (class_names[cls - 1] if class_names and
                         cls - 1 < len(class_names) else str(cls))
                rows.append({
                    "index":    idx,
                    "filename": p.name,
                    "class_id": cls,
                    "class_name": cname,
                    "area_px":  int(area),
                    "cx":       cx, "cy": cy,
                    "bbox_x1":  x1, "bbox_y1": y1,
                    "bbox_x2":  x1 + w, "bbox_y2": y1 + h,
                })

        # Side-by-side
        canvas = np.hstack([img_bgr, vis])
        cv2.imwrite(str(out_dir / (p.stem + "_unet.png")), canvas)

        if save_masks:
            cv2.imwrite(str(out_dir / (p.stem + "_mask.png")), mask)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ui.log("OK", f"UNet prediction done — {len(files)} image(s), "
                 f"{len(rows)} detections → {out_dir}")

    return _ok(
        data={
            "out_dir":       str(out_dir),
            "n_images":      len(files),
            "n_detections":  len(rows),
            "csv_path":      str(csv_path),
        },
        message=f"Prediction complete. Results in {out_dir}",
    )

"""
src/trainer/yolo_trainer.py
───────────────────────────
YOLO training and COCO→YOLO conversion utilities for SimuScan datasets.

Converted datasets live INSIDE the COCO folder as hidden subfolders:
  <timestamp>_COCO/
    images/  annotations/
    _yolo_det/   ← YOLO detection  (images/ labels/ dataset.yaml)
    _yolo_seg/   ← YOLO segment    (images/ labels/ dataset.yaml)
    _unet/       ← UNet format     (images/ masks/)
    dataset_info.json  ← class names, n_classes (written on first conversion)

Public API
----------
convert_coco_to_yolo_dual(coco_folder)  → (det_folder, seg_folder)
train_yolo_model(coco_folder, ...)      → dict
get_training_results(run_dir)           → dict
check_gpu()                             → dict
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.bridge.ui_bridge import ui

# ── Path helpers ───────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _datasets_root() -> Path:
    return _repo_root() / "data" / "Datasets"


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


# ══════════════════════════════════════════════════════════════════════════════
# GPU check
# ══════════════════════════════════════════════════════════════════════════════

def check_gpu() -> dict:
    """
    Return GPU availability and device info.

    Returns
    -------
    dict  {"status": "ok", "data": {"gpu_available": bool, "device": str,
                                     "device_name": str | None,
                                     "cuda_version": str | None}}
    """
    try:
        import torch
        available = torch.cuda.is_available()
        name      = torch.cuda.get_device_name(0) if available else None
        cuda_ver  = torch.version.cuda if available else None
        device    = "0" if available else "cpu"
        return _ok(
            data={
                "gpu_available": available,
                "device":        device,
                "device_name":   name,
                "cuda_version":  cuda_ver,
            },
            message=f"{'GPU detected: ' + (name or '') if available else 'No GPU detected — CPU will be used.'}",
        )
    except ImportError:
        return _err("PyTorch is not installed. Cannot check GPU availability.")


# ══════════════════════════════════════════════════════════════════════════════
# COCO → YOLO dual conversion
# ══════════════════════════════════════════════════════════════════════════════

def _yolo_det_folder(coco_folder: str) -> str:
    return os.path.join(coco_folder, "_yolo_det")


def _yolo_seg_folder(coco_folder: str) -> str:
    return os.path.join(coco_folder, "_yolo_seg")


def _save_dataset_info(coco_folder: str, class_names: list) -> None:
    """Write dataset_info.json into the COCO folder (idempotent)."""
    info_path = Path(coco_folder) / "dataset_info.json"
    if not info_path.exists():
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({"class_names": class_names, "n_classes": len(class_names)}, f)


def load_dataset_info(coco_folder: str) -> dict:
    """Return dataset_info.json contents, or empty dict if not present."""
    info_path = Path(coco_folder) / "dataset_info.json"
    if info_path.is_file():
        try:
            with open(info_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def convert_coco_to_yolo_dual(coco_folder: str) -> Tuple[str, str]:
    """
    Convert a COCO dataset into two YOLO datasets stored INSIDE the COCO folder:
      <coco_folder>/_yolo_det/   detection  (cls xc yc w h)
      <coco_folder>/_yolo_seg/   segmentation (cls x1 y1 … xn yn)

    If the subfolders already exist (dataset.yaml present) the conversion is
    skipped and the existing paths are returned immediately.

    Parameters
    ----------
    coco_folder : str
        Absolute or relative path to a folder whose name ends with '_COCO'.

    Returns
    -------
    (yolo_det_folder, yolo_seg_folder) : Tuple[str, str]
    """
    coco_folder = str(Path(coco_folder).resolve())

    if not coco_folder.endswith("_COCO"):
        raise ValueError("The input folder name must end with '_COCO'")

    coco_images_dir = os.path.join(coco_folder, "images")
    coco_ann_dir    = os.path.join(coco_folder, "annotations")

    if not os.path.isdir(coco_ann_dir):
        raise FileNotFoundError(f"Annotations folder not found: {coco_ann_dir}")

    yolo_det_folder = _yolo_det_folder(coco_folder)
    yolo_seg_folder = _yolo_seg_folder(coco_folder)

    # Skip if already converted
    if (os.path.isfile(os.path.join(yolo_det_folder, "dataset.yaml")) and
            os.path.isfile(os.path.join(yolo_seg_folder, "dataset.yaml"))):
        return yolo_det_folder, yolo_seg_folder

    for out_root in (yolo_det_folder, yolo_seg_folder):
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    class_names: List[str]      = []
    catid_to_idx: Dict[int, int] = {}

    def clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    def write_yaml(out_root: str, task: str) -> None:
        data = {
            "path":  out_root.replace("\\", "/"),
            "train": "images/train",
            "val":   "images/val",
            "names": class_names,
            "nc":    len(class_names),
            "task":  task,
        }
        test_dir = os.path.join(out_root, "images", "test")
        if os.path.isdir(test_dir) and os.listdir(test_dir):
            data["test"] = "images/test"
        with open(os.path.join(out_root, "dataset.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    splits_processed = []

    for split in ("train", "val", "test"):
        ann_file   = os.path.join(coco_ann_dir, f"instances_{split}.json")
        img_folder = os.path.join(coco_images_dir, split)

        if not (os.path.isfile(ann_file) and os.path.isdir(img_folder)):
            continue

        splits_processed.append(split)

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        if not class_names:
            cats_sorted  = sorted(coco["categories"], key=lambda c: c["id"])
            class_names  = [c["name"] for c in cats_sorted]
            catid_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}

        images_info: Dict[int, dict] = {im["id"]: im for im in coco["images"]}

        anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_id, img in images_info.items():
            fname    = img["file_name"]
            W, H     = img["width"], img["height"]
            src_path = os.path.join(img_folder, fname)

            if not os.path.isfile(src_path):
                continue

            for dst_dir in (
                os.path.join(yolo_det_folder, "images", split),
                os.path.join(yolo_seg_folder, "images", split),
            ):
                dst = os.path.join(dst_dir, fname)
                if not os.path.isfile(dst):
                    shutil.copyfile(src_path, dst)

            stem         = os.path.splitext(fname)[0]
            lab_det_path = os.path.join(yolo_det_folder, "labels", split, stem + ".txt")
            lab_seg_path = os.path.join(yolo_seg_folder, "labels", split, stem + ".txt")

            det_lines: List[str] = []
            seg_lines: List[str] = []

            for ann in anns_by_img.get(img_id, []):
                cls = catid_to_idx[ann["category_id"]]

                # Detection: COCO xywh → YOLO normalised xc yc w h
                bbox = ann.get("bbox")
                if bbox:
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        xc = clamp01((x + w / 2) / W)
                        yc = clamp01((y + h / 2) / H)
                        wn = clamp01(w / W)
                        hn = clamp01(h / H)
                        det_lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

                # Segmentation: polygon list → YOLO normalised coords
                seg = ann.get("segmentation", [])
                if isinstance(seg, list) and seg and isinstance(seg[0], list):
                    for poly in seg:
                        if len(poly) < 6:
                            continue
                        pts = [
                            f"{clamp01(poly[i]/W):.6f} {clamp01(poly[i+1]/H):.6f}"
                            for i in range(0, len(poly), 2)
                        ]
                        seg_lines.append(f"{cls} " + " ".join(pts))

            with open(lab_det_path, "w", encoding="utf-8") as f:
                f.write("\n".join(det_lines) + ("\n" if det_lines else ""))

            with open(lab_seg_path, "w", encoding="utf-8") as f:
                f.write("\n".join(seg_lines) + ("\n" if seg_lines else ""))

    if not class_names:
        raise RuntimeError("No valid splits found — class names could not be determined.")

    write_yaml(yolo_det_folder, task="detect")
    write_yaml(yolo_seg_folder, task="segment")
    _save_dataset_info(coco_folder, class_names)

    return yolo_det_folder, yolo_seg_folder


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

# Default model names per task
_DEFAULT_MODELS = {
    "detect":  "yolov8m.pt",
    "segment": "yolov8m-seg.pt",
}


@dataclass
class TrainResult:
    status:      str
    message:     str
    run_dir:     str
    task:        str
    model_path:  str          # best.pt path
    epochs_done: int
    metrics:     dict


def _detect_device() -> str:
    """Return '0' if a CUDA GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _read_results_csv(run_dir: str) -> dict:
    """
    Parse the results.csv written by Ultralytics during training.
    Returns a dict of {metric_name: last_value}.
    """
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(csv_path):
        return {}
    try:
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        if not rows:
            return {}
        last = rows[-1]
        # Strip whitespace from keys/values
        return {k.strip(): _try_float(v.strip()) for k, v in last.items()}
    except Exception:
        return {}


def _try_float(v: str):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def train_yolo_model(
    coco_folder:  str,
    task:         str   = "segment",
    model_name:   str | None = None,
    epochs:       int   = 100,
    imgsz:        int   = 512,
    batch:        int   = 32,
) -> dict:
    """
    Train a YOLO model on a converted dataset derived from a COCO folder.

    The function resolves the correct YOLO dataset automatically based on `task`,
    selects the appropriate default model, detects GPU/CPU, and saves results
    to data/out_yolo/<timestamp>.

    Parameters
    ----------
    coco_folder : str
        Path to the original *_COCO folder (used to resolve _YOLO_DET/_YOLO_SEG).
    task : str
        'segment' (default) or 'detect'.
    model_name : str | None
        Override the base model, e.g. 'yolov8l-seg.pt'. If None the default
        for the selected task is used (yolov8m-seg.pt / yolov8m.pt).
    epochs : int
        Number of training epochs (default 100).
    imgsz : int
        Input image size in pixels (default 512).
    batch : int
        Batch size (default 32).

    Returns
    -------
    dict  {"status": "ok"|"error", "message": str, "data": TrainResult-as-dict}
    """
    # ── Validate task ─────────────────────────────────────────────────────────
    task = task.lower()
    if task not in ("detect", "segment"):
        return _err(f"Invalid task '{task}'. Must be 'detect' or 'segment'.")

    # ── Resolve YOLO dataset folder ───────────────────────────────────────────
    coco_folder = str(Path(coco_folder).resolve())
    if not coco_folder.endswith("_COCO"):
        return _err("coco_folder must end with '_COCO'.")

    yolo_folder = (_yolo_det_folder(coco_folder) if task == "detect"
                   else _yolo_seg_folder(coco_folder))
    yaml_path   = os.path.join(yolo_folder, "dataset.yaml")

    if not os.path.isfile(yaml_path):
        return _err(
            f"dataset.yaml not found at '{yaml_path}'. "
            "Run the COCO→YOLO conversion first."
        )

    # ── Select model ──────────────────────────────────────────────────────────
    base_model = model_name or _DEFAULT_MODELS[task]

    # ── Resolve output directory ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{timestamp}_{task}"
    out_dir   = str(_out_yolo_root() / run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Detect device ─────────────────────────────────────────────────────────
    device = _detect_device()

    # ── Pre-flight notice ─────────────────────────────────────────────────────
    # Emitted BEFORE the slow part (model download + dataset scan inside YOLO()),
    # so the user immediately sees that training was launched and what it will do.
    ui.log("INFO",
        f"🚀 Launching YOLO training — task={task}  dataset={Path(coco_folder).name}  "
        f"epochs={epochs}  imgsz={imgsz}  batch={batch}  device={device}  model={base_model}")
    ui.log("INFO",
        "Preparing training (model download + dataset scan)… this may take a while. "
        "Per-epoch progress and a live loss curve will appear below as it runs.")

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO

        model = YOLO(base_model)  # downloads automatically if not cached

        # Accumulators for the live (single, in-place) loss curve
        live_epochs: list[int]   = []
        live_loss:   list[float] = []

        def _update_live_curve(total_epochs: int) -> None:
            """Redraw the live training-loss curve (one point per epoch).

            Sent via ui.plot_live() so it occupies a single, in-place slot in the
            Plot panel and is NOT recorded to the session gallery.
            """
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(live_epochs, live_loss, color="#f85149", linewidth=1.5,
                        marker="o", markersize=3)
                ax.set_title("YOLO — training loss (live)", fontsize=10, color="#c9d1d9")
                ax.set_xlabel("Epoch", color="#8b949e", fontsize=8)
                ax.set_ylabel("loss", color="#8b949e", fontsize=8)
                if total_epochs:
                    ax.set_xlim(1, max(int(total_epochs), len(live_epochs)))
                ax.set_facecolor("#0e1117")
                ax.tick_params(colors="#8b949e", labelsize=7)
                ax.grid(True, color="#21262d", linewidth=0.5)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#30363d")
                fig.patch.set_facecolor("#161b22")
                fig.tight_layout()
                ui.plot_live(fig, focus=(len(live_epochs) <= 1))  # Live view; auto-shows on 1st epoch
            except Exception:
                pass  # non-critical

        # ── GUI callbacks — route progress to the log panel ───────────────────
        def _on_train_start(trainer):
            ui.log("INFO", f"Training started — {trainer.epochs} epochs  "
                           f"device={device}  imgsz={imgsz}  batch={batch}")

        def _on_train_epoch_end(trainer):
            e    = trainer.epoch + 1
            tot  = trainer.epochs
            #loss = float(trainer.loss) if hasattr(trainer, "loss") else 0.0
            loss = float(trainer.loss.detach()) if hasattr(trainer, "loss") else 0.0
            # Try to get box/seg loss components if available
            metrics = getattr(trainer, "metrics", {}) or {}
            mAP     = metrics.get("metrics/mAP50(B)", metrics.get("metrics/mAP50", None))
            if mAP is not None:
                ui.log("INFO", f"Epoch {e:>4}/{tot}  loss={loss:.4f}  mAP50={mAP:.4f}")
            else:
                ui.log("INFO", f"Epoch {e:>4}/{tot}  loss={loss:.4f}")
            # Live loss curve — single updating plot, NOT saved to the gallery
            live_epochs.append(e)
            live_loss.append(loss)
            _update_live_curve(tot)

        def _on_train_end(trainer):
            metrics = getattr(trainer, "metrics", {}) or {}
            mAP50   = metrics.get("metrics/mAP50(B)", metrics.get("metrics/mAP50", "—"))
            mAP5095 = metrics.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP50-95", "—"))
            ui.log("OK", f"Training complete — "
                         f"mAP50={mAP50}  mAP50-95={mAP5095}  "
                         f"best.pt → {out_dir}/weights/best.pt")
            # Show final training curves + Ultralytics plots
            _show_final_plots(out_dir)
            # Generate + save + show test predictions
            evaluate_yolo_model(out_dir)

        def _show_final_plots(run_dir: str):
            """Display training curves and any PNG plots Ultralytics generated."""
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                import csv as _csv
                run_path = Path(run_dir)

                # 1) Training curves from results.csv
                csv_path = run_path / "results.csv"
                if csv_path.is_file():
                    rows = []
                    with open(csv_path, newline="", encoding="utf-8") as f:
                        reader = _csv.DictReader(f)
                        rows = list(reader)
                    if len(rows) >= 2:
                        all_keys  = [k.strip() for k in rows[0].keys()]
                        loss_keys = [k for k in all_keys if "loss" in k.lower()]
                        map_keys  = [k for k in all_keys
                                     if "map50" in k.lower() and "95" not in k.lower()]
                        plot_keys = loss_keys + map_keys
                        if plot_keys:
                            epochs_x = list(range(1, len(rows) + 1))
                            cols  = min(3, len(plot_keys))
                            rows_n = (len(plot_keys) + cols - 1) // cols
                            fig, axes = plt.subplots(rows_n, cols,
                                                     figsize=(cols * 4, rows_n * 3))
                            axes_flat = ([axes] if len(plot_keys) == 1
                                         else list(axes.flat if hasattr(axes, "flat") else axes))
                            for ax, key in zip(axes_flat, plot_keys):
                                vals = []
                                for r in rows:
                                    try:
                                        vals.append(float(r[key]))
                                    except Exception:
                                        vals.append(float("nan"))
                                color = "#f85149" if "loss" in key.lower() else "#3fb950"
                                ax.plot(epochs_x, vals, color=color, linewidth=1.5)
                                ax.set_title(key.strip(), fontsize=9, color="#c9d1d9")
                                ax.set_xlabel("Epoch", color="#8b949e", fontsize=7)
                                ax.set_facecolor("#0e1117")
                                ax.tick_params(colors="#8b949e", labelsize=7)
                                for spine in ax.spines.values():
                                    spine.set_edgecolor("#30363d")
                            for ax in axes_flat[len(plot_keys):]:
                                ax.set_visible(False)
                            fig.patch.set_facecolor("#161b22")
                            fig.suptitle("YOLO — training curves (final)", fontsize=10,
                                         color="#c9d1d9")
                            fig.tight_layout()
                            ui.plot(fig)
                            plt.close(fig)

                # 2) Ultralytics PNG plots (confusion matrix, PR curve, etc.)
                for png in sorted(run_path.glob("*.png")):
                    try:
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.imshow(mpimg.imread(str(png)))
                        ax.set_title(png.stem.replace("_", " "), fontsize=9,
                                     color="#c9d1d9")
                        ax.axis("off")
                        fig.patch.set_facecolor("#161b22")
                        plt.tight_layout()
                        ui.plot(fig)
                        plt.close(fig)
                    except Exception:
                        pass
            except Exception:
                pass

        def _show_test_predictions(run_dir: str, yolo_dir: str, n_samples: int = 4):
            """Run inference on test images with best.pt and show overlays."""
            try:
                import random
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg

                best_pt = Path(run_dir) / "weights" / "best.pt"
                if not best_pt.is_file():
                    return

                test_img_dir = Path(yolo_dir) / "images" / "test"
                split_name   = "test"
                if not test_img_dir.is_dir():
                    test_img_dir = Path(yolo_dir) / "images" / "val"
                    split_name   = "val"
                if not test_img_dir.is_dir():
                    return

                exts   = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
                imgs   = [p for p in test_img_dir.iterdir() if p.suffix.lower() in exts]
                if not imgs:
                    return
                sample = random.sample(imgs, min(n_samples, len(imgs)))

                from ultralytics import YOLO as _YOLO
                pred_model = _YOLO(str(best_pt))

                ui.log("INFO", f"Showing {len(sample)} {split_name} predictions...")
                for img_path in sample:
                    results = pred_model.predict(str(img_path), verbose=False,
                                                 conf=0.25, save=False)
                    if not results:
                        continue
                    rendered = results[0].plot()[:, :, ::-1]  # BGR→RGB
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].imshow(mpimg.imread(str(img_path)), cmap="gray")
                    axes[0].set_title(f"[{split_name}] {img_path.name}",
                                      fontsize=8, color="#c9d1d9")
                    axes[0].axis("off")
                    axes[1].imshow(rendered)
                    axes[1].set_title(f"Prediction — {img_path.name}",
                                      fontsize=8, color="#c9d1d9")
                    axes[1].axis("off")
                    fig.patch.set_facecolor("#161b22")
                    plt.tight_layout()
                    ui.plot(fig)
                    plt.close(fig)
            except Exception as e:
                ui.log("WARN", f"Test preview failed: {e}")

        def _on_fit_epoch_end(trainer):
            # Intermediate training curve every 10 epochs (skip last — shown in _on_train_end)
            try:
                e = trainer.epoch + 1
                if e % 10 == 0 and e != trainer.epochs:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import csv as _csv
                    csv_path = str(Path(out_dir) / "results.csv")
                    if not Path(csv_path).is_file():
                        return
                    rows = []
                    with open(csv_path, newline="", encoding="utf-8") as f:
                        reader = _csv.DictReader(f)
                        rows = list(reader)
                    if len(rows) < 2:
                        return
                    epochs_x = list(range(1, len(rows) + 1))
                    # Find loss and mAP columns
                    all_keys = [k.strip() for k in rows[0].keys()]
                    loss_keys = [k for k in all_keys if "loss" in k.lower() and "val" not in k.lower()]
                    map_keys  = [k for k in all_keys if "map50" in k.lower() and "95" not in k.lower()]
                    n_plots = len(loss_keys) + len(map_keys)
                    if n_plots == 0:
                        return
                    cols = min(3, n_plots)
                    rows_n = (n_plots + cols - 1) // cols
                    fig, axes = plt.subplots(rows_n, cols,
                                             figsize=(cols * 4, rows_n * 3))
                    axes = [axes] if n_plots == 1 else list(
                        axes.flat if hasattr(axes, "flat") else axes
                    )
                    for ax, key in zip(axes, loss_keys + map_keys):
                        vals = []
                        for r in rows:
                            try:
                                vals.append(float(r[key]))
                            except Exception:
                                vals.append(float("nan"))
                        color = "#f85149" if "loss" in key.lower() else "#3fb950"
                        ax.plot(epochs_x, vals, color=color, linewidth=1.5)
                        ax.set_title(key.strip(), fontsize=9, color="#c9d1d9")
                        ax.set_facecolor("#0e1117")
                        ax.tick_params(colors="#8b949e", labelsize=7)
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#30363d")
                    for ax in axes[n_plots:]:
                        ax.set_visible(False)
                    fig.patch.set_facecolor("#161b22")
                    fig.suptitle(f"Training curves — epoch {e}", fontsize=10,
                                 color="#c9d1d9")
                    fig.tight_layout()
                    ui.plot(fig)
            except Exception:
                pass  # non-critical

        model.add_callback("on_train_start",     _on_train_start)
        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        model.add_callback("on_train_end",       _on_train_end)

        model.train(
            data            = yaml_path,
            epochs          = epochs,
            imgsz           = imgsz,
            batch           = batch,
            device          = device,
            project         = str(_out_yolo_root()),
            name            = run_name,
            exist_ok        = True,
            #verbose         = False,   # silence Ultralytics console output
            # Augmentation policy tuned for AFM synthetic data
            degrees         = 0.0,
            translate       = 0.0,
            scale           = 0.0,
            shear           = 0.0,
            perspective     = 0.0,
            flipud          = 0.5,
            fliplr          = 0.5,
            mosaic          = 0.0,
            mixup           = 0.0,
            hsv_v           = 0.1,
            label_smoothing = 0.05,
        )

    except Exception as exc:
        return _err(f"Training failed: {exc}", detail=str(exc))

    # ── Collect best.pt path ──────────────────────────────────────────────────
    best_pt = os.path.join(out_dir, "weights", "best.pt")
    if not os.path.isfile(best_pt):
        # Ultralytics may put weights one level up depending on version
        best_pt = os.path.join(str(_out_yolo_root()), run_name, "weights", "best.pt")

    metrics = _read_results_csv(out_dir)

    result = TrainResult(
        status      = "ok",
        message     = f"Training completed. Results saved to '{out_dir}'.",
        run_dir     = out_dir,
        task        = task,
        model_path  = best_pt if os.path.isfile(best_pt) else "not found",
        epochs_done = epochs,
        metrics     = metrics,
    )

    return _ok(data=asdict(result), message=result.message)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_yolo_model(run_dir: str, n_samples: int = 4) -> dict:
    """
    Load best.pt from a YOLO run and generate prediction images on test/val.
    Saves PNGs to <run_dir>/test_predictions/ and shows them in the plot panel.
    The dataset folder is resolved from args.yaml written by Ultralytics.
    """
    import random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        return _err(f"Run directory not found: '{run_dir}'")

    best_pt = run_path / "weights" / "best.pt"
    if not best_pt.is_file():
        return _err("best.pt not found in run directory.")

    # Resolve dataset folder from args.yaml (written by Ultralytics at training start)
    args_yaml = run_path / "args.yaml"
    yolo_folder: str | None = None
    if args_yaml.is_file():
        try:
            with open(args_yaml, encoding="utf-8") as f:
                args = yaml.safe_load(f) or {}
            data_path = args.get("data", "")
            if data_path and Path(data_path).is_file():
                yolo_folder = str(Path(data_path).parent)
        except Exception:
            pass

    if not yolo_folder or not os.path.isdir(yolo_folder):
        return _err("Cannot locate YOLO dataset folder from args.yaml.")

    test_img_dir = Path(yolo_folder) / "images" / "test"
    split_name   = "test"
    if not test_img_dir.is_dir():
        test_img_dir = Path(yolo_folder) / "images" / "val"
        split_name   = "val"
    if not test_img_dir.is_dir():
        return _err("No test/val images folder found.")

    exts  = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    imgs  = [p for p in test_img_dir.iterdir() if p.suffix.lower() in exts]
    if not imgs:
        return _err("No images found in test/val folder.")

    sample = random.sample(imgs, min(n_samples, len(imgs)))

    try:
        from ultralytics import YOLO as _YOLO
        pred_model    = _YOLO(str(best_pt))
        out_pred_dir  = run_path / "test_predictions"
        out_pred_dir.mkdir(exist_ok=True)

        ui.log("INFO", f"YOLO evaluation — {len(sample)} {split_name} predictions...")
        n_saved = 0
        for img_path in sample:
            results = pred_model.predict(str(img_path), verbose=False, conf=0.25, save=False)
            if not results:
                continue
            rendered = results[0].plot()[:, :, ::-1]  # BGR→RGB
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(mpimg.imread(str(img_path)), cmap="gray")
            axes[0].set_title(f"[{split_name}] {img_path.name}", fontsize=8, color="#c9d1d9")
            axes[0].axis("off")
            axes[1].imshow(rendered)
            axes[1].set_title(f"Prediction — {img_path.name}", fontsize=8, color="#c9d1d9")
            axes[1].axis("off")
            fig.patch.set_facecolor("#161b22")
            plt.tight_layout()
            save_path = out_pred_dir / (img_path.stem + "_pred.png")
            fig.savefig(str(save_path), dpi=100, bbox_inches="tight")
            n_saved += 1
            ui.plot(fig)
            plt.close(fig)

        ui.log("OK", f"YOLO evaluation done — {n_saved} prediction(s) → {out_pred_dir}")
        return _ok(
            data={"n_predictions": n_saved, "split": split_name,
                  "out_dir": str(out_pred_dir)},
            message=f"YOLO evaluation complete — {n_saved} predictions ({split_name})",
        )
    except Exception as e:
        return _err(f"YOLO evaluation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Post-training inspection
# ══════════════════════════════════════════════════════════════════════════════

def get_training_results(run_dir: str) -> dict:
    """
    Read and return the training metrics from a completed run directory.
    Also shows training curves and all Ultralytics PNG plots in the plot panel.

    Parameters
    ----------
    run_dir : str
        Path to a specific run folder inside data/out_yolo/.

    Returns
    -------
    dict with last-epoch metrics, best.pt location, and run metadata.
    """
    run_dir = str(Path(run_dir).resolve())

    if not os.path.isdir(run_dir):
        return _err(f"Run directory not found: '{run_dir}'")

    metrics   = _read_results_csv(run_dir)
    best_pt   = os.path.join(run_dir, "weights", "best.pt")
    last_pt   = os.path.join(run_dir, "weights", "last.pt")
    args_yaml = os.path.join(run_dir, "args.yaml")

    train_args: dict = {}
    if os.path.isfile(args_yaml):
        try:
            with open(args_yaml, encoding="utf-8") as f:
                train_args = yaml.safe_load(f) or {}
        except Exception:
            pass

    data = {
        "run_dir":        run_dir,
        "best_pt":        best_pt if os.path.isfile(best_pt) else None,
        "last_pt":        last_pt if os.path.isfile(last_pt) else None,
        "last_epoch_metrics": metrics,
        "train_args": {
            k: train_args.get(k)
            for k in ("model", "data", "epochs", "imgsz", "batch", "device", "task")
            if k in train_args
        },
    }

    # ── Show training curves + Ultralytics PNG plots ──────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import csv as _csv
        run_path = Path(run_dir)

        # Training curves from results.csv
        csv_path = run_path / "results.csv"
        if csv_path.is_file():
            rows = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                rows = list(reader)
            if len(rows) >= 2:
                all_keys  = [k.strip() for k in rows[0].keys()]
                loss_keys = [k for k in all_keys if "loss" in k.lower()]
                map_keys  = [k for k in all_keys
                             if "map50" in k.lower() and "95" not in k.lower()]
                plot_keys = loss_keys + map_keys
                if plot_keys:
                    epochs_x = list(range(1, len(rows) + 1))
                    cols   = min(3, len(plot_keys))
                    rows_n = (len(plot_keys) + cols - 1) // cols
                    fig, axes = plt.subplots(rows_n, cols,
                                             figsize=(cols * 4, rows_n * 3))
                    axes_flat = ([axes] if len(plot_keys) == 1
                                 else list(axes.flat if hasattr(axes, "flat") else axes))
                    for ax, key in zip(axes_flat, plot_keys):
                        vals = []
                        for r in rows:
                            try:
                                vals.append(float(r[key]))
                            except Exception:
                                vals.append(float("nan"))
                        color = "#f85149" if "loss" in key.lower() else "#3fb950"
                        ax.plot(epochs_x, vals, color=color, linewidth=1.5)
                        ax.set_title(key.strip(), fontsize=9, color="#c9d1d9")
                        ax.set_xlabel("Epoch", color="#8b949e", fontsize=7)
                        ax.set_facecolor("#0e1117")
                        ax.tick_params(colors="#8b949e", labelsize=7)
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#30363d")
                    for ax in axes_flat[len(plot_keys):]:
                        ax.set_visible(False)
                    fig.patch.set_facecolor("#161b22")
                    fig.suptitle("YOLO — training curves", fontsize=10, color="#c9d1d9")
                    fig.tight_layout()
                    ui.plot(fig)
                    plt.close(fig)

        # Ultralytics PNG plots (confusion matrix, PR curve, F1 curve, etc.)
        for png in sorted(run_path.glob("*.png")):
            try:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.imshow(mpimg.imread(str(png)))
                ax.set_title(png.stem.replace("_", " "), fontsize=9,
                             color="#c9d1d9")
                ax.axis("off")
                fig.patch.set_facecolor("#161b22")
                plt.tight_layout()
                ui.plot(fig)
                plt.close(fig)
            except Exception:
                pass
    except Exception as e:
        ui.log("WARN", f"Could not load plots for run '{run_dir}': {e}")

    # Show saved prediction images (or generate them if missing)
    pred_dir  = Path(run_dir) / "test_predictions"
    pred_pngs = sorted(pred_dir.glob("*_pred.png")) if pred_dir.is_dir() else []
    if pred_pngs:
        for p in pred_pngs:
            try:
                import matplotlib.image as _mpi
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(_mpi.imread(str(p)))
                ax.set_title(p.stem.replace("_", " "), fontsize=8, color="#c9d1d9")
                ax.axis("off")
                fig.patch.set_facecolor("#161b22")
                plt.tight_layout()
                ui.plot(fig)
                plt.close(fig)
            except Exception:
                pass
    else:
        ui.log("INFO", "No saved predictions found — running evaluation now...")
        evaluate_yolo_model(run_dir)

    return _ok(data=data, message=f"Results loaded from '{run_dir}'.")


def list_training_runs() -> dict:
    """
    List all training run folders inside data/out_yolo/ with basic metadata.

    Returns
    -------
    dict with a list of run summaries sorted newest-first.
    """
    out_root = _out_yolo_root()
    if not out_root.exists():
        return _ok(data=[], message="No training runs found (out_yolo folder does not exist).")

    runs = []
    for p in sorted(out_root.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        best_pt = p / "weights" / "best.pt"
        last_pt = p / "weights" / "last.pt"
        csv_exists = (p / "results.csv").is_file()
        runs.append({
            "name":          p.name,
            "path":          str(p),
            "has_best_pt":   best_pt.is_file(),
            "has_last_pt":   last_pt.is_file(),
            "has_results":   csv_exists,
        })
    return _ok(data=runs, message=f"{len(runs)} training run(s) found.")


def list_datasets() -> dict:
    """
    List all COCO dataset folders inside data/Datasets/.

    Each entry shows per-split image counts, class names, and whether the
    derived formats (_yolo_det, _yolo_seg, _unet) are already converted.
    """
    ds_root = _datasets_root()
    if not ds_root.exists():
        return _ok(data=[], message="No datasets found (Datasets folder does not exist).")

    summaries = []
    for p in sorted(ds_root.iterdir(), reverse=True):
        if not p.is_dir() or not p.name.endswith("_COCO"):
            continue

        name = p.name

        # Count images per split
        split_counts: dict = {}
        img_dir = p / "images"
        if img_dir.is_dir():
            for split_dir in img_dir.iterdir():
                if split_dir.is_dir():
                    n = sum(
                        1 for f in split_dir.iterdir()
                        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
                    )
                    split_counts[split_dir.name] = n

        # Class names from dataset_info.json (written during conversion) or annotations
        info = load_dataset_info(str(p))
        class_names: list = info.get("class_names", [])
        if not class_names:
            for ann_split in ("train", "val"):
                ann_file = p / "annotations" / f"instances_{ann_split}.json"
                if ann_file.is_file():
                    try:
                        with open(ann_file, encoding="utf-8") as f:
                            coco = json.load(f)
                        class_names = [
                            c["name"]
                            for c in sorted(coco.get("categories", []), key=lambda c: c["id"])
                        ]
                        break
                    except Exception:
                        pass

        # Conversion status
        yolo_det_ready = (p / "_yolo_det" / "dataset.yaml").is_file()
        yolo_seg_ready = (p / "_yolo_seg" / "dataset.yaml").is_file()
        unet_ready     = (p / "_unet" / "dataset_info.json").is_file()

        summaries.append({
            "name":           name,
            "path":           str(p),
            "split_counts":   split_counts,
            "total_images":   sum(split_counts.values()),
            "classes":        class_names,
            "n_classes":      len(class_names),
            "yolo_det_ready": yolo_det_ready,
            "yolo_seg_ready": yolo_seg_ready,
            "unet_ready":     unet_ready,
        })

    return _ok(data=summaries, message=f"{len(summaries)} dataset(s) found.")
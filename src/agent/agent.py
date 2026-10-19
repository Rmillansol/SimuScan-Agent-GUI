"""
src/agent/agent.py
──────────────────
Agent loop — GUI version.

All output goes through the UIBridge singleton:
  ui.log("LEVEL", "message")   → Log panel
  ui.chat("markdown text")     → Chat panel
  (matplotlib figures go via ui.plot() in the visualizer files)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.bridge.ui_bridge import ui
from src.simuscan.syntheticAFM_COCO import visualize_coco_samples, generate_data_for_training
from src.trainer.yolo_trainer import convert_coco_to_yolo_dual, train_yolo_model, check_gpu
from src.trainer.unet_trainer import train_unet_model, predict_folder_unet
from src.trainer.yolo_visualizer import visualize_yolo_dataset, predict_folder, predict_gwy_folder, get_model_info
from src.trainer.triplet_trainer import run_triplet_training
from src.trainer.triplet_inference import run_triplet_library
from src.trainer.triplet_visualizer import (
    show_crop_grid, show_embedding_umap,
    get_triplet_model_info, find_latest_triplet_pt,
)
from src.automation.automation import (
    connect_microscope     as _connect_microscope,
    run_exploration        as _run_exploration,
    get_exploration_status as _get_exploration_status,
    stop_exploration       as _stop_exploration,
)
from src.simuscan.image_shape import generate_mask_from_image

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent.parent
_CHATS_DIR  = _ROOT / "data" / "chats"
_CONFIG_DIR = _ROOT / "data" / "config"
_ENV        = os.environ.copy()
_ENV["PYTHONPATH"] = str(_ROOT)

_EXIT_WORDS = {"exit", "quit", "salir", "adios", "adiós", "bye", "goodbye", "q"}

# ── Chat logger ───────────────────────────────────────────────────────────────

class ChatLogger:
    def __init__(self, session_id: str) -> None:
        self.session_id  = session_id
        self.session_dir = _CHATS_DIR / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log_path    = self.session_dir / "conversation.json"
        self._log_file   = self.session_dir / "session.log"
        self._images_file = self.session_dir / "images.json"
        self._plots_dir  = self.session_dir / "plots"
        self._entries:      list[dict] = []
        self._image_entries: list[dict] = []
        self._saved_yamls:  set[str]   = set()

    # ── Conversation JSON ─────────────────────────────────────────────────────

    def log(self, role: str, content: str | None = None,
            tool_calls: list | None = None, tool_results: list | None = None) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "role": role,
        }
        if content:      entry["content"]      = content
        if tool_calls:   entry["tool_calls"]   = tool_calls
        if tool_results: entry["tool_results"] = tool_results
        self._entries.append(entry)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)

    # ── Session log (panel output) ────────────────────────────────────────────

    def log_line(self, level: str, text: str) -> None:
        """Append one log-panel line to session.log."""
        ts = datetime.now().strftime("%H:%M:%S")
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {level:<5} {text}\n")

    # ── Figures / images ──────────────────────────────────────────────────────

    def record_figure(self, src_path: str) -> str:
        """
        Copy a matplotlib figure PNG into plots/ and record it in images.json.
        Returns the destination path.
        """
        self._plots_dir.mkdir(exist_ok=True)
        dst = self._plots_dir / Path(src_path).name
        try:
            shutil.copy2(src_path, dst)
        except Exception:
            dst = Path(src_path)  # fallback: keep original path
        self._image_entries.append({
            "type":      "figure",
            "path":      str(dst),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self._flush_images()
        return str(dst)

    def record_image_path(self, path: str, label: str = "") -> None:
        """
        Record a reference to an existing image file (ref images, datasets).
        The file is NOT copied — only the path is stored.
        """
        self._image_entries.append({
            "type":      "reference",
            "path":      str(path),
            "label":     label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self._flush_images()

    def _flush_images(self) -> None:
        with open(self._images_file, "w", encoding="utf-8") as f:
            json.dump(self._image_entries, f, ensure_ascii=False, indent=2)

    # ── Config YAMLs ─────────────────────────────────────────────────────────

    def copy_yaml(self, yaml_filename: str) -> Path | None:
        if yaml_filename in self._saved_yamls:
            return None
        src = _CONFIG_DIR / yaml_filename
        if not src.exists():
            return None
        dst = self.session_dir / yaml_filename
        shutil.copy2(src, dst)
        self._saved_yamls.add(yaml_filename)
        return dst

# ── MCP tool conversion ───────────────────────────────────────────────────────

def _mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    result = []
    for t in mcp_tools:
        schema = t.inputSchema if hasattr(t, "inputSchema") else {}
        result.append({
            "type": "function",
            "function": {
                "name":        t.name,
                "description": t.description or "",
                "parameters":  schema,
            },
        })
    return result

# ── Prompts (imported from prompts.py — edit that file to change behaviour) ──

from src.agent.prompts import SYSTEM_PROMPT, build_image_prompt  # noqa: E402

# ── Phantom tool handlers ─────────────────────────────────────────────────────

def _handle_generate_temp(d: dict, mcp_session) -> tuple[str, str | None]:
    temp_dir = Path(d["base_output_dir"]) / d["run_name"]
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    ui.log("INFO", "Generating TEMP dataset...")
    dataset_path = generate_data_for_training(
        config_yaml_path = d["config_path"],
        base_output_dir  = d["base_output_dir"],
        run_name         = d["run_name"],
        show_preview     = True,
        n_images         = 10,
    )
    return json.dumps({
        "status":  "ok",
        "message": f"Temp dataset generated at '{dataset_path}'.",
        "data":    {"dataset_path": str(dataset_path)},
    }), None


def _handle_generate_final(d: dict) -> tuple[str, str | None]:
    ui.log("INFO", "Generating FINAL dataset...")
    dataset_path = generate_data_for_training(
        config_yaml_path = d["config_path"],
        base_output_dir  = d["base_output_dir"],
        run_name         = d["run_name"],
        show_preview     = True,
        n_images         = None,
    )
    return json.dumps({
        "status":  "ok",
        "message": f"Dataset generated at '{dataset_path}'.",
        "data":    {"dataset_path": str(dataset_path)},
    }), str(dataset_path)


def _handle_preview_image(d: dict) -> str:
    """Render ONE synthetic image from the temp config and show it in the panel."""
    from src.simuscan.syntheticAFM_COCO import generate_preview_image
    import numpy as np
    seed = d.get("seed", 0)
    ui.log("INFO", f"Generating quick preview — 1 image, full resolution, seed={seed}…")
    try:
        img = generate_preview_image(config_yaml_path=d["config_path"], seed=seed)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Same min-max grayscale normalisation the dataset uses when saving PNGs,
        # so the preview matches what will actually be generated.
        arr = np.asarray(img, dtype=np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        norm = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

        # Monotonic counter so each render is visibly distinct in the title even
        # with a fixed seed (otherwise consecutive previews look identical).
        _handle_preview_image.n = getattr(_handle_preview_image, "n", 0) + 1
        n = _handle_preview_image.n

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(norm, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Synthetic preview #{n} — {arr.shape[1]}×{arr.shape[0]} px  "
                     f"(seed={seed}, all enabled effects forced)",
                     fontsize=9, color="#c9d1d9")
        ax.axis("off")
        fig.patch.set_facecolor("#161b22")
        fig.tight_layout()
        ui.plot_preview(fig)   # append as a new image so previews can be compared

        ui.log("OK", "Preview updated in the Visualization panel.")
        return json.dumps({"status": "ok", "message": "Preview image generated."})
    except Exception as exc:
        ui.log("ERR", f"Preview failed: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_convert(coco_folder: str) -> str:
    ui.log("INFO", "Converting COCO → YOLO (DET + SEG)...")
    try:
        det_folder, seg_folder = convert_coco_to_yolo_dual(coco_folder)
        ui.log("OK", f"YOLO DET → {det_folder}")
        ui.log("OK", f"YOLO SEG → {seg_folder}")
        return json.dumps({
            "status":  "ok",
            "message": "COCO→YOLO conversion completed.",
            "data": {"yolo_det_folder": det_folder, "yolo_seg_folder": seg_folder},
        })
    except Exception as exc:
        ui.log("ERR", f"Conversion failed: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_train(d: dict) -> str:
    gpu = check_gpu()
    ui.log("INFO", gpu.get("message", ""))
    ui.log("INFO",
        f"Training YOLO — task={d['task']}  model={d['model_name']}  "
        f"epochs={d['epochs']}  imgsz={d['imgsz']}  batch={d['batch']}")
    try:
        result = train_yolo_model(
            coco_folder = d["coco_folder"],
            task        = d["task"],
            model_name  = d["model_name"],
            epochs      = d["epochs"],
            imgsz       = d["imgsz"],
            batch       = d["batch"],
        )
        if result.get("status") == "ok":
            rd = result["data"]
            ui.log("OK", f"Training done. Run dir: {rd.get('run_dir')}")
            ui.log("OK", f"best.pt: {rd.get('model_path')}")
        else:
            ui.log("ERR", f"Training failed: {result.get('message')}")
        return json.dumps(result)
    except Exception as exc:
        ui.log("ERR", f"Training error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_train_unet(d: dict) -> str:
    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    ui.log("INFO",
        f"Training UNet — epochs={d['epochs']}  batch={d['batch_size']}  "
        f"lr={d['lr']}  imgsz={d['image_size']}  device={device}")
    try:
        result = train_unet_model(
            coco_folder = d["coco_folder"],
            epochs      = d["epochs"],
            batch_size  = d["batch_size"],
            lr          = d["lr"],
            image_size  = d["image_size"],
        )
        if result.get("status") == "ok":
            rd = result["data"]
            ui.log("OK", f"UNet training done. Run dir: {rd.get('run_dir')}")
            ui.log("OK", f"best_val_loss: {rd.get('best_val_loss')}")
        else:
            ui.log("ERR", f"UNet training failed: {result.get('message')}")
        return json.dumps(result)
    except Exception as exc:
        ui.log("ERR", f"UNet training error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_predict_unet(d: dict) -> str:
    ui.log("INFO", f"Running UNet prediction on '{d['image_folder']}'...")
    try:
        result = predict_folder_unet(
            image_folder = d["image_folder"],
            weights_path = d["weights_path"],
            out_channels = d["out_channels"],
            class_names  = d.get("class_names"),
            image_size   = d.get("image_size", 512),
            alpha        = d.get("alpha", 0.5),
            save_masks   = d.get("save_masks", False),
        )
        if result.get("status") == "ok":
            rd = result["data"]
            ui.log("OK", f"Prediction done: {rd.get('n_detections')} detections "
                         f"→ {rd.get('out_dir')}")
        else:
            ui.log("ERR", f"Prediction failed: {result.get('message')}")
        return json.dumps(result)
    except Exception as exc:
        ui.log("ERR", f"UNet prediction error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_visualize_dataset(d: dict) -> str:
    ui.log("INFO", f"Visualizing dataset: {d['dataset_name']}")
    ui.log("INFO", f"Splits: {d['splits']}  N/split: {d['n_per_split']}")
    result = visualize_yolo_dataset(
        yolo_folder  = d["yolo_folder"],
        splits       = d["splits"],
        n_per_split  = d["n_per_split"],
        show_boxes   = d["show_boxes"],
        show_masks   = d["show_masks"],
        show_labels  = d["show_labels"],
    )
    if result.get("status") == "ok":
        rd = result["data"]
        ui.log("OK", f"Displayed {rd['total_shown']} image(s): {rd['shown_per_split']}")
    else:
        ui.log("ERR", f"Visualization failed: {result.get('message')}")
    return json.dumps(result)


def _handle_predict_folder_gwy(d: dict) -> str:
    ui.log("INFO", f"GWY analysis: {d['n_gwy_files']} file(s) in '{d['gwy_folder']}'")
    ui.log("INFO", f"Model: {Path(d['model_path']).name}  "
                   f"scan_direction: {d['scan_direction']}  flatten_perc: {d['flatten_perc']}")
    result = predict_gwy_folder(
        gwy_folder     = d["gwy_folder"],
        model_path     = d["model_path"],
        scan_direction = d["scan_direction"],
        flatten_perc   = d["flatten_perc"],
        show_boxes     = d["show_boxes"],
        show_masks     = d["show_masks"],
        show_labels    = d["show_labels"],
        conf_threshold = d["conf_threshold"],
    )
    if result.get("status") == "ok":
        rd = result["data"]
        ui.log("OK", f"GWY analysis done: {rd['n_processed']}/{rd['n_gwy_files']} files, "
                     f"{rd['total_detections']} detection(s)")
        ui.log("OK", f"Output → {rd['output_dir']}")
    else:
        ui.log("ERR", f"GWY analysis failed: {result.get('message')}")
    return json.dumps(result)


def _handle_predict_folder(d: dict) -> str:
    info = get_model_info(d["model_path"])
    if info.get("status") == "ok":
        mi = info["data"]
        ui.log("INFO", f"Model: {mi['model_path']}")
        ui.log("INFO", f"Task: {mi['task']}  Classes: {mi['class_names']}")
    ui.log("INFO", f"Running inference on {d['n_images']} image(s) in {d['image_folder']}")
    result = predict_folder(
        image_folder   = d["image_folder"],
        model_path     = d["model_path"],
        show_boxes     = d["show_boxes"],
        show_masks     = d["show_masks"],
        show_labels    = d["show_labels"],
        conf_threshold = d["conf_threshold"],
    )
    if result.get("status") == "ok":
        rd = result["data"]
        ui.log("OK", f"Inference complete: {rd['n_images_processed']} images, "
                     f"{rd['total_detections']} detections")
        ui.log("OK", f"Output: {rd['output_dir']}")
    else:
        ui.log("ERR", f"Prediction failed: {result.get('message')}")
    return json.dumps(result)


def _handle_train_triplet(data: dict) -> str:
    ui.log("INFO", f"Triplet training — YOLO_SEG: {data['yolo_seg_folder']}")
    ui.log("INFO", f"Epochs: {data['triplet_epochs']}  Latent: {data['latent_dim']}  "
                   f"max_iou: {data['max_iou']}  min_sep: {data.get('min_separation_px', 10)}px")
    result = run_triplet_training(
        yolo_seg_folder   = data["yolo_seg_folder"],
        yolo_model_path   = data["yolo_model_path"],
        max_iou           = data.get("max_iou", 0.10),
        min_separation_px = data.get("min_separation_px", 10),
        border_margin_pct = data.get("border_margin_pct", 0.05),
        crop_size         = data.get("crop_size", 64),
        otsu_min_frac     = data.get("otsu_min_frac", 0.10),
        latent_dim        = data.get("latent_dim", 32),
        triplet_epochs    = data.get("triplet_epochs", 40),
        triplet_batch     = data.get("triplet_batch", 64),
        triplet_lr        = data.get("triplet_lr", 1e-3),
        triplet_margin    = data.get("triplet_margin", 0.5),
        knn_k             = data.get("knn_k", 5),
    )
    if result.get("status") == "ok":
        d = result["data"]
        ui.log("OK", f"Triplet done. Run dir: {d['run_dir']}")
        ui.log("OK", f"Crops: {d['n_crops']}  Classes: {d['n_classes']}")
        ui.log("OK", f"Val acc: {d['final_val_acc']*100:.1f}%  "
                     f"Silhouette: {d['silhouette_global']:.4f}  "
                     f"kNN: {d['knn_accuracy']*100:.1f}%")
    else:
        ui.log("ERR", f"Triplet training failed: {result.get('message')}")
    return json.dumps(result)


def _handle_build_triplet_library(data: dict) -> str:
    ui.log("INFO", f"Building triplet library — folder: {data['image_folder']}")
    ui.log("INFO", f"Images: {data['n_images']}  yolo_conf: {data.get('yolo_conf', 0.5)}"
                   f"  pixel_size_nm: {data.get('pixel_size_nm', 0.0)}")
    result = run_triplet_library(
        image_folder      = data["image_folder"],
        triplet_run_name  = data.get("triplet_run_name"),
        yolo_run_name     = data.get("yolo_run_name"),
        pixel_size_nm     = data.get("pixel_size_nm", 1.0),
        n_clusters        = data.get("n_clusters"),
        yolo_conf         = data.get("yolo_conf", 0.50),
        border_margin_pct = data.get("border_margin_pct", 0.0),
        min_separation_px = data.get("min_separation_px", 0),
    )
    if result.get("status") == "ok":
        d = result["data"]
        ui.log("OK", f"Library built: {d['n_objects']} objects, "
                     f"{d['n_clusters']} clusters → {Path(d['run_dir']).name}")
    else:
        ui.log("ERR", f"Library build failed: {result.get('message')}")
    return json.dumps(result)


def _handle_show_crop_grid(data: dict) -> str:
    ui.log("INFO", f"Showing crop grid from {data['run_dir']}")
    result = show_crop_grid(
        run_dir   = data["run_dir"],
        class_idx = data.get("class_idx"),
        max_items = data.get("max_items", 64),
        save_path = None,
    )
    return json.dumps(result)


def _handle_show_embedding_umap(data: dict) -> str:
    ui.log("INFO", f"Showing UMAP from {data['run_dir']}")
    result = show_embedding_umap(
        run_dir   = data["run_dir"],
        save_path = None,
    )
    return json.dumps(result)


def _handle_show_triplet_library_explorer(data: dict) -> str:
    from src.trainer.triplet_explorer import _resolve_explorer_dir, _err, _ok
    run_name = data.get("library_run_name")
    lib_dir = _resolve_explorer_dir(run_name)
    if lib_dir is None or not lib_dir.is_dir():
        result = _err(f"Library not found: '{run_name or 'latest'}'")
        return json.dumps(result)
    ui.log("INFO", f"Opening UMAP explorer — {lib_dir.name}")
    ui.open_explorer_dialog(str(lib_dir))  # blocks until window closed
    result = _ok(message=f"Explorer closed — {lib_dir.name}")
    return json.dumps(result)


def _handle_connect_microscope() -> str:
    ui.log("INFO", "Connecting to Nanosurf Studio...")
    try:
        result = _connect_microscope()
        if result.get("status") == "ok":
            ui.log("OK", "Connected to Nanosurf Studio.")
        else:
            ui.log("ERR", f"Connection failed: {result.get('message')}")
        return json.dumps(result)
    except Exception as exc:
        ui.log("ERR", f"Connection error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_get_status() -> str:
    try:
        result = _get_exploration_status()
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_stop_exploration() -> str:
    ui.log("INFO", "Sending stop signal...")
    try:
        result = _stop_exploration()
        if result.get("status") == "ok":
            ui.log("OK", "Stop signal sent. Session will finish after current frame.")
        else:
            ui.log("WARN", f"Stop failed: {result.get('message')}")
        return json.dumps(result)
    except Exception as exc:
        ui.log("ERR", f"Stop error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_run_exploration(d: dict) -> str:
    ed = d.get("exploration_dict", {})
    ui.log("INFO", f"Starting AFM exploration — strategy: {ed.get('strategy','?')}")
    ui.log("INFO", f"Area: {ed.get('area_to_cover_um',0):.0f}µm²  "
                   f"Frame: {ed.get('scan_size_um',0):.1f}µm")
    _tc = ed.get('target_classes', {}) or {}
    _cls_txt = ", ".join(f"{k}:{v.get('n','?')}" for k, v in _tc.items()) if _tc else "(all)"
    ui.log("INFO", f"Targets: {_cls_txt}")
    try:
        result = _run_exploration(ed)
        if result.get("status") == "ok":
            ui.log("OK", "Exploration thread launched.")
        else:
            ui.log("ERR", f"Could not start: {result.get('message')}")
        return json.dumps(result)
    except Exception as exc:
        ui.log("ERR", f"Exploration error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def _handle_show_ref_images(d: dict) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    pairs = d.get("pairs", [])
    if not pairs:
        return json.dumps({"status": "error", "message": "No image pairs to display."})

    for pair in pairs:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            try:
                axes[0].imshow(mpimg.imread(pair["image_path"]))
            except Exception:
                axes[0].text(0.5, 0.5, "Cannot load", ha="center", va="center")
            axes[0].set_title(pair["image_file"], fontsize=8)
            axes[0].axis("off")

            if pair["mask_exists"]:
                try:
                    axes[1].imshow(mpimg.imread(pair["mask_path"]), cmap="gray")
                except Exception:
                    axes[1].text(0.5, 0.5, "Cannot load", ha="center", va="center")
            else:
                axes[1].text(0.5, 0.5, "No mask", ha="center", va="center",
                             color="red", fontsize=10)
            axes[1].set_title(pair["mask_file"], fontsize=8)
            axes[1].axis("off")

            plt.tight_layout()
            ui.plot(fig)
            plt.close(fig)
        except Exception as e:
            ui.log("WARN", f"Cannot display {pair['image_file']}: {e}")

    n = len(pairs)
    shown = [p["image_file"] for p in pairs]
    ui.log("OK", f"Displayed {n} reference image pair(s) — use prev/next to browse.")
    return json.dumps({
        "status":  "ok",
        "message": f"Displayed {n} reference image pair(s) (use ← → to browse).",
        "data":    {"shown": shown},
    })


def _handle_extract_pdf_images(d: dict) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    images = d.get("images", [])
    if not images:
        return json.dumps({"status": "ok", "message": "No images extracted from PDF."})

    pdf_name = d.get("pdf", "PDF")
    for info in images:
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            img_arr = mpimg.imread(info["path"])
            ax.imshow(img_arr)
            ax.set_title(
                f"[{info['index']}]  {pdf_name}  —  page {info['page']}  —  "
                f"{info['width']}×{info['height']} px",
                fontsize=8,
            )
            ax.axis("off")
            plt.tight_layout()
            ui.plot(fig)
            plt.close(fig)
        except Exception as e:
            ui.log("WARN", f"Cannot display image {info['index']}: {e}")

    n = len(images)
    ui.log("OK", f"Displayed {n} PDF image(s) — use prev/next to browse. Ask the user which index is an AFM image.")
    return json.dumps({
        "status":  "ok",
        "message": f"Displayed {n} image(s) one by one (use ← → to browse). Ask the user which index to use as AFM reference.",
        "data":    {"count": n, "images": images},
    })


def _handle_run_generator(d: dict) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    folder  = d.get("image_folder", "")
    files   = d.get("image_files",  [])
    count   = d.get("count",        0)

    repo_root = Path(__file__).resolve().parents[2]
    ref_dir   = repo_root / "ref_objects" / folder

    for fname in files:
        img_path = ref_dir / fname
        try:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(mpimg.imread(str(img_path)), cmap="gray")
            ax.set_title(fname, fontsize=8)
            ax.axis("off")
            plt.tight_layout()
            ui.plot(fig)
            plt.close(fig)
        except Exception as e:
            ui.log("WARN", f"Cannot display {fname}: {e}")

    ui.log("OK", f"Generated {count} image(s) → ref_objects/{folder}/")
    return json.dumps({
        "status":  "ok",
        "message": f"{count} image(s) generated and saved to ref_objects/{folder}/.",
        "data":    d,
    })


def _zero_background_and_save(img_path: Path, mask_path: Path) -> None:
    """
    Apply Otsu threshold to separate the object (bright) from the background
    (dark), zero all background pixels in the image, and save both the cleaned
    image and the binary mask.

    This ensures the PNG stored in ref_objects/image_files/ has a truly black
    background rather than the near-black gray produced by diffusion models.
    """
    import cv2 as _cv2
    import numpy as _np

    img = _cv2.imread(str(img_path), _cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")

    # Otsu threshold: bright object on dark background → THRESH_BINARY
    _, mask = _cv2.threshold(img, 0, 255, _cv2.THRESH_BINARY | _cv2.THRESH_OTSU)

    # Clean up mask edges
    k3 = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3))
    mask = _cv2.dilate(mask, k3, iterations=1)
    k5 = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5, 5))
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_CLOSE, k5, iterations=2)
    mask = _cv2.morphologyEx(mask, _cv2.MORPH_OPEN,  k5, iterations=1)

    # Zero background pixels in the image
    img_clean = img.copy()
    img_clean[mask == 0] = 0

    # Save cleaned image (overwrite raw download) and mask
    _cv2.imwrite(str(img_path),  img_clean)
    _cv2.imwrite(str(mask_path), mask)


def _apply_distance_height_profile(img_path: Path, mask_path=None, *,
                                   order: float = 2.0, dh0: float = 0.0,
                                   texture: float = 0.4,
                                   output_path=None, save_mask: bool = True) -> str:
    """
    Reshape a grayscale image into a physically-plausible AFM height map.

    The object height is rebuilt as a dome envelope that grows with distance
    from the object edge (distance transform), modulated by the original image
    texture so diffusion-model detail is preserved:

        height = dome * ((1 - texture) + texture * normalized_intensity)

    texture = 0 → pure dome (no shading kept); 1 → texture modulates fully.

    mask_path   : reused if the file exists; otherwise an Otsu mask is computed
                  from the image and (if save_mask) written to mask_path.
    output_path : destination PNG. None → overwrite img_path.
    Returns the path written.
    """
    import cv2 as _cv2
    import numpy as _np
    from src.simuscan.mask_tools import mask_to_height_map

    img = _cv2.imread(str(img_path), _cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    img = img.astype(_np.float32)

    # ── Mask: reuse if provided/exists, else Otsu ────────────────────────────
    binmask = None
    if mask_path and Path(mask_path).is_file():
        m = _cv2.imread(str(mask_path), _cv2.IMREAD_GRAYSCALE)
        if m is not None:
            binmask = (m > 0).astype(_np.uint8)
    if binmask is None:
        _, m = _cv2.threshold(img.astype(_np.uint8), 0, 255,
                              _cv2.THRESH_BINARY | _cv2.THRESH_OTSU)
        k5 = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5, 5))
        m  = _cv2.morphologyEx(m, _cv2.MORPH_CLOSE, k5, iterations=2)
        m  = _cv2.morphologyEx(m, _cv2.MORPH_OPEN,  k5, iterations=1)
        binmask = (m > 0).astype(_np.uint8)
        if save_mask and mask_path:
            _cv2.imwrite(str(mask_path), binmask * 255)

    if int(binmask.sum()) == 0:
        raise ValueError("Empty mask — cannot apply height profile.")

    # ── Dome envelope (grows from edge to center) ────────────────────────────
    dome = mask_to_height_map(binmask, height_max=1.0, order=order, dh0=dh0)

    # ── Model texture normalized to [0,1] inside the mask ────────────────────
    tex = _np.zeros_like(img, dtype=_np.float32)
    obj = img[binmask > 0]
    if obj.size and float(obj.max()) > float(obj.min()):
        tex[binmask > 0] = (img[binmask > 0] - obj.min()) / (obj.max() - obj.min())

    # ── Blend: dome dominates the profile, texture modulates ─────────────────
    w = float(_np.clip(texture, 0.0, 1.0))
    height = dome * ((1.0 - w) + w * tex)
    height[binmask == 0] = 0.0

    hmax = float(height.max())
    if hmax > 0:
        out = (_np.clip(height / hmax, 0, 1) * 255).astype(_np.uint8)
    else:
        out = height.astype(_np.uint8)

    dst = Path(output_path) if output_path else Path(img_path)
    _cv2.imwrite(str(dst), out)
    return str(dst)


def _handle_apply_height_profile(d: dict) -> str:
    """Apply the distance-based AFM height profile to an existing image."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img_path    = d["image_path"]
    mask_path   = d.get("mask_path")
    output_path = d["output_path"]
    ui.log("INFO", f"Applying height profile → {Path(output_path).name}")
    try:
        out = _apply_distance_height_profile(
            Path(img_path), mask_path,
            order=d.get("order", 2.0),
            dh0=d.get("dh0", 0.0),
            texture=d.get("texture", 0.4),
            output_path=output_path,
            save_mask=d.get("save_mask", True),
        )
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Height profile failed: {exc}"})

    # Show before/after in the plot panel
    try:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(mpimg.imread(str(img_path)), cmap="afmhot")
        axes[0].set_title("Original", fontsize=9); axes[0].axis("off")
        axes[1].imshow(mpimg.imread(out), cmap="afmhot")
        axes[1].set_title("Height profile", fontsize=9); axes[1].axis("off")
        plt.tight_layout()
        ui.plot(fig)
        plt.close(fig)
    except Exception:
        pass

    return json.dumps({
        "status": "ok",
        "message": f"Height profile applied. Saved to '{out}'.",
        "data": {"output_path": out},
    })


def _handle_generate_ref_image(d: dict, api_key: str) -> str:
    """
    Calls OpenAI image generation API, saves PNGs inside a per-class subfolder
    under ref_objects/image_files/<superclass_class>/, and sends them to the
    plot panel.

    Folder/file naming convention:
        ref_objects/image_files/<superclass>_<class>/
            <superclass>_<class>_NNN.png
            mask_<superclass>_<class>_NNN.png

    If the folder already exists, new images are appended and the user is
    informed how many files were already there.
    """
    import base64
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    class_name      = d["class_name"]
    superclass_name  = d["superclass_name"]
    prompt           = d["prompt"]
    n_images         = d.get("n_images", 1)
    reference_image  = d.get("reference_image")   # full path str or None

    # Distance-based AFM height profile (applied by default after generation)
    apply_hp = d.get("apply_height_profile", True)
    hp_order = d.get("height_profile_order", 2.0)
    hp_dh0   = d.get("height_profile_dh0", 0.0)
    hp_tex   = d.get("height_profile_texture", 0.4)

    folder_name = f"{superclass_name}_{class_name}"
    img_dir     = _ROOT / "ref_objects" / folder_name
    already_existed = img_dir.exists()
    img_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images to determine next index
    existing = sorted(img_dir.glob(f"{folder_name}_*.png"))
    existing = [p for p in existing if not p.name.startswith("mask_")]
    next_idx = len(existing)

    if already_existed and next_idx > 0:
        ui.log("INFO",
               f"Folder '{folder_name}' already has {next_idx} image(s). "
               f"Adding {n_images} more (indices {next_idx}–{next_idx + n_images - 1}).")

    if reference_image:
        ui.log("INFO", f"Using reference image: {Path(reference_image).name} (images.edit mode)")

    openai_client = OpenAI(api_key=api_key)
    saved_pairs: list[dict] = []

    # Diffusion image model from settings (.env), default gpt-image-1
    try:
        from src.ui.dialogs.settings_dialog import load_settings
        img_model = load_settings().get("OPENAI_IMAGE_MODEL", "gpt-image-1")
    except Exception:
        img_model = "gpt-image-1"
    ui.log("INFO", f"Image model: {img_model}")

    for i in range(n_images):
        idx       = next_idx + i
        img_name  = f"{folder_name}_{idx:03d}.png"
        mask_name = f"mask_{folder_name}_{idx:03d}.png"
        img_path  = img_dir / img_name
        mask_path = img_dir / mask_name

        ui.log("INFO", f"Generating image {i + 1}/{n_images}: {img_name} ...")
        try:
            if reference_image:
                # images.edit — pass the crop directly as visual reference.
                # The model sees the full image and generates a similar clean
                # AFM version guided by the prompt.
                with open(reference_image, "rb") as ref_f:
                    response = openai_client.images.edit(
                        model=img_model,
                        image=ref_f,
                        prompt=build_image_prompt(prompt),
                        n=1,
                        size="1024x1024",
                    )
            else:
                # images.generate — text only
                response = openai_client.images.generate(
                    model=img_model,
                    prompt=build_image_prompt(prompt),
                    n=1,
                    size="1024x1024",
                    output_format="png",
                )

            img_b64 = response.data[0].b64_json
            if img_b64:
                img_bytes = base64.b64decode(img_b64)
            else:
                import urllib.request
                img_bytes = urllib.request.urlopen(response.data[0].url).read()

            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # Zero the background and save cleaned image + mask
            _zero_background_and_save(img_path, mask_path)

            # Reshape into a distance-based AFM height map (dome grows from edge)
            if apply_hp:
                _apply_distance_height_profile(
                    img_path, mask_path,
                    order=hp_order, dh0=hp_dh0, texture=hp_tex,
                    save_mask=False,
                )
                ui.log("OK", f"Saved {img_name} (height profile applied) and {mask_name}")
            else:
                ui.log("OK", f"Saved {img_name} (background zeroed) and {mask_name}")

            saved_pairs.append({"image_file": img_name, "mask_file": mask_name})
            if ui._session_logger:
                ui._session_logger.record_image_path(str(img_path),  label=img_name)
                ui._session_logger.record_image_path(str(mask_path), label=mask_name)

        except Exception as exc:
            ui.log("ERR", f"Image generation failed for {img_name}: {exc}")
            return json.dumps({"status": "error", "message": str(exc)})

    # ── Visualize in plot panel ───────────────────────────────────────────────
    n = len(saved_pairs)
    if n > 0:
        fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
        if n == 1:
            axes = [axes]
        for row, pair in enumerate(saved_pairs):
            img_arr  = mpimg.imread(str(img_dir / pair["image_file"]))
            mask_arr = mpimg.imread(str(img_dir / pair["mask_file"]))
            axes[row][0].imshow(img_arr, cmap="gray")
            axes[row][0].set_title(pair["image_file"], fontsize=8)
            axes[row][0].axis("off")
            axes[row][1].imshow(mask_arr, cmap="gray")
            axes[row][1].set_title(pair["mask_file"], fontsize=8)
            axes[row][1].axis("off")
        fig.suptitle(f"Reference images: {folder_name}", fontsize=10)
        plt.tight_layout()
        ui.plot(fig)
        plt.close(fig)

    # Collect all image_files now in the folder for the return value
    all_imgs = sorted(img_dir.glob(f"{folder_name}_*.png"))
    all_imgs = [p.name for p in all_imgs if not p.name.startswith("mask_")]

    return json.dumps({
        "status":  "ok",
        "message": (
            f"Generated {n} image(s) for '{folder_name}'. "
            f"Folder now contains {len(all_imgs)} image(s) total. "
            f"Use add_image_object(label='{class_name}', "
            f"image_folder='{folder_name}', "
            f"image_files={all_imgs}) to add to config."
        ),
        "data": {
            "folder_name":  folder_name,
            "image_folder": folder_name,
            "image_files":  all_imgs,
            "new_files":    [p["image_file"] for p in saved_pairs],
            "directory":    str(img_dir),
        },
    })


# ── Main agent loop ───────────────────────────────────────────────────────────

async def run_gui(api_key: str, model: str, input_queue: Queue) -> None:
    """
    Async agent loop for GUI use.
    Reads user input from input_queue.
    All output goes via ui.log() / ui.chat().
    """
    client     = OpenAI(api_key=api_key)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger     = ChatLogger(session_id)
    ui.set_session_logger(logger)          # wire log/plot persistence
    logger.log("system", content=f"[GUI SESSION STARTED] model={model}")
    ui.log("SYS", f"Session started: {session_id}")

    server_params = StdioServerParameters(
        command = sys.executable,
        args    = ["-m", "src.mcp_server"],
        env     = _ENV,
        cwd     = str(_ROOT),
    )

    import asyncio
    loop = asyncio.get_event_loop()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()
            ui.log("OK", "MCP server connected.")

            tools_resp  = await mcp.list_tools()
            openai_tools = _mcp_tools_to_openai(tools_resp.tools)
            ui.log("INFO", f"{len(openai_tools)} tools loaded.")

            # Auto-index docs/ at session start so the knowledge base always
            # reflects the latest docs. Only NEW/CHANGED files are (re)indexed
            # (unchanged ones are skipped), so this is cheap after the first run.
            try:
                ui.log("INFO", "Indexing documentation (docs/)...")
                idx_res  = await mcp.call_tool("index_documents", {})
                idx_text = "".join(b.text for b in idx_res.content if hasattr(b, "text"))
                idx_data = json.loads(idx_text)
                if idx_data.get("status") == "ok":
                    ui.log("OK", f"Docs indexed: {idx_data.get('message', 'done')}")
                else:
                    ui.log("WARN", f"Doc indexing skipped: {idx_data.get('message')}")
            except Exception as e:
                ui.log("WARN", f"Doc indexing failed at startup: {e}")

            messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
            saved_yamls: list[str] = []

            while True:
                user_input = await loop.run_in_executor(None, input_queue.get)
                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.lower().split()[0] in _EXIT_WORDS:
                    ui.log("SYS", "Session ended by user.")
                    ui.chat("Session ended. Goodbye!")
                    break

                logger.log("user", content=user_input)
                messages.append({"role": "user", "content": user_input})

                # LLM loop
                while True:
                    response = client.chat.completions.create(
                        model       = model,
                        messages    = messages,
                        tools       = openai_tools if openai_tools else None,
                        tool_choice = "auto",
                    )
                    msg = response.choices[0].message
                    messages.append(msg.model_dump(exclude_unset=True))

                    if not msg.tool_calls:
                        reply = msg.content or ""
                        ui.chat(reply)
                        logger.log("assistant", content=reply)
                        break

                    tool_call_log        = []
                    tool_result_messages = []

                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        try:
                            fn_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            fn_args = {}

                        ui.log("TOOL", f"[{fn_name}] {json.dumps(fn_args, ensure_ascii=False)}")

                        # Call MCP server
                        mcp_result  = await mcp.call_tool(fn_name, fn_args)
                        result_text = "".join(
                            b.text for b in mcp_result.content if hasattr(b, "text")
                        )

                        # ── Intercepts ─────────────────────────────────────────
                        if fn_name == "save_config" and fn_args.get("filename"):
                            fname = fn_args["filename"]
                            if fname not in saved_yamls:
                                saved_yamls.append(fname)
                            logger.copy_yaml(fname)

                        elif fn_name == "tool_generate_temp_dataset":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    d = rd["data"]
                                    save_res = await mcp.call_tool(
                                        "save_temp_config", {}
                                    )
                                    save_text = "".join(
                                        b.text for b in save_res.content if hasattr(b, "text")
                                    )
                                    if json.loads(save_text).get("status") == "ok":
                                        result_text, _ = _handle_generate_temp(d, mcp)
                            except Exception as e:
                                ui.log("WARN", f"temp dataset intercept: {e}")

                        elif fn_name == "tool_preview_image":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    d = rd["data"]
                                    save_res = await mcp.call_tool("save_temp_config", {})
                                    save_text = "".join(
                                        b.text for b in save_res.content if hasattr(b, "text")
                                    )
                                    if json.loads(save_text).get("status") == "ok":
                                        result_text = _handle_preview_image(d)
                            except Exception as e:
                                ui.log("WARN", f"preview intercept: {e}")

                        elif fn_name == "tool_generate_final_dataset":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text, coco_folder = _handle_generate_final(rd["data"])
                                    if coco_folder:
                                        conv = _handle_convert(coco_folder)
                                        try:
                                            conv_data = json.loads(conv)
                                            main_data = json.loads(result_text)
                                            if conv_data.get("status") == "ok":
                                                main_data["data"]["yolo_conversion"] = conv_data["data"]
                                                main_data["message"] += " YOLO datasets created."
                                            result_text = json.dumps(main_data)
                                        except Exception:
                                            pass
                            except Exception as e:
                                ui.log("WARN", f"final dataset intercept: {e}")

                        elif fn_name == "tool_convert_coco_to_yolo":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_convert(rd["data"]["coco_folder"])
                            except Exception as e:
                                ui.log("WARN", f"convert intercept: {e}")

                        elif fn_name == "tool_train_yolo":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_train(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"train intercept: {e}")

                        elif fn_name == "tool_visualize_dataset":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_visualize_dataset(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"visualize intercept: {e}")

                        elif fn_name == "tool_predict_folder":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_predict_folder(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"predict intercept: {e}")

                        elif fn_name == "tool_train_unet":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_train_unet(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"train_unet intercept: {e}")

                        elif fn_name == "tool_predict_folder_gwy":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_predict_folder_gwy(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"predict_gwy intercept: {e}")

                        elif fn_name == "tool_predict_folder_unet":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_predict_unet(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"predict_unet intercept: {e}")

                        elif fn_name == "tool_train_triplet":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_train_triplet(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"triplet intercept: {e}")

                        elif fn_name == "tool_build_triplet_library":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_build_triplet_library(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"triplet library intercept: {e}")

                        elif fn_name == "tool_show_crop_grid":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_show_crop_grid(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"crop grid intercept: {e}")

                        elif fn_name == "tool_show_embedding_umap":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_show_embedding_umap(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"umap intercept: {e}")

                        elif fn_name == "tool_show_triplet_library_explorer":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("phantom"):
                                    result_text = _handle_show_triplet_library_explorer(rd)
                            except Exception as e:
                                ui.log("WARN", f"library explorer intercept: {e}")

                        elif fn_name == "automation_connect_microscope":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_connect_microscope()
                            except Exception as e:
                                ui.log("WARN", f"connect intercept: {e}")

                        elif fn_name == "automation_get_status":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_get_status()
                            except Exception as e:
                                ui.log("WARN", f"status intercept: {e}")

                        elif fn_name == "automation_stop":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_stop_exploration()
                            except Exception as e:
                                ui.log("WARN", f"stop intercept: {e}")

                        elif fn_name == "tool_run_exploration":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_run_exploration(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"exploration intercept: {e}")

                        elif fn_name == "tool_generate_ref_image":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_generate_ref_image(rd["data"], api_key)
                            except Exception as e:
                                ui.log("WARN", f"ref image intercept: {e}")

                        elif fn_name == "tool_apply_height_profile":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_apply_height_profile(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"apply height profile intercept: {e}")

                        elif fn_name == "tool_show_ref_images":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_show_ref_images(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"show ref images intercept: {e}")

                        elif fn_name == "extract_pdf_images":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_extract_pdf_images(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"extract pdf images intercept: {e}")

                        elif fn_name in ("open_crop_tool", "open_crop_from_files"):
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    d = rd["data"]
                                    ui.open_crop_dialog(
                                        json.dumps(d),
                                        "", "", d["prefix"],
                                    )
                            except Exception as e:
                                ui.log("WARN", f"crop tool intercept: {e}")

                        elif fn_name == "run_generator":
                            try:
                                rd = json.loads(result_text)
                                if rd.get("status") == "ok":
                                    result_text = _handle_run_generator(rd["data"])
                            except Exception as e:
                                ui.log("WARN", f"run_generator intercept: {e}")

                        elif fn_name == "tool_pid_tuner":
                            try:
                                from src.simuscan_tools.pid_tuner import launch_pid_tuner
                                launch_pid_tuner()
                                result_text = '{"status":"ok","message":"PID Tuner window opened."}'
                            except Exception as e:
                                result_text = f'{{"status":"error","message":"PID Tuner failed: {e}"}}'

                        # Log result
                        try:
                            rd = json.loads(result_text)
                            lvl = "OK" if rd.get("status") == "ok" else "ERR"
                            ui.log(lvl, f"  └─ {rd.get('message', rd.get('status','?'))}")
                        except Exception:
                            pass

                        tool_call_log.append({
                            "tool":   fn_name,
                            "args":   fn_args,
                            "result": result_text,
                        })
                        tool_result_messages.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      result_text,
                        })

                    logger.log("assistant", tool_calls=tool_call_log)
                    logger.log("tool",      tool_results=tool_call_log)
                    messages.extend(tool_result_messages)

    ui.set_session_logger(None)  # detach logger when session ends

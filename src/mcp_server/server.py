"""
src/mcp_server/server.py
──────────────────────────
FastMCP server that exposes SimuScanConfig and YOLO trainer/visualizer as MCP tools.
Run as a standalone process (stdio transport).

Entry point:  python -m src.mcp_server
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Make sure project root is on sys.path when run as __main__ ────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env", override=False)  # populate env if not already set by parent

from fastmcp import FastMCP
from simuscan_tools.simuscan_config import SimuScanConfig
from trainer.yolo_trainer import (
    check_gpu,
    evaluate_yolo_model,
    get_training_results,
    list_datasets,
    list_training_runs,
)
from trainer.yolo_visualizer import (
    find_latest_best_pt,
    get_model_info,
)
from trainer.triplet_trainer import (
        list_triplet_runs,
        get_triplet_results,
    )
from trainer.triplet_inference import (
        run_triplet_library,
        list_triplet_libraries,
        get_triplet_library_results,
    )
from trainer.triplet_visualizer import (
    find_latest_triplet_pt,
    get_triplet_model_info,
)
from trainer.unet_trainer import (
    convert_coco_to_unet,
    evaluate_unet_model,
    train_unet_model,
    get_unet_training_results,
    list_unet_training_runs,
    predict_folder_unet,
)

from simuscan_tools.automation_config import AutomationConfig
from automation.automation import (
        get_exploration_status as _get_exploration_status,
        stop_exploration       as _stop_exploration,
    )

from mcp_server.rag.indexer import (
    index_docs_folder,
    index_single_file,
    list_indexed_documents,
)
from mcp_server.rag.retriever import retrieve as _rag_retrieve

# ── Server instance ───────────────────────────────────────────────────────────
mcp = FastMCP(
    name="simuscan-config-server",
    instructions=(
        "You are a configuration assistant for SimuScan, a synthetic AFM dataset "
        "generator. You manage YAML configuration files, generate datasets, and "
        "orchestrate YOLO training and visualization workflows via the tools provided. "
        "Never modify default.yaml directly — always work on a named copy."
    ),
)

# ── Active config state (one per server process) ──────────────────────────────
_cfg: SimuScanConfig | None = None

# Automation config: pre-loaded from default at startup
try:
    _auto_cfg: AutomationConfig = AutomationConfig()
except Exception:
    _auto_cfg = None

def _require_cfg() -> dict:
    if _cfg is None:
        return {
            "status": "error",
            "message": "No configuration loaded. Call load_config first.",
        }
    return {}

# ── Path helpers ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def _datasets_root() -> Path:
    return _repo_root() / "data" / "Datasets"

def _config_dir() -> Path:
    return _repo_root() / "data" / "config" / "conf_simuscan"

def _out_yolo_root() -> Path:
    return _repo_root() / "data" / "out_yolo"

_TEMP_CONFIG   = "temp.yaml"
_TEMP_RUN_NAME = "temp"

# ── Result helpers ────────────────────────────────────────────────────────────

def _ok(data: Any = None, message: str = "OK") -> dict:
    result: dict = {"status": "ok", "message": message}
    if data is not None:
        result["data"] = data
    return result

def _err(message: str, detail: Any = None) -> dict:
    result: dict = {"status": "error", "message": message}
    if detail is not None:
        result["detail"] = detail
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Session / file management tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def load_config(filename: str | None = None) -> dict:
    """
    Load a SimuScan YAML configuration into the active session.

    Parameters
    ----------
    filename : str | None
        YAML file name (e.g. 'my_config.yaml') located in data/config/conf_simuscan/.
        If None, a fresh in-memory copy of default.yaml is loaded.
    """
    global _cfg
    if filename and Path(filename).name == _TEMP_CONFIG:
        return {"status": "error",
                "message": ("temp.yaml is an internal preview snapshot and cannot "
                            "be loaded. Load a real config (e.g. 'config_1.yaml') "
                            "or pass None to start from default.")}
    try:
        _cfg = SimuScanConfig(filename)
        return _cfg.summary()
    except FileNotFoundError as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def save_config(filename: str) -> dict:
    """
    Save the current configuration to data/config/conf_simuscan/<filename>.
    Never overwrites default.yaml or the internal temp.yaml.

    Parameters
    ----------
    filename : str   Target YAML file name, e.g. 'my_experiment.yaml'.
    """
    err = _require_cfg()
    if err:
        return err
    if Path(filename).name == _TEMP_CONFIG:
        return {"status": "error",
                "message": ("temp.yaml is internal — never save to it. Use a real "
                            "filename (e.g. 'my_experiment.yaml'). Previews are "
                            "handled automatically by tool_generate_temp_dataset.")}
    return _cfg.save(filename)


@mcp.tool()
def save_temp_config() -> dict:
    """
    INTERNAL — snapshot the current in-memory config to temp.yaml for previews.
    Does NOT change the active named file. The agent should not call this
    directly; tool_generate_temp_dataset uses it automatically.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.save_temp()


@mcp.tool()
def list_config_files() -> dict:
    """List all YAML configuration files available in data/config/conf_simuscan/."""
    try:
        tmp = _cfg if _cfg is not None else SimuScanConfig()
        return tmp.list_files()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def list_stl_files() -> dict:
    """List all STL files available in data/STL_files/."""
    try:
        tmp = _cfg if _cfg is not None else SimuScanConfig()
        return tmp.list_stl_files()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def get_summary() -> dict:
    """
    Return a high-level summary of the currently loaded configuration:
    file name, image size, number of images, and which objects are enabled.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.summary()


# ══════════════════════════════════════════════════════════════════════════════
# Read tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_section(section: str) -> dict:
    """
    Return all parameters of a top-level configuration section.

    Parameters
    ----------
    section : str
        One of: config, tip_conf, substrate_conf, artifacts_conf, pid_conf,
        domains_conf, shapes, objects.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.get_section(section)


@mcp.tool()
def get_param(section: str, param: str) -> dict:
    """
    Return the value of a single parameter from an editable flat section.

    Parameters
    ----------
    section : str   One of: config, tip_conf, substrate_conf, artifacts_conf,
                    pid_conf, domains_conf.
    param   : str   Parameter name within that section.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.get_param(section, param)


@mcp.tool()
def list_objects() -> dict:
    """Return all object labels with their shape and enabled/disabled status."""
    err = _require_cfg()
    if err:
        return err
    return _cfg.list_objects()


@mcp.tool()
def get_object(label: str) -> dict:
    """
    Return all parameters of a specific object by its label.

    Parameters
    ----------
    label : str   Object label, e.g. 'hexagon', 'bacillus', 'linearDNA'.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.get_object(label)


@mcp.tool()
def get_object_param(label: str, param: str) -> dict:
    """
    Return the value of a single parameter of an object.

    Parameters
    ----------
    label : str   Object label.
    param : str   Parameter name within the object.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.get_object_param(label, param)


@mcp.tool()
def list_shapes() -> dict:
    """Return the list of valid shape names that can be assigned to objects."""
    try:
        tmp = _cfg if _cfg is not None else SimuScanConfig()
        return tmp.list_shapes()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def describe_section(section: str) -> dict:
    """
    Return parameter names and their current types for a section.

    Parameters
    ----------
    section : str   Any top-level section name.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.describe_section(section)


# ══════════════════════════════════════════════════════════════════════════════
# Write tools — flat sections
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def set_param(section: str, param: str, value: Any) -> dict:
    """
    Edit a single parameter in a flat editable section.
    The new value must match the existing type (bool, int, float, str, list).
    The 'shapes' section is read-only.

    Parameters
    ----------
    section : str   One of: config, tip_conf, substrate_conf, artifacts_conf,
                    pid_conf, domains_conf.
    param   : str   Parameter name.
    value   : Any   New value (type-checked against the current value).
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.set_param(section, param, value)


@mcp.tool()
def set_params(section: str, updates: dict[str, Any]) -> dict:
    """
    Edit multiple parameters in one flat editable section at once.

    Parameters
    ----------
    section : str   One of the four editable flat sections.
    updates : dict  {param_name: new_value, ...}
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.set_params(section, updates)


# ══════════════════════════════════════════════════════════════════════════════
# Write tools — objects section
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def set_object_param(label: str, param: str, value: Any) -> dict:
    """
    Edit a single parameter of an object identified by its label.

    IMPORTANT — use EXACT field names as they appear in get_object() output:
      "enable" (NOT "enabled"), "n_objects", "size_shape", "height",
      "probability", "shape", "image_file", "mask_file".
    Call get_object(label) first if unsure of the field name.

    Parameters
    ----------
    label : str   Object label, e.g. 'hexagon'.
    param : str   Exact parameter name from get_object() output.
    value : Any   New value (must match the existing type).
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.set_object_param(label, param, value)


@mcp.tool()
def set_object_params(label: str, updates: dict[str, Any]) -> dict:
    """
    Edit multiple parameters of an object at once.

    IMPORTANT — use EXACT field names as they appear in get_object() output:
      "enable" (NOT "enabled"), "n_objects", "size_shape", "height",
      "probability", "shape", "image_file", "mask_file".
    Call get_object(label) first if unsure of any field name.

    Parameters
    ----------
    label   : str    Object label.
    updates : dict   {exact_param_name: new_value, ...}
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.set_object_params(label, updates)


@mcp.tool()
def rename_object(old_label: str, new_label: str) -> dict:
    """
    Rename an object's label (the class name used in COCO/YOLO annotations).

    Use THIS to rename a class — do NOT use set_object_param(label,'label',...),
    which corrupts the internal index. Keeps the config consistent and validates
    that new_label is unique.

    Parameters
    ----------
    old_label : str   Current object label, e.g. 'bacillus_new'.
    new_label : str   New shorter/unique label, e.g. 'bac'.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.rename_object(old_label, new_label)


@mcp.tool()
def add_stl_object(
    stl_files:      list[str],
    label:          str | None        = None,
    stl_weights:    list[float] | None = None,
    template_label: str | None        = None,
) -> dict:
    """
    Add a new STL-based object to the objects section of the active configuration.

    Parameters
    ----------
    stl_files      : list[str]         .stl file names in data/STL_files/.
    label          : str | None        Unique label; auto-generated if None.
    stl_weights    : list[float] | None Weights summing to 1.0; equal if None.
    template_label : str | None        Base object to copy params from.
    """
    err = _require_cfg()
    if err:
        return err
    return _cfg.add_stl_object(
        stl_files=stl_files,
        label=label,
        stl_weights=stl_weights,
        template_label=template_label,
    )


@mcp.tool()
def add_image_object(
    class_name:      str,
    superclass_name: str,
    image_folder:    str,
    image_files:     list,
    image_weights:   list       = None,
    n_objects:       int        = 5,
    size_shape:      list[int]  = None,
    height:          list[float]= None,
    template_label:  str | None = None,
) -> dict:
    """
    Add a new image-based object (shape="image") to the active configuration.

    Images must already exist in ref_objects/<image_folder>/.
    Use tool_generate_ref_image first to create them.

    Parameters
    ----------
    class_name      : Unique label for this object class (e.g. "ribosome").
    superclass_name : Category label (e.g. "protein"). Used for naming only.
    image_folder    : Subfolder name inside ref_objects/
                      (format: superclass_class, e.g. "protein_ribosome").
    image_files     : List of PNG filenames inside image_folder, e.g.
                      ["protein_ribosome_000.png", "protein_ribosome_001.png"].
                      Each must have a corresponding mask_<filename> in the same folder.
    image_weights   : Optional sampling probability per file (must sum to 1.0).
                      If None, all files are sampled with equal probability.
    n_objects       : Mean number of objects per synthetic image (default 5).
    size_shape      : [width_px, height_px] of the object patch (default [40, 40]).
    height          : [min, max] height range in normalised units (default [0.8, 1.0]).
    template_label  : Optional existing object label to copy other params from.
    """
    err = _require_cfg()
    if err:
        return err

    if size_shape is None:
        size_shape = [40, 40]
    if height is None:
        height = [0.8, 1.0]

    # Verify folder and files exist
    folder_path = _repo_root() / "ref_objects" / image_folder
    if not folder_path.exists():
        return _err(
            f"Folder '{image_folder}' not found in ref_objects/. "
            "Run tool_generate_ref_image first."
        )
    missing = []
    for fname in image_files:
        if not (folder_path / fname).exists():
            missing.append(fname)
        mask_name = f"mask_{fname}"
        if not (folder_path / mask_name).exists():
            missing.append(mask_name)
    if missing:
        return _err(f"Missing files in '{image_folder}': {missing}")

    return _cfg.add_image_object(
        label          = class_name,
        image_folder   = image_folder,
        image_files    = image_files,
        image_weights  = image_weights,
        n_objects      = n_objects,
        size_shape     = size_shape,
        height         = height,
        template_label = template_label,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dataset generation — phantom tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def tool_generate_temp_dataset() -> dict:
    """
    Phantom tool — generate a preview synthetic AFM dataset from the current
    temp config (~10 images). No YOLO conversion is run for temp datasets.

    Returns parameters for the agent to call generate_data_for_training().
    """
    config_path = _config_dir() / _TEMP_CONFIG
    return _ok(
        data={
            "config_path":     str(config_path),
            "base_output_dir": str(_datasets_root()),
            "run_name":        _TEMP_RUN_NAME,
        },
        message=(
            f"Preconditions met. Agent will generate the temp dataset from "
            f"'{config_path}' into data/Datasets/temp."
        ),
    )


@mcp.tool()
def tool_preview_image(seed: int = 0) -> dict:
    """
    Preview a SINGLE synthetic AFM image from the CURRENT in-memory config, shown
    in the Visualization panel. Fast: renders one full-resolution image through the
    full pipeline (objects + substrate + tip + PID + artifacts + flatten) — NO
    dataset, NO annotations, nothing written to disk.

    Use this to iterate on parameters quickly, e.g. "show me a preview", "now lower
    the dirt and preview again", "remove the PID and show it". You do NOT need to
    save_config first — it previews the live in-memory config.

    It FORCES every ENABLED effect/object to appear at once (ignoring each effect's
    per-image probability) so the effect of each parameter is visible; this is
    preview-only and does NOT alter the config or the final dataset. For the
    realistic look (effects appearing at their configured probabilities) use
    tool_generate_temp_dataset (the ~10-image preview).

    seed : fixed layout seed (default 0). Keep it the same across tweaks so a
           parameter change shows only that change; pass a different seed for a new
           random arrangement of objects.
    """
    err = _require_cfg()
    if err:
        return err
    config_path = _config_dir() / _TEMP_CONFIG
    return _ok(
        data={"config_path": str(config_path), "seed": int(seed)},
        message=(f"Preconditions met. Agent will render a single-image preview from "
                 f"'{config_path}' (seed={seed})."),
    )


@mcp.tool()
def tool_generate_final_dataset(yaml_filename: str) -> dict:
    """
    Phantom tool — generate the definitive synthetic AFM dataset from a saved
    config file. After generation the agent automatically runs COCO→YOLO
    conversion (both DET and SEG).

    Requirements: config must be saved with save_config() first.
    Do NOT use temp.yaml.

    Parameters
    ----------
    yaml_filename : str   Config file name in data/config/conf_simuscan/, e.g. 'hexagons.yaml'.
    """
    config_path = _config_dir() / yaml_filename
    run_name    = datetime.now().strftime("%Y%m%d_%H%M%S") + "_COCO"

    return _ok(
        data={
            "config_path":     str(config_path),
            "base_output_dir": str(_datasets_root()),
            "run_name":        run_name,
        },
        message=(
            f"Config '{yaml_filename}' validated. Agent will generate the "
            f"final dataset as '{run_name}' and then convert to YOLO format."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Trainer — phantom tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def tool_convert_coco_to_yolo(coco_folder_name: str) -> dict:
    """
    Phantom tool — validate that a COCO dataset exists and return the
    parameters the agent needs to call convert_coco_to_yolo_dual().

    NOTE: this is called automatically after tool_generate_final_dataset.
    Only call manually if a COCO dataset already exists without YOLO conversion.

    Parameters
    ----------
    coco_folder_name : str
        Name of the COCO folder inside data/Datasets/, e.g. '20260402_104409_COCO'.
    """
    if not coco_folder_name.endswith("_COCO"):
        return _err("Folder name must end with '_COCO'.")

    coco_path = _datasets_root() / coco_folder_name
    if not coco_path.is_dir():
        return _err(
            f"COCO folder '{coco_folder_name}' not found in data/Datasets/.",
            detail={"expected_path": str(coco_path)},
        )
    if not (coco_path / "annotations").is_dir():
        return _err(
            f"No 'annotations' subfolder found inside '{coco_folder_name}'."
        )

    return _ok(
        data={
            "coco_folder":     str(coco_path),
            "yolo_det_folder": str(coco_path / "_yolo_det"),
            "yolo_seg_folder": str(coco_path / "_yolo_seg"),
        },
        message=f"COCO dataset '{coco_folder_name}' validated.",
    )


@mcp.tool()
def tool_train_yolo(
    coco_folder_name: str,
    task:             str        = "segment",
    model_name:       str | None = None,
    epochs:           int        = 100,
    imgsz:            int        = 512,
    batch:            int        = 32,
) -> dict:
    """
    Phantom tool — validate training preconditions and return parameters
    for the agent to call train_yolo_model().

    Parameters
    ----------
    coco_folder_name : str        Name of the *_COCO folder in data/Datasets/.
    task             : str        'segment' (default) or 'detect'.
    model_name       : str | None Override base model, e.g. 'yolov8l-seg.pt'.
                                  Default: yolov8m-seg.pt / yolov8m.pt.
    epochs           : int        Training epochs (default 100).
    imgsz            : int        Image size in pixels (default 512).
    batch            : int        Batch size (default 32).
    """
    task = task.lower()
    if task not in ("detect", "segment"):
        return _err(f"Invalid task '{task}'. Must be 'detect' or 'segment'.")
    if not coco_folder_name.endswith("_COCO"):
        return _err("coco_folder_name must end with '_COCO'.")

    coco_path = _datasets_root() / coco_folder_name
    if not coco_path.is_dir():
        return _err(f"COCO folder '{coco_folder_name}' not found in data/Datasets/.")

    subfolder = "_yolo_det" if task == "detect" else "_yolo_seg"
    yolo_path = coco_path / subfolder
    yaml_path = yolo_path / "dataset.yaml"

    if not yaml_path.is_file():
        return _err(
            f"YOLO dataset not yet converted for task='{task}'. "
            "Run tool_convert_coco_to_yolo first.",
            detail={"expected_yaml": str(yaml_path)},
        )

    default_models = {"detect": "yolov8m.pt", "segment": "yolov8m-seg.pt"}
    resolved_model = model_name or default_models[task]
    run_name       = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task}"

    return _ok(
        data={
            "coco_folder":      str(coco_path),
            "yolo_folder":      str(yolo_path),
            "task":             task,
            "model_name":       resolved_model,
            "epochs":           epochs,
            "imgsz":            imgsz,
            "batch":            batch,
            "expected_run_dir": str(_out_yolo_root() / run_name),
        },
        message=(
            f"Preconditions met. Agent will train with "
            f"task='{task}', model='{resolved_model}', epochs={epochs}."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Trainer — query tools (execute directly)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_yolo_datasets() -> dict:
    """
    List all dataset folders in data/Datasets/ (COCO, YOLO_DET, YOLO_SEG)
    with per-split image counts and class names.
    """
    return list_datasets()


@mcp.tool()
def list_yolo_training_runs() -> dict:
    """
    List all training run folders in data/out_yolo/ with presence of
    best.pt, last.pt, and results.csv for each run.
    """
    return list_training_runs()


@mcp.tool()
def get_yolo_training_results(run_name: str) -> dict:
    """
    Return last-epoch metrics, model weight paths, and training arguments
    for a specific run in data/out_yolo/.
    Also shows training curves, Ultralytics PNG plots, and saved test predictions
    (or re-generates predictions if missing).

    Parameters
    ----------
    run_name : str
        Run folder name, e.g. '20260408_143022_segment'.
        Use list_yolo_training_runs() to see available runs.
    """
    return get_training_results(str(_out_yolo_root() / run_name))


@mcp.tool()
def evaluate_yolo_model_tool(run_name: str) -> dict:
    """
    Load best.pt from a YOLO run and generate prediction images on test/val split.
    Saves PNGs to <run>/test_predictions/ and shows them in the plot panel.
    Use this to re-evaluate a model or generate predictions for a specific run.

    Parameters
    ----------
    run_name : str
        Run folder name, e.g. '20260408_143022_segment'.
        Use list_yolo_training_runs() to see available runs.
    """
    return evaluate_yolo_model(str(_out_yolo_root() / run_name))


@mcp.tool()
def check_gpu_availability() -> dict:
    """
    Check whether a CUDA GPU is available for training.
    Returns device name, CUDA version, and which device will be used.
    """
    return check_gpu()


# ══════════════════════════════════════════════════════════════════════════════
# Visualizer — phantom tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def tool_visualize_dataset(
    coco_folder_name: str | None       = None,
    task:             str              = "segment",
    splits:           list[str] | None = None,
    n_per_split:      int  = 3,
    show_boxes:       bool = True,
    show_masks:       bool = True,
    show_labels:      bool = True,
) -> dict:
    """
    Phantom tool — validate a YOLO dataset and return parameters so the
    agent can call visualize_yolo_dataset() locally (matplotlib is blocking).

    Parameters
    ----------
    coco_folder_name : str | None
        Name of the *_COCO folder in data/Datasets/.
        If None, the most recent COCO folder is used automatically.
    task        : str               'segment' (default) or 'detect'.
    splits      : list[str] | None  Splits to show. Default: ['train'].
    n_per_split : int               Images per split (default 3).
    show_boxes  : bool              Draw bounding boxes (default True).
    show_masks  : bool              Draw segmentation polygons (default True).
    show_labels : bool              Draw class name labels (default True).
    """
    ds_root = _datasets_root()

    if coco_folder_name is None:
        candidates = sorted(
            [p.name for p in ds_root.iterdir()
             if p.is_dir() and p.name.endswith("_COCO")],
            reverse=True,
        )
        if not candidates:
            return _err(
                "No COCO dataset found in data/Datasets/. "
                "Generate a dataset first."
            )
        coco_folder_name = candidates[0]

    coco_path = ds_root / coco_folder_name
    if not coco_path.is_dir():
        return _err(f"COCO folder '{coco_folder_name}' not found in data/Datasets/.")

    subfolder = "_yolo_seg" if task.lower() == "segment" else "_yolo_det"
    yolo_path = coco_path / subfolder
    if not (yolo_path / "dataset.yaml").is_file():
        return _err(
            f"YOLO dataset not yet converted (task='{task}'). "
            "Run tool_convert_coco_to_yolo first."
        )

    return _ok(
        data={
            "yolo_folder":  str(yolo_path),
            "splits":       splits or ["train"],
            "n_per_split":  n_per_split,
            "show_boxes":   show_boxes,
            "show_masks":   show_masks,
            "show_labels":  show_labels,
            "dataset_name": coco_folder_name,
        },
        message=(
            f"Dataset '{coco_folder_name}' ({subfolder}) validated. "
            f"Agent will display {n_per_split} image(s) "
            f"from {splits or ['train']}."
        ),
    )


@mcp.tool()
def tool_predict_folder(
    image_folder:   str,
    model_path:     str | None = None,
    show_boxes:     bool  = True,
    show_masks:     bool  = True,
    show_labels:    bool  = True,
    conf_threshold: float = 0.25,
) -> dict:
    """
    Phantom tool — validate inputs and return parameters so the agent can
    call predict_folder() locally (saves side-by-side PNGs + CSV to disk).

    Parameters
    ----------
    image_folder   : str        Folder containing images to predict.
    model_path     : str | None .pt file. Auto-selects latest best.pt if None.
    show_boxes     : bool       Draw bounding boxes in overlay (default True).
    show_masks     : bool       Draw segmentation masks in overlay (default True).
    show_labels    : bool       Draw class + confidence text (default True).
    conf_threshold : float      Minimum confidence (default 0.25).
    """
    img_dir = Path(image_folder)
    if not img_dir.is_dir():
        return _err(f"Image folder not found: '{image_folder}'.")

    img_files = [
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    ]
    if not img_files:
        return _err(f"No images found in '{image_folder}'.")

    resolved_model = model_path
    auto_resolved  = False
    if resolved_model is None:
        resolved_model = find_latest_best_pt()
        auto_resolved  = True

    if resolved_model is None:
        return _err(
            "No trained model found in data/out_yolo/. "
            "Train a model first or provide a model_path."
        )
    if not Path(resolved_model).is_file():
        return _err(f"Model file not found: '{resolved_model}'.")

    return _ok(
        data={
            "image_folder":   str(img_dir.resolve()),
            "model_path":     resolved_model,
            "auto_model":     auto_resolved,
            "n_images":       len(img_files),
            "show_boxes":     show_boxes,
            "show_masks":     show_masks,
            "show_labels":    show_labels,
            "conf_threshold": conf_threshold,
        },
        message=(
            f"Found {len(img_files)} image(s). "
            f"Model: '{Path(resolved_model).name}' "
            f"{'(auto-selected)' if auto_resolved else '(user-specified)'}."
        ),
    )


@mcp.tool()
def tool_predict_folder_gwy(
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
    Phantom tool — validate inputs so the agent can run the full GWY analysis
    pipeline locally: load .gwy files → flatten topography → YOLO inference.

    Output is saved in gwy_folder/gwy_analysis_<timestamp>/:
      extracted_png/     — flattened PNG images (one per .gwy file)
      overlays/          — side-by-side (raw | prediction) PNGs
      detections.csv     — all detections with pixel + physical µm columns
      analysis_config.json — model path, channel, flatten_perc, thresholds, etc.

    Flattening uses the same line-by-line OLS method as the AFM automation.

    Parameters
    ----------
    gwy_folder     : str        Folder containing .gwy files.
    model_path     : str | None YOLO .pt file; auto-selects latest best.pt if None.
    scan_direction : str        'forward' | 'backward' — the single topo channel to read (default 'forward').
    flatten_perc   : float      Percentile for line-flattening background selection (default 40).
    show_boxes     : bool       Draw bounding boxes on overlays (default True).
    show_masks     : bool       Draw segmentation masks on overlays (default True).
    show_labels    : bool       Draw class + confidence labels (default True).
    conf_threshold : float      Minimum YOLO confidence (default 0.25).
    """
    gwy_dir = Path(gwy_folder)
    if not gwy_dir.is_dir():
        return _err(f"Folder not found: '{gwy_folder}'.")

    if str(scan_direction).strip().lower() not in ("forward", "backward"):
        return _err(f"scan_direction must be 'forward' or 'backward', got '{scan_direction}'.")

    gwy_files = list(gwy_dir.glob("*.gwy"))
    if not gwy_files:
        return _err(f"No .gwy files found in '{gwy_folder}'.")

    resolved_model = model_path
    auto_resolved  = False
    if resolved_model is None:
        resolved_model = find_latest_best_pt()
        auto_resolved  = True
    if resolved_model is None:
        return _err(
            "No trained YOLO model found in data/out_yolo/. "
            "Train a model first or provide a model_path."
        )
    if not Path(resolved_model).is_file():
        return _err(f"Model file not found: '{resolved_model}'.")

    return _ok(
        data={
            "gwy_folder":    str(gwy_dir.resolve()),
            "model_path":    resolved_model,
            "auto_model":    auto_resolved,
            "scan_direction": scan_direction,
            "flatten_perc":  flatten_perc,
            "n_gwy_files":   len(gwy_files),
            "show_boxes":    show_boxes,
            "show_masks":    show_masks,
            "show_labels":   show_labels,
            "conf_threshold": conf_threshold,
        },
        message=(
            f"Found {len(gwy_files)} .gwy file(s). "
            f"Model: '{Path(resolved_model).name}' "
            f"{'(auto-selected)' if auto_resolved else '(user-specified)'}. "
            f"Scan direction: '{scan_direction}', flatten_perc={flatten_perc}."
        ),
    )


@mcp.tool()
def tool_get_model_info(model_path: str | None = None) -> dict:
    """
    Return metadata about a YOLO model: task, class names, file size, run dir.
    Auto-selects the most recent best.pt if model_path is None.

    Parameters
    ----------
    model_path : str | None   Path to a .pt file, or None for auto-select.
    """
    resolved = model_path or find_latest_best_pt()
    if resolved is None:
        return _err("No trained model found. Train a model first or provide model_path.")
    return get_model_info(resolved)


# ══════════════════════════════════════════════════════════════════════════════
# UNet Trainer tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def tool_train_unet(
    coco_folder_name: str,
    epochs:     int   = 25,
    batch_size: int   = 8,
    lr:         float = 1e-3,
    image_size: int   = 256,
) -> dict:
    """
    Phantom tool — validate preconditions and return parameters so the agent
    can call train_unet_model().

    Converts the COCO dataset to UNet format automatically if needed.
    Detects number of output channels from dataset_info.json.

    Parameters
    ----------
    coco_folder_name : str   Name of the *_COCO folder in data/Datasets/.
    epochs           : int   Training epochs (default 25).
    batch_size       : int   Batch size (default 8).
    lr               : float Learning rate (default 1e-3).
    image_size       : int   Square resize size (default 512).
    """
    if not coco_folder_name.endswith("_COCO"):
        return _err("coco_folder_name must end with '_COCO'.")

    coco_path = _datasets_root() / coco_folder_name
    if not coco_path.is_dir():
        return _err(f"COCO folder '{coco_folder_name}' not found in data/Datasets/.")
    if not (coco_path / "annotations").is_dir():
        return _err(f"No 'annotations' subfolder found in '{coco_folder_name}'.")

    return _ok(
        data={
            "coco_folder": str(coco_path),
            "epochs":      epochs,
            "batch_size":  batch_size,
            "lr":          lr,
            "image_size":  image_size,
        },
        message=(
            f"Preconditions met. Agent will train UNet on '{coco_folder_name}' "
            f"for {epochs} epochs."
        ),
    )


@mcp.tool()
def list_unet_training_runs_tool() -> dict:
    """
    List all UNet training run folders in data/out_unet/ with best_val_loss,
    class names, and presence of weight files.
    """
    return list_unet_training_runs()


@mcp.tool()
def get_unet_training_results_tool(run_name: str) -> dict:
    """
    Return full results (loss history, IoU/Dice metrics, class names) for a UNet run.
    Shows loss curve, metrics bar, confusion matrix, and test predictions.
    Re-runs evaluation automatically if plots are missing.

    Parameters
    ----------
    run_name : str
        Run folder name, e.g. '20260601_120000_unet'.
        Use list_unet_training_runs_tool() to see available runs.
    """
    run_dir = str(_out_unet_root() / run_name)
    return get_unet_training_results(run_dir)


@mcp.tool()
def evaluate_unet_model_tool(run_name: str) -> dict:
    """
    Load saved UNet weights and run full evaluation on test (or val) split.
    Generates and saves: metrics_bar.png, confusion_matrix.png, test_pred_0..3.png.
    Updates results.json with fresh test_metrics.
    Use this to re-evaluate a model or generate evaluation plots for a specific run.

    Parameters
    ----------
    run_name : str
        Run folder name, e.g. '20260601_120000_unet'.
        Use list_unet_training_runs_tool() to see available runs.
    """
    run_dir = str(_out_unet_root() / run_name)
    return evaluate_unet_model(run_dir)


@mcp.tool()
def tool_predict_folder_unet(
    image_folder:  str,
    weights_path:  str | None = None,
    run_name:      str | None = None,
    image_size:    int   = 512,
    alpha:         float = 0.5,
    save_masks:    bool  = False,
) -> dict:
    """
    Phantom tool — validate inputs so the agent can call predict_folder_unet().
    Saves side-by-side overlay PNGs + detections.csv in
    <image_folder>/out_unet_prediction_<timestamp>/.

    Parameters
    ----------
    image_folder  : str        Folder with images to predict.
    weights_path  : str | None Explicit path to .pth file.
    run_name      : str | None Run folder name (auto-locates best.pth).
                               Used if weights_path is None.
    image_size    : int        Resize size (must match training, default 512).
    alpha         : float      Overlay transparency (default 0.5).
    save_masks    : bool       Also save raw mask PNGs (default False).
    """
    img_dir = Path(image_folder)
    if not img_dir.is_dir():
        return _err(f"Image folder not found: '{image_folder}'.")

    # Resolve weights and class info
    resolved_weights = weights_path
    class_names: list = []
    out_channels: int = 0

    if resolved_weights is None:
        if run_name is None:
            # Auto-select latest run with best.pth
            runs = sorted(_out_unet_root().iterdir(), reverse=True)
            for r in runs:
                if (r / "unet_best.pth").is_file():
                    resolved_weights = str(r / "unet_best.pth")
                    run_name = r.name
                    break
        else:
            run_dir = _out_unet_root() / run_name
            bst = run_dir / "unet_best.pth"
            lst = run_dir / "unet_last.pth"
            resolved_weights = str(bst if bst.is_file() else lst)

    if not resolved_weights or not Path(resolved_weights).is_file():
        return _err("No UNet weights found. Train a model first or provide weights_path.")

    # Load class info from results.json of the run
    if run_name:
        rj = _out_unet_root() / run_name / "results.json"
        if rj.is_file():
            try:
                import json as _json
                with open(rj, encoding="utf-8") as f:
                    rd = _json.load(f)
                class_names  = rd.get("class_names", [])
                out_channels = rd.get("out_channels", 0)
            except Exception:
                pass

    if out_channels == 0:
        return _err(
            "Cannot determine out_channels. Provide run_name or ensure "
            "results.json is present in the run folder."
        )

    return _ok(
        data={
            "image_folder":  str(img_dir.resolve()),
            "weights_path":  resolved_weights,
            "out_channels":  out_channels,
            "class_names":   class_names,
            "image_size":    image_size,
            "alpha":         alpha,
            "save_masks":    save_masks,
        },
        message=(
            f"Found {sum(1 for f in img_dir.iterdir() if f.suffix.lower() in {'.png','.jpg','.jpeg','.tiff','.bmp'})} "
            f"image(s). Model: '{Path(resolved_weights).name}' "
            f"({out_channels} channels, classes: {class_names})."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Triplet Trainer — phantom tool
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def tool_train_triplet(
    coco_folder_name     : str,
    yolo_model_run_name  : str | None = None,
    max_iou              : float = 0.10,
    min_separation_px    : int   = 10,
    border_margin_pct    : float = 0.05,
    crop_size            : int   = 64,
    otsu_min_frac        : float = 0.10,
    latent_dim           : int   = 32,
    triplet_epochs       : int   = 40,
    triplet_batch        : int   = 64,
    triplet_lr           : float = 1e-3,
    triplet_margin       : float = 0.5,
    knn_k                : int   = 5,
) -> dict:
    """
    Phantom tool — validate preconditions and return parameters so the agent
    can call run_triplet_training() locally (training is blocking).

    Requirements:
      - A *_COCO dataset with a _yolo_seg/ subfolder must exist in data/Datasets/.
      - A trained YOLO segmentation model (.pt) must exist in data/out_yolo/.

    Parameters
    ----------
    coco_folder_name     : str        Name of the *_COCO folder in data/Datasets/.
    yolo_model_run_name  : str | None Run name in data/out_yolo/ (auto-selects latest if None).
    max_iou              : float      Max pairwise bbox IoU; above this the lower-conf
                                      object is dropped (default 0.10).
    min_separation_px    : int        Minimum pixel distance between any two objects'
                                      bboxes; closer pairs drop the lower-conf object
                                      (default 10).
    border_margin_pct    : float      Objects whose bbox is within this fraction of
                                      min(H,W) from any image edge are dropped
                                      (default 0.05 = 5%).
    crop_size            : int        Pixel size of object crops (default 64).
    otsu_min_frac        : float      Floor for Otsu threshold on normalised crop
                                      (default 0.10). Prevents over-thresholding on
                                      low-contrast objects.
    latent_dim           : int        Encoder embedding dimension (default 32).
    triplet_epochs       : int        Training epochs (default 40).
    triplet_batch        : int        Batch size (default 64).
    triplet_lr           : float      Learning rate (default 1e-3).
    triplet_margin       : float      Triplet loss margin (default 0.5).
    knn_k                : int        k for kNN evaluation (default 5).
    """
    from pathlib import Path

    ds_root = _datasets_root()

    # ── Resolve YOLO_SEG folder (lives inside the COCO folder) ───────────────
    if not coco_folder_name.endswith("_COCO"):
        return _err("coco_folder_name must end with '_COCO'.")
    coco_path = ds_root / coco_folder_name
    if not coco_path.is_dir():
        return _err(f"COCO folder '{coco_folder_name}' not found in data/Datasets/.")
    seg_path = coco_path / "_yolo_seg"
    if not seg_path.is_dir():
        return _err(
            f"_yolo_seg subfolder not found inside '{coco_folder_name}'. "
            "Run tool_convert_coco_to_yolo first.",
            detail={"expected": str(seg_path)},
        )
    if not (seg_path / "dataset.yaml").is_file():
        return _err(f"dataset.yaml not found in '{coco_folder_name}/_yolo_seg'.")

    # ── Resolve YOLO model ────────────────────────────────────────────────────
    out_yolo = _out_yolo_root()
    if yolo_model_run_name is not None:
        model_path = out_yolo / yolo_model_run_name / "weights" / "best.pt"
    else:
        model_path = None
        for p in sorted(out_yolo.iterdir(), reverse=True):
            if p.is_dir():
                pt = p / "weights" / "best.pt"
                if pt.is_file():
                    model_path = pt
                    break

    if model_path is None or not Path(str(model_path)).is_file():
        return _err(
            "No trained YOLO model found in data/out_yolo/. "
            "Train a YOLO model first, or provide yolo_model_run_name.",
            detail={"expected": str(model_path) if model_path else "none"},
        )

    return _ok(
        data={
            "yolo_seg_folder"  : str(seg_path),
            "yolo_model_path"  : str(model_path),
            "max_iou"          : max_iou,
            "min_separation_px": min_separation_px,
            "border_margin_pct": border_margin_pct,
            "crop_size"        : crop_size,
            "otsu_min_frac"    : otsu_min_frac,
            "latent_dim"       : latent_dim,
            "triplet_epochs"   : triplet_epochs,
            "triplet_batch"    : triplet_batch,
            "triplet_lr"       : triplet_lr,
            "triplet_margin"   : triplet_margin,
            "knn_k"            : knn_k,
        },
        message=(
            f"Preconditions met. Agent will train TripletEncoder on "
            f"'{coco_folder_name}/_yolo_seg' using model '{Path(str(model_path)).parent.parent.name}', "
            f"epochs={triplet_epochs}, latent_dim={latent_dim}, "
            f"max_iou={max_iou}, min_sep={min_separation_px}px, border={border_margin_pct*100:.0f}%."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Triplet visualiser — phantom tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def tool_show_crop_grid(
    run_name  : str | None = None,
    class_idx : int | None = None,
    max_items : int        = 64,
) -> dict:
    """
    Phantom tool — validate a triplet run and return parameters so the agent
    can call show_crop_grid() locally (matplotlib is blocking).

    Parameters
    ----------
    run_name  : str | None  Folder name inside data/out_triplet/ (e.g. 'time_20260410_120000').
                             Auto-selects latest if None.
    class_idx : int | None  Show only this class index (None = all classes).
    max_items : int         Maximum crops to show (default 64).
    """
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parent.parent.parent / "data" / "out_triplet"

    if run_name is None:
        if not root.exists():
            return _err("No triplet runs found. Train a triplet encoder first.")
        candidates = sorted(
            [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("time_")],
            reverse=True,
        )
        if not candidates:
            return _err("No triplet runs found. Train a triplet encoder first.")
        run_name = candidates[0]

    run_dir = root / run_name
    if not run_dir.is_dir():
        return _err(f"Run folder '{run_name}' not found in data/out_triplet/.")
    if not (run_dir / "crops").is_dir():
        return _err(f"Crops folder not found in '{run_name}'. "
                    "Was save_crops=True during training?")

    return _ok(
        data={
            "run_dir"  : str(run_dir),
            "class_idx": class_idx,
            "max_items": max_items,
        },
        message=(
            f"Run '{run_name}' validated. Agent will display crop grid"
            + (f" for class {class_idx}" if class_idx is not None else " (all classes)")
            + f" (max {max_items} items)."
        ),
    )


@mcp.tool()
def tool_show_embedding_umap(run_name: str | None = None) -> dict:
    """
    Phantom tool — validate a triplet run and return parameters so the agent
    can call show_embedding_umap() locally (matplotlib is blocking).

    Parameters
    ----------
    run_name : str | None  Folder name in data/out_triplet/. Auto-selects latest if None.
    """
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parent.parent.parent / "data" / "out_triplet"

    if run_name is None:
        if not root.exists():
            return _err("No triplet runs found.")
        candidates = sorted(
            [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("time_")],
            reverse=True,
        )
        if not candidates:
            return _err("No triplet runs found.")
        run_name = candidates[0]

    run_dir = root / run_name
    if not run_dir.is_dir():
        return _err(f"Run folder '{run_name}' not found in data/out_triplet/.")

    emb_csv = run_dir / "results" / "embeddings.csv"
    if not emb_csv.is_file():
        return _err(f"embeddings.csv not found in '{run_name}/results/'. "
                    "Re-run training to generate embeddings.")

    return _ok(
        data={"run_dir": str(run_dir)},
        message=f"Run '{run_name}' validated. Agent will display interactive UMAP.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Triplet Trainer — query tools (execute directly, no phantom needed)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_triplet_training_runs() -> dict:
    """
    List all triplet encoder training runs in data/out_triplet/
    with presence of model file and metrics.
    """
    return list_triplet_runs()


@mcp.tool()
def get_triplet_training_results(run_name: str) -> dict:
    """
    Return metrics, model path, and output file locations for a specific
    triplet training run.

    Parameters
    ----------
    run_name : str  Folder name inside data/out_triplet/, e.g. 'time_20260410_120000'.
                    Use list_triplet_training_runs() to see available runs.
    """
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parent.parent.parent / "data" / "out_triplet"
    return get_triplet_results(str(root / run_name))


@mcp.tool()
def tool_build_triplet_library(
    image_folder      : str,
    triplet_run_name  : str | None = None,
    yolo_run_name     : str | None = None,
    pixel_size_nm     : float = 1.0,
    n_clusters        : int | None = None,
    yolo_conf         : float = 0.50,
    border_margin_pct : float = 0.0,
    min_separation_px : int   = 0,
) -> dict:
    """
    Phantom tool — validate inputs and return parameters so the agent can call
    run_triplet_library() locally (processing is blocking).

    Processes every image in image_folder through the full YOLO → Otsu → align
    → encode pipeline, builds an embedding library and saves all outputs to
    data/out_triplet_lib/<timestamp>/.

    Parameters
    ----------
    image_folder      : str        Absolute path to folder with PNG/JPG images.
    triplet_run_name  : str | None Run name in data/out_triplet/ (auto = latest).
    yolo_run_name     : str | None Run name in data/out_yolo/ (auto = latest).
    pixel_size_nm     : float      Physical pixel size in nm. 1.0 = report in pixels.
    n_clusters        : int | None KMeans clusters. None = auto (silhouette, max 8).
    yolo_conf         : float      YOLO confidence threshold (default 0.50).
    border_margin_pct : float      Fraction of image width/height to exclude near
                                   borders (0.0 = disabled). E.g. 0.05 = 5%.
    min_separation_px : int        Minimum pixel distance between object centres.
                                   Objects closer than this are discarded (0 = disabled).
    """
    from pathlib import Path as _Path

    img_folder = _Path(image_folder)
    if not img_folder.is_dir():
        return _err(f"Image folder not found: '{image_folder}'")

    imgs = (list(img_folder.glob("*.png")) + list(img_folder.glob("*.jpg"))
            + list(img_folder.glob("*.gwy")))
    if not imgs:
        return _err(f"No PNG/JPG/GWY images found in '{image_folder}'")

    # Validate triplet model
    triplet_root = _Path(__file__).resolve().parent.parent.parent / "data" / "out_triplet"
    if triplet_run_name:
        pt = triplet_root / triplet_run_name / "triplet_encoder.pt"
        if not pt.is_file():
            return _err(f"triplet_encoder.pt not found in run '{triplet_run_name}'.")
    else:
        found = any(
            (p / "triplet_encoder.pt").is_file()
            for p in triplet_root.iterdir() if p.is_dir()
        ) if triplet_root.exists() else False
        if not found:
            return _err("No trained TripletEncoder found. Train one first.")

    # Validate YOLO model
    yolo_root = _Path(__file__).resolve().parent.parent.parent / "data" / "out_yolo"
    if yolo_run_name:
        yolo_pt = yolo_root / yolo_run_name / "weights" / "best.pt"
        if not yolo_pt.is_file():
            return _err(f"best.pt not found in YOLO run '{yolo_run_name}'.")

    return _ok(
        data={
            "image_folder"     : str(img_folder),
            "triplet_run_name" : triplet_run_name,
            "yolo_run_name"    : yolo_run_name,
            "pixel_size_nm"    : pixel_size_nm,
            "n_clusters"       : n_clusters,
            "yolo_conf"        : yolo_conf,
            "border_margin_pct": border_margin_pct,
            "min_separation_px": min_separation_px,
            "n_images"         : len(imgs),
            "input_type"       : "gwy" if all(p.suffix == ".gwy" for p in imgs) else "png",
        },
        message=(
            f"Ready to build library: {len(imgs)} "
            f"{'GWY' if all(p.suffix == '.gwy' for p in imgs) else 'PNG/JPG'} files, "
            f"triplet={'auto' if not triplet_run_name else triplet_run_name}, "
            f"yolo={'auto' if not yolo_run_name else yolo_run_name}."
        ),
    )


@mcp.tool()
def list_triplet_library_runs() -> dict:
    """
    List all triplet embedding library runs in data/out_triplet_lib/
    sorted newest-first, with object count and cluster count.
    """
    return list_triplet_libraries()


@mcp.tool()
def get_triplet_library_results_tool(run_name: str) -> dict:
    """
    Return summary stats and show UMAP plots for a specific triplet library run.

    Parameters
    ----------
    run_name : str  Folder name inside data/out_triplet_lib/, e.g.
                    'time_20260410_120000'. Use list_triplet_library_runs() to
                    see available runs.
    """
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parent.parent.parent / "data" / "out_triplet_lib"
    return get_triplet_library_results(str(root / run_name))


@mcp.tool()
def tool_get_triplet_model_info(model_path: str | None = None) -> dict:
    """
    Return metadata about a trained TripletEncoder: latent dim, number of
    classes, val accuracy, silhouette score, kNN accuracy.
    Auto-selects the most recent model if model_path is None.

    Parameters
    ----------
    model_path : str | None  Path to a triplet_encoder.pt, or None for auto-select.
    """
    resolved = model_path or find_latest_triplet_pt()
    if resolved is None:
        return _err("No trained triplet model found. Train one first.")
    return get_triplet_model_info(resolved)


@mcp.tool()
def tool_show_triplet_library_explorer(
    library_run_name: str | None = None,
) -> dict:
    """
    ⚠ PHANTOM TOOL — handled locally by agent.py ⚠

    Open an interactive UMAP explorer window for a triplet embedding library.
    Three tabs show the UMAP of the combined descriptor, shape features, and
    embedding respectively.  Clicking any point displays the crop image and
    metadata (class, cluster, novelty score, shape features) in a side panel.

    Works on BOTH:
      - Experimental libraries in data/out_triplet_lib/ (built from a folder
        of PNG/GWY images via tool_build_triplet_library).
      - Training runs in data/out_triplet/ — every triplet training run now
        also writes a library.json from the synthetic crops, so its embeddings
        can be explored interactively too.

    Parameters
    ----------
    library_run_name : str | None
        Folder name inside data/out_triplet_lib/ OR data/out_triplet/, e.g.
        'time_20260618_120842'. If None, the most recent run with a library.json
        across both folders is used automatically.
    """
    return {
        "phantom"          : True,
        "library_run_name" : library_run_name,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Automation — active config state
# ══════════════════════════════════════════════════════════════════════════════
 
# _auto_cfg is declared at module level alongside _cfg (see instructions above)
 
def _require_auto_cfg() -> dict:
    if _auto_cfg is None:
        return {
            "status": "error",
            "message": "Automation config not available (default YAML missing). Call load_automation_config('<file>') to load one.",
        }
    return {}
 
def _auto_config_dir() -> Path:
    return _repo_root() / "data" / "config" / "conf_automation"
 
# ══════════════════════════════════════════════════════════════════════════════
# Automation — session / file management tools
# ══════════════════════════════════════════════════════════════════════════════
 
@mcp.tool()
def load_automation_config(filename: str | None = None) -> dict:
    """
    Load an AFM automation YAML configuration into the active session.
 
    Parameters
    ----------
    filename : str | None
        YAML file name (e.g. 'my_session.yaml') in data/config/conf_automation/.
        If None, loads a fresh in-memory copy of automation_default.yaml.
    """
    global _auto_cfg
    try:
        _auto_cfg = AutomationConfig(filename)
        return _auto_cfg.summary()
    except FileNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
 
 
@mcp.tool()
def save_automation_config(filename: str) -> dict:
    """
    Save the current automation configuration to
    data/config/conf_automation/<filename>.
    Never overwrites automation_default.yaml.
 
    Parameters
    ----------
    filename : str   Target YAML file name, e.g. 'session_hexagons.yaml'.
    """
    err = _require_auto_cfg()
    if err:
        return err
    return _auto_cfg.save(filename)
 
 
@mcp.tool()
def list_automation_configs() -> dict:
    """List all YAML files available in data/config/conf_automation/."""
    try:
        tmp = _auto_cfg if _auto_cfg is not None else AutomationConfig()
        return tmp.list_files()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
 
 
@mcp.tool()
def get_automation_summary() -> dict:
    """
    Return a high-level summary of the currently loaded automation config:
    gallery_dir, yolo_model_path, target_classes, strategy, scan sizes, etc.
    """
    err = _require_auto_cfg()
    if err:
        return err
    return _auto_cfg.summary()
 
 
@mcp.tool()
def get_automation_all() -> dict:
    """Return all parameters of the currently loaded automation config."""
    err = _require_auto_cfg()
    if err:
        return err
    return _auto_cfg.get_all()
 
 
@mcp.tool()
def get_automation_param(param: str) -> dict:
    """
    Return the value of a single automation parameter.

    Parameters
    ----------
    param : str   Parameter name, e.g. 'gallery_dir', 'yolo_model_path',
                  'scan_size_um', 'strategy', 'target_classes'.
    """
    err = _require_auto_cfg()
    if err:
        return err
    return _auto_cfg.get_param(param)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Automation — write tools
# ══════════════════════════════════════════════════════════════════════════════
 
@mcp.tool()
def set_automation_param(param: str, value: Any) -> dict:
    """
    Edit a single automation parameter.
    Type-safe against the stored value.
 
    Special behaviour: setting 'yolo_model_path' automatically reads and
    updates 'target_classes' from the model file (all classes with n=10, enabled=True).

    Parameters
    ----------
    param : str   Parameter name.
    value : Any   New value (must match the existing type).

    Common parameters
    -----------------
    gallery_dir          : str    Path where Nanosurf saves .gwy files.
    yolo_model_path      : str    Absolute path to best.pt  (auto-fills target_classes).
    target_classes       : dict   {class_name: {n: int, enabled: bool}, ...}.
                                  n = desired captures; enabled = False to skip a class.
    scan_size_um         : float  Exploration frame size in µm.
    scan_rate_hz         : float  Line scan rate in Hz.
    scan_resolution_px   : int    Pixels per side for exploration frames.
    area_to_cover_um     : float  Total area to explore in µm² (used when grid_n_x/y=0).
    overlap_fraction     : float  Frame overlap fraction (negative = gap between frames).
    strategy                     : str    'raster' | 'probabilistic'.
    max_jump_um                  : float  Max position jump between frames (probabilistic only, µm).
    prob_alpha_init              : float  Initial alpha for Beta maps (prior, default 1.0).
    prob_beta_init               : float  Initial beta for Beta maps (prior, default 1.0).
    prob_spatial_sigma_um        : float  Gaussian propagation radius for evidence (µm).
    prob_kappa_uncertainty       : float  Uncertainty bonus weight in utility score.
    prob_lambda_move             : float  Movement distance penalty in utility score.
    prob_epsilon_unknown         : float  Probability of choosing most-uncertain cell.
    prob_negative_update_scale   : float  Weight of negative evidence (no detection).
    prob_initial_random_n        : int    Random scans near center before active search.
    prob_initial_center_radius_um: float  Radius for initial random scans (µm).
    isolation_max_bbox_overlap   : float  Max bbox overlap to count as isolated (0=off).
    isolation_min_distance_um    : float  Min center distance to count as isolated (µm, 0=off).
    border_filter_enabled        : bool   Edge-touching detections are cut-off, so
                                  their shape is unreliable: they do NOT count toward
                                  targets, the library, or zoom/crops. But if YOLO
                                  detected them (enough visible) they still feed the
                                  probabilistic map, so it is not biased toward
                                  "empty" at frame edges. Drawn in red on the overlay.
    zoom_enabled                 : bool   Enable zoom scans on detections.
    zoom_max_rate_hz     : float  Max scan rate for zoom frames.
    zoom_max_size_um     : float  Hard cap on zoom frame size in µm.
    zoom_bbox_factors    : list   [{min_bbox_um: float, factor: float}, ...]
                                  Zoom size = factor × bbox_size, rule matched from top.
    max_large_frames     : int    Probabilistic stop condition: end after this many
                                  frames (0/negative = no limit). Ignored by raster,
                                  which always scans its full grid.
    triplet_model_path   : str    Path to triplet_encoder.pt (optional).
    stage_x_min_um / stage_x_max_um / stage_y_min_um / stage_y_max_um : float
                                  Stage travel limits (same frame as origin, µm). The
                                  grid is clipped to this reachable rectangle so frames
                                  are never queued where the positioner stalls (which
                                  causes .gwy timeouts). 0 (max<=min) disables clipping.
    stage_edge_margin_um : float  Safety margin kept from the hard travel limit.
    stage_auto_range     : bool   Try to read the range from the hardware (Nanosurf
                                  Studio does not expose it → keep false, set bounds).
    Note: the .gwy wait timeout is computed automatically from scan_resolution_px
    and scan_rate_hz — there is no gwy_wait_timeout_s parameter.
    """
    err = _require_auto_cfg()
    if err:
        return err
    return _auto_cfg.set_param(param, value)
 
 
@mcp.tool()
def set_automation_params(updates: dict[str, Any]) -> dict:
    """
    Edit multiple automation parameters at once.
 
    Parameters
    ----------
    updates : dict   {param_name: new_value, ...}
    """
    err = _require_auto_cfg()
    if err:
        return err
    return _auto_cfg.set_params(updates)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Automation — microscope connection (phantom — executed locally by agent)
# ══════════════════════════════════════════════════════════════════════════════
 
@mcp.tool()
def automation_connect_microscope() -> dict:
    """
    Phantom tool — returns a signal for the agent to call connect_microscope()
    locally, so the Nanosurf Studio connection and _ACTIVE_SESSION are created
    in the same process that will later call run_exploration().
 
    Must be called before tool_run_exploration.
    Safe to call multiple times — reuses existing connection if already connected.
    """
    return _ok(
        data={"action": "connect_microscope"},
        message="Agent will connect to Nanosurf Studio locally.",
    )
 
 
@mcp.tool()
def automation_get_status() -> dict:
    """
    Phantom tool — returns a signal for the agent to call
    get_exploration_status() locally, where _ACTIVE_SESSION lives.
 
    Returns phase, frame counts, detections per class, and last message.
    """
    return _ok(
        data={"action": "get_exploration_status"},
        message="Agent will query exploration status locally.",
    )
 
 
@mcp.tool()
def automation_stop() -> dict:
    """
    Phantom tool — returns a signal for the agent to call
    stop_exploration() locally, where _ACTIVE_SESSION lives.
 
    Sends a graceful stop: the session finishes the current frame then stops.
    """
    return _ok(
        data={"action": "stop_exploration"},
        message="Agent will send stop signal locally.",
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Automation — phantom tool: run exploration
# ══════════════════════════════════════════════════════════════════════════════
 
@mcp.tool()
def tool_run_exploration(filename: str | None = None) -> dict:
    """
    Phantom tool — validate the automation configuration and return the
    exploration dict so the agent can call run_exploration() locally
    (in a background thread, non-blocking).
 
    The agent MUST call automation_connect_microscope() before this tool.
 
    Can be called in two ways:
      1. With a filename: loads the YAML from conf_automation/ and validates it.
      2. Without filename: uses the currently loaded automation config in memory.
 
    Parameters
    ----------
    filename : str | None
        YAML file in data/config/conf_automation/. If None, uses the
        in-memory config loaded with load_automation_config().
 
    Validation checks
    -----------------
    - gallery_dir must be set (non-empty string)
    - yolo_model_path must be set if zoom_enabled is True
    - target_classes must be non-empty if zoom_enabled is True
    - The config file (if given) must exist
    """
    global _auto_cfg
 
    # ── Resolve which config to use ───────────────────────────────────────────
    if filename is not None:
        # Load from file (updates the in-memory session config too)
        try:
            _auto_cfg = AutomationConfig(filename)
        except FileNotFoundError as exc:
            return {"status": "error", "message": str(exc)}
        config_yaml_path = str(_auto_config_dir() / filename)
    else:
        if _auto_cfg is None:
            return {
                "status": "error",
                "message": (
                    "No automation config loaded and no filename provided. "
                    "Call load_automation_config() first or provide a filename."
                ),
            }
        # Use the current in-memory config; derive yaml path if it has a name
        config_yaml_path = (
            str(_auto_config_dir() / _auto_cfg._filename)
            if _auto_cfg._filename
            else ""
        )
 
    # ── Validation ────────────────────────────────────────────────────────────
    d = _auto_cfg._data

    if not d.get("gallery_dir", "").strip():
        return _err(
            "'gallery_dir' is not set. "
            "Use set_automation_param('gallery_dir', '<path>') to configure it."
        )

    zoom_enabled = d.get("zoom_enabled", False)
    yolo_enabled = d.get("yolo_enabled", False)
    if zoom_enabled or yolo_enabled:
        if not d.get("yolo_model_path", "").strip():
            return _err(
                "'yolo_model_path' is required when yolo_enabled or zoom_enabled is True. "
                "Use set_automation_param('yolo_model_path', '<path>') to set it."
            )
        if not d.get("target_classes"):
            return _err(
                "'target_classes' is empty. "
                "Set 'yolo_model_path' to auto-fill, or set target_classes manually: "
                "{'class_name': {'n': 10, 'enabled': True}, ...}"
            )

    strategy = d.get("strategy", "raster")
    if strategy == "probabilistic" and not yolo_enabled:
        return _err(
            "strategy='probabilistic' requires yolo_enabled=True "
            "(per-class probability maps need YOLO detections)."
        )

    # ── Build exploration dict (µm → m conversion) ────────────────────────────
    exploration_dict = _auto_cfg.to_exploration_dict(
        config_yaml_path=config_yaml_path
    )

    # ── Build human-readable summary ──────────────────────────────────────────
    size_um  = d.get("scan_size_um", 50.0)
    area_um  = d.get("area_to_cover_um", 10000.0)
    target   = d.get("target_classes", {})
    max_fr   = d.get("max_large_frames", 1000)

    cls_summary = {k: v.get("n") for k, v in target.items()
                   if isinstance(v, dict) and v.get("enabled", True)}

    return _ok(
        data={"exploration_dict": exploration_dict},
        message=(
            f"Config validated. Agent will start exploration: "
            f"strategy='{strategy}', area={area_um}µm², frame={size_um}µm, "
            f"target_classes={cls_summary}, max_frames={max_fr}, "
            f"yolo={yolo_enabled}, zoom={zoom_enabled}."
        ),
    )
 

# ══════════════════════════════════════════════════════════════════════════════
# Reference image generation tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def tool_generate_ref_image(
    class_name: str,
    superclass_name: str,
    prompt: str,
    n_images: int = 1,
    add_to_config: bool = True,
    reference_image: str | None = None,
    apply_height_profile: bool = True,
    height_profile_order: float = 2.0,
    height_profile_dh0: float = 0.0,
    height_profile_texture: float = 0.4,
) -> dict:
    """
    Generate reference images of a single object using OpenAI image generation,
    then create binary masks automatically and zero the background.

    Images are saved inside a dedicated subfolder:
      ref_objects/<superclass_name>_<class_name>/
        <superclass_name>_<class_name>_000.png
        mask_<superclass_name>_<class_name>_000.png
        ...

    If the folder already exists, new images are APPENDED with the next index.

    After generation, call add_image_object() with the returned image_folder
    and image_files list to register the object in the active configuration.

    Args:
        class_name:       Short class identifier (e.g. "ribosome").
        superclass_name:  Category name (e.g. "protein").
        prompt:           Description of the object for the image model.
                          AFM style constraints are appended automatically.
        n_images:         Number of images to generate (1–5).
        add_to_config:    Informational flag — agent should call add_image_object
                          after generation if True.
        reference_image:  Optional path to an existing image to use as visual
                          reference. Can be an absolute path or a filename
                          relative to ref_objects/<folder>/
                          (e.g. "protein_ribosome_000.png"). When provided,
                          uses images.edit() instead of images.generate() so the
                          model produces variants consistent with that image.
                          Use list_ref_images() to find available filenames.
        apply_height_profile: After generation, reshape each image into a
                          physically-plausible AFM height map: a dome envelope
                          that grows with distance from the object edge, so the
                          topography is no longer the model's lighting/shading.
                          Default True. Set False to keep the raw model output.
                          (Disable when the user says "keep it as the model
                          generated it" / "don't apply the height profile".)
        height_profile_order: Dome decay order (low ≈ flat plateau, high ≈ sharp
                          peak). Default 2.0.
        height_profile_dh0: Relative height at the object edge (0 = drops to
                          zero, >0 = step/plateau edge). Default 0.0.
        height_profile_texture: Blend weight 0–1. 0 = pure dome (no model
                          texture), 1 = model texture fully modulates the dome.
                          Default 0.4 (keeps diffusion detail while enforcing the
                          dome relation).
    """
    n_images = max(1, min(5, n_images))
    if not class_name.strip() or not superclass_name.strip():
        return _err("class_name and superclass_name must not be empty.")
    if not prompt.strip():
        return _err("prompt must not be empty.")

    # Validate reference_image path if provided.
    # Accepts either a full absolute path or a filename relative to ref_objects/<folder>/.
    ref_path = None
    if reference_image:
        candidate = Path(reference_image)
        if candidate.is_absolute():
            ref_path = candidate
        else:
            folder_name = f"{superclass_name.strip()}_{class_name.strip()}"
            ref_path = _repo_root() / "ref_objects" / folder_name / reference_image
        if not ref_path.exists():
            return _err(
                f"Reference image not found: {ref_path}. "
                "Use list_ref_images() to see available files."
            )

    return _ok(
        data={
            "class_name":       class_name.strip(),
            "superclass_name":  superclass_name.strip(),
            "prompt":           prompt.strip(),
            "n_images":         n_images,
            "add_to_config":    add_to_config,
            "reference_image":  str(ref_path) if ref_path else None,
            "apply_height_profile":   apply_height_profile,
            "height_profile_order":   height_profile_order,
            "height_profile_dh0":     height_profile_dh0,
            "height_profile_texture": height_profile_texture,
        },
        message=(
            f"Ready to generate {n_images} reference image(s) for "
            f"'{superclass_name}/{class_name}'"
            + (f" using '{reference_image}' as visual reference (images.edit)." if reference_image else ".")
        ),
    )


@mcp.tool()
def tool_apply_height_profile(
    image_path: str,
    order: float = 2.0,
    dh0: float = 0.0,
    texture: float = 0.4,
    output_path: str | None = None,
    save_mask: bool = True,
) -> dict:
    """
    Apply the distance-based AFM height profile to an ALREADY-EXISTING image,
    without generating anything. Use this when the user provides their own image
    (a photo, a crop, a download) and wants it turned into realistic AFM
    topography.

    The object is reshaped into a dome envelope that grows with distance from its
    edge, modulated by the original image texture:
        height = dome * ((1 - texture) + texture * normalized_intensity)

    A new image is written by default (the original is NOT overwritten).

    Args:
        image_path : Path to the source image. Absolute, or a filename/relative
                     path inside ref_objects/.
        order      : Dome decay order (low ≈ flat plateau, high ≈ sharp peak).
        dh0        : Relative height at the edge (0 = drops to zero, >0 = step).
        texture    : Blend 0–1. 0 = pure dome, 1 = texture modulates fully.
                     Default 0.4 (keeps detail while enforcing the dome relation).
        output_path: Destination PNG. None → '<stem>_height.png' next to source.
        save_mask  : If no sibling mask exists, save the computed Otsu mask.
    """
    from pathlib import Path as _Path

    p = _Path(image_path)
    if not p.is_absolute():
        cand = _repo_root() / "ref_objects" / image_path
        if cand.exists():
            p = cand
    if not p.is_file():
        return _err(f"Image not found: {image_path}")

    out = _Path(output_path) if output_path else p.with_name(f"{p.stem}_height.png")

    # Reuse a sibling mask (mask_<name>) if present, else compute next to output
    mask_sib = p.with_name(f"mask_{p.name}")
    if mask_sib.is_file():
        mask_path = str(mask_sib)
    else:
        mask_path = str(out.with_name(f"mask_{out.name}"))

    return _ok(
        data={
            "image_path":  str(p),
            "mask_path":   mask_path,
            "order":       order,
            "dh0":         dh0,
            "texture":     texture,
            "output_path": str(out),
            "save_mask":   save_mask,
        },
        message=f"Ready to apply height profile to '{p.name}' → '{out.name}'.",
    )


@mcp.tool()
def tool_show_ref_images(
    filter_name: str | None = None,
    max_images: int = 5,
) -> dict:
    """
    Display existing reference images (and their masks) from
    ref_objects/ in the plot panel.

    Images are organised in per-class subfolders:
      ref_objects/<superclass_class>/

    Args:
        filter_name: Optional substring to filter by folder or file name
                     (e.g. "protein" shows only protein_* folders/files).
        max_images:  Maximum number of image pairs to show (1–5, default 5).
    """
    root = _repo_root() / "ref_objects"
    if not root.exists():
        return _err("ref_objects/ does not exist yet.")

    max_images = max(1, min(5, max_images))

    # Collect all PNGs from subfolders (new structure) and root (legacy)
    pngs = []
    for p in sorted(root.rglob("*.png")):
        if not p.name.startswith("mask_"):
            pngs.append(p)

    if filter_name:
        pngs = [p for p in pngs if filter_name.lower() in str(p).lower()]
    if not pngs:
        return _err(
            "No reference images found"
            + (f" matching '{filter_name}'" if filter_name else "")
            + ". Run tool_generate_ref_image first."
        )

    selected = pngs[:max_images]
    pairs = []
    for p in selected:
        mask_path = p.parent / f"mask_{p.name}"
        pairs.append({
            "image_file":  p.name,
            "mask_file":   f"mask_{p.name}",
            "image_path":  str(p),
            "mask_path":   str(mask_path),
            "mask_exists": mask_path.exists(),
            "folder":      p.parent.name,
        })

    return _ok(
        data={"pairs": pairs, "directory": str(root)},
        message=f"Agent will display {len(pairs)} reference image pair(s).",
    )


@mcp.tool()
def list_ref_images() -> dict:
    """
    List all reference images in ref_objects/, grouped by class folder.

    Returns a dict with one entry per subfolder (class), each containing:
      - folder:        subfolder name (superclass_class)
      - image_files:   list of PNG filenames in that folder
      - count:         number of images
    """
    root = _repo_root() / "ref_objects"
    if not root.exists():
        return _ok(data={"folders": []}, message="ref_objects/ does not exist yet.")

    folders = {}
    for p in sorted(root.rglob("*.png")):
        if p.name.startswith("mask_"):
            continue
        folder = p.parent.name
        folders.setdefault(folder, []).append(p.name)

    result = [
        {"folder": k, "image_files": v, "count": len(v)}
        for k, v in sorted(folders.items())
    ]
    total = sum(r["count"] for r in result)
    return _ok(
        data={"folders": result},
        message=f"{total} reference image(s) in {len(result)} class folder(s).",
    )


@mcp.tool()
def delete_ref_image(
    folder_name: str,
    image_file:  str,
) -> dict:
    """
    Delete a single reference image (and its companion mask) from a class folder.

    Parameters
    ----------
    folder_name : Class folder name inside ref_objects/ (e.g. "protein_ribosome").
    image_file  : PNG filename to delete (e.g. "protein_ribosome_002.png").
                  The corresponding mask_<image_file> is also deleted automatically.
    """
    root   = _repo_root() / "ref_objects" / folder_name
    if not root.exists():
        return _err(f"Folder '{folder_name}' not found in ref_objects/.")

    img_path  = root / image_file
    mask_path = root / f"mask_{image_file}"

    if not img_path.exists():
        return _err(f"File '{image_file}' not found in '{folder_name}'.")

    deleted = []
    img_path.unlink()
    deleted.append(image_file)
    if mask_path.exists():
        mask_path.unlink()
        deleted.append(f"mask_{image_file}")

    remaining = sorted(p.name for p in root.glob("*.png") if not p.name.startswith("mask_"))
    return _ok(
        data={"deleted": deleted, "remaining": remaining, "folder": folder_name},
        message=(
            f"Deleted {deleted} from '{folder_name}'. "
            f"{len(remaining)} image(s) remaining."
        ),
    )


@mcp.tool()
def delete_ref_folder(
    folder_name: str,
    confirm: bool = False,
) -> dict:
    """
    Delete an entire class folder (all images and masks) from ref_objects/.

    This is irreversible. You MUST pass confirm=True to proceed.

    Parameters
    ----------
    folder_name : Class folder name inside ref_objects/ (e.g. "protein_ribosome").
    confirm     : Must be True to actually delete. If False, returns a dry-run
                  listing what would be deleted without touching any files.
    """
    import shutil

    root = _repo_root() / "ref_objects" / folder_name
    if not root.exists():
        return _err(f"Folder '{folder_name}' not found in ref_objects/.")

    files = sorted(p.name for p in root.glob("*.png"))

    if not confirm:
        return _ok(
            data={"folder": folder_name, "files": files, "count": len(files)},
            message=(
                f"DRY RUN: '{folder_name}' contains {len(files)} file(s): {files}. "
                "Call again with confirm=True to permanently delete the folder."
            ),
        )

    shutil.rmtree(root)
    return _ok(
        data={"deleted_folder": folder_name, "files_deleted": files},
        message=f"Folder '{folder_name}' and {len(files)} file(s) permanently deleted.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PDF reference tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_reference_pdfs() -> dict:
    """
    List all PDF files in docs/references/ available for image extraction and
    RAG indexing.
    """
    refs_dir = _repo_root() / "docs" / "references"
    if not refs_dir.exists():
        return _ok(
            data={"pdfs": []},
            message="docs/references/ does not exist yet. Create it and add PDF papers.",
        )
    pdfs = sorted(p.name for p in refs_dir.glob("*.pdf"))
    return _ok(
        data={"pdfs": pdfs, "count": len(pdfs), "directory": str(refs_dir)},
        message=f"{len(pdfs)} PDF(s) available in docs/references/.",
    )


@mcp.tool()
def extract_pdf_images(
    pdf_name: str,
    pages: list[int] | None = None,
    min_size_px: int = 100,
) -> dict:
    """
    Extract embedded images from a PDF in docs/references/ and display them
    in the plot panel so the user can select which ones are AFM images.

    Images are saved to docs/references/images_<pdf_stem>/ and assigned
    an index. Pass that index to open_crop_tool() so the user can crop
    the regions of interest interactively.

    Args:
        pdf_name:    Filename of the PDF in docs/references/ (e.g. "nature_afm.pdf").
        pages:       List of page numbers to extract from (1-based). If None,
                     extracts from all pages.
        min_size_px: Minimum image dimension in pixels to include (filters out
                     icons, logos, decorative elements). Default 100.
    """
    try:
        import fitz
    except ImportError:
        return _err("pymupdf not installed. Run: uv add pymupdf")

    pdf_path = _repo_root() / "docs" / "references" / pdf_name
    if not pdf_path.exists():
        return _err(f"PDF '{pdf_name}' not found in docs/references/.")

    out_dir = _repo_root() / "docs" / "references" / f"images_{pdf_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
        page_range = [p - 1 for p in pages] if pages else range(total_pages)

        img_global_idx = 0
        for page_idx in page_range:
            if page_idx < 0 or page_idx >= total_pages:
                continue
            page = doc[page_idx]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                w, h = base_image["width"], base_image["height"]
                if w < min_size_px or h < min_size_px:
                    continue
                ext  = base_image["ext"]
                data = base_image["image"]
                fname = f"img_{img_global_idx:03d}_p{page_idx + 1}.{ext}"
                fpath = out_dir / fname
                fpath.write_bytes(data)
                extracted.append({
                    "index":    img_global_idx,
                    "filename": fname,
                    "path":     str(fpath),
                    "page":     page_idx + 1,
                    "width":    w,
                    "height":   h,
                    "format":   ext,
                })
                img_global_idx += 1

    if not extracted:
        return _ok(
            data={"images": [], "pdf": pdf_name},
            message=f"No images found in '{pdf_name}' (pages checked: {list(page_range)}).",
        )

    return _ok(
        data={
            "images":    extracted,
            "count":     len(extracted),
            "pdf":       pdf_name,
            "directory": str(out_dir),
        },
        message=(
            f"Extracted {len(extracted)} image(s) from '{pdf_name}'. "
            "Agent will display them. Tell the user to choose which ones are AFM images "
            "and pass the index to use_pdf_image_as_reference()."
        ),
    )


@mcp.tool()
def use_pdf_image_as_reference(
    pdf_name: str,
    image_index: int,
    class_name: str,
    superclass_name: str,
    prompt: str,
    n_images: int = 1,
) -> dict:
    """
    Use an image previously extracted from a PDF as a visual reference to
    generate synthetic AFM reference images via OpenAI images.edit().

    The model receives the real AFM image from the paper and generates
    n_images synthetic variants with black background, grayscale, and
    top-down view — ready to use as SimuScan reference objects.

    Args:
        pdf_name:       PDF filename (e.g. "nature_afm.pdf"). Used to locate
                        the extracted image in docs/references/images_<pdf_stem>/.
        image_index:    Index of the extracted image (from extract_pdf_images()).
        class_name:     Class label for the new object (e.g. "ribosome").
        superclass_name: Category label (e.g. "protein").
        prompt:         Description to guide the generation. AFM constraints
                        are appended automatically.
        n_images:       Number of synthetic variants to generate (1–5).
    """
    from pathlib import Path as _Path

    pdf_stem = _Path(pdf_name).stem
    img_dir  = _repo_root() / "docs" / "references" / f"images_{pdf_stem}"

    if not img_dir.exists():
        return _err(
            f"No extracted images found for '{pdf_name}'. "
            "Call extract_pdf_images() first."
        )

    candidates = [
        p for p in img_dir.iterdir()
        if p.name.startswith(f"img_{image_index:03d}_")
    ]
    if not candidates:
        return _err(
            f"Image index {image_index} not found in docs/references/images_{pdf_stem}/. "
            "Check the index with extract_pdf_images()."
        )

    raw_ref_path = candidates[0]
    n_images = max(1, min(5, n_images))

    # If a cropped version already exists in ref_objects/, prefer it as reference
    folder_name = f"{superclass_name.strip()}_{class_name.strip()}"
    ref_obj_dir = _repo_root() / "ref_objects" / folder_name
    cropped = sorted(p for p in ref_obj_dir.iterdir()
                     if p.suffix == ".png" and not p.name.startswith("mask_")) \
              if ref_obj_dir.exists() else []
    reference_image = str(cropped[-1]) if cropped else str(raw_ref_path)

    return _ok(
        data={
            "class_name":      class_name.strip(),
            "superclass_name": superclass_name.strip(),
            "prompt":          prompt.strip(),
            "n_images":        n_images,
            "add_to_config":   False,
            "reference_image": reference_image,
        },
        message=(
            f"Ready to generate {n_images} synthetic variant(s) from "
            f"'{Path(reference_image).name}' using images.edit(). "
            "Agent will call OpenAI image API."
        ),
    )


@mcp.tool()
def open_crop_tool(
    pdf_name: str,
    image_index: int,
    class_name: str,
    superclass_name: str,
) -> dict:
    """
    Open an interactive crop dialog so the user can manually select the region
    of interest from a raw PDF image.

    The dialog shows the raw image; the user drags a rectangle over the AFM
    structure of interest and clicks Confirm. The selected region is then:
      1. Converted to grayscale and made square (black padding).
      2. Background zeroed with Otsu thresholding.
      3. Saved to docs/references/images_<pdf_stem>/<superclass>_<class>_<N>.png
         (and the corresponding mask).

    When the user confirms, a [CROP DONE] message is automatically sent back
    to the agent with the saved paths — wait for it before proceeding.

    Args:
        pdf_name:        PDF filename (e.g. "paper.pdf").
        image_index:     Index from extract_pdf_images().
        class_name:      Object class label (e.g. "ribosome").
        superclass_name: Superclass label (e.g. "protein").
    """
    pdf_stem = pdf_name.rsplit(".", 1)[0]
    img_dir  = _repo_root() / "docs" / "references" / f"images_{pdf_stem}"

    if not img_dir.exists():
        return _err(f"No extracted images for '{pdf_name}'. Call extract_pdf_images() first.")

    candidates = [p for p in img_dir.iterdir() if p.name.startswith(f"img_{image_index:03d}_")]
    if not candidates:
        return _err(f"Image index {image_index} not found. Check with extract_pdf_images().")

    # Collect all raw images for this article (for navigation in the dialog)
    all_raw = sorted(
        p for p in img_dir.iterdir()
        if p.name.startswith("img_") and not p.name.startswith("mask_")
    )
    if not all_raw:
        return _err(f"No raw images found for '{pdf_name}'. Call extract_pdf_images() first.")

    # Crops are saved directly to ref_objects/<folder>/ — they ARE reference images
    folder_name = f"{superclass_name.strip()}_{class_name.strip()}"
    out_dir     = _repo_root() / "ref_objects" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return _ok(
        data={
            "raw_images":   [str(p) for p in all_raw],
            "start_index":  min(image_index, len(all_raw) - 1),
            "out_dir":      str(out_dir),
            "prefix":       folder_name,
            "image_folder": folder_name,
        },
        message=(
            f"Opening crop dialog with {len(all_raw)} image(s) from '{pdf_name}'. "
            "User can navigate and save multiple crops. "
            "Wait for [CROP DONE] before proceeding."
        ),
    )


@mcp.tool()
def list_experimental_images(subfolder: str = "") -> dict:
    """
    List AFM images available in data/experimental/ (or a subfolder of it).

    Args:
        subfolder: Optional subfolder inside data/experimental/ to list.
                   Leave empty to list the root and all subfolders.
    """
    exp_root = _repo_root() / "data" / "experimental"
    target   = (exp_root / subfolder) if subfolder else exp_root

    if not target.exists():
        exp_root.mkdir(parents=True, exist_ok=True)
        return _ok(
            data={"images": [], "folders": []},
            message=f"data/experimental/{subfolder} is empty or does not exist.",
        )

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images  = sorted(p for p in target.rglob("*") if p.suffix.lower() in exts)
    folders = sorted({p.parent.relative_to(exp_root) for p in images
                      if p.parent != exp_root})

    return _ok(
        data={
            "images":  [str(p) for p in images],
            "folders": [str(f) for f in folders],
            "count":   len(images),
            "root":    str(exp_root),
        },
        message=f"{len(images)} experimental image(s) found in data/experimental/{subfolder}.",
    )


@mcp.tool()
def open_crop_from_files(
    class_name: str,
    superclass_name: str,
    subfolder: str = "",
) -> dict:
    """
    Open the interactive crop dialog using experimental AFM images from
    data/experimental/ (or a subfolder of it).

    Identical workflow to open_crop_tool but the source images are experimental
    AFM acquisitions instead of PDF-extracted images.
    The user can navigate all images in the folder, draw multiple crops, and
    save them. Crops are saved directly to ref_objects/<superclass>_<class>/
    ready to use with add_image_object().

    Args:
        class_name:      Object class label (e.g. "ribosome").
        superclass_name: Superclass label (e.g. "protein").
        subfolder:       Optional subfolder inside data/experimental/ to use.
                         Leave empty to use all images in data/experimental/.
    """
    exp_root = _repo_root() / "data" / "experimental"
    target   = (exp_root / subfolder) if subfolder else exp_root
    exp_root.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted(p for p in target.rglob("*") if p.suffix.lower() in exts)

    if not images:
        return _err(
            f"No images found in data/experimental/{subfolder}. "
            "Add AFM images there and try again."
        )

    folder_name = f"{superclass_name.strip()}_{class_name.strip()}"
    out_dir     = _repo_root() / "ref_objects" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return _ok(
        data={
            "raw_images":   [str(p) for p in images],
            "start_index":  0,
            "out_dir":      str(out_dir),
            "prefix":       folder_name,
            "image_folder": folder_name,
        },
        message=(
            f"Opening crop dialog with {len(images)} experimental image(s) "
            f"from data/experimental/{subfolder}. "
            "User can navigate and save multiple crops. "
            "Wait for [CROP DONE] before proceeding."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# User generator tools
# ══════════════════════════════════════════════════════════════════════════════

def _generators_dir() -> Path:
    d = _repo_root() / "data" / "user_generators"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_generator_meta(path: Path) -> dict:
    """Extract function names and their docstrings from a .py file."""
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return {"file": path.name, "error": str(e), "functions": []}

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node) or ""
            # Extract param names from signature
            args = [a.arg for a in node.args.args if a.arg != "self"]
            functions.append({
                "name":    node.name,
                "args":    args,
                "docstring": doc[:400] if doc else "",
            })

    module_doc = ast.get_docstring(tree) or ""
    return {
        "file":       path.name,
        "module":     path.stem,
        "module_doc": module_doc[:600] if module_doc else "",
        "functions":  functions,
    }


@mcp.tool()
def list_generators() -> dict:
    """
    List all user-defined generator scripts in data/user_generators/.
    Returns each file's name, module docstring, and available functions
    with their signatures and docstrings.
    """
    files = sorted(_generators_dir().glob("*.py"))
    if not files:
        return _ok(
            data={"generators": []},
            message="No generator scripts found in data/user_generators/.",
        )
    generators = [_parse_generator_meta(f) for f in files]
    return _ok(
        data={"generators": generators, "count": len(generators)},
        message=f"{len(generators)} generator script(s) found.",
    )


@mcp.tool()
def run_generator(
    module_name: str,
    function_name: str,
    class_name: str,
    superclass_name: str,
    n_images: int = 1,
    params: dict | None = None,
) -> dict:
    """
    Run a user-defined generator function and save the results to
    ref_objects/<superclass>_<class>/ as PNG images with binary masks.

    The function must return (image, mask) as uint8 numpy arrays (HxW).
    It is called n_images times — pass a seed in params for reproducibility,
    or leave it out for random variation each call.

    Args:
        module_name:     Filename without .py (e.g. "benzene").
        function_name:   Function name inside the module (e.g. "generate").
        class_name:      Object class label (e.g. "benzene").
        superclass_name: Superclass label (e.g. "molecule").
        n_images:        Number of images to generate (1–20).
        params:          Dict of parameters passed to the function.
                         Leave None to use function defaults.
    """
    import importlib.util, sys

    gen_path = _generators_dir() / f"{module_name}.py"
    if not gen_path.exists():
        return _err(f"Generator '{module_name}.py' not found in data/user_generators/.")

    # Dynamic import
    spec   = importlib.util.spec_from_file_location(module_name, gen_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return _err(f"Error loading '{module_name}.py': {e}")

    fn = getattr(module, function_name, None)
    if fn is None:
        return _err(f"Function '{function_name}' not found in '{module_name}.py'.")

    n_images = max(1, min(20, n_images))
    folder_name = f"{superclass_name.strip()}_{class_name.strip()}"
    out_dir     = _repo_root() / "ref_objects" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(p for p in out_dir.iterdir()
                      if p.suffix == ".png" and not p.name.startswith("mask_"))
    next_idx = len(existing)

    saved = []
    for i in range(n_images):
        try:
            result = fn(params or {})
        except Exception as e:
            return _err(f"Generator function raised an error: {e}")

        if not (isinstance(result, (tuple, list)) and len(result) == 2):
            return _err(
                f"'{function_name}' must return (image, mask) — got {type(result)}."
            )

        image, mask = result
        import numpy as _np
        import cv2 as _cv2

        for arr, name in ((image, "image"), (mask, "mask")):
            if not isinstance(arr, _np.ndarray):
                return _err(f"'{name}' returned by '{function_name}' is not a numpy array.")
            if arr.dtype != _np.uint8:
                return _err(f"'{name}' must be uint8, got {arr.dtype}.")
            if arr.ndim != 2:
                return _err(f"'{name}' must be 2D (HxW grayscale), got shape {arr.shape}.")

        idx       = next_idx + i
        img_name  = f"{folder_name}_{idx:03d}.png"
        mask_name = f"mask_{folder_name}_{idx:03d}.png"

        _cv2.imwrite(str(out_dir / img_name),  image)
        _cv2.imwrite(str(out_dir / mask_name), mask)
        saved.append({"image_file": img_name, "mask_file": mask_name})

    return _ok(
        data={
            "image_folder": folder_name,
            "image_files":  [s["image_file"] for s in saved],
            "saved":        saved,
            "count":        len(saved),
        },
        message=(
            f"Generated {len(saved)} image(s) with '{module_name}.{function_name}' "
            f"→ ref_objects/{folder_name}/."
        ),
    )


@mcp.tool()
def tool_pid_tuner() -> dict:
    """
    Open the interactive PID Topography Tuner window.

    Shows a synthetic circular pillar (512×512) processed through the PID
    feedback simulation.  The window has:
      - Left panel  : original ideal pillar image
      - Right panel : PID-distorted image
      - Bottom plot : central horizontal profile comparing ideal vs PID
      - Parameter controls for all PID and physical tapping-mode parameters

    Use this tool to help the user tune pid_conf values before setting them
    with set_params(section='pid_conf', ...).
    """
    return {"status": "ok", "phantom": True,
            "message": "Opening PID Tuner window."}


@mcp.tool()
def propose_generator(module_name: str, code: str) -> dict:
    """
    Stage a new generator script for user review BEFORE saving it.

    Writes the code to data/user_generators/pending/<module_name>.py and
    returns the full code so the user can inspect it. The script is NOT
    available to run_generator() until the user confirms via save_generator().

    Workflow
    --------
    1. Call propose_generator(module_name, code)   — shows code to user.
    2. User reads and approves (or rejects).
    3. Call save_generator(module_name)             — moves to user_generators/.
       OR discard_generator(module_name)            — deletes the pending file.

    The generator function must accept a dict of params and return
    (image, mask) as uint8 2-D numpy arrays (HxW grayscale).
    """
    pending_dir = _generators_dir() / "pending"
    pending_dir.mkdir(exist_ok=True)
    path = pending_dir / f"{module_name}.py"
    path.write_text(code, encoding="utf-8")
    return _ok(
        data={"module_name": module_name, "code": code, "pending_path": str(path)},
        message=(
            f"Generator '{module_name}.py' staged for review. "
            "Show the full code to the user. "
            "Call save_generator() if they approve, or discard_generator() to cancel."
        ),
    )


@mcp.tool()
def save_generator(module_name: str) -> dict:
    """
    Confirm and permanently save a previously proposed generator script.

    Moves data/user_generators/pending/<module_name>.py →
    data/user_generators/<module_name>.py, making it available to
    list_generators() and run_generator().

    Only call this after the user has explicitly approved the code shown
    by propose_generator().
    """
    pending_path = _generators_dir() / "pending" / f"{module_name}.py"
    if not pending_path.exists():
        return _err(
            f"No pending generator '{module_name}.py'. "
            "Call propose_generator() first."
        )
    dest = _generators_dir() / f"{module_name}.py"
    pending_path.rename(dest)
    return _ok(
        data={"module_name": module_name, "path": str(dest)},
        message=(
            f"Generator '{module_name}.py' saved to data/user_generators/. "
            "Ready to use with run_generator()."
        ),
    )


@mcp.tool()
def discard_generator(module_name: str) -> dict:
    """
    Discard a proposed generator script without saving it.
    Deletes data/user_generators/pending/<module_name>.py.
    """
    pending_path = _generators_dir() / "pending" / f"{module_name}.py"
    if not pending_path.exists():
        return _err(f"No pending generator '{module_name}.py' found.")
    pending_path.unlink()
    return _ok(message=f"Generator '{module_name}.py' discarded.")


# ══════════════════════════════════════════════════════════════════════════════
# RAG tools
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def index_documents(
    force_reindex: bool = False,
    chunk_size: int = 512,
    overlap: int = 64,
) -> dict:
    """
    Index the docs/ folder into the vector database so they can be searched with
    search_documents. Detects changes by content hash: NEW files are indexed,
    CHANGED files are automatically re-indexed, and UNCHANGED files are skipped —
    so calling this after editing a doc picks up the edits without force_reindex.

    Args:
        force_reindex: If True, drop and rebuild the entire index from scratch
                       (rarely needed now that changed files are detected).
        chunk_size:    Approximate character size of each text chunk (default 512).
        overlap:       Overlap in characters between consecutive chunks (default 64).
    """
    return index_docs_folder(
        chunk_size=chunk_size,
        overlap=overlap,
        force_reindex=force_reindex,
    )


@mcp.tool()
def index_document(file_path: str) -> dict:
    """
    Index or re-index a single document file.

    Args:
        file_path: Filename relative to docs/ (e.g. "simuscan_manual.md")
                   or an absolute path.
    """
    return index_single_file(file_path)


@mcp.tool()
def list_documents() -> dict:
    """
    List all documents currently indexed in the vector database,
    along with the total number of stored chunks.
    """
    return list_indexed_documents()


@mcp.tool()
def search_documents(
    query: str,
    n_results: int = 5,
    source_filter: str | None = None,
) -> dict:
    """
    Search the indexed documentation using semantic similarity.
    Returns the most relevant text chunks with their source and relevance score.

    Args:
        query:         Natural-language question or keywords to search for.
        n_results:     Number of chunks to return (default 5).
        source_filter: Optional filename to restrict search to one document
                       (e.g. "parameter_reference.md").
    """
    return _rag_retrieve(
        query=query,
        n_results=n_results,
        source_filter=source_filter,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="stdio")
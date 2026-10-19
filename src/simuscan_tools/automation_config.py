"""
src/simuscan_tools/automation_config.py
────────────────────────────────────────
Module for managing AFM Automation YAML configuration files.

Three operation modes:
  Mode 1 — Scan only       : yolo_enabled=false, zoom_enabled=false
  Mode 2 — Scan + detect   : yolo_enabled=true,  zoom_enabled=false
  Mode 3 — Scan + detect + zoom : yolo_enabled=true, zoom_enabled=true

Two strategies:
  raster        : deterministic grid, supports negative overlap (gap)
  probabilistic : adaptive sampling weighted by per-class detection maps

Grid size:
  If grid_n_x > 0 and grid_n_y > 0, those values define the scan grid
  directly (columns × rows). Otherwise area_to_cover_um is used.
  Real covered area = (n*step + scan_size)²  where step = scan_size*(1-overlap)
  Negative overlap_fraction → gap between frames.
"""

from __future__ import annotations

import copy
import difflib
from pathlib import Path
from typing import Any

import yaml


# ── Path helpers ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _auto_config_dir() -> Path:
    return _repo_root() / "data" / "config" / "conf_automation"


def _default_yaml_path() -> Path:
    return _auto_config_dir() / "automation_default.yaml"


# ── Parameters that store µm values and need metre conversions ────────────────
_UM_TO_M: dict[str, str] = {
    "scan_size_um"   : "scan_size_m",
    "area_to_cover_um": "area_to_cover_m",
    "origin_x_um"    : "origin_x_m",
    "origin_y_um"    : "origin_y_m",
    "max_jump_um"    : "max_jump_m",
    "zoom_max_size_um": "zoom_max_size_m",
}

# Parameters that pass through directly to ExplorationConfig
_DIRECT_PARAMS: list[str] = [
    "gallery_dir",
    "gwy_base_name",
    "scan_direction",
    "scan_rate_hz",
    "scan_resolution_px",
    "yolo_enabled",
    "yolo_model_path",
    "target_classes",
    "yolo_conf_threshold",
    "yolo_overlap_threshold",
    "border_filter_enabled",
    "zoom_enabled",
    "zoom_bbox_factors",
    "zoom_max_rate_hz",
    "zoom_resolution_px",
    "max_large_frames",
    "strategy",
    "overlap_fraction",
    "grid_n_x",
    "grid_n_y",
    "prob_alpha_init",
    "prob_beta_init",
    "prob_spatial_sigma_um",
    "prob_kappa_uncertainty",
    "prob_lambda_move",
    "prob_epsilon_unknown",
    "prob_negative_update_scale",
    "prob_initial_random_n",
    "prob_initial_center_radius_um",
    "isolation_max_bbox_overlap",
    "isolation_min_distance_um",
    "flatten_perc",
    "gwy_poll_interval_s",
    "triplet_model_path",
    "stage_auto_range",
    "stage_x_min_um",
    "stage_x_max_um",
    "stage_y_min_um",
    "stage_y_max_um",
    "stage_edge_margin_um",
]


# ── Type helpers ──────────────────────────────────────────────────────────────

def _type_name(v: Any) -> str:
    if isinstance(v, bool):  return "bool"
    if isinstance(v, int):   return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str):   return "str"
    if isinstance(v, list):  return "list"
    if isinstance(v, dict):  return "dict"
    if v is None:            return "null"
    return type(v).__name__


def _coerce(new_val: Any, original_val: Any) -> tuple[Any, str | None]:
    if original_val is None:
        return new_val, None
    if isinstance(original_val, bool):
        if isinstance(new_val, bool):
            return new_val, None
        return None, f"Expected bool, got {_type_name(new_val)}."
    if isinstance(original_val, int):
        if isinstance(new_val, bool):
            return None, "Expected int, got bool."
        if isinstance(new_val, int):
            return new_val, None
        return None, f"Expected int, got {_type_name(new_val)}."
    if isinstance(original_val, float):
        if isinstance(new_val, bool):
            return None, "Expected float, got bool."
        if isinstance(new_val, (int, float)):
            return float(new_val), None
        return None, f"Expected float, got {_type_name(new_val)}."
    if isinstance(original_val, str):
        if isinstance(new_val, str):
            return new_val, None
        return None, f"Expected str, got {_type_name(new_val)}."
    if isinstance(original_val, list):
        if isinstance(new_val, list):
            return new_val, None
        return None, f"Expected list, got {_type_name(new_val)}."
    if isinstance(original_val, dict):
        if isinstance(new_val, dict):
            return new_val, None
        return None, f"Expected dict, got {_type_name(new_val)}."
    if isinstance(new_val, type(original_val)):
        return new_val, None
    return None, f"Expected {_type_name(original_val)}, got {_type_name(new_val)}."


# ── Result builders ───────────────────────────────────────────────────────────

def _ok(data: Any = None, message: str = "OK") -> dict:
    r: dict = {"status": "ok", "message": message}
    if data is not None:
        r["data"] = data
    return r


def _err(message: str, detail: Any = None) -> dict:
    r: dict = {"status": "error", "message": message}
    if detail is not None:
        r["detail"] = detail
    return r


def _not_found_msg(param: str, valid: Any) -> str:
    """'not found' message with a closest-name suggestion when available."""
    matches = difflib.get_close_matches(param, list(valid), n=1, cutoff=0.6)
    if matches:
        return f"Parameter '{param}' not found. Did you mean '{matches[0]}'?"
    return f"Parameter '{param}' not found."


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class AutomationConfig:
    """
    Manages an AFM Automation YAML configuration.

    Usage
    -----
    cfg = AutomationConfig()                   # loads automation_default.yaml
    cfg = AutomationConfig("my_session.yaml")  # loads existing user file

    Never overwrites automation_default.yaml.
    """

    _DEFAULT_NAME = "automation_default.yaml"

    def __init__(self, filename: str | None = None) -> None:
        self._config_dir = _auto_config_dir()
        self._is_default = filename is None
        self._filename   = filename

        if filename is None:
            self._data = self._load_yaml(_default_yaml_path())
        else:
            path = self._config_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Automation config not found: {path}")
            self._data = self._load_yaml(path)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filename: str | None = None) -> dict:
        target = filename or self._filename
        if target is None:
            return _err(
                "Cannot overwrite automation_default.yaml. "
                "Provide a filename to save as a new file."
            )
        if target == self._DEFAULT_NAME:
            return _err(f"{self._DEFAULT_NAME} is read-only.")

        self._config_dir.mkdir(parents=True, exist_ok=True)
        target_path = self._config_dir / target
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        self._filename   = target
        self._is_default = False
        return _ok(message=f"Automation config saved to {target_path}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all(self) -> dict:
        return _ok(data=copy.deepcopy(self._data))

    def get_param(self, param: str) -> dict:
        if param not in self._data:
            return _err(_not_found_msg(param, self._data.keys()))
        return _ok(data={param: copy.deepcopy(self._data[param])})

    def list_files(self) -> dict:
        files = [p.name for p in self._config_dir.glob("*.yaml") if p.is_file()]
        return _ok(data=sorted(files))

    def summary(self) -> dict:
        d = self._data
        yolo_en = d.get("yolo_enabled", False)
        zoom_en = d.get("zoom_enabled", False)
        if not yolo_en:
            mode = "Mode 1 — Scan only"
        elif not zoom_en:
            mode = "Mode 2 — Scan + YOLO detect"
        else:
            mode = "Mode 3 — Scan + YOLO detect + Zoom"

        nx = d.get("grid_n_x", 0)
        ny = d.get("grid_n_y", 0)
        if nx > 0 and ny > 0:
            grid_info = f"{nx}×{ny} = {nx*ny} positions (explicit grid)"
        else:
            grid_info = f"auto from area {d.get('area_to_cover_um', 10000)}µm²"

        target = d.get("target_classes", {})
        enabled_cls = {k: v for k, v in target.items()
                       if isinstance(v, dict) and v.get("enabled", True)}

        return _ok(data={
            "loaded_file"       : self._filename or f"{self._DEFAULT_NAME} (in-memory)",
            "is_default"        : self._is_default,
            "mode"              : mode,
            "gallery_dir"       : d.get("gallery_dir", ""),
            "yolo_enabled"      : yolo_en,
            "yolo_model_path"   : d.get("yolo_model_path", ""),
            "target_classes"    : target,
            "enabled_classes"   : list(enabled_cls.keys()),
            "zoom_enabled"      : zoom_en,
            "strategy"          : d.get("strategy", "raster"),
            "grid"              : grid_info,
            "scan_size_um"      : d.get("scan_size_um", 50.0),
            "overlap_fraction"  : d.get("overlap_fraction", 0.1),
            "max_jump_um"       : d.get("max_jump_um", 270.0),
            "max_large_frames"  : d.get("max_large_frames", 1000),
            "triplet_model_path": d.get("triplet_model_path", ""),
        })

    # ── Write ─────────────────────────────────────────────────────────────────

    def set_param(self, param: str, value: Any) -> dict:
        """
        Edit a single parameter. Type-safe against the stored value.

        Special behaviour
        -----------------
        - Setting 'yolo_model_path' auto-updates 'target_classes' from the model.
        - Setting 'yolo_enabled=True' warns if yolo_model_path is empty.
        - Setting 'zoom_enabled=True' warns if yolo_enabled is False.
        """
        if param not in self._data:
            return _err(_not_found_msg(param, self._data.keys()))

        coerced, err = _coerce(value, self._data[param])
        if err:
            return _err(
                f"Type mismatch for '{param}': {err}",
                detail={"expected": _type_name(self._data[param]),
                        "got": _type_name(value)},
            )

        self._data[param] = coerced
        msg = f"'{param}' updated to {coerced!r}."

        if param == "yolo_model_path" and coerced:
            classes_msg = self._try_load_yolo_classes(coerced)
            if classes_msg:
                msg += f" {classes_msg}"

        if param == "yolo_enabled" and coerced is True:
            if not self._data.get("yolo_model_path", ""):
                msg += " WARNING: yolo_model_path is empty — set it before running."
        if param == "zoom_enabled" and coerced is True:
            if not self._data.get("yolo_enabled", False):
                msg += " WARNING: zoom_enabled=True but yolo_enabled=False."

        return _ok(message=msg)

    def set_params(self, updates: dict[str, Any]) -> dict:
        results = {}
        for param, value in updates.items():
            results[param] = self.set_param(param, value)
        failed = {p: r.get("message", "failed")
                  for p, r in results.items() if r["status"] != "ok"}
        if not failed:
            return _ok(data=results)
        applied = [p for p in results if p not in failed]
        msg = "Failed: " + "; ".join(f"{p} — {m}" for p, m in failed.items())
        if applied:
            msg += f". Applied OK: {applied}"
        return _err(msg, detail=results)

    # ── Conversion to ExplorationConfig dict ──────────────────────────────────

    def to_exploration_dict(self, config_yaml_path: str = "") -> dict:
        """
        Build the kwargs dict expected by ExplorationConfig in automation.py.
        Converts µm → metres.
        """
        d = self._data
        # Keys must match ExplorationConfig field names exactly so that
        # run_exploration() can pass them as **kwargs without losing values.
        # No unit conversion here — ExplorationConfig stores µm and converts
        # internally via properties (scan_size_m, area_to_cover_m2, etc.).
        return {
            "gallery_dir"           : d.get("gallery_dir", ""),
            "gwy_base_name"         : d.get("gwy_base_name", "scan"),
            "scan_direction"        : d.get("scan_direction", "forward"),
            "origin_x_m"            : d.get("origin_x_um", 0.0) * 1e-6,
            "origin_y_m"            : d.get("origin_y_um", 0.0) * 1e-6,
            "scan_size_um"          : d.get("scan_size_um", 50.0),
            "scan_rate_hz"          : d.get("scan_rate_hz", 100.0),
            "scan_resolution_px"    : d.get("scan_resolution_px", 256),
            # Grid
            "grid_n_x"              : d.get("grid_n_x", 0),
            "grid_n_y"              : d.get("grid_n_y", 0),
            "area_to_cover_um"      : d.get("area_to_cover_um", 10000.0),
            "overlap_fraction"      : d.get("overlap_fraction", 0.1),
            # Strategy
            "strategy"              : d.get("strategy", "raster"),
            "max_jump_um"           : d.get("max_jump_um", 270.0),
            # Probabilistic (defaults tuned to the validated simulation)
            "prob_alpha_init"              : d.get("prob_alpha_init", 1.0),
            "prob_beta_init"               : d.get("prob_beta_init", 1.0),
            "prob_spatial_sigma_um"        : d.get("prob_spatial_sigma_um", 90.0),
            "prob_kappa_uncertainty"       : d.get("prob_kappa_uncertainty", 0.6),
            "prob_lambda_move"             : d.get("prob_lambda_move", 0.018),
            "prob_epsilon_unknown"         : d.get("prob_epsilon_unknown", 0.10),
            "prob_negative_update_scale"   : d.get("prob_negative_update_scale", 1.4),
            "prob_initial_random_n"        : d.get("prob_initial_random_n", 5),
            "prob_initial_center_radius_um": d.get("prob_initial_center_radius_um", 200.0),
            # Isolation filter
            "isolation_max_bbox_overlap"   : d.get("isolation_max_bbox_overlap", 0.3),
            "isolation_min_distance_um"    : d.get("isolation_min_distance_um", 0.0),
            # YOLO
            "yolo_enabled"          : d.get("yolo_enabled", False),
            "yolo_model_path"       : d.get("yolo_model_path", ""),
            "target_classes"        : d.get("target_classes", {}),
            "yolo_conf_threshold"   : d.get("yolo_conf_threshold", 0.7),
            "yolo_overlap_threshold": d.get("yolo_overlap_threshold", 0.3),
            "border_filter_enabled" : d.get("border_filter_enabled", True),
            # Zoom
            "zoom_enabled"          : d.get("zoom_enabled", False),
            "zoom_resolution_px"    : d.get("zoom_resolution_px", 512),
            "zoom_max_rate_hz"      : d.get("zoom_max_rate_hz", 2.0),
            "zoom_max_size_um"      : d.get("zoom_max_size_um", 25.0),
            "zoom_bbox_factors"     : d.get("zoom_bbox_factors", [
                {"min_bbox_um": 5.0, "factor": 1.5},
                {"min_bbox_um": 2.0, "factor": 2.0},
                {"min_bbox_um": 0.0, "factor": 3.0},
            ]),
            # Stopping
            "max_large_frames"      : d.get("max_large_frames", 1000),
            # Misc
            "flatten_perc"          : d.get("flatten_perc", 40.0),
            "gwy_poll_interval_s"   : d.get("gwy_poll_interval_s", 2.0),
            "triplet_model_path"    : d.get("triplet_model_path", ""),
            # Stage travel limits
            "stage_auto_range"      : d.get("stage_auto_range", True),
            "stage_x_min_um"        : d.get("stage_x_min_um", 0.0),
            "stage_x_max_um"        : d.get("stage_x_max_um", 0.0),
            "stage_y_min_um"        : d.get("stage_y_min_um", 0.0),
            "stage_y_max_um"        : d.get("stage_y_max_um", 0.0),
            "stage_edge_margin_um"  : d.get("stage_edge_margin_um", 300.0),
            "config_yaml_path"      : config_yaml_path,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _try_load_yolo_classes(self, model_path: str) -> str:
        """Auto-populate target_classes from the YOLO model file."""
        try:
            from src.trainer.yolo_visualizer import get_model_info
            result = get_model_info(model_path)
            if result.get("status") == "ok":
                classes = result["data"].get("class_names", [])
                if classes:
                    existing = self._data.get("target_classes", {})
                    # Keep existing n/enabled for classes already configured;
                    # add new ones with defaults
                    updated = {}
                    for cls in classes:
                        if cls in existing:
                            updated[cls] = existing[cls]
                        else:
                            updated[cls] = {"n": 10, "enabled": True}
                    self._data["target_classes"] = updated
                    return (
                        f"target_classes auto-updated from model: {classes}. "
                        "Use set_param('target_classes', {{...}}) to adjust n or enable/disable."
                    )
                return "Model loaded but no class names found — target_classes unchanged."
            return f"Could not read model info: {result.get('message', 'unknown error')}."
        except Exception as exc:
            return f"Could not auto-load classes (non-critical): {exc}"

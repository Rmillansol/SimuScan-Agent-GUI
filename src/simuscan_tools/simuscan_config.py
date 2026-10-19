"""
simuscan_config.py
──────────────────
Module for managing SimuScan YAML configuration files.

Sections
--------
- config          : editable key-value parameters
- tip_conf        : editable key-value parameters
- substrate_conf  : editable key-value parameters
- artifacts_conf  : editable key-value parameters
- domains_conf    : editable key-value parameters (grain/domain generator)
- shapes          : read-only (fixed, copied from default)
- objects         : list of objects identified by `label`;
                    each object's parameters are editable

All outputs are plain dicts serialisable to JSON so that they
can be used directly as MCP tool responses consumed by an LLM.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# ── Path helpers ──────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    """Return the project root (the folder that contains src/ and data/)."""
    return Path(__file__).resolve().parent.parent.parent


def _default_yaml_path() -> Path:
    return _repo_root() / "data" / "config" / "conf_simuscan" / "default.yaml"


def _config_dir() -> Path:
    return _repo_root() / "data" / "config" / "conf_simuscan"


def _stl_dir() -> Path:
    return _repo_root() / "data" / "STL_files"


# ── Read-only section names ────────────────────────────────────────────────────

_READ_ONLY_SECTIONS = {"shapes"}
_EDITABLE_SECTIONS  = {"config", "tip_conf", "substrate_conf", "artifacts_conf",
                       "pid_conf", "domains_conf"}
_VALID_SHAPE_NAMES: set[str] = set()   # populated on load


# ── Type helpers ──────────────────────────────────────────────────────────────

def _type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if v is None:
        return "null"
    return type(v).__name__


def _coerce(new_val: Any, original_val: Any) -> tuple[Any, str | None]:
    """
    Try to keep *new_val* compatible with the type of *original_val*.
    Returns (coerced_value, error_message_or_None).

    Special case: if original_val is None the new value is accepted as-is
    (type is inferred from the value itself); further semantic validation
    is done by the caller.
    """
    if original_val is None:
        # Accept any value; semantic checks happen in caller
        return new_val, None

    orig_type = type(original_val)

    # bool must be checked before int (bool is subclass of int in Python)
    if isinstance(original_val, bool):
        if isinstance(new_val, bool):
            return new_val, None
        return None, (
            f"Expected bool, got {_type_name(new_val)}."
        )

    if isinstance(original_val, int):
        if isinstance(new_val, bool):
            return None, "Expected int, got bool."
        if isinstance(new_val, int):
            return new_val, None
        return None, f"Expected int, got {_type_name(new_val)}."

    if isinstance(original_val, float):
        # Accept int→float coercion
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

    # fallback: same type required
    if isinstance(new_val, orig_type):
        return new_val, None
    return None, (
        f"Expected {_type_name(original_val)}, got {_type_name(new_val)}."
    )


# ── Result builders ───────────────────────────────────────────────────────────

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


# ── Main class ────────────────────────────────────────────────────────────────

class SimuScanConfig:
    """
    Manages a SimuScan YAML configuration.

    Usage
    -----
    cfg = SimuScanConfig()                  # loads default.yaml
    cfg = SimuScanConfig("my_config.yaml")  # loads an existing user file

    All public methods return a JSON-serialisable dict with keys:
        "status"  : "ok" | "error"
        "message" : human-readable description
        "data"    : payload (present on success when there is data to return)
    """

    # ── Construction & loading ────────────────────────────────────────────────

    def __init__(self, filename: str | None = None) -> None:
        """
        Load configuration.

        Parameters
        ----------
        filename : str | None
            YAML file name (not full path) inside data/config/.
            If None the default.yaml is loaded (read-only origin).
        """
        self._default_path = _default_yaml_path()
        self._config_dir   = _config_dir()
        self._stl_dir      = _stl_dir()

        self._is_default   = filename is None
        self._filename     = filename  # None → working on a copy of default

        # Load default first to get shape names
        raw_default = self._load_yaml(self._default_path)
        global _VALID_SHAPE_NAMES
        _VALID_SHAPE_NAMES = {s["name"] for s in raw_default.get("shapes", [])}

        if filename is None:
            self._data = copy.deepcopy(raw_default)
        else:
            path = self._config_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            self._data = self._load_yaml(path)

        # Build fast lookup: label → index in objects list
        self._obj_index: dict[str, int] = {
            obj["label"]: i
            for i, obj in enumerate(self._data.get("objects", []))
        }

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_temp(self) -> dict:
        """
        Save the current configuration to data/config/temp.yaml,
        overwriting it every time it is called.
        Useful for quick previews without committing to a permanent file.
        """
        return self._write_yaml("temp.yaml")

    def save(self, filename: str | None = None) -> dict:
        """
        Save the current configuration to data/config/<filename>.

        Parameters
        ----------
        filename : str | None
            Target file name.  If None the file that was loaded is
            overwritten (only allowed for user files, never for default.yaml).

        Returns
        -------
        dict  {"status": "ok"|"error", ...}
        """
        target_name = filename or self._filename

        if target_name is None:
            return _err(
                "Cannot overwrite default.yaml. "
                "Provide a filename to save as a new user file."
            )
        if target_name == "default.yaml":
            return _err("default.yaml is read-only and cannot be overwritten.")

        return self._write_yaml(target_name)

    def _write_yaml(self, filename: str) -> dict:
        """Write current config to data/config/<filename>."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        target_path = self._config_dir / filename

        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self._data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        if filename != "temp.yaml":
            self._filename   = filename
            self._is_default = False

        return _ok(message=f"Configuration saved to {target_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # READ methods
    # ══════════════════════════════════════════════════════════════════════════

    def get_section(self, section: str) -> dict:
        """
        Return all parameters of a top-level section.

        Parameters
        ----------
        section : str
            One of: config, tip_conf, substrate_conf, artifacts_conf, domains_conf,
                    shapes, objects.
        """
        if section not in self._data:
            return _err(f"Section '{section}' not found.")
        return _ok(data=copy.deepcopy(self._data[section]))

    def get_param(self, section: str, param: str) -> dict:
        """
        Return the value of a single parameter inside an editable section.

        Parameters
        ----------
        section : str  One of the four editable flat sections.
        param   : str  Parameter name.
        """
        if section in _READ_ONLY_SECTIONS:
            return _err(f"Section '{section}' is read-only; use get_section.")
        if section not in _EDITABLE_SECTIONS:
            return _err(
                f"'{section}' is not a flat editable section. "
                "For objects use get_object / get_object_param."
            )
        sec = self._data.get(section)
        if sec is None:
            return _err(f"Section '{section}' not found.")
        if param not in sec:
            return _err(f"Parameter '{param}' not found in section '{section}'.")
        return _ok(data={param: copy.deepcopy(sec[param])})

    def list_objects(self) -> dict:
        """Return a list of all object labels and their enabled status."""
        objs = self._data.get("objects", [])
        summary = [
            {"label": o["label"], "shape": o.get("shape"), "enable": o.get("enable")}
            for o in objs
        ]
        return _ok(data=summary)

    def get_object(self, label: str) -> dict:
        """Return all parameters of a specific object by label."""
        if label not in self._obj_index:
            return _err(f"Object with label '{label}' not found.")
        idx  = self._obj_index[label]
        return _ok(data=copy.deepcopy(self._data["objects"][idx]))

    def get_object_param(self, label: str, param: str) -> dict:
        """Return the value of a single parameter of an object."""
        if label not in self._obj_index:
            return _err(f"Object with label '{label}' not found.")
        obj = self._data["objects"][self._obj_index[label]]
        if param not in obj:
            return _err(f"Parameter '{param}' not found in object '{label}'.")
        return _ok(data={param: copy.deepcopy(obj[param])})

    def list_shapes(self) -> dict:
        """Return the list of valid shape names (read-only)."""
        return _ok(data=sorted(_VALID_SHAPE_NAMES))

    def list_files(self) -> dict:
        """Return the list of YAML config files available in data/config/."""
        files = [
            p.name
            for p in self._config_dir.glob("*.yaml")
            if p.is_file()
        ]
        return _ok(data=sorted(files))

    def list_stl_files(self) -> dict:
        """Return the list of STL files available in data/STL_files/."""
        if not self._stl_dir.exists():
            return _ok(data=[], message="STL directory does not exist yet.")
        files = [p.name for p in self._stl_dir.glob("*.stl") if p.is_file()]
        return _ok(data=sorted(files))

    # ══════════════════════════════════════════════════════════════════════════
    # WRITE methods — flat editable sections
    # ══════════════════════════════════════════════════════════════════════════

    def set_param(self, section: str, param: str, value: Any) -> dict:
        """
        Edit a parameter in one of the four editable flat sections.

        Type-safety: the new value must match the type of the current value.
        The section 'shapes' cannot be edited.

        Parameters
        ----------
        section : str
        param   : str
        value   : Any  Must match the type already stored.
        """
        if section in _READ_ONLY_SECTIONS:
            return _err(f"Section '{section}' is read-only and cannot be edited.")
        if section not in _EDITABLE_SECTIONS:
            return _err(
                f"'{section}' is not a flat editable section. "
                "For objects use set_object_param."
            )
        sec = self._data.get(section)
        if sec is None:
            return _err(f"Section '{section}' not found.")
        if param not in sec:
            return _err(f"Parameter '{param}' not found in section '{section}'.")

        coerced, err = _coerce(value, sec[param])
        if err:
            return _err(
                f"Type mismatch for '{section}.{param}': {err}",
                detail={"expected": _type_name(sec[param]), "got": _type_name(value)},
            )

        sec[param] = coerced
        return _ok(message=f"'{section}.{param}' updated to {coerced!r}.")

    def set_params(self, section: str, updates: dict[str, Any]) -> dict:
        """
        Edit multiple parameters in one flat editable section at once.

        Parameters
        ----------
        section : str
        updates : dict  {param_name: new_value, ...}

        Returns a summary dict with per-parameter results.
        """
        results = {}
        for param, value in updates.items():
            results[param] = self.set_param(section, param, value)
        all_ok = all(r["status"] == "ok" for r in results.values())
        return _ok(data=results) if all_ok else _err(
            "Some parameters could not be updated.", detail=results
        )

    # ══════════════════════════════════════════════════════════════════════════
    # WRITE methods — objects section
    # ══════════════════════════════════════════════════════════════════════════

    def set_object_param(self, label: str, param: str, value: Any) -> dict:
        """
        Edit a single parameter of an object identified by its label.

        Special validations
        -------------------
        - param='shape'      : value must be in the valid shape names.
        - param='stl_files'  : list of str; each file must exist in data/STL_files/.
        - param='stl_weights': list of float; len must match stl_files; must sum to 1.0.
        - When shape=='STL'  : stl_files and stl_weights must both be non-None lists.
        """
        if label not in self._obj_index:
            return _err(f"Object with label '{label}' not found.")

        obj = self._data["objects"][self._obj_index[label]]

        # Alias: the LLM sometimes sends "enabled" instead of "enable"
        if param == "enabled":
            param = "enable"

        if param not in obj:
            return _err(f"Parameter '{param}' not found in object '{label}'.")

        # ── shape validation ──────────────────────────────────────────────────
        if param == "shape":
            if not isinstance(value, str):
                return _err("'shape' must be a string.")
            # "image", "STL" and "domains" are valid renderer types that are
            # not listed in the shapes catalog (they have their own renderer)
            _self_rendered = {"image", "STL", "domains"}
            if value not in _VALID_SHAPE_NAMES and value not in _self_rendered:
                return _err(
                    f"'{value}' is not a valid shape name.",
                    detail={"valid_shapes": sorted(_VALID_SHAPE_NAMES | _self_rendered)},
                )
            # If switching away from STL, clear stl fields
            obj[param] = value
            if value != "STL":
                obj["stl_files"]   = None
                obj["stl_weights"] = None
            return _ok(message=f"Object '{label}'.shape updated to '{value}'.")

        # ── stl_files validation ──────────────────────────────────────────────
        if param == "stl_files":
            err = self._validate_stl_files(value)
            if err:
                return _err(err)
            # Check weight list length consistency if weights already set
            weights = obj.get("stl_weights")
            if weights is not None and len(weights) != len(value):
                return _err(
                    f"stl_files has {len(value)} entries but stl_weights "
                    f"has {len(weights)}. Update stl_weights first or together."
                )
            obj[param] = value
            return _ok(message=f"Object '{label}'.stl_files updated.")

        # ── stl_weights validation ────────────────────────────────────────────
        if param == "stl_weights":
            err = self._validate_stl_weights(value)
            if err:
                return _err(err)
            files = obj.get("stl_files")
            if files is not None and len(files) != len(value):
                return _err(
                    f"stl_weights has {len(value)} entries but stl_files "
                    f"has {len(files)}. Lengths must match."
                )
            obj[param] = value
            return _ok(message=f"Object '{label}'.stl_weights updated.")

        # ── general parameter ─────────────────────────────────────────────────
        coerced, err = _coerce(value, obj[param])
        if err:
            return _err(
                f"Type mismatch for object '{label}'.{param}: {err}",
                detail={"expected": _type_name(obj[param]), "got": _type_name(value)},
            )

        obj[param] = coerced

        # Post-edit: if shape is STL and stl_files/weights are missing, warn
        if obj.get("shape") == "STL":
            warning = self._stl_completeness_warning(obj)
            if warning:
                return _ok(
                    message=f"Object '{label}'.{param} updated. WARNING: {warning}"
                )

        return _ok(message=f"Object '{label}'.{param}' updated to {coerced!r}.")

    def set_object_params(self, label: str, updates: dict[str, Any]) -> dict:
        """
        Edit multiple parameters of an object at once.

        Parameters
        ----------
        label   : str
        updates : dict  {param_name: new_value, ...}
        """
        # If stl_files and stl_weights are updated together, validate jointly
        if "stl_files" in updates and "stl_weights" in updates:
            files   = updates["stl_files"]
            weights = updates["stl_weights"]
            err = self._validate_stl_files(files) or self._validate_stl_weights(weights)
            if err:
                return _err(err)
            if len(files) != len(weights):
                return _err(
                    f"stl_files ({len(files)} items) and "
                    f"stl_weights ({len(weights)} items) must have equal length."
                )

        results = {}
        for param, value in updates.items():
            results[param] = self.set_object_param(label, param, value)

        all_ok = all(r["status"] == "ok" for r in results.values())
        return _ok(data=results) if all_ok else _err(
            "Some parameters could not be updated.", detail=results
        )

    def rename_object(self, old_label: str, new_label: str) -> dict:
        """
        Rename an object's label (the class name used in COCO/YOLO annotations).

        Updates both the object's 'label' field and the internal label→index
        lookup so the config stays consistent. Use this instead of
        set_object_param(label, 'label', ...), which would corrupt the index.
        """
        old_label = str(old_label).strip()
        new_label = str(new_label).strip()

        if not new_label:
            return _err("new_label must not be empty.")
        if old_label not in self._obj_index:
            return _err(f"Object with label '{old_label}' not found.")
        if new_label == old_label:
            return _ok(message=f"Label '{old_label}' unchanged.")
        if new_label in self._obj_index:
            return _err(
                f"Label '{new_label}' is already used by another object. "
                "Choose a unique name."
            )

        idx = self._obj_index[old_label]
        self._data["objects"][idx]["label"] = new_label
        del self._obj_index[old_label]
        self._obj_index[new_label] = idx
        return _ok(message=f"Object label '{old_label}' renamed to '{new_label}'.")

    # ══════════════════════════════════════════════════════════════════════════
    # ADD object
    # ══════════════════════════════════════════════════════════════════════════

    def add_stl_object(
        self,
        stl_files: list[str],
        label: str | None = None,
        stl_weights: list[float] | None = None,
        template_label: str | None = None,
    ) -> dict:
        """
        Add a new STL-based object to the objects section.

        The new object is a deep copy of a template object with shape and
        STL fields overwritten according to the rules below.

        Parameters
        ----------
        stl_files : list[str]
            Non-empty list of .stl file names that must exist in data/STL_files/.
        label : str | None
            Unique label for the new object.  If None or already taken, a
            unique name is auto-generated (User01, User02, …).
        stl_weights : list[float] | None
            One weight per stl_files entry; must sum to 1.0.
            If None, equal weights are assigned automatically.
        template_label : str | None
            Label of the object to copy parameters from.
            If None: use the first enabled object; if none is enabled,
            use the first object in the list.

        Returns
        -------
        dict  {"status": "ok"|"error", ...}
        """
        # ── 1. Validate stl_files ─────────────────────────────────────────────
        err = self._validate_stl_files(stl_files)
        if err:
            return _err(err)

        # ── 2. Resolve / auto-generate stl_weights ────────────────────────────
        if stl_weights is None:
            n = len(stl_files)
            w = round(1.0 / n, 10)
            stl_weights = [w] * n
            # Fix floating-point rounding so sum is exactly 1.0
            stl_weights[-1] = round(1.0 - sum(stl_weights[:-1]), 10)
        else:
            err = self._validate_stl_weights(stl_weights)
            if err:
                return _err(err)
            if len(stl_weights) != len(stl_files):
                return _err(
                    f"stl_files has {len(stl_files)} entries but "
                    f"stl_weights has {len(stl_weights)}."
                )

        # ── 3. Resolve label ──────────────────────────────────────────────────
        label = self._resolve_label(label)

        # ── 4. Pick template object ───────────────────────────────────────────
        objs = self._data.get("objects", [])
        if not objs:
            return _err("No objects in configuration to use as a template.")

        if template_label is not None:
            if template_label not in self._obj_index:
                return _err(f"Template object '{template_label}' not found.")
            template = objs[self._obj_index[template_label]]
        else:
            # First enabled object, fallback to first object
            template = next(
                (o for o in objs if o.get("enable")), objs[0]
            )

        # ── 5. Build new object ───────────────────────────────────────────────
        new_obj = copy.deepcopy(template)
        new_obj["label"]       = label
        new_obj["shape"]       = "STL"
        new_obj["stl_files"]   = stl_files
        new_obj["stl_weights"] = stl_weights
        new_obj["enable"]      = True

        # ── 6. Insert and update index ────────────────────────────────────────
        self._data["objects"].append(new_obj)
        self._obj_index[label] = len(self._data["objects"]) - 1

        return _ok(
            data={
                "label"      : label,
                "shape"      : "STL",
                "stl_files"  : stl_files,
                "stl_weights": stl_weights,
                "template"   : template["label"],
                "enable"     : True,
            },
            message=f"Object '{label}' created from template '{template['label']}'.",
        )

    def add_image_object(
        self,
        label:          str,
        image_folder:   str,
        image_files:    list,
        image_weights:  list        = None,
        n_objects:      int         = 5,
        size_shape:     list        = None,
        height:         list        = None,
        template_label: str | None  = None,
    ) -> dict:
        """
        Add a new image-based object (shape="image") to the objects section.

        Parameters
        ----------
        label          : Unique label for the new object.
        image_folder   : Subfolder inside ref_objects/image_files/ (format: superclass_class).
        image_files    : List of PNG filenames inside image_folder,
                         e.g. ["protein_ribosome_000.png", "protein_ribosome_001.png"].
                         Companion masks must exist as mask_<filename> in the same folder.
        image_weights  : Optional sampling weights per file (must sum to 1.0).
                         If None, all files are sampled uniformly.
        n_objects      : Mean objects per image.
        size_shape     : [width_px, height_px].
        height         : [min, max] normalised height range.
        template_label : Object to copy other params from (optional).
        """
        if size_shape is None:
            size_shape = [40, 40]
        if height is None:
            height = [0.8, 1.0]

        if not image_files:
            return _err("image_files list must not be empty.")

        # Validate weights if provided
        if image_weights is not None:
            if len(image_weights) != len(image_files):
                return _err(
                    f"image_weights length ({len(image_weights)}) must match "
                    f"image_files length ({len(image_files)})."
                )
            if abs(sum(image_weights) - 1.0) > 1e-3:
                return _err("image_weights must sum to 1.0.")

        # Resolve label uniqueness
        label = self._resolve_label(label)

        # Pick template
        objs = self._data.get("objects", [])
        if not objs:
            return _err("No objects in configuration to use as a template.")
        if template_label is not None:
            if template_label not in self._obj_index:
                return _err(f"Template object '{template_label}' not found.")
            template = objs[self._obj_index[template_label]]
        else:
            template = next((o for o in objs if o.get("enable")), objs[0])

        # Build new object
        new_obj = copy.deepcopy(template)
        new_obj["label"]          = label
        new_obj["shape"]          = "image"
        new_obj["image_folder"]   = image_folder
        new_obj["image_files"]    = image_files
        new_obj["image_weights"]  = image_weights
        new_obj["n_objects"]      = n_objects
        new_obj["size_shape"]     = size_shape
        new_obj["height"]         = height
        new_obj["enable"]         = True
        # Clear legacy single-file and STL fields
        new_obj.pop("image_file",  None)
        new_obj.pop("mask_file",   None)
        new_obj.pop("stl_files",   None)
        new_obj.pop("stl_weights", None)

        self._data["objects"].append(new_obj)
        self._obj_index[label] = len(self._data["objects"]) - 1

        return _ok(
            data={
                "label":         label,
                "shape":         "image",
                "image_folder":  image_folder,
                "image_files":   image_files,
                "image_weights": image_weights,
                "n_objects":     n_objects,
                "size_shape":    size_shape,
                "height":        height,
                "template":      template["label"],
            },
            message=(
                f"Image object '{label}' added with {len(image_files)} file(s) "
                f"from folder '{image_folder}'. "
                "Remember to save_config() before generating."
            ),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Introspection helpers (useful for LLM tool discovery)
    # ══════════════════════════════════════════════════════════════════════════

    def describe_section(self, section: str) -> dict:
        """
        Return parameter names and their current types for a section.
        Useful for an LLM to discover what can be edited.
        """
        if section not in self._data:
            return _err(f"Section '{section}' not found.")
        raw = self._data[section]
        if isinstance(raw, list):
            # objects or shapes
            if section == "shapes":
                return _ok(
                    data={"note": "read-only", "shape_names": sorted(_VALID_SHAPE_NAMES)}
                )
            # objects list
            return _ok(
                data=[
                    {k: _type_name(v) for k, v in obj.items()} for obj in raw
                ]
            )
        return _ok(
            data={k: {"type": _type_name(v), "value": v} for k, v in raw.items()}
        )

    def summary(self) -> dict:
        """
        Return a high-level summary of the loaded configuration.
        """
        objs    = self._data.get("objects", [])
        enabled = [o["label"] for o in objs if o.get("enable")]
        return _ok(
            data={
                "loaded_file": self._filename or "default.yaml (in-memory copy)",
                "is_default" : self._is_default,
                "image_size" : self._data.get("config", {}).get("image_size"),
                "n_images"   : self._data.get("config", {}).get("n_images"),
                "total_objects"  : len(objs),
                "enabled_objects": enabled,
                "valid_shapes"   : sorted(_VALID_SHAPE_NAMES),
            }
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _resolve_label(self, label: str | None) -> str:
        """Return *label* if unique, otherwise auto-generate User01, User02 …"""
        if label and label not in self._obj_index:
            return label
        # Auto-generate
        i = 1
        while True:
            candidate = f"User{i:02d}"
            if candidate not in self._obj_index:
                return candidate
            i += 1

    def _validate_stl_files(self, value: Any) -> str | None:
        """Return an error string, or None if valid."""
        if not isinstance(value, list) or not value:
            return "stl_files must be a non-empty list of file name strings."
        for item in value:
            if not isinstance(item, str):
                return f"stl_files entries must be strings, got {_type_name(item)}."
            stl_path = self._stl_dir / item
            if not stl_path.exists():
                return (
                    f"STL file '{item}' not found in {self._stl_dir}. "
                    "Use list_stl_files() to see available files."
                )
        return None

    def _validate_stl_weights(self, value: Any) -> str | None:
        """Return an error string, or None if valid."""
        if not isinstance(value, list) or not value:
            return "stl_weights must be a non-empty list of floats."
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return f"stl_weights entries must be floats, got {_type_name(item)}."
        total = sum(float(w) for w in value)
        if abs(total - 1.0) > 1e-6:
            return (
                f"stl_weights must sum to 1.0 (current sum: {total:.6f})."
            )
        return None

    def _stl_completeness_warning(self, obj: dict) -> str | None:
        """Return a warning message if STL shape is missing required fields."""
        files   = obj.get("stl_files")
        weights = obj.get("stl_weights")
        if not files or not weights:
            return (
                "Shape is 'STL' but stl_files or stl_weights are not set. "
                "Set both before saving."
            )
        return None
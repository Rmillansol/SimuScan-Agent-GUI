from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from openai import OpenAI


# =============================================================================
# SIMUSCAN PHASE 1 CONTROLLER
# =============================================================================
# What this file does
# -------------------
# Phase 1 supports:
#   - Query current state
#   - Enable/disable existing objects
#   - Modify properties in: config, objects, tip_conf, substrate_conf, artifacts_conf
#   - Generate preview runs using a temporary YAML config
#   - Confirm/reject previews
#   - Generate final dataset
#
# What this file intentionally does NOT do yet
# --------------------------------------------
#   - Create new shapes
#   - Create new objects
#   - Add STL-based objects
#   - Add free-form section geometry
#
# Design idea
# -----------
# The LLM does NOT write the final YAML directly.
# The LLM returns a structured JSON command.
# This script:
#   1) loads the current YAML config as a Python dict,
#   2) sends a compact state summary + strict command schema instructions to the model,
#   3) receives a JSON command,
#   4) validates and applies the command,
#   5) writes a temporary/final YAML,
#   6) optionally executes SimuScan,
#   7) stores session state.
#
# Notes
# -----
# - This implementation uses a strong prompt contract plus local validation.
# - It is ready to evolve later toward true response_format / json_schema output.
# - For now it is designed to be practical and editable.
# =============================================================================


# =============================================================================
# USER SETTINGS
# =============================================================================
MODEL_NAME = "gpt-5"
STATE_FILENAME = ".simuscan_phase1_state.json"
DEFAULT_PREVIEW_COUNT = 4
DEFAULT_DATASET_COUNT = 100
MAX_STATE_SUMMARY_OBJECTS = 50


# =============================================================================
# TYPES
# =============================================================================
Intent = Literal[
    "query_state",
    "modify_config",
    "modify_and_preview",
    "generate_preview",
    "confirm_preview",
    "reject_preview",
    "generate_dataset",
    "request_clarification",
]

Operation = Literal[
    "enable_object",
    "disable_object",
    "set_object_property",
    "set_general_property",
    "set_tip_property",
    "set_substrate_property",
    "set_artifact_property",
]


@dataclass
class SessionState:
    base_config_path: str
    current_config_path: str
    working_dir: str
    simuscan_executable: Optional[str] = None
    preview_output_dir: Optional[str] = None
    final_output_dir: Optional[str] = None
    config_dirty: bool = False
    preview_generated: bool = False
    preview_accepted: bool = False
    dataset_generated: bool = False
    last_prompt: Optional[str] = None
    last_preview_paths: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_config_path": self.base_config_path,
            "current_config_path": self.current_config_path,
            "working_dir": self.working_dir,
            "simuscan_executable": self.simuscan_executable,
            "preview_output_dir": self.preview_output_dir,
            "final_output_dir": self.final_output_dir,
            "config_dirty": self.config_dirty,
            "preview_generated": self.preview_generated,
            "preview_accepted": self.preview_accepted,
            "dataset_generated": self.dataset_generated,
            "last_prompt": self.last_prompt,
            "last_preview_paths": self.last_preview_paths,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(**data)


# =============================================================================
# YAML / CONFIG HELPERS
# =============================================================================
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data

def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def load_state(state_path: str | Path) -> Optional[SessionState]:
    p = Path(state_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SessionState.from_dict(data)

def save_state(state: SessionState, state_path: str | Path) -> None:
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)

def find_objects(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = config.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("config['objects'] must be a list")
    return objects

def find_object_by_label(config: Dict[str, Any], label: str) -> Dict[str, Any]:
    for obj in find_objects(config):
        if obj.get("label") == label:
            return obj
    raise KeyError(f"Object with label '{label}' not found")

def list_object_labels(config: Dict[str, Any]) -> List[str]:
    return [obj.get("label") for obj in find_objects(config) if isinstance(obj, dict) and "label" in obj]

def list_shape_names(config: Dict[str, Any]) -> List[str]:
    shapes = config.get("shapes", [])
    if not isinstance(shapes, list):
        return []
    result = []
    for shape in shapes:
        if isinstance(shape, dict) and "name" in shape:
            result.append(shape["name"])
    return result

# =============================================================================
# STATE SUMMARY FOR THE MODEL
# =============================================================================
def summarize_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    keys_of_interest = [
        "label",
        "enable",
        "shape",
        "n_objects",
        "size_shape",
        "height",
        "apply_colony",
        "stl_files",
    ]
    out: Dict[str, Any] = {}
    for k in keys_of_interest:
        if k in obj:
            out[k] = obj[k]
    return out

def build_state_summary(config: Dict[str, Any], session: SessionState) -> Dict[str, Any]:
    objects = find_objects(config)
    enabled_labels = [obj.get("label") for obj in objects if obj.get("enable") is True]

    summary = {
        "session": {
            "base_config_path": session.base_config_path,
            "current_config_path": session.current_config_path,
            "config_dirty": session.config_dirty,
            "preview_generated": session.preview_generated,
            "preview_accepted": session.preview_accepted,
            "dataset_generated": session.dataset_generated,
            "last_preview_paths": session.last_preview_paths,
        },
        "general": config.get("config", {}),
        "tip_conf": config.get("tip_conf", {}),
        "substrate_conf": config.get("substrate_conf", {}),
        "artifacts_conf": config.get("artifacts_conf", {}),
        "available_shapes": list_shape_names(config),
        "available_object_labels": list_object_labels(config),
        "enabled_object_labels": enabled_labels,
        "objects_brief": [summarize_object(obj) for obj in objects[:MAX_STATE_SUMMARY_OBJECTS]],
        "phase1_capabilities": {
            "can_query": True,
            "can_modify_existing_objects": True,
            "can_modify_general": True,
            "can_modify_tip": True,
            "can_modify_substrate": True,
            "can_modify_artifacts": True,
            "can_generate_preview": True,
            "can_confirm_preview": True,
            "can_reject_preview": True,
            "can_generate_dataset": True,
            "can_add_shapes": False,
            "can_add_objects": False,
            "can_add_stl": False,
            "can_create_sections_geometry": False,
        },
    }
    return summary

# =============================================================================
# MODEL PROMPT CONTRACT
# =============================================================================
SYSTEM_PROMPT = """
You are a command interpreter for a SimuScan configuration editor.

Your job is NOT to write YAML.
Your job is to read the user's natural language request and return exactly one JSON object
that follows the contract below.

PHASE 1 SCOPE
-------------
You are only allowed to support these behaviors:
1) Query current state
2) Enable/disable EXISTING objects
3) Modify properties in:
   - general config block (config)
   - existing objects
   - tip_conf
   - substrate_conf
   - artifacts_conf
4) Generate preview
5) Confirm preview
6) Reject preview
7) Generate final dataset
8) Ask for clarification when needed

NOT ALLOWED IN PHASE 1
----------------------
- Do not create new shapes
- Do not create new objects
- Do not create STL entries
- Do not create section geometry
- Do not invent unsupported operations

GENERAL RULES
-------------
- Return JSON only.
- No markdown.
- No explanations outside JSON.
- Do not invent field names.
- Use only operation names listed in the contract.
- If the user asks for something outside phase 1, return request_clarification and explain that
  phase 1 supports editing existing config only.
- If the user asks something ambiguous, return request_clarification.
- When the user wants to see examples after modifications, use intent=modify_and_preview.
- When the user only wants to inspect the state, use intent=query_state.
- When the user says to generate many images/final dataset, use intent=generate_dataset.
- If the user says a preview is good/accepted, use confirm_preview.
- If the user rejects a preview without specifying new changes, use reject_preview.

TOP-LEVEL JSON CONTRACT
-----------------------
{
  "intent": one of [
    "query_state",
    "modify_config",
    "modify_and_preview",
    "generate_preview",
    "confirm_preview",
    "reject_preview",
    "generate_dataset",
    "request_clarification"
  ],
  "queries": [ ... ],                    # only for query_state
  "actions": [ ... ],                    # for modify_config or modify_and_preview
  "num_images": integer,                 # optional for preview/dataset
  "output_dir": string,                  # optional for dataset
  "question": string                     # only for request_clarification
}

ALLOWED OPERATIONS
------------------
Each action must be one of:

1) {
     "op": "enable_object",
     "label": string,
     "value": true
   }

2) {
     "op": "disable_object",
     "label": string
   }

3) {
     "op": "set_object_property",
     "label": string,
     "field": string,
     "value": any JSON value
   }

4) {
     "op": "set_general_property",
     "field": string,
     "value": any JSON value
   }

5) {
     "op": "set_tip_property",
     "field": string,
     "value": any JSON value
   }

6) {
     "op": "set_substrate_property",
     "field": string,
     "value": any JSON value
   }

7) {
     "op": "set_artifact_property",
     "field": string,
     "value": any JSON value
   }

QUERY CONTRACT
--------------
For intent=query_state, queries must be a list of objects like:
{
  "target": one of ["general", "tip_conf", "substrate_conf", "artifacts_conf", "objects", "enabled_objects", "object_details"],
  "label": optional string for object_details
}

IMPORTANT
---------
- Prefer minimal commands.
- If the user asks to change several things and preview, combine them in modify_and_preview.
- If a user asks to enable/disable an object and also change its parameters, include both actions.
- Never output comments.
- Never output trailing commas.
""".strip()


def build_user_message(user_prompt: str, state_summary: Dict[str, Any]) -> str:
    payload = {
        "state_summary": state_summary,
        "user_prompt": user_prompt,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# =============================================================================
# LLM CLIENT
# =============================================================================
class SimuScanLLMInterpreter:
    def __init__(self, client: OpenAI, model: str = MODEL_NAME) -> None:
        self.client = client
        self.model = model

    def interpret(self, user_prompt: str, state_summary: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(user_prompt, state_summary)},
            ],
        )

        text = getattr(response, "output_text", None)
        if not text:
            raise RuntimeError("Model did not return output_text")

        try:
            command = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model returned invalid JSON: {exc}\nRaw output:\n{text}") from exc

        validate_command_shape(command)
        return command


# =============================================================================
# COMMAND VALIDATION
# =============================================================================
ALLOWED_INTENTS = {
    "query_state",
    "modify_config",
    "modify_and_preview",
    "generate_preview",
    "confirm_preview",
    "reject_preview",
    "generate_dataset",
    "request_clarification",
}

ALLOWED_OPS = {
    "enable_object",
    "disable_object",
    "set_object_property",
    "set_general_property",
    "set_tip_property",
    "set_substrate_property",
    "set_artifact_property",
}

ALLOWED_QUERY_TARGETS = {
    "general",
    "tip_conf",
    "substrate_conf",
    "artifacts_conf",
    "objects",
    "enabled_objects",
    "object_details",
}



def validate_command_shape(command: Dict[str, Any]) -> None:
    if not isinstance(command, dict):
        raise ValueError("Command must be a dict")
    intent = command.get("intent")
    if intent not in ALLOWED_INTENTS:
        raise ValueError(f"Invalid intent: {intent}")

    if intent == "query_state":
        queries = command.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError("query_state requires a non-empty 'queries' list")
        for q in queries:
            if not isinstance(q, dict):
                raise ValueError("Each query must be a dict")
            target = q.get("target")
            if target not in ALLOWED_QUERY_TARGETS:
                raise ValueError(f"Invalid query target: {target}")

    if intent in {"modify_config", "modify_and_preview"}:
        actions = command.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"{intent} requires a non-empty 'actions' list")
        for action in actions:
            validate_action_shape(action)

    if intent in {"generate_preview", "modify_and_preview", "generate_dataset"}:
        num_images = command.get("num_images")
        if num_images is not None:
            if not isinstance(num_images, int) or num_images <= 0:
                raise ValueError("num_images must be a positive integer")

    if intent == "generate_dataset":
        output_dir = command.get("output_dir")
        if output_dir is not None and not isinstance(output_dir, str):
            raise ValueError("output_dir must be a string if provided")

    if intent == "request_clarification":
        question = command.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("request_clarification requires a non-empty 'question'")



def validate_action_shape(action: Dict[str, Any]) -> None:
    if not isinstance(action, dict):
        raise ValueError("Each action must be a dict")
    op = action.get("op")
    if op not in ALLOWED_OPS:
        raise ValueError(f"Unsupported action op: {op}")

    if op in {"enable_object", "disable_object", "set_object_property"}:
        label = action.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Action {op} requires a non-empty label")

    if op == "enable_object":
        if action.get("value") is not True:
            raise ValueError("enable_object requires value=true")

    if op in {
        "set_object_property",
        "set_general_property",
        "set_tip_property",
        "set_substrate_property",
        "set_artifact_property",
    }:
        field_name = action.get("field")
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError(f"Action {op} requires a non-empty field")
        if "value" not in action:
            raise ValueError(f"Action {op} requires a value")


# =============================================================================
# SEMANTIC VALIDATION FOR CONFIG EDITING
# =============================================================================
def validate_known_top_blocks(config: Dict[str, Any]) -> None:
    required_blocks = ["config", "tip_conf", "substrate_conf", "artifacts_conf", "objects"]
    for block in required_blocks:
        if block not in config:
            raise ValueError(f"Missing required top-level block: {block}")

def validate_existing_object_label(config: Dict[str, Any], label: str) -> None:
    labels = list_object_labels(config)
    if label not in labels:
        raise KeyError(f"Unknown object label '{label}'. Available: {labels}")

def validate_field_exists(target_block: Dict[str, Any], field_name: str, context: str) -> None:
    if field_name not in target_block:
        available = sorted(target_block.keys())
        raise KeyError(f"Unknown field '{field_name}' in {context}. Available fields: {available}")

def validate_config_after_edit(config: Dict[str, Any]) -> None:
    validate_known_top_blocks(config)

    # Basic semantic consistency: object.shape should exist in shapes if shapes list is present.
    shape_names = set(list_shape_names(config))
    for obj in find_objects(config):
        if not isinstance(obj, dict):
            continue
        shape = obj.get("shape")
        if isinstance(shape, str) and shape and shape != "STL" and shape_names:
            if shape not in shape_names:
                raise ValueError(
                    f"Object '{obj.get('label')}' references unknown shape '{shape}'. "
                    f"Available shapes: {sorted(shape_names)}"
                )


# =============================================================================
# ACTION APPLICATION
# =============================================================================
def apply_actions(config: Dict[str, Any], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    new_config = copy.deepcopy(config)
    validate_known_top_blocks(new_config)

    for action in actions:
        op = action["op"]

        if op == "enable_object":
            label = action["label"]
            validate_existing_object_label(new_config, label)
            obj = find_object_by_label(new_config, label)
            obj["enable"] = True

        elif op == "disable_object":
            label = action["label"]
            validate_existing_object_label(new_config, label)
            obj = find_object_by_label(new_config, label)
            obj["enable"] = False

        elif op == "set_object_property":
            label = action["label"]
            field_name = action["field"]
            value = action["value"]
            validate_existing_object_label(new_config, label)
            obj = find_object_by_label(new_config, label)
            validate_field_exists(obj, field_name, f"object '{label}'")
            obj[field_name] = value

        elif op == "set_general_property":
            field_name = action["field"]
            value = action["value"]
            block = new_config["config"]
            validate_field_exists(block, field_name, "config")
            block[field_name] = value

        elif op == "set_tip_property":
            field_name = action["field"]
            value = action["value"]
            block = new_config["tip_conf"]
            validate_field_exists(block, field_name, "tip_conf")
            block[field_name] = value

        elif op == "set_substrate_property":
            field_name = action["field"]
            value = action["value"]
            block = new_config["substrate_conf"]
            validate_field_exists(block, field_name, "substrate_conf")
            block[field_name] = value

        elif op == "set_artifact_property":
            field_name = action["field"]
            value = action["value"]
            block = new_config["artifacts_conf"]
            validate_field_exists(block, field_name, "artifacts_conf")
            block[field_name] = value

        else:
            raise ValueError(f"Unhandled operation: {op}")

    validate_config_after_edit(new_config)
    return new_config


# =============================================================================
# QUERY RESOLUTION
# =============================================================================
def resolve_queries(config: Dict[str, Any], queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    all_objects = find_objects(config)

    for query in queries:
        target = query["target"]

        if target == "general":
            result["general"] = config.get("config", {})

        elif target == "tip_conf":
            result["tip_conf"] = config.get("tip_conf", {})

        elif target == "substrate_conf":
            result["substrate_conf"] = config.get("substrate_conf", {})

        elif target == "artifacts_conf":
            result["artifacts_conf"] = config.get("artifacts_conf", {})

        elif target == "objects":
            result["objects"] = all_objects

        elif target == "enabled_objects":
            result["enabled_objects"] = [obj for obj in all_objects if obj.get("enable") is True]

        elif target == "object_details":
            label = query.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("object_details query requires 'label'")
            result.setdefault("object_details", {})[label] = find_object_by_label(config, label)

        else:
            raise ValueError(f"Unknown query target: {target}")

    return result


# =============================================================================
# SIMUSCAN EXECUTION
# =============================================================================
def write_temp_preview_config(config: Dict[str, Any], working_dir: Path, num_images: int) -> Path:
    preview_cfg = copy.deepcopy(config)
    preview_cfg["config"]["n_images"] = num_images
    out_path = working_dir / "preview_config.yaml"
    save_yaml(preview_cfg, out_path)
    return out_path



def write_final_config(config: Dict[str, Any], working_dir: Path, name: str = "final_config.yaml") -> Path:
    out_path = working_dir / name
    save_yaml(config, out_path)
    return out_path



def run_external_simuscan(executable: str, config_path: Path, working_dir: Path) -> subprocess.CompletedProcess:
    # Adjust this command if SimuScan expects another CLI contract.
    # Example expected usage:
    #   simuscan.exe --config path/to/config.yaml
    cmd = [executable, "--config", str(config_path)]
    return subprocess.run(
        cmd,
        cwd=str(working_dir),
        text=True,
        capture_output=True,
        check=False,
    )



def collect_preview_images(preview_dir: Path) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if not preview_dir.exists():
        return []
    images = [str(p) for p in sorted(preview_dir.rglob("*")) if p.suffix.lower() in exts]
    return images


# =============================================================================
# ORCHESTRATOR
# =============================================================================
class SimuScanPhase1Controller:
    def __init__(
        self,
        client: OpenAI,
        base_config_path: str,
        working_dir: str,
        model: str = MODEL_NAME,
        simuscan_executable: Optional[str] = None,
        state_file: Optional[str] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.interpreter = SimuScanLLMInterpreter(client=client, model=model)
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = str(self.working_dir / (state_file or STATE_FILENAME))

        loaded_state = load_state(self.state_file)
        if loaded_state is None:
            self.session = SessionState(
                base_config_path=str(Path(base_config_path).resolve()),
                current_config_path=str(Path(base_config_path).resolve()),
                working_dir=str(self.working_dir),
                simuscan_executable=simuscan_executable,
            )
            save_state(self.session, self.state_file)
        else:
            self.session = loaded_state
            if simuscan_executable is not None:
                self.session.simuscan_executable = simuscan_executable
                save_state(self.session, self.state_file)

    def get_current_config(self) -> Dict[str, Any]:
        return load_yaml(self.session.current_config_path)

    def set_current_config(self, config: Dict[str, Any], filename: str = "current_config.yaml") -> Path:
        path = self.working_dir / filename
        save_yaml(config, path)
        self.session.current_config_path = str(path)
        self.session.config_dirty = True
        self.session.preview_accepted = False
        self.session.dataset_generated = False
        save_state(self.session, self.state_file)
        return path

    def handle_prompt(self, user_prompt: str) -> Dict[str, Any]:
        current_config = self.get_current_config()
        state_summary = build_state_summary(current_config, self.session)
        command = self.interpreter.interpret(user_prompt, state_summary)

        result = self.execute_command(command, current_config)
        self.session.last_prompt = user_prompt
        self.session.history.append(
            {
                "user_prompt": user_prompt,
                "command": command,
                "result_summary": result.get("summary", ""),
            }
        )
        save_state(self.session, self.state_file)
        return {
            "command": command,
            "result": result,
        }

    def execute_command(self, command: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
        intent = command["intent"]

        if intent == "query_state":
            data = resolve_queries(current_config, command["queries"])
            return {
                "status": "ok",
                "summary": "State queried successfully.",
                "data": data,
            }

        if intent == "request_clarification":
            return {
                "status": "clarification_needed",
                "summary": command["question"],
            }

        if intent == "modify_config":
            new_config = apply_actions(current_config, command["actions"])
            path = self.set_current_config(new_config)
            return {
                "status": "ok",
                "summary": f"Configuration updated and saved to {path}",
                "config_path": str(path),
            }

        if intent == "generate_preview":
            return self.generate_preview(current_config, command.get("num_images", DEFAULT_PREVIEW_COUNT))

        if intent == "modify_and_preview":
            new_config = apply_actions(current_config, command["actions"])
            path = self.set_current_config(new_config)
            preview_result = self.generate_preview(new_config, command.get("num_images", DEFAULT_PREVIEW_COUNT))
            preview_result["config_path"] = str(path)
            preview_result["summary"] = f"Configuration updated and preview requested. {preview_result['summary']}"
            return preview_result

        if intent == "confirm_preview":
            self.session.preview_accepted = True
            save_state(self.session, self.state_file)
            return {
                "status": "ok",
                "summary": "Preview accepted.",
            }

        if intent == "reject_preview":
            self.session.preview_accepted = False
            save_state(self.session, self.state_file)
            return {
                "status": "ok",
                "summary": "Preview rejected. You can now refine the configuration.",
            }

        if intent == "generate_dataset":
            return self.generate_dataset(
                current_config=current_config,
                num_images=command.get("num_images", DEFAULT_DATASET_COUNT),
                output_dir=command.get("output_dir"),
            )

        raise ValueError(f"Unhandled intent: {intent}")

    def generate_preview(self, current_config: Dict[str, Any], num_images: int) -> Dict[str, Any]:
        preview_cfg_path = write_temp_preview_config(current_config, self.working_dir, num_images=num_images)
        preview_dir = self.working_dir / "preview_output"
        preview_dir.mkdir(exist_ok=True)
        self.session.preview_output_dir = str(preview_dir)

        if not self.session.simuscan_executable:
            self.session.preview_generated = True
            self.session.last_preview_paths = []
            save_state(self.session, self.state_file)
            return {
                "status": "dry_run",
                "summary": (
                    "Preview config created, but no SimuScan executable was configured. "
                    f"Config saved at {preview_cfg_path}"
                ),
                "preview_config_path": str(preview_cfg_path),
                "preview_images": [],
            }

        completed = run_external_simuscan(
            executable=self.session.simuscan_executable,
            config_path=preview_cfg_path,
            working_dir=self.working_dir,
        )

        images = collect_preview_images(preview_dir)
        self.session.preview_generated = True
        self.session.preview_accepted = False
        self.session.last_preview_paths = images
        save_state(self.session, self.state_file)

        return {
            "status": "ok" if completed.returncode == 0 else "execution_error",
            "summary": f"Preview run finished with code {completed.returncode}. Found {len(images)} image(s).",
            "preview_config_path": str(preview_cfg_path),
            "preview_images": images,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def generate_dataset(self, current_config: Dict[str, Any], num_images: int, output_dir: Optional[str]) -> Dict[str, Any]:
        dataset_cfg = copy.deepcopy(current_config)
        dataset_cfg["config"]["n_images"] = num_images

        if output_dir:
            dataset_cfg["config"]["dataset_path"] = output_dir
            final_output_dir = Path(output_dir)
        else:
            final_output_dir = self.working_dir / "dataset_output"
            dataset_cfg["config"]["dataset_path"] = str(final_output_dir)

        final_cfg_path = write_final_config(dataset_cfg, self.working_dir, name="dataset_config.yaml")
        self.session.final_output_dir = str(final_output_dir)

        if not self.session.simuscan_executable:
            self.session.dataset_generated = True
            save_state(self.session, self.state_file)
            return {
                "status": "dry_run",
                "summary": (
                    "Dataset config created, but no SimuScan executable was configured. "
                    f"Config saved at {final_cfg_path}"
                ),
                "dataset_config_path": str(final_cfg_path),
                "output_dir": str(final_output_dir),
            }

        completed = run_external_simuscan(
            executable=self.session.simuscan_executable,
            config_path=final_cfg_path,
            working_dir=self.working_dir,
        )
        self.session.dataset_generated = completed.returncode == 0
        save_state(self.session, self.state_file)

        return {
            "status": "ok" if completed.returncode == 0 else "execution_error",
            "summary": f"Dataset run finished with code {completed.returncode}.",
            "dataset_config_path": str(final_cfg_path),
            "output_dir": str(final_output_dir),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


# =============================================================================
# CLI / EXAMPLE USAGE
# =============================================================================
def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def interactive_loop(controller: SimuScanPhase1Controller) -> None:
    print("SimuScan Phase 1 controller")
    print("Type 'exit' to quit.")
    while True:
        try:
            user_prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            break

        try:
            response = controller.handle_prompt(user_prompt)
            print_json(response)
        except Exception as exc:
            print_json({"status": "error", "message": str(exc)})



def main(base_config_path: Path, working_dir: Path, simuscan_executable: Path) -> None:
    # Environment variable recommended:
    #   export OPENAI_API_KEY=...
    # Example:
    #   python simuscan_phase1_controller.py /path/to/default.yaml /path/to/workdir /path/to/simuscan_executable

    
    # base_config_path = sys.argv[1]
    # working_dir = sys.argv[2]
    # simuscan_executable = sys.argv[3] if len(sys.argv) >= 4 else None

    client = OpenAI()
    controller = SimuScanPhase1Controller(
        client=client,
        base_config_path=base_config_path,
        working_dir=working_dir,
        simuscan_executable=simuscan_executable,
        model=MODEL_NAME,
    )
    interactive_loop(controller)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_config_path",
        default=r"C:\Users\z78\Documents\Projects\AppsPy\AFM_data_analysis\src\OpenAI\config\default.yaml"
    )

    parser.add_argument(
        "--working_dir",
        default=r"C:\Users\z78\Documents\Projects\AppsPy\AFM_data_analysis\src\OpenAI\workdir"
    )

    parser.add_argument(
        "--simuscan_executable",
        default=r"C:\Users\z78\Documents\Projects\AppsPy\AFM_data_analysis\src\OpenAI\simuscan.exe"
    )

    args = parser.parse_args()

    main(
        base_config=Path(args.base_config),
        working_dir=Path(args.working_dir),
        simuscan_exe=Path(args.simuscan_exe)
    )

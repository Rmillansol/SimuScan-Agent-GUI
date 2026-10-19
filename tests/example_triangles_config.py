"""
tests/example_triangles_config.py
──────────────────────────────────
Example: create a new config file from default where:
  - All objects are disabled except 'triangle'
  - n_images  = 25
  - image_size = 256
  - Double-tip effect enabled (is_double=True, double_probability=1.0)
"""

import sys
import json
from pathlib import Path

#sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simuscan_tools.simuscan_config import SimuScanConfig


def pp(label: str, result: dict) -> None:
    print(f"\n{label}")
    print(json.dumps(result, indent=2, default=str))


def main():
    # ── 1. Load a fresh copy of default ──────────────────────────────────────
    cfg = SimuScanConfig()
    pp("Initial summary", cfg.summary())

    # ── 2. Disable all objects ────────────────────────────────────────────────
    all_objects = cfg.list_objects()["data"]
    for obj in all_objects:
        result = cfg.set_object_param(obj["label"], "enable", False)
        print(f"  disable '{obj['label']}': {result['message']}")

    # ── 3. Enable only triangle ───────────────────────────────────────────────
    pp("Enable triangle", cfg.set_object_param("triangle", "enable", True))

    # ── 4. Update general config ──────────────────────────────────────────────
    pp("Set n_images=25",    cfg.set_param("config", "n_images",   25))
    pp("Set image_size=256", cfg.set_param("config", "image_size", 256))

    # ── 5. Configure double tip ───────────────────────────────────────────────
    pp("Enable double tip",        cfg.set_param("tip_conf", "is_double",         True))
    pp("Set double_probability=1", cfg.set_param("tip_conf", "double_probability", 1.0))

    # ── 6. Verify before saving ───────────────────────────────────────────────
    pp("Final summary", cfg.summary())

    enabled = [
        o["label"]
        for o in cfg.list_objects()["data"]
        if o["enable"]
    ]
    print(f"\nEnabled objects: {enabled}")
    assert enabled == ["triangle"], f"Expected only triangle, got: {enabled}"

    assert cfg.get_param("config", "n_images")["data"]["n_images"]     == 25
    assert cfg.get_param("config", "image_size")["data"]["image_size"] == 256
    assert cfg.get_param("tip_conf", "is_double")["data"]["is_double"] == True
    assert cfg.get_param("tip_conf", "double_probability")["data"]["double_probability"] == 1.0

    # ── 7. Save ───────────────────────────────────────────────────────────────
    pp("Save", cfg.save("triangles_only.yaml"))
    print("\nDone — file saved to data/config/triangles_only.yaml")


if __name__ == "__main__":
    main()
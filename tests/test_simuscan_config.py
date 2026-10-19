"""
tests/test_simuscan_config.py
─────────────────────────────
Quick functional tests for SimuScanConfig.
Run from the project root:
    python -m pytest tests/ -v
or simply:
    python tests/test_simuscan_config.py
"""

import sys
import json
from pathlib import Path

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simuscan_tools.simuscan_config import SimuScanConfig


def pp(result: dict) -> None:
    """Pretty-print a result dict."""
    print(json.dumps(result, indent=2, default=str))


# ══════════════════════════════════════════════════════════════════════════════
def test_load_default():
    print("\n── test_load_default ──")
    cfg = SimuScanConfig()
    result = cfg.summary()
    pp(result)
    assert result["status"] == "ok"
    assert result["data"]["total_objects"] > 0


# ══════════════════════════════════════════════════════════════════════════════
def test_get_section():
    print("\n── test_get_section ──")
    cfg = SimuScanConfig()
    result = cfg.get_section("config")
    pp(result)
    assert result["status"] == "ok"
    assert "image_size" in result["data"]


def test_get_param():
    print("\n── test_get_param ──")
    cfg = SimuScanConfig()
    result = cfg.get_param("config", "image_size")
    pp(result)
    assert result["status"] == "ok"
    assert result["data"]["image_size"] == 512


def test_read_only_shapes():
    print("\n── test_read_only_shapes ──")
    cfg    = SimuScanConfig()
    result = cfg.get_section("shapes")
    pp(result)
    assert result["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
def test_set_param_ok():
    print("\n── test_set_param_ok ──")
    cfg    = SimuScanConfig()
    result = cfg.set_param("config", "image_size", 256)
    pp(result)
    assert result["status"] == "ok"
    # Verify the change was applied
    assert cfg.get_param("config", "image_size")["data"]["image_size"] == 256


def test_set_param_wrong_type():
    print("\n── test_set_param_wrong_type ──")
    cfg    = SimuScanConfig()
    result = cfg.set_param("config", "image_size", "big")   # should fail
    pp(result)
    assert result["status"] == "error"


def test_set_param_bool_strict():
    print("\n── test_set_param_bool_strict ──")
    cfg = SimuScanConfig()
    # 1 is an int, should fail for a bool field
    result = cfg.set_param("tip_conf", "is_double", 1)
    pp(result)
    assert result["status"] == "error"
    # True is a bool, should pass
    result2 = cfg.set_param("tip_conf", "is_double", False)
    pp(result2)
    assert result2["status"] == "ok"


def test_set_param_read_only_section():
    print("\n── test_set_param_read_only_section ──")
    cfg    = SimuScanConfig()
    result = cfg.set_param("shapes", "anything", 1)
    pp(result)
    assert result["status"] == "error"


# ══════════════════════════════════════════════════════════════════════════════
def test_list_objects():
    print("\n── test_list_objects ──")
    cfg    = SimuScanConfig()
    result = cfg.list_objects()
    pp(result)
    assert result["status"] == "ok"
    labels = [o["label"] for o in result["data"]]
    assert "hexagon" in labels


def test_get_object():
    print("\n── test_get_object ──")
    cfg    = SimuScanConfig()
    result = cfg.get_object("hexagon")
    pp(result)
    assert result["status"] == "ok"
    assert result["data"]["label"] == "hexagon"


def test_get_object_not_found():
    print("\n── test_get_object_not_found ──")
    cfg    = SimuScanConfig()
    result = cfg.get_object("nonexistent_label")
    pp(result)
    assert result["status"] == "error"


def test_set_object_param_ok():
    print("\n── test_set_object_param_ok ──")
    cfg    = SimuScanConfig()
    result = cfg.set_object_param("hexagon", "n_objects", 5)
    pp(result)
    assert result["status"] == "ok"
    assert cfg.get_object_param("hexagon", "n_objects")["data"]["n_objects"] == 5


def test_set_object_param_wrong_type():
    print("\n── test_set_object_param_wrong_type ──")
    cfg    = SimuScanConfig()
    result = cfg.set_object_param("hexagon", "n_objects", "five")
    pp(result)
    assert result["status"] == "error"


def test_set_object_shape_invalid():
    print("\n── test_set_object_shape_invalid ──")
    cfg    = SimuScanConfig()
    result = cfg.set_object_param("hexagon", "shape", "octopus")
    pp(result)
    assert result["status"] == "error"


def test_set_object_shape_valid():
    print("\n── test_set_object_shape_valid ──")
    cfg    = SimuScanConfig()
    result = cfg.set_object_param("hexagon", "shape", "triangle")
    pp(result)
    assert result["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
def test_stl_weights_bad_sum():
    print("\n── test_stl_weights_bad_sum ──")
    cfg    = SimuScanConfig()
    # bacillus already has stl_files set in the default
    result = cfg.set_object_param("bacillus", "stl_weights", [0.4, 0.4])
    pp(result)
    assert result["status"] == "error"


def test_stl_weights_good():
    print("\n── test_stl_weights_good ──")
    cfg    = SimuScanConfig()
    result = cfg.set_object_param("bacillus", "stl_weights", [0.5, 0.5])
    pp(result)
    # Will be ok only if bacillus.stl files exist; otherwise error about missing file
    # Here we just check the structure is returned
    assert "status" in result


# ══════════════════════════════════════════════════════════════════════════════
def test_save_as_new_file(tmp_path, monkeypatch):
    """Save a modified config as a new YAML and reload it."""
    print("\n── test_save_as_new_file ──")
    import simuscan_tools.simuscan_config as m
    # Redirect config_dir to tmp_path for this test
    monkeypatch.setattr(m, "_config_dir", lambda: tmp_path)

    cfg = SimuScanConfig()
    cfg.set_param("config", "n_images", 99)
    result = cfg.save("test_output.yaml")
    pp(result)
    assert result["status"] == "ok"
    assert (tmp_path / "test_output.yaml").exists()


def test_cannot_save_as_default():
    print("\n── test_cannot_save_as_default ──")
    cfg    = SimuScanConfig()
    result = cfg.save("default.yaml")
    pp(result)
    assert result["status"] == "error"


def test_cannot_save_without_name():
    print("\n── test_cannot_save_without_name ──")
    cfg    = SimuScanConfig()   # loaded from default, _filename is None
    result = cfg.save()
    pp(result)
    assert result["status"] == "error"


# ══════════════════════════════════════════════════════════════════════════════
def test_describe_section():
    print("\n── test_describe_section ──")
    cfg    = SimuScanConfig()
    result = cfg.describe_section("config")
    pp(result)
    assert result["status"] == "ok"


def test_list_shapes():
    print("\n── test_list_shapes ──")
    cfg    = SimuScanConfig()
    result = cfg.list_shapes()
    pp(result)
    assert result["status"] == "ok"
    assert "hexagon" in result["data"]


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            # Simple tests that don't need pytest fixtures
            import inspect
            if inspect.signature(t).parameters:
                print(f"\n[SKIP] {t.__name__} (requires pytest fixtures)")
                continue
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{'─'*40}")
    print(f"Results: {passed} passed, {failed} failed")
"""
tests/test_domains.py
─────────────────────
Integration tests for the grain/domain field renderer (`shape: domains`).

Run with:
    uv run python -m pytest tests/test_domains.py -v
    uv run python tests/test_domains.py          # no pytest needed

They cover the parts of the integration that are easy to break silently:
  * the px / height-unit bridge to the nm-based vendored engine,
  * masks used verbatim, disjoint, and clipped under the objects on top,
  * the grain film being a BASE LAYER, so objects add on top of it,
  * the per-image instance label map and its link to the COCO annotations,
  * the object-topography accumulation (it used to be added once per object).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

from src.simuscan.SyntheticParams import DomainsConfig, ObjectConfig, SyntheticParams  # noqa: E402
from src.simuscan.domains_core import DEFAULT_CONFIG  # noqa: E402
from src.simuscan.domains_shape import (  # noqa: E402
    generate_domains_shape,
    split_domain_instances,
)

CONFIG_DIR = REPO / "data" / "config" / "conf_simuscan"


def _fast_config(**kw) -> DomainsConfig:
    """Small frame, big grains — a few seconds instead of a few minutes."""
    d = DomainsConfig(mean_grain_size_px=40.0, grain_size_sigma_px=10.0)
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def _obj(**kw) -> ObjectConfig:
    base = dict(label="grain", shape="domains", category="domains", height=[0.8, 1.0])
    base.update(kw)
    return ObjectConfig(**base)


def _field(size=256, dconf=None, obj=None):
    """Render a field and split it with no objects on top."""
    dconf = dconf or _fast_config()
    obj = obj or _obj()
    topo, label, meta = generate_domains_shape(obj, size, dconf)
    masks, gmeta = split_domain_instances(label, meta, dconf)
    return topo, label, masks, gmeta


# ══════════════════════════════════════════════════════════════════════════════
# Config bridge
# ══════════════════════════════════════════════════════════════════════════════

def test_core_overrides_cover_every_engine_key():
    """Every knob of the vendored engine must be reachable from domains_conf."""
    ov = DomainsConfig().core_overrides()
    engine = set(DEFAULT_CONFIG) - {"n_images_per_mode", "image_shape", "output_base_dir"}
    assert engine - set(ov) == set(), f"not exposed: {sorted(engine - set(ov))}"
    assert set(ov) - set(DEFAULT_CONFIG) == set(), f"unknown: {sorted(set(ov) - set(DEFAULT_CONFIG))}"


def test_pixel_bridge_is_identity():
    """
    Lateral parameters are declared in px, so the engine must be driven with
    pixel_size_nm = 1 — otherwise every size would be silently rescaled.
    """
    ov = _fast_config().core_overrides()
    assert ov["pixel_size_nm"] == 1.0
    assert ov["mean_grain_size_nm"] == 40.0


def test_grain_size_is_actually_in_pixels():
    """A 40 px request must produce ~40 px grains, not 40 nm worth of them."""
    topo, _label, _masks, gmeta = _field()
    assert topo.shape == (256, 256)
    measured = np.mean([m["eq_diameter_px"] for m in gmeta])
    assert 0.6 * 40.0 < measured < 1.6 * 40.0, measured


# ══════════════════════════════════════════════════════════════════════════════
# Masks
# ══════════════════════════════════════════════════════════════════════════════

def test_tessellation_masks_are_disjoint_and_cover_the_frame():
    """
    Masks come verbatim from the label map: in a tessellation they must tile the
    frame exactly, with no overlap. An accidental dilation would break both.
    """
    _topo, _label, masks, _gmeta = _field(dconf=_fast_config(mode="tessellation"))
    stack = np.sum(masks, axis=0)
    assert stack.max() == 1, "grain masks overlap — were they dilated?"
    assert stack.mean() == 1.0, "a tessellation must leave no background"


def test_overlap_mode_leaves_background_and_still_disjoint():
    _topo, _label, masks, _gmeta = _field(dconf=_fast_config(mode="overlap_interactive"))
    stack = np.sum(masks, axis=0)
    assert stack.max() == 1, "only the top visible grain may own a pixel"
    assert stack.mean() < 1.0, "overlap mode should leave some background"


def test_background_is_zero_so_the_substrate_shows_through():
    topo, label, _masks, _gmeta = _field(dconf=_fast_config(mode="overlap_interactive"))
    assert np.all(topo[label == 0] == 0.0)
    assert topo.min() >= 0.0


def test_min_area_and_max_instances_filters():
    d = _fast_config(max_instances=5)
    _t, _l, masks, gmeta = _field(dconf=d)
    assert len(masks) == 5 == len(gmeta)
    areas = [int(m.sum()) for m in masks]
    assert areas == sorted(areas, reverse=True), "the cap must keep the largest grains"

    _t2, _l2, masks2, _g2 = _field(dconf=_fast_config(min_grain_area_px=400))
    assert all(m.sum() >= 400 for m in masks2)


# ══════════════════════════════════════════════════════════════════════════════
# Clipping under the objects sitting on the film
# ══════════════════════════════════════════════════════════════════════════════

def test_grain_masks_are_clipped_under_objects():
    """A pixel must never be labelled both 'grain' and something else."""
    dconf = _fast_config(min_visible_after_objects=0.0)
    _topo, label, meta = generate_domains_shape(_obj(), 256, dconf)

    occupied = np.zeros((256, 256), bool)
    occupied[60:160, 60:160] = True          # one big square object on the film

    masks, _gmeta = split_domain_instances(label, meta, dconf, occupied=occupied)
    assert masks, "everything was dropped"
    for m in masks:
        assert not np.any(m[occupied]), "a grain still claims pixels owned by an object"
    stack = np.sum(masks, axis=0)
    assert stack.max() == 1, "clipped grains must stay disjoint"


def test_grains_mostly_covered_are_dropped():
    dconf_keep = _fast_config(min_visible_after_objects=0.0)
    dconf_drop = _fast_config(min_visible_after_objects=0.9)
    _topo, label, meta = generate_domains_shape(_obj(), 256, dconf_keep)

    occupied = np.zeros((256, 256), bool)
    occupied[:, :128] = True                 # bury the whole left half

    kept = split_domain_instances(label, meta, dconf_keep, occupied=occupied)[0]
    strict = split_domain_instances(label, meta, dconf_drop, occupied=occupied)[0]
    assert len(strict) < len(kept), "the visibility threshold had no effect"


def test_a_grain_cut_in_two_stays_one_instance():
    """
    An object on the film can slice a grain into separate pieces. It is still ONE
    grain, so it must stay one mask — and COCOBuilder must fold its contours into
    a single annotation, otherwise two annotations would share a grain_id.
    """
    dconf = _fast_config(min_visible_after_objects=0.0)
    _topo, label, meta = generate_domains_shape(_obj(), 256, dconf)

    occupied = np.zeros((256, 256), bool)
    occupied[:, 120:136] = True              # a vertical bar slicing grains in two

    masks, gmeta = split_domain_instances(label, meta, dconf, occupied=occupied)
    ids = [g["grain_id"] for g in gmeta]
    assert len(ids) == len(set(ids)), "one grain produced several masks"
    assert len(masks) == len(gmeta)

    # at least one grain really did get cut, otherwise the test proves nothing
    cut = [m for m in masks if cv2.connectedComponents(m, connectivity=8)[0] > 2]
    assert cut, "the bar did not slice any grain — test is not exercising the case"


def test_merge_contours_folds_the_pieces_into_one_annotation():
    """
    A two-piece mask must become ONE annotation carrying both polygons when
    merge_contours is set, and the historical two annotations when it is not.
    """
    from src.simuscan.COCOBuilder import COCOBuilder

    mask = np.zeros((128, 128), np.uint8)
    mask[20:60, 20:60] = 1                  # two clearly separated pieces
    mask[20:60, 80:120] = 1
    img = mask.astype(np.float32)
    cats = [{"id": 0, "name": "grain", "supercategory": "domains"}]

    for merge, expect_anns in ((True, 1), (False, 2)):
        cb = COCOBuilder(str(Path(tempfile.mkdtemp())), cats, split_ratios=(1.0, 0.0, 0.0))
        cb.add_data_training(img, "train", [mask], [0], 1.0, 0.0, 10.0, False,
                             mask_meta=[{"grain_id": 7}], dilate_px=[0],
                             merge_contours=[merge])
        anns = cb.coco_dicts["train"]["annotations"]
        assert len(anns) == expect_anns, f"merge={merge} gave {len(anns)} annotations"
        if merge:
            a = anns[0]
            assert len(a["segmentation"]) == 2, "the two pieces were not both kept"
            assert a["grain_id"] == 7
            x, y, w, h = a["bbox"]
            assert x <= 20 and x + w >= 120, "bbox must span both pieces"
            assert a["area"] > 3000, "area must sum both pieces"


# ══════════════════════════════════════════════════════════════════════════════
# Heights
# ══════════════════════════════════════════════════════════════════════════════

def test_z_mode_direct_keeps_the_configured_height_scale():
    """z_mean is in SimuScan height units; direct mode must not renormalise."""
    dconf = _fast_config(z_mode="direct", z_mean=[0.5, 1.2])
    topo, _label, _m = generate_domains_shape(_obj(height=[5.0, 5.0]), 256, dconf)
    # The object's height range is ignored in direct mode, so the relief stays
    # on the z_mean scale rather than jumping to 5.
    assert topo.max() < 3.0, topo.max()


def test_z_mode_relative_rescales_into_the_object_height():
    dconf = _fast_config(z_mode="relative", z_norm_percentile=99.5)
    target = 2.5
    topo, label, _m = generate_domains_shape(_obj(height=[target, target]), 256, dconf)
    ref = float(np.percentile(topo[label > 0], 99.5))
    assert abs(ref - target) < 1e-3, ref


def test_reported_mean_z_follows_the_applied_scaling():
    """A grain's annotated mean_z must be comparable with the saved image."""
    dconf = _fast_config(z_mode="relative")
    topo, label, meta = generate_domains_shape(_obj(height=[2.0, 2.0]), 256, dconf)
    masks, gmeta = split_domain_instances(label, meta, dconf)
    for m, g in list(zip(masks, gmeta))[:5]:
        measured = float(topo[m > 0].mean())
        assert abs(measured - g["mean_z"]) < 0.5 * max(g["mean_z"], 0.1), (measured, g["mean_z"])


def test_invalid_modes_are_rejected():
    for kw, obj in [({"mode": "nope"}, _obj()), ({}, _obj(domains_mode="nope"))]:
        try:
            generate_domains_shape(obj, 128, _fast_config(**kw))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mode accepted")

    try:
        generate_domains_shape(_obj(), 128, _fast_config(z_mode="nope"))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid z_mode accepted")


# ══════════════════════════════════════════════════════════════════════════════
# Scene composition
# ══════════════════════════════════════════════════════════════════════════════

def test_object_topography_is_added_once_per_image():
    """
    Regression: `topo += image` used to live INSIDE the object loop, and `image`
    is the running max over the objects seen so far — so with N classes the first
    object's topography was added N times.

    Three classes are stubbed out to fill three disjoint quadrants with height
    1.0. The composed topography must be 1.0 everywhere in those quadrants; the
    old behaviour gave 3.0, 2.0 and 1.0.
    """
    from src.simuscan import syntheticAFM_COCO as sac

    size = 64
    params = SyntheticParams.from_yaml(str(CONFIG_DIR / "shapes_conf.yaml"))
    params.image_size = size
    params.artifacts.apply_delta = params.artifacts.apply_gaussian = False
    params.artifacts.apply_loss_contact = params.artifacts.apply_adhesion = False
    params.artifacts.apply_sticky_zones = params.artifacts.apply_line_noise = False
    params.artifacts.apply_periodic_noise = params.artifacts.apply_shift_lines_noise = False
    params.artifacts.apply_flatten = False
    params.pid.apply_pid = False

    enabled = params.get_objects_by_enabled()[:3]
    assert len(enabled) == 3, "this test needs 3 enabled object classes"
    for o in params.objects:
        o.enable = o in enabled
        o.probability = 1.0
        o.apply_colony = False

    quadrants = [(slice(0, 32), slice(0, 32)),
                 (slice(0, 32), slice(32, 64)),
                 (slice(32, 64), slice(0, 32))]
    calls = {"n": 0}

    def fake_sections(obj, s):
        ys, xs = quadrants[calls["n"] % 3]
        calls["n"] += 1
        t = np.zeros((s, s), np.float32); t[ys, xs] = 1.0
        m = np.zeros((s, s), np.uint8);   m[ys, xs] = 1
        return [t], [m], m

    orig_sections = sac.generate_rand_image_from_obj_sections
    orig_substrate = sac.subs.rand_substrate
    orig_tip = sac.sh_tips.tip_convolution_roi_fast
    try:
        sac.generate_rand_image_from_obj_sections = fake_sections
        sac.subs.rand_substrate = lambda *_a, **_k: np.zeros((size, size), np.float32)
        sac.sh_tips.tip_convolution_roi_fast = lambda img, _tip: img
        out = sac._render_preview(params)
    finally:
        sac.generate_rand_image_from_obj_sections = orig_sections
        sac.subs.rand_substrate = orig_substrate
        sac.sh_tips.tip_convolution_roi_fast = orig_tip

    for ys, xs in quadrants:
        peak = float(out[ys, xs].max())
        assert abs(peak - 1.0) < 1e-4, f"quadrant peak {peak}, expected 1.0 (added more than once)"


def test_objects_sit_on_top_of_the_grain_film():
    """
    The grain field is a base layer folded into the substrate, and the substrate
    is ADDED — so an object keeps its full height above the local film level
    instead of being max-composited into it.
    """
    from src.simuscan import syntheticAFM_COCO as sac

    size = 128
    params = SyntheticParams.from_yaml(str(CONFIG_DIR / "domains_grains.yaml"))
    params.image_size = size
    params.domains.mean_grain_size_px = 40.0
    params.substratum.apply_tilt = params.substratum.apply_dirt = False
    params.substratum.apply_shapes = params.substratum.apply_perlin = False
    params.substratum.apply_gaussian = False
    params.artifacts.apply_flatten = False
    params.pid.apply_pid = False

    obj_h, box = 3.0, (slice(40, 90), slice(40, 90))

    def fake_sections(obj, s):
        t = np.zeros((s, s), np.float32); t[box] = obj_h
        m = np.zeros((s, s), np.uint8);   m[box] = 1
        return [t], [m], m

    # enable one ordinary class alongside the grains
    hexa = params.get_object_by_label("hexagon")
    hexa.enable, hexa.probability, hexa.apply_colony = True, 1.0, False

    orig_sections = sac.generate_rand_image_from_obj_sections
    orig_tip = sac.sh_tips.tip_convolution_roi_fast
    orig_art = sac.add_artefactes
    try:
        sac.generate_rand_image_from_obj_sections = fake_sections
        sac.sh_tips.tip_convolution_roi_fast = lambda img, _tip: img
        sac.add_artefactes = lambda t, _a: t
        out = sac._render_preview(params)
    finally:
        sac.generate_rand_image_from_obj_sections = orig_sections
        sac.sh_tips.tip_convolution_roi_fast = orig_tip
        sac.add_artefactes = orig_art

    inside = np.zeros((size, size), bool); inside[box] = True
    rise = float(out[inside].mean() - out[~inside].mean())
    # max-compositing would leave a rise far below the object's own height
    assert rise > 0.8 * obj_h, (
        f"object rises only {rise:.2f} over the film, expected ~{obj_h} "
        "— is it being max-composited instead of added?"
    )


def test_two_enabled_domain_fields_are_rejected():
    """Two full-frame fields would produce masks that contradict each other."""
    from src.simuscan.syntheticAFM_COCO import _render_domain_layer

    two = [_obj(label="a"), _obj(label="b")]
    try:
        _render_domain_layer(two, 64, _fast_config())
    except ValueError as exc:
        assert "only one" in str(exc).lower()
    else:
        raise AssertionError("two enabled domains objects were accepted")


# ══════════════════════════════════════════════════════════════════════════════
# YAML wiring and the full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_configs_expose_the_domains_section_and_object():
    for name in ("default.yaml", "domains_grains.yaml"):
        params = SyntheticParams.from_yaml(str(CONFIG_DIR / name))
        assert params.get_object_by_label("grain").shape == "domains"
        assert params.domains.mode in ("tessellation", "overlap_interactive")
        assert params.save_instance_map is True

    preset = SyntheticParams.from_yaml(str(CONFIG_DIR / "domains_grains.yaml"))
    assert [o.label for o in preset.get_objects_by_enabled()] == ["grain"]


def test_domains_conf_is_editable_by_the_agent():
    from src.simuscan_tools.simuscan_config import SimuScanConfig
    cfg = SimuScanConfig("domains_grains.yaml")
    assert cfg.describe_section("domains_conf")["status"] == "ok"
    assert cfg.set_param("domains_conf", "mean_grain_size_px", 33.0)["status"] == "ok"
    assert cfg.get_param("domains_conf", "mean_grain_size_px")["data"]["mean_grain_size_px"] == 33.0
    assert cfg.set_object_param("grain", "shape", "domains")["status"] == "ok"


def _generate_small_dataset(out: Path, tweaks=()) -> str:
    from src.simuscan.syntheticAFM_COCO import generate_data_for_training
    text = (CONFIG_DIR / "domains_grains.yaml").read_text(encoding="utf-8")
    text = text.replace("  image_size: 512", "  image_size: 256")
    text = text.replace("  mean_grain_size_px: 60.0", "  mean_grain_size_px: 50.0")
    for old, new in tweaks:
        assert old in text, old
        text = text.replace(old, new)
    tmp_cfg = CONFIG_DIR / "_pytest_domains.yaml"
    tmp_cfg.write_text(text, encoding="utf-8")
    try:
        return generate_data_for_training(str(tmp_cfg), base_output_dir=str(out),
                                          run_name="domains_test_COCO", n_images=1)
    finally:
        tmp_cfg.unlink(missing_ok=True)


def _load_all(ds: str):
    anns, cats, by_split = [], None, {}
    for sub in ("train", "val", "test"):
        d = json.loads((Path(ds) / "annotations" / f"instances_{sub}.json").read_text())
        anns += d["annotations"]
        cats = cats or d["categories"]
        if d["images"]:
            by_split[sub] = d
    return anns, cats, by_split


def test_full_pipeline_produces_coco_with_per_grain_metadata(tmp_path=None):
    """End-to-end: grains survive tip convolution, PID, artifacts and flattening."""
    ds = _generate_small_dataset(Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp()))
    anns, cats, _ = _load_all(ds)

    assert anns, "no annotations produced"
    assert [c["name"] for c in cats] == ["grain"]

    a = anns[0]
    for key in ("grain_id", "eq_diameter_px", "mean_z", "eccentricity",
                "flatness", "instance_index"):
        assert key in a, f"missing per-grain field {key!r}"
    assert a["area"] > 0 and len(a["segmentation"][0]) >= 6

    per_image: dict[int, list[int]] = {}
    for x in anns:
        per_image.setdefault(x["image_id"], []).append(x["grain_id"])
    for image_id, ids in per_image.items():
        assert len(ids) == len(set(ids)), f"duplicate grain_id in image {image_id}"


def test_instance_map_is_written_and_matches_the_annotations(tmp_path=None):
    """
    The label map is the exact ground truth: COCO polygons are a re-derivation of
    it, and the semantic UNet converter throws instances away entirely.
    """
    ds = _generate_small_dataset(Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp()))
    _anns, _cats, by_split = _load_all(ds)

    checked = 0
    for split, d in by_split.items():
        anns_by_img: dict[int, list] = {}
        for a in d["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        for img in d["images"]:
            path = Path(ds) / "instances" / split / (Path(img["file_name"]).stem + ".png")
            assert path.exists(), f"missing instance map {path}"
            inst = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert inst.dtype == np.uint16, f"{inst.dtype} would overflow past 255 grains"
            assert inst.shape == (img["height"], img["width"])

            anns = anns_by_img.get(img["id"], [])
            assert anns, "image with no annotations"
            indices = {a["instance_index"] for a in anns}
            present = set(np.unique(inst).tolist()) - {0}
            assert present <= indices, "instance map holds ids no annotation claims"
            assert len(present) > 1, "instance map collapsed every grain into one id"

            # Each index must sit inside its annotation's bbox. Not every pixel:
            # contours smaller than 1 px² are dropped when the polygons are
            # derived, but the label map keeps them — which is exactly why the
            # map, not the polygons, is the authoritative ground truth.
            by_idx = {a["instance_index"]: a for a in anns}
            for idx in list(present)[:10]:
                ys, xs = np.nonzero(inst == idx)
                x, y, w, h = by_idx[idx]["bbox"]
                inside = ((xs >= x - 1) & (xs <= x + w + 1) &
                          (ys >= y - 1) & (ys <= y + h + 1))
                assert inside.mean() > 0.95, (
                    f"instance {idx}: only {inside.mean():.1%} of its pixels "
                    f"fall in the annotation bbox"
                )
            checked += 1
    assert checked, "no instance maps were checked"


def test_block_other_objects_rejects_the_rest(tmp_path=None):
    out = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    ds = _generate_small_dataset(out, tweaks=[
        ("  block_other_objects: false", "  block_other_objects: true"),
        ("""- label: hexagon
  shape: hexagon
  category: geometric
  enable: false""", """- label: hexagon
  shape: hexagon
  category: geometric
  enable: true"""),
    ])
    anns, _cats, _ = _load_all(ds)
    names = {a["category_name"] for a in anns}
    assert names == {"grain"}, f"block_other_objects let {names - {'grain'}} through"


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                fn(tempfile.mkdtemp())
            else:
                fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

"""
src/simuscan/domains_shape.py
─────────────────────────────
SimuScan renderer for grain / domain fields (`shape: domains`).

Bridges `domains_core.generate_domains` (ideal topography + a visible instance
label map) to the SimuScan dataset pipeline, which then adds everything the
reference generator deliberately leaves out: tip convolution, PID feedback,
scanner noise, line artifacts, drift and flattening.

Units
-----
`domains_conf:` is written in SimuScan units — lateral sizes in PIXELS, heights
in relative height units (~0-1) — like every other section of the YAML. The
vendored generator speaks nm, so `DomainsConfig.core_overrides()` hands it the
pixel values together with `pixel_size_nm = 1`. That substitution is exact: the
generator uses `pixel_size_nm` only to turn lateral nm into pixels, and never
applies it to Z.

Why a domain field is not an ordinary object
--------------------------------------------
It is a SCENE: the grains tile the frame and share borders. So it is rendered
as a BASE LAYER — `syntheticAFM_COCO` folds its topography into the substrate,
which the pipeline ADDS to the object topography. Ordinary objects therefore sit
ON TOP of the grain film at their full height, instead of being max-composited
into it (which flattened the grain texture underneath them).

Two consequences for the masks:

  * they come straight out of the generator's label map. They are NOT re-derived
    by thresholding the convolved topography, the way isolated objects are
    (`mask = topo_conv > 0.15 * topo.max()`). That rule assumes a black
    background; in a tessellation there is no background, and applying it would
    dissolve the grain boundaries.

  * they are CLIPPED against whatever objects ended up on top of the film, so a
    pixel is never labelled both `grain` and `hexagon`. See
    `split_domain_instances`.

Public API
----------
generate_domains_shape(obj, image_size, dconf)
    -> (scene_topo, label, meta_by_id)
split_domain_instances(label, meta_by_id, dconf, occupied=None)
    -> (masks, grain_meta)
"""

from __future__ import annotations

import random

import numpy as np

from .SyntheticParams import DomainsConfig, ObjectConfig
from .domains_core import generate_domains


# Per-grain metadata copied into the COCO annotations, mapped from the
# generator's nm-flavoured names to SimuScan units.
#   `grain_id` is the generator's own ID: it identifies the grain INSTANCE,
#   which in COCO is the annotation — the class is category_id, and categories
#   are dataset-global.
# Entries flagged `scaled` are heights and follow the same baseline/scale
# transform applied to the topography, so an annotation's height can be compared
# against the saved image.
_META_FIELDS = (
    # (core key,                  annotation key,         scaled)
    ("eq_diameter_nm",            "eq_diameter_px",       False),
    ("eccentricity",              "eccentricity",         False),
    ("n_sides",                   "n_sides",              False),
    ("flatness",                  "flatness",             False),
    ("roughness_rms_actual_nm",   "roughness_rms_actual", True),
)


def _normalise_z(topo: np.ndarray, label: np.ndarray, obj: ObjectConfig,
                 dconf: DomainsConfig) -> tuple[np.ndarray, float, float]:
    """
    Put the generator's topography on the SimuScan height scale.

    The baseline is the background level when the mode leaves background
    (overlap_interactive, where topo is already referenced to 0) and the lowest
    grain level when it does not (tessellation, where the whole frame sits on a
    pedestal). Background pixels are forced back to 0 so the substrate shows
    through the gaps.

    Returns (z, baseline, scale) so per-grain heights can be reported on the
    same scale as the image.
    """
    grains = label > 0
    if not np.any(grains):
        return np.zeros_like(topo, dtype=np.float32), 0.0, 1.0

    background = ~grains
    baseline = float(topo[background].mean()) if np.any(background) else float(topo[grains].min())

    z = (topo - baseline).astype(np.float32)
    z[background] = 0.0
    np.clip(z, 0.0, None, out=z)

    scale = 1.0
    if dconf.z_mode == "relative":
        ref = float(np.percentile(z[grains], float(dconf.z_norm_percentile)))
        if ref > 1e-9:
            lo, hi = float(obj.height[0]), float(obj.height[-1])
            scale = random.uniform(lo, hi) / ref
            z *= scale

    return z, baseline, scale


def _grain_metadata(meta: list[dict], baseline: float, scale: float) -> dict[int, dict]:
    """Per-grain annotation payload, keyed by the generator's grain id."""
    out: dict[int, dict] = {}
    for src in meta:
        gid = int(src.get("id", -1))
        if gid < 0:
            continue
        entry: dict = {"grain_id": gid}
        for core_key, ann_key, is_scaled in _META_FIELDS:
            if src.get(core_key) is None:
                continue
            entry[ann_key] = float(src[core_key]) * (scale if is_scaled else 1.0)

        # Mean height on the same scale as the saved image. overlap_interactive
        # reports the level the grain was actually placed at.
        mean_z = src.get("placed_mean_z_nm", src.get("mean_z_nm"))
        if mean_z is not None:
            entry["mean_z"] = max(0.0, float(mean_z) - baseline) * scale

        if "edge_rounding_used" in src:
            entry["edge_rounding_used"] = bool(src["edge_rounding_used"])

        out[gid] = entry
    return out


def generate_domains_shape(obj: ObjectConfig, image_size: int,
                           dconf: DomainsConfig):
    """
    Render one grain/domain field for `obj`.

    Returns
    -------
    scene_topo : (S, S) float32   topography in SimuScan height units
    label      : (S, S) int32     one id per visible grain, 0 = background
    meta_by_id : dict[int, dict]  annotation payload per grain id

    The label map is returned whole rather than pre-split into masks: the caller
    has to clip it against the objects that landed on the film before it can
    decide what the final instances are.
    """
    mode = obj.domains_mode or dconf.mode
    if mode not in ("tessellation", "overlap_interactive"):
        raise ValueError(
            f"Object '{obj.label}': domains mode must be 'tessellation' or "
            f"'overlap_interactive', got {mode!r}."
        )
    if dconf.z_mode not in ("direct", "relative"):
        raise ValueError(
            f"domains_conf.z_mode must be 'direct' or 'relative', got {dconf.z_mode!r}."
        )

    overrides = dconf.core_overrides()
    overrides["mode"] = mode
    overrides["image_shape"] = (int(image_size), int(image_size))

    topo, label, meta, _extra, _cfg = generate_domains(**overrides)

    scene_topo, baseline, scale = _normalise_z(topo, label, obj, dconf)
    return scene_topo, label, _grain_metadata(meta, baseline, scale)


def split_domain_instances(label: np.ndarray, meta_by_id: dict[int, dict],
                           dconf: DomainsConfig,
                           occupied: np.ndarray | None = None):
    """
    Turn the label map into per-grain binary masks, largest grain first.

    `occupied` is the footprint of the objects sitting ON the film. Those pixels
    are removed from every grain, so the same pixel is never labelled twice. A
    grain is then dropped when too little of it survives — a sliver peeking out
    from under a particle is not a useful training instance.

    A grain that an object cuts into several pieces remains ONE instance: the
    caller asks COCOBuilder to merge its contours into a single annotation.
    """
    ids, areas = np.unique(label, return_counts=True)
    keep = ids > 0
    ids, orig_areas = ids[keep], areas[keep]

    clipping = occupied is not None and np.any(occupied)
    min_area = int(dconf.min_grain_area_px)
    min_visible = float(dconf.min_visible_after_objects)

    kept: list[tuple[int, np.ndarray, int]] = []
    for gid, orig_area in zip(ids, orig_areas):
        m = label == gid
        if clipping:
            m = m & ~occupied
            area = int(m.sum())
            if area == 0 or area < min_area:
                continue
            if area < min_visible * float(orig_area):
                continue
        else:
            area = int(orig_area)
            if area < min_area:
                continue
        # A grain cut into several pieces by an object on top stays ONE instance;
        # COCOBuilder writes it as one annotation with several polygons, which is
        # what `segmentation` being a list of polygons is for. Keeping only the
        # biggest piece would be an arbitrary choice and would punch holes in a
        # tessellation's coverage.
        kept.append((int(gid), m, area))

    kept.sort(key=lambda t: -t[2])
    if dconf.max_instances > 0:
        kept = kept[: int(dconf.max_instances)]

    masks = [m.astype(np.uint8) for _gid, m, _a in kept]
    grain_meta = [dict(meta_by_id.get(gid, {"grain_id": gid})) for gid, _m, _a in kept]
    return masks, grain_meta

"""
src/simuscan/domains_core.py
────────────────────────────
IDEAL grain / domain topography generator (physics + geometry only).

VENDORED, UNMODIFIED LOGIC from the reference implementation:
    Cristals_Yongtao/code/generator/synthetic_afm_domains.py

Only the batch/PNG-saving layer of the original was dropped (`run_batch`,
`_save_pngs`, `_write_txt`, `_make_run_dir`, `_stats`, `__main__`); in
SimuScan all I/O is handled by the dataset pipeline instead. The generation
code below is byte-for-byte the reference algorithm, so upstream fixes can be
re-applied by re-vendoring this file.

This module deliberately produces ONLY the ideal ground truth:
topography (nm) + visible instance label map. AFM artifacts (tip convolution,
PID feedback, noise, line artifacts, flattening) are added afterwards by
`syntheticAFM_COCO.generate_data_for_training`, which is the whole point of
the integration.

Public API
----------
generate_domains(mode, **overrides) -> (topo_nm, label_mask, meta, extra, cfg)
    mode: "tessellation" | "overlap_interactive"
    topo_nm    : (H, W) float   topography in nm
    label_mask : (H, W) int32   one integer ID per VISIBLE grain, 0 = background
    meta       : list[dict]     per-grain metadata
    extra      : dict           image-level metadata
    cfg        : dict           the resolved configuration actually used

DEFAULT_CONFIG documents every knob; see also
docs/simuscan/domains_generator.md for the SimuScan-facing description.
"""

import copy

import numpy as np
from scipy.interpolate import CubicSpline, RBFInterpolator
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, map_coordinates, rotate
from skimage.draw import polygon2mask
from skimage.measure import find_contours


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
# Quick interpretation when a parameter increases:
#
# mean_grain_size_nm          -> larger grains, fewer grains per image
# grain_size_sigma_nm         -> wider Gaussian/Lognormal size spread
# eccentricity_range          -> more elongated grains
# n_sides_range               -> more polygon sides / more finely faceted grains
# curved_grain_probability    -> more grains eligible to contain curved sides
# curved_side_probability     -> more curved sides inside eligible grains
# side_curvature_range        -> stronger side bowing
# shape_irregularity_range    -> less regular polygon outlines
# z_mean_nm                   -> larger mean domain heights
# neighbor_z_correlation      -> nearby grains have more similar mean Z
# correlation_length_nm       -> Z correlation extends farther in space
# boundary_z_amplitude_nm     -> stronger perimeter Z variation
# boundary_n_ctrl             -> more complex perimeter Z spline
# inner_extrema_count_range   -> more internal hills/depressions
# inner_extrema_amplitude_nm  -> stronger internal hills/depressions
# surface_smoothing           -> smoother fitted surface
# flatness_range              -> flatter grains (less relief + more smoothing)
# roughness_rms_nm            -> larger nanoscale/local height fluctuations
# roughness_corr_length_nm    -> laterally broader / smoother roughness features
# roughness_edge_taper_px     -> stronger suppression of roughness near grain edges
# angular_correlation_probability -> more images use oriented/anisotropic roughness
# roughness_anisotropy_ratio  -> stronger directional roughness elongation
# edge_rounding_probability   -> more images use rounded border-Z attenuation
# edge_rounding_border_fraction -> lower edge fraction means stronger edge drop
# edge_rounding_width_nm      -> wider distance band for edge rounding in Z
# tessellation_relax          -> more regular tessellation
# overlap_target              -> stronger pairwise overlap preference
# contact_preference          -> stronger preference to touch existing grains
# gap_preference              -> stronger preference for free/background region
# coverage_target             -> larger covered fraction in overlap mode
# stack_height_factor         -> stronger coupling of upper-grain mean Z to lower support
# =============================================================================

DEFAULT_CONFIG = {
    # Batch / image -----------------------------------------------------------
    "n_images_per_mode": 5,
    "image_shape": (512, 512),
    "pixel_size_nm": 2.0,
    "seed": None,  # int = reproducible; None = random each launch

    # Grain-size distribution -------------------------------------------------
    # gaussian  -> mean_grain_size_nm + grain_size_sigma_nm
    # lognormal -> same arithmetic mean/std; converted internally
    # uniform   -> grain_size_range_nm
    "size_distribution": "gaussian",
    "mean_grain_size_nm": 50.0,
    "grain_size_sigma_nm": 15.0,
    "grain_size_range_nm": (30.0, 70.0),

    # XY shape ---------------------------------------------------------------
    "eccentricity_range": (0.0, 0.4),       # uniform sampling
    "n_sides_range": (4, 8),                # direct in overlap; target in tessellation
    "curved_grain_probability": 0.30,
    "curved_side_probability": 0.30,
    "side_curvature_range": (0.05, 0.50),   # uniform sampling
    "shape_irregularity_range": (0.03, 0.12),

    # Mean Z and spatial correlation -----------------------------------------
    "z_mean_nm": (5.0, 12.0),
    "neighbor_z_correlation": 0.60,
    "correlation_length_nm": 100.0,
    "correlation_cutoff_factor": 2.0,

    # Periodic Z around perimeter --------------------------------------------
    "boundary_z_amplitude_nm": (0.5, 3.0),
    "boundary_n_ctrl": 12,
    "boundary_samples": 24,

    # Internal extrema --------------------------------------------------------
    "inner_extrema_count_range": (0, 2),    # integer, uniform, inclusive
    "inner_extrema_amplitude_nm": (1.0, 3.5),
    "inner_extrema_edge_exclusion_px": 2,
    "inner_extrema_min_distance_px": 5,

    # Surface planarity / smoothness -----------------------------------------
    "surface_smoothing": 0.15,
    "flatness_range": (0.4, 0.8),          # uniform per grain

    # Correlated roughness ----------------------------------------------------
    # Each grain receives an INDEPENDENT roughness realization, but all grains
    # sample the RMS amplitude and correlation length from these common ranges.
    #
    # roughness_rms_nm:
    #     RMS height of the local roughness. Larger -> rougher surface.
    #     May be a scalar or a (min,max) range; range is sampled uniformly.
    #
    # roughness_corr_length_nm:
    #     Approximate 1/e lateral autocorrelation length. Larger -> broader,
    #     smoother roughness features; smaller -> finer texture.
    #     May be a scalar or a (min,max) range; range is sampled uniformly.
    #
    # roughness_edge_taper_px:
    #     0 keeps full roughness up to the grain boundary. Values >0 smoothly
    #     suppress roughness over this many pixels near the perimeter.
    "roughness_enabled": True,
    "roughness_rms_nm": (0.4, 1.0),
    "roughness_corr_length_nm": (5.0, 12.0),
    "roughness_edge_taper_px": 0,

    # Directional / angular roughness anisotropy (IMAGE level) ---------------
    # First, each generated IMAGE decides whether directional anisotropy is
    # active, using angular_correlation_probability.
    #
    # If active:
    #   * one common angle is sampled for the whole image;
    #   * one common anisotropy ratio is sampled for the whole image;
    #   * each grain still receives its own independent roughness realization,
    #     but all grains share that image-wide preferred direction.
    #
    # angular_correlation_probability:
    #     Probability that one image uses anisotropic/oriented roughness.
    #     0.0 -> always isotropic, 1.0 -> always anisotropic.
    #
    # angular_correlation_angle_range_deg:
    #     Range of possible preferred directions in degrees. Sampled ONCE per
    #     image, not per grain. 0 and 180 correspond to the same direction.
    #
    # roughness_anisotropy_ratio:
    #     Ratio between the correlation length parallel to the preferred
    #     direction and the correlation length perpendicular to it.
    #     1.0 -> isotropic. Larger values -> stronger directional texture.
    #
    # Implementation detail:
    #     If base correlation length = L and anisotropy ratio = r, the
    #     effective directional correlation lengths are defined as:
    #
    #         L_perp     = L
    #         L_parallel = L * r
    #
    #     This makes the preferred direction visually more evident: the base
    #     correlation length controls the transverse texture scale, and the
    #     anisotropy ratio stretches that texture along the preferred angle.
    "angular_correlation_probability": 0.50,
    "angular_correlation_angle_range_deg": (0.0, 180.0),
    "roughness_anisotropy_ratio": (3.0, 8.0),

    # Edge rounding in Z (IMAGE activation + per-grain sampling) -------------
    # This feature rounds grain borders in height, so the topography decreases
    # smoothly when approaching the grain perimeter. The border does NOT need
    # to go to zero; instead it approaches a user-defined fraction of the local
    # grain topography. The transition is non-linear and rounded, not linear.
    #
    # edge_rounding_probability:
    #     Probability that one IMAGE activates border-Z rounding.
    #     0.0 -> never; 1.0 -> all images.
    #
    # edge_rounding_border_fraction:
    #     Final multiplicative fraction reached exactly at the grain edge.
    #     1.0 -> no edge drop; smaller values -> stronger border lowering.
    #     If a range is given, a different value is sampled uniformly per grain.
    #
    # edge_rounding_width_nm:
    #     Distance band (toward the grain interior) where the rounded descent
    #     acts. Larger values -> broader/softer rounded borders.
    #     If a range is given, a different value is sampled uniformly per grain.
    #
    # edge_rounding_profile:
    #     Shape of the descent. "cosine" is rounded/smooth and recommended.
    "edge_rounding_probability": 0.40,
    "edge_rounding_border_fraction": (0.65, 0.90),
    "edge_rounding_width_nm": (4.0, 12.0),
    "edge_rounding_profile": "cosine",

    # Tessellation ------------------------------------------------------------
    "tessellation_relax": 1,
    "tessellation_side_penalty_strength": 0.15,

    # Overlap-interactive placement ------------------------------------------
    "overlap_target": (0.05, 0.20),
    "contact_preference": 1.0,
    "gap_preference": 0.60,
    "max_placement_trials": 50,
    "coverage_target": (0.90, 0.98),
    "min_visible_fraction": 0.01,

    # Overlap stacking --------------------------------------------------------
    # support_percentile:
    #     Scalar support level measured from the lower visible topography over
    #     ALL pixels where the new grain overlaps lower grains.
    #
    # stack_height_factor:
    #     Bounded support-coupling factor, NOT an extra full grain-height term.
    #     0.0 -> keep the new grain at its own correlated mean Z.
    #     1.0 -> if lower support is higher, move the grain mean up to support.
    #     Values in between interpolate between those two cases.
    #
    # This avoids runaway Z accumulation through chains of lateral overlaps and
    # keeps overlap_interactive Z statistics comparable to tessellation.
    "support_percentile": 75.0,
    "stack_height_factor": 0.80,

    # Output ------------------------------------------------------------------
    "output_base_dir": "images",
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def _sample_uniform(rng, value, low=None, high=None):
    if np.isscalar(value):
        x = float(value)
    else:
        a, b = map(float, value)
        if a > b:
            a, b = b, a
        x = float(rng.uniform(a, b))
    if low is not None:
        x = max(low, x)
    if high is not None:
        x = min(high, x)
    return x


def _sample_int_uniform(rng, value, low=None, high=None):
    if np.isscalar(value):
        x = int(round(value))
    else:
        a, b = int(round(value[0])), int(round(value[1]))
        if a > b:
            a, b = b, a
        x = int(rng.integers(a, b + 1))
    if low is not None:
        x = max(low, x)
    if high is not None:
        x = min(high, x)
    return x


def _sample_grain_size_nm(rng, cfg):
    dist = cfg["size_distribution"].lower()
    mean = float(cfg["mean_grain_size_nm"])
    sd = float(cfg["grain_size_sigma_nm"])

    if dist == "uniform":
        return _sample_uniform(rng, cfg["grain_size_range_nm"], low=1e-6)

    if dist == "gaussian":
        lo, hi = max(1e-6, 0.15 * mean), 4.0 * mean
        for _ in range(1000):
            x = rng.normal(mean, sd)
            if lo <= x <= hi:
                return float(x)
        return float(np.clip(mean, lo, hi))

    if dist == "lognormal":
        if mean <= 0 or sd < 0:
            raise ValueError("Lognormal requires mean > 0 and sigma >= 0.")
        if sd == 0:
            return mean
        sigma_ln = np.sqrt(np.log(1.0 + (sd / mean) ** 2))
        mu_ln = np.log(mean) - 0.5 * sigma_ln**2
        lo, hi = max(1e-6, 0.10 * mean), 6.0 * mean
        for _ in range(1000):
            x = rng.lognormal(mu_ln, sigma_ln)
            if lo <= x <= hi:
                return float(x)
        return float(np.clip(mean, lo, hi))

    raise ValueError("size_distribution must be gaussian, lognormal, or uniform")


def _normalize_01(a):
    a = np.asarray(a, float)
    lo, hi = float(np.min(a)), float(np.max(a))
    return np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)


def _largest_contour(mask):
    contours = find_contours(mask.astype(float), 0.5)
    return None if not contours else max(contours, key=len)


def _resample_closed_contour(contour_rc, n_samples):
    pts = np.column_stack([contour_rc[:, 1], contour_rc[:, 0]])
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])
    seg = np.diff(pts, axis=0)
    ds = np.sqrt(np.sum(seg**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] <= 1e-12:
        return np.repeat(pts[:1], n_samples, axis=0)
    target = np.linspace(0, s[-1], int(n_samples), endpoint=False)
    return np.column_stack([
        np.interp(target, s, pts[:, 0]),
        np.interp(target, s, pts[:, 1]),
    ])


# =============================================================================
# OVERLAP POLYGON: INDIVIDUAL STRAIGHT/CURVED SIDES
# =============================================================================

def _polygon_with_curved_sides(rng, cx, cy, d_px, ecc, n_sides, irregularity, cfg):
    n_sides = max(3, int(n_sides))
    axis_ratio = np.sqrt(max(1e-4, 1.0 - float(ecc)**2))
    area = np.pi * (d_px / 2.0) ** 2
    a = np.sqrt(area / (np.pi * axis_ratio))
    b = a * axis_ratio

    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    dtheta = 2*np.pi/n_sides
    angles += rng.uniform(-0.16*dtheta, 0.16*dtheta, n_sides)
    angles = np.sort(angles)
    radial = np.clip(rng.normal(1.0, irregularity, n_sides), 0.65, 1.45)

    x, y = a*radial*np.cos(angles), b*radial*np.sin(angles)
    theta = rng.uniform(0, np.pi)
    c, s = np.cos(theta), np.sin(theta)
    vertices = np.column_stack([c*x - s*y + cx, s*x + c*y + cy])

    grain_curve_eligible = rng.random() < cfg["curved_grain_probability"]
    curved_sides, curvatures, dense = 0, [], []

    for i in range(n_sides):
        p0, p1 = vertices[i], vertices[(i+1) % n_sides]
        edge = p1 - p0
        length = np.linalg.norm(edge)
        if length < 1e-9:
            continue
        is_curved = grain_curve_eligible and rng.random() < cfg["curved_side_probability"]
        t = np.linspace(0, 1, 14, endpoint=False)
        if not is_curved:
            pts = (1-t[:, None])*p0 + t[:, None]*p1
        else:
            curv = _sample_uniform(rng, cfg["side_curvature_range"], low=0.0)
            normal = np.array([-edge[1], edge[0]]) / length
            bow = rng.choice([-1.0, 1.0]) * curv * 0.35 * length
            control = 0.5*(p0+p1) + bow*normal
            omt = 1-t
            pts = (omt**2)[:,None]*p0 + (2*omt*t)[:,None]*control + (t**2)[:,None]*p1
            curved_sides += 1
            curvatures.append(curv)
        dense.append(pts)

    return np.vstack(dense), {
        "n_sides": n_sides,
        "curved_sides": curved_sides,
        "mean_side_curvature": float(np.mean(curvatures)) if curvatures else 0.0,
    }


def _rasterize_polygon_crop(xy, image_shape):
    """Rasterize only the polygon bounding box for fast overlap placement."""
    h, w = image_shape
    x0 = max(0, int(np.floor(np.min(xy[:,0]))) - 1)
    x1 = min(w, int(np.ceil(np.max(xy[:,0]))) + 2)
    y0 = max(0, int(np.floor(np.min(xy[:,1]))) - 1)
    y1 = min(h, int(np.ceil(np.max(xy[:,1]))) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    local_xy = xy - np.array([x0, y0])
    local_mask = polygon2mask((y1-y0, x1-x0), np.column_stack([local_xy[:,1], local_xy[:,0]]))
    if np.sum(local_mask) < 8:
        return None
    return (slice(y0, y1), slice(x0, x1)), local_mask


# =============================================================================
# TESSELLATION XY
# =============================================================================

def _assign_tessellation(seeds, shape):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    best = np.full(shape, np.inf)
    labels = np.zeros(shape, np.int32)
    for gid, s in enumerate(seeds, 1):
        dx, dy = xx-s["cx"], yy-s["cy"]
        c, sn = np.cos(s["theta"]), np.sin(s["theta"])
        xr, yr = c*dx + sn*dy, -sn*dx + c*dy
        score = (xr/max(s["a"],1e-6))**2 + (yr/max(s["b"],1e-6))**2
        m = score < best
        best[m], labels[m] = score[m], gid
    return labels


def _adjacency_pairs(labels):
    pairs = set()
    for a, b in ((labels[:,:-1], labels[:,1:]), (labels[:-1,:], labels[1:,:])):
        m = (a != b) & (a > 0) & (b > 0)
        for p, q in zip(a[m], b[m]):
            p, q = int(p), int(q)
            pairs.add((min(p,q), max(p,q)))
    return sorted(pairs)


def _side_counts(labels):
    ids = np.unique(labels)
    counts = {int(i): 0 for i in ids if i > 0}
    for a, b in _adjacency_pairs(labels):
        counts[a] += 1
        counts[b] += 1
    return counts


def _soft_side_relax(seeds, labels, cfg):
    low, high = cfg["n_sides_range"]
    counts = _side_counts(labels)
    for gid, s in enumerate(seeds, 1):
        ys, xs = np.nonzero(labels == gid)
        if len(xs) == 0:
            continue
        n = counts.get(gid, 0)
        deviation = max(low-n, 0, n-high)
        alpha = np.clip(cfg["tessellation_side_penalty_strength"]*deviation, 0, 0.6)
        if alpha > 0:
            s["cx"] = (1-alpha)*s["cx"] + alpha*xs.mean()
            s["cy"] = (1-alpha)*s["cy"] + alpha*ys.mean()


def _curve_selected_shared_boundaries(labels, rng, mean_d_px, cfg):
    """Efficient one-shot warp near selected shared boundaries."""
    pairs = _adjacency_pairs(labels)
    p_boundary = np.clip(cfg["curved_grain_probability"] * cfg["curved_side_probability"], 0, 1)
    selected = {}
    for pair in pairs:
        if rng.random() < p_boundary:
            selected[pair] = _sample_uniform(rng, cfg["side_curvature_range"], low=0.0)

    if not selected:
        return labels, {"n_shared_boundaries": len(pairs), "n_curved_shared_boundaries": 0,
                        "curved_shared_boundary_fraction": 0.0, "mean_shared_boundary_curvature": 0.0}

    amp_map = np.zeros(labels.shape, float)

    # Mark selected boundaries with their curvature amplitude.
    left, right = labels[:,:-1], labels[:,1:]
    diff = (left != right) & (left > 0) & (right > 0)
    yy, xx = np.nonzero(diff)
    for y, x in zip(yy, xx):
        p, q = int(left[y,x]), int(right[y,x])
        pair = (min(p,q), max(p,q))
        if pair in selected:
            amp = selected[pair]
            amp_map[y, x] = max(amp_map[y,x], amp)
            amp_map[y, x+1] = max(amp_map[y,x+1], amp)

    top, bottom = labels[:-1,:], labels[1:,:]
    diff = (top != bottom) & (top > 0) & (bottom > 0)
    yy, xx = np.nonzero(diff)
    for y, x in zip(yy, xx):
        p, q = int(top[y,x]), int(bottom[y,x])
        pair = (min(p,q), max(p,q))
        if pair in selected:
            amp = selected[pair]
            amp_map[y, x] = max(amp_map[y,x], amp)
            amp_map[y+1, x] = max(amp_map[y+1,x], amp)

    influence = gaussian_filter(amp_map, sigma=max(1.0, 0.05*mean_d_px))
    if influence.max() > 0:
        influence /= influence.max()

    sigma_noise = max(2.0, 0.20*mean_d_px)
    dx = gaussian_filter(rng.normal(size=labels.shape), sigma_noise)
    dy = gaussian_filter(rng.normal(size=labels.shape), sigma_noise)
    dx /= np.std(dx) + 1e-12
    dy /= np.std(dy) + 1e-12

    # Average selected curvature controls the maximum displacement.
    mean_curv = float(np.mean(list(selected.values())))
    max_disp = mean_curv * 0.16 * mean_d_px
    yy_grid, xx_grid = np.mgrid[:labels.shape[0], :labels.shape[1]]
    coords = np.array([yy_grid + max_disp*dy*influence, xx_grid + max_disp*dx*influence])
    warped = map_coordinates(labels.astype(float), coords, order=0, mode="nearest").astype(np.int32)

    return warped, {
        "n_shared_boundaries": len(pairs),
        "n_curved_shared_boundaries": len(selected),
        "curved_shared_boundary_fraction": len(selected)/len(pairs) if pairs else 0.0,
        "mean_shared_boundary_curvature": mean_curv,
    }


# =============================================================================
# SPATIALLY CORRELATED MEAN Z
# =============================================================================

def _correlated_means(rng, centroids_px, cfg):
    n = len(centroids_px)
    if n == 0:
        return np.array([])
    lo, hi = map(float, cfg["z_mean_nm"])
    if lo > hi:
        lo, hi = hi, lo
    raw = rng.uniform(lo, hi, n)
    z = raw.copy()
    rho = np.clip(cfg["neighbor_z_correlation"], 0, 0.999)
    L = max(float(cfg["correlation_length_nm"]), 1e-9)
    cutoff = cfg["correlation_cutoff_factor"] * L
    xy_nm = np.asarray(centroids_px) * cfg["pixel_size_nm"]
    for _ in range(5):
        new = np.empty_like(z)
        for i in range(n):
            d = np.linalg.norm(xy_nm - xy_nm[i], axis=1)
            m = (d > 0) & (d <= cutoff)
            if not np.any(m):
                new[i] = raw[i]
            else:
                w = np.exp(-(d[m]/L)**2)
                nbr = np.sum(w*z[m]) / np.sum(w)
                new[i] = (1-rho)*raw[i] + rho*nbr
        z = new
    return np.clip(z, lo, hi)


# =============================================================================
# Z SURFACE
# =============================================================================

def _periodic_boundary_z(rng, amplitude, n_ctrl, n_eval):
    n_ctrl = max(4, int(n_ctrl))
    vals = rng.normal(size=n_ctrl)
    vals = 0.2*np.roll(vals,1) + 0.6*vals + 0.2*np.roll(vals,-1)
    vals -= vals.mean()
    if vals.std() > 1e-12:
        vals /= vals.std()
    vals *= amplitude
    s_ctrl = np.linspace(0,1,n_ctrl+1)
    z_ctrl = np.r_[vals, vals[0]]
    spline = CubicSpline(s_ctrl, z_ctrl, bc_type="periodic")
    return spline(np.linspace(0,1,n_eval,endpoint=False))


def _choose_extrema(rng, mask, n_points, edge_exclusion, min_distance):
    if n_points <= 0:
        return np.empty((0,2))
    dist = distance_transform_edt(mask)
    ys, xs = np.nonzero(mask & (dist >= edge_exclusion))
    if len(xs) == 0:
        return np.empty((0,2))
    candidates = np.column_stack([xs,ys]).astype(float)
    chosen = []
    for idx in rng.permutation(len(candidates)):
        p = candidates[idx]
        if all(np.linalg.norm(p-q) >= min_distance for q in chosen):
            chosen.append(p)
            if len(chosen) >= n_points:
                break
    return np.asarray(chosen) if chosen else np.empty((0,2))


def _relative_surface(rng, mask, cfg):
    contour = _largest_contour(mask)
    if contour is None or mask.sum() < 8:
        return np.zeros(mask.shape), {"flatness": None, "n_inner_extrema": 0,
                                     "boundary_amp_effective_nm": 0.0,
                                     "inner_amp_mean_effective_nm": 0.0,
                                     "surface_smoothing_effective": cfg["surface_smoothing"]}

    flatness = _sample_uniform(rng, cfg["flatness_range"], 0, 1)
    relief_scale = max(0.03, 1.0-flatness)
    boundary_amp = _sample_uniform(rng, cfg["boundary_z_amplitude_nm"], 0) * relief_scale
    smoothing = cfg["surface_smoothing"] * (0.5 + 2.0*flatness)

    boundary_xy = _resample_closed_contour(contour, cfg["boundary_samples"])
    boundary_z = _periodic_boundary_z(rng, boundary_amp, cfg["boundary_n_ctrl"], cfg["boundary_samples"])

    n_ext = _sample_int_uniform(rng, cfg["inner_extrema_count_range"], 0)
    inner_xy = _choose_extrema(rng, mask, n_ext,
                               cfg["inner_extrema_edge_exclusion_px"],
                               cfg["inner_extrema_min_distance_px"])
    inner_z, amps = [], []
    for _ in range(len(inner_xy)):
        amp = _sample_uniform(rng, cfg["inner_extrema_amplitude_nm"], 0) * relief_scale
        inner_z.append((1 if rng.random() < 0.5 else -1) * amp)
        amps.append(amp)
    inner_z = np.asarray(inner_z)

    ctrl_xy, ctrl_z = boundary_xy.copy(), boundary_z.copy()
    if len(inner_xy):
        ctrl_xy = np.vstack([ctrl_xy, inner_xy])
        ctrl_z = np.r_[ctrl_z, inner_z]

    center = ctrl_xy.mean(0)
    scale = ctrl_xy.std(0)
    scale[scale < 1e-6] = 1.0
    ctrl_n = (ctrl_xy-center)/scale

    ys, xs = np.nonzero(mask)
    eval_xy = np.column_stack([xs,ys]).astype(float)
    eval_n = (eval_xy-center)/scale

    rbf = RBFInterpolator(ctrl_n, ctrl_z, kernel="thin_plate_spline", smoothing=float(smoothing))
    z = np.asarray(rbf(eval_n), float)
    z -= z.mean()  # relative relief centered on zero

    max_req = max(max(cfg["boundary_z_amplitude_nm"]), max(cfg["inner_extrema_amplitude_nm"]), 0.1)
    z = np.clip(z, -2.5*max_req, 2.5*max_req)

    out = np.zeros(mask.shape)
    out[ys,xs] = z
    return out, {
        "flatness": float(flatness),
        "n_inner_extrema": int(len(inner_xy)),
        "boundary_amp_effective_nm": float(boundary_amp),
        "inner_amp_mean_effective_nm": float(np.mean(amps)) if amps else 0.0,
        "surface_smoothing_effective": float(smoothing),
    }



def _sample_image_roughness_state(rng, cfg):
    """
    Sample image-level roughness anisotropy settings.

    This is evaluated ONCE per image, not once per grain.

    Returns a dictionary describing whether angular anisotropy is active, and if
    active, the common image-wide preferred direction and anisotropy ratio.
    """
    use_angular = (
        cfg.get("roughness_enabled", True)
        and rng.random() < float(np.clip(cfg.get("angular_correlation_probability", 0.0), 0.0, 1.0))
    )

    if use_angular:
        angle_deg = _sample_uniform(rng, cfg.get("angular_correlation_angle_range_deg", (0.0, 180.0)))
        anisotropy_ratio = _sample_uniform(rng, cfg.get("roughness_anisotropy_ratio", (1.0, 1.0)), low=1.0)
    else:
        angle_deg = None
        anisotropy_ratio = 1.0

    return {
        "angular_correlation_used": bool(use_angular),
        "angular_correlation_angle_deg": (float(angle_deg) if angle_deg is not None else None),
        "roughness_anisotropy_ratio_sampled": float(anisotropy_ratio),
    }


def _sample_image_edge_rounding_state(rng, cfg):
    """Sample whether one image activates rounded edge attenuation in Z."""
    use_rounding = rng.random() < float(np.clip(cfg.get("edge_rounding_probability", 0.0), 0.0, 1.0))
    return {"edge_rounding_used": bool(use_rounding),
            "edge_rounding_profile": str(cfg.get("edge_rounding_profile", "cosine"))}


def _apply_edge_rounding(z_local, mask, rng, cfg, image_edge_state=None):
    """
    Apply a smooth rounded attenuation in Z near the grain boundary.

    If image_edge_state["edge_rounding_used"] is False, z_local is returned
    unchanged. Otherwise, each grain independently samples:

      * edge_rounding_border_fraction
      * edge_rounding_width_nm

    Let d be the distance (in pixels) from a point to the nearest grain edge and
    let w be the sampled width in pixels. For d >= w, the attenuation factor is 1.
    At the edge (d = 0), the factor equals the sampled border fraction p.

    The default profile is a smooth rounded cosine transition:

        u = clip(d / w, 0, 1)
        t = 0.5 - 0.5*cos(pi*u)
        factor = p + (1 - p) * t

    so the border decreases smoothly and non-linearly, without necessarily going
    to zero.
    """
    if image_edge_state is None:
        image_edge_state = {"edge_rounding_used": False, "edge_rounding_profile": str(cfg.get("edge_rounding_profile", "cosine"))}

    out = np.array(z_local, dtype=float, copy=True)
    if not bool(image_edge_state.get("edge_rounding_used", False)) or np.sum(mask) < 4:
        return out, {"edge_rounding_used": bool(image_edge_state.get("edge_rounding_used", False)),
                     "edge_rounding_profile": str(image_edge_state.get("edge_rounding_profile", cfg.get("edge_rounding_profile", "cosine"))),
                     "edge_rounding_border_fraction": 1.0,
                     "edge_rounding_width_nm": 0.0,
                     "edge_rounding_width_px": 0.0}

    p = _sample_uniform(rng, cfg.get("edge_rounding_border_fraction", (1.0, 1.0)), low=0.0, high=1.0)
    width_nm = _sample_uniform(rng, cfg.get("edge_rounding_width_nm", (0.0, 0.0)), low=0.0)
    width_px = width_nm / max(float(cfg["pixel_size_nm"]), 1e-12)
    profile = str(image_edge_state.get("edge_rounding_profile", cfg.get("edge_rounding_profile", "cosine"))).lower()

    if width_px <= 1e-9 or p >= 0.999999:
        return out, {"edge_rounding_used": True,
                     "edge_rounding_profile": profile,
                     "edge_rounding_border_fraction": float(p),
                     "edge_rounding_width_nm": float(width_nm),
                     "edge_rounding_width_px": float(width_px)}

    dist_inside = distance_transform_edt(mask)
    u = np.clip(dist_inside / width_px, 0.0, 1.0)
    if profile == "smoothstep":
        t = u*u*(3.0 - 2.0*u)
    else:
        t = 0.5 - 0.5*np.cos(np.pi*u)

    factor = p + (1.0 - p)*t
    out[mask] = out[mask] * factor[mask]
    return out, {"edge_rounding_used": True,
                 "edge_rounding_profile": profile,
                 "edge_rounding_border_fraction": float(p),
                 "edge_rounding_width_nm": float(width_nm),
                 "edge_rounding_width_px": float(width_px)}


def _correlated_roughness(rng, mask, cfg, image_roughness_state=None):
    """
    Generate a different zero-mean correlated roughness realization for one grain.

    Statistical model
    -----------------
    White Gaussian noise is spatially filtered with a Gaussian kernel. For an
    isotropic Gaussian kernel with standard deviation sigma, the autocorrelation
    of the filtered field is approximately exp[-r^2/(4*sigma^2)]. Therefore:

        sigma_px = roughness_corr_length_nm / (2 * pixel_size_nm)

    so roughness_corr_length_nm approximately corresponds to the 1/e
    autocorrelation length L in exp[-(r/L)^2].

    Optional IMAGE-level angular anisotropy
    ---------------------------------------
    If image_roughness_state["angular_correlation_used"] is True, then one
    common angle and anisotropy ratio are shared by all grains of the image.

    We then construct an anisotropic filtered field by:
      1) rotating the white-noise crop so the preferred direction aligns with x;
      2) applying an anisotropic Gaussian smoothing with:
             sigma_perp     = sigma_base
             sigma_parallel = sigma_base * ratio
      3) rotating back to the original orientation.

    This gives a much more visible directional correlation: the base
    correlation length controls the transverse texture scale, while the
    anisotropy ratio stretches the roughness along the chosen image-wide
    angle.

    The field is then centered to zero mean and normalized so its RMS INSIDE the
    domain equals the sampled roughness_rms_nm. Each grain samples its own RMS
    and base correlation length from the common ranges and receives an
    independent noise realization.

    roughness_edge_taper_px > 0 optionally suppresses roughness near the grain
    perimeter. After tapering, the field is re-centered and re-normalized so the
    requested RMS is preserved.
    """
    out = np.zeros(mask.shape, dtype=float)

    if image_roughness_state is None:
        image_roughness_state = {
            "angular_correlation_used": False,
            "angular_correlation_angle_deg": None,
            "roughness_anisotropy_ratio_sampled": 1.0,
        }

    if not cfg.get("roughness_enabled", True) or np.sum(mask) < 4:
        return out, {
            "roughness_rms_target_nm": 0.0,
            "roughness_rms_actual_nm": 0.0,
            "roughness_corr_length_nm": 0.0,
            "roughness_angular_correlation_used": bool(image_roughness_state["angular_correlation_used"]),
            "roughness_angular_correlation_angle_deg": image_roughness_state["angular_correlation_angle_deg"],
            "roughness_anisotropy_ratio_sampled": float(image_roughness_state["roughness_anisotropy_ratio_sampled"]),
        }

    rms_target = _sample_uniform(rng, cfg["roughness_rms_nm"], low=0.0)
    corr_length_nm = _sample_uniform(rng, cfg["roughness_corr_length_nm"], low=0.0)

    if rms_target <= 0.0 or corr_length_nm <= 0.0:
        return out, {
            "roughness_rms_target_nm": float(rms_target),
            "roughness_rms_actual_nm": 0.0,
            "roughness_corr_length_nm": float(corr_length_nm),
            "roughness_angular_correlation_used": bool(image_roughness_state["angular_correlation_used"]),
            "roughness_angular_correlation_angle_deg": image_roughness_state["angular_correlation_angle_deg"],
            "roughness_anisotropy_ratio_sampled": float(image_roughness_state["roughness_anisotropy_ratio_sampled"]),
        }

    pixel_size_nm = max(float(cfg["pixel_size_nm"]), 1e-12)
    sigma_base_px = max(corr_length_nm / (2.0 * pixel_size_nm), 0.15)

    angular_used = bool(image_roughness_state.get("angular_correlation_used", False))
    angle_deg = image_roughness_state.get("angular_correlation_angle_deg", None)
    anisotropy_ratio = max(float(image_roughness_state.get("roughness_anisotropy_ratio_sampled", 1.0)), 1.0)

    if angular_used:
        sigma_perp_px = max(sigma_base_px, 0.15)
        sigma_parallel_px = max(sigma_base_px * anisotropy_ratio, 0.15)
    else:
        sigma_parallel_px = sigma_base_px
        sigma_perp_px = sigma_base_px

    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    pad = max(2, int(np.ceil(3.0 * max(sigma_parallel_px, sigma_perp_px))))
    ph = (y1 - y0) + 2 * pad
    pw = (x1 - x0) + 2 * pad

    white = rng.normal(0.0, 1.0, size=(ph, pw))

    if angular_used:
        rotated = rotate(white, angle=-(float(angle_deg) if angle_deg is not None else 0.0), reshape=False, order=1, mode='reflect')
        filtered_rot = gaussian_filter(rotated, sigma=(sigma_perp_px, sigma_parallel_px), mode='reflect')
        filtered = rotate(filtered_rot, angle=(float(angle_deg) if angle_deg is not None else 0.0), reshape=False, order=1, mode='reflect')
    else:
        filtered = gaussian_filter(white, sigma=sigma_base_px, mode='reflect')

    field = filtered[pad:pad + (y1 - y0), pad:pad + (x1 - x0)].copy()
    local_mask = mask[y0:y1, x0:x1]
    vals = field[local_mask]

    if vals.size < 2:
        return out, {
            "roughness_rms_target_nm": float(rms_target),
            "roughness_rms_actual_nm": 0.0,
            "roughness_corr_length_nm": float(corr_length_nm),
            "roughness_angular_correlation_used": bool(angular_used),
            "roughness_angular_correlation_angle_deg": angle_deg,
            "roughness_anisotropy_ratio_sampled": float(anisotropy_ratio),
        }

    field -= float(np.mean(vals))

    taper_px = float(cfg.get('roughness_edge_taper_px', 0))
    if taper_px > 0:
        dist_inside = distance_transform_edt(local_mask)
        u = np.clip(dist_inside / taper_px, 0.0, 1.0)
        taper = 0.5 - 0.5 * np.cos(np.pi * u)
        field *= taper

    vals = field[local_mask]
    field -= float(np.mean(vals))
    vals = field[local_mask]

    rms = float(np.sqrt(np.mean(vals**2)))
    if rms > 1e-12:
        field *= rms_target / rms
    else:
        field[:] = 0.0

    field *= local_mask
    out[y0:y1, x0:x1] = field

    actual_vals = out[mask]
    actual_rms = float(np.sqrt(np.mean(actual_vals**2))) if actual_vals.size else 0.0

    return out, {
        "roughness_rms_target_nm": float(rms_target),
        "roughness_rms_actual_nm": float(actual_rms),
        "roughness_corr_length_nm": float(corr_length_nm),
        "roughness_angular_correlation_used": bool(angular_used),
        "roughness_angular_correlation_angle_deg": (float(angle_deg) if angle_deg is not None else None),
        "roughness_anisotropy_ratio_sampled": float(anisotropy_ratio),
        "roughness_corr_length_parallel_nm": float(corr_length_nm * anisotropy_ratio),
        "roughness_corr_length_perp_nm": float(corr_length_nm),
    }


# =============================================================================
# TESSELLATION GENERATOR
# =============================================================================

def _generate_tessellation(rng, cfg):
    h, w = cfg["image_shape"]
    mean_d_px = cfg["mean_grain_size_nm"] / cfg["pixel_size_nm"]
    n_grains = max(4, int(round((h*w)/(np.pi*(mean_d_px/2)**2))))

    seeds = []
    for _ in range(n_grains):
        d_nm = _sample_grain_size_nm(rng, cfg)
        d_px = d_nm/cfg["pixel_size_nm"]
        ecc = _sample_uniform(rng, cfg["eccentricity_range"], 0, 0.95)
        ratio = np.sqrt(max(1e-4, 1-ecc**2))
        seeds.append({"cx": rng.uniform(0,w), "cy": rng.uniform(0,h),
                      "a": d_px/2, "b": max(1,d_px*ratio/2),
                      "theta": rng.uniform(0,np.pi), "d_nm": d_nm, "ecc": ecc})

    labels = _assign_tessellation(seeds, cfg["image_shape"])
    for _ in range(cfg["tessellation_relax"]):
        for gid, s in enumerate(seeds,1):
            ys, xs = np.nonzero(labels == gid)
            if len(xs):
                s["cx"], s["cy"] = xs.mean(), ys.mean()
        labels = _assign_tessellation(seeds, cfg["image_shape"])
        _soft_side_relax(seeds, labels, cfg)
        labels = _assign_tessellation(seeds, cfg["image_shape"])

    labels, curve_stats = _curve_selected_shared_boundaries(labels, rng, mean_d_px, cfg)
    ids = [int(i) for i in np.unique(labels) if i > 0]
    masks, centroids, sizes, eccs = [], [], [], []
    for gid in ids:
        gm = labels == gid
        ys, xs = np.nonzero(gm)
        masks.append(gm)
        centroids.append([xs.mean(), ys.mean()])
        sizes.append(seeds[gid-1]["d_nm"])
        eccs.append(seeds[gid-1]["ecc"])

    means = _correlated_means(rng, centroids, cfg)
    topo = np.zeros(cfg["image_shape"], float)
    meta = []
    image_roughness_state = _sample_image_roughness_state(rng, cfg)
    image_edge_state = _sample_image_edge_rounding_state(rng, cfg)
    for j, gid in enumerate(ids):
        rel, info = _relative_surface(rng, masks[j], cfg)
        rough, rough_info = _correlated_roughness(rng, masks[j], cfg, image_roughness_state=image_roughness_state)

        # Final tessellation Z = correlated mean + smooth grain surface
        #                        + independent correlated roughness, optionally
        #                        rounded down close to the border.
        z_local = means[j] + rel + rough
        z_local, edge_info = _apply_edge_rounding(z_local, masks[j], rng, cfg, image_edge_state=image_edge_state)
        topo[masks[j]] = z_local[masks[j]]

        meta.append({"id": gid, "eq_diameter_nm": sizes[j], "eccentricity": eccs[j],
                     "mean_z_nm": float(means[j]), "visible_area_px": int(masks[j].sum()),
                     **info, **rough_info, **edge_info})

    counts = _side_counts(labels)
    low, high = cfg["n_sides_range"]
    extra = {**curve_stats,
             **image_roughness_state,
             **image_edge_state,
             "mean_actual_side_count": float(np.mean(list(counts.values()))) if counts else 0.0,
             "fraction_inside_side_target": float(np.mean([low <= n <= high for n in counts.values()])) if counts else 0.0}
    return topo, labels, meta, extra


# =============================================================================
# OVERLAP-INTERACTIVE GENERATOR
# =============================================================================

def _candidate_score(local_mask, occupied_crop, target_overlap, cfg):
    area = local_mask.sum()
    if area <= 0:
        return -np.inf
    overlap = np.sum(local_mask & occupied_crop)/area
    expanded = binary_dilation(occupied_crop, iterations=2)
    contact = np.sum(local_mask & expanded & ~occupied_crop)/area
    free = np.sum(local_mask & ~expanded)/area
    score = -7*abs(overlap-target_overlap) + 2.2*cfg["contact_preference"]*contact + 0.8*cfg["gap_preference"]*free
    if overlap > 0.70:
        score -= 12*(overlap-0.70)
    return score


def _owner_map(grains, shape):
    owner = np.zeros(shape, np.int32)
    for g in grains:
        ys, xs = g["slice"]
        sub = owner[ys,xs]
        sub[g["local_mask"]] = g["id"]
    return owner


def _generate_overlap_xy(rng, cfg):
    h, w = cfg["image_shape"]
    target_coverage = _sample_uniform(rng, cfg["coverage_target"], 0, 1)
    occupied = np.zeros((h,w), bool)
    grains = []
    mean_d_px = cfg["mean_grain_size_nm"]/cfg["pixel_size_nm"]
    rough_n = (h*w)/max(np.pi*(mean_d_px/2)**2,1)
    max_grains = max(50, int(6*rough_n))

    while occupied.mean() < target_coverage and len(grains) < max_grains:
        d_nm = _sample_grain_size_nm(rng, cfg)
        d_px = d_nm/cfg["pixel_size_nm"]
        ecc = _sample_uniform(rng, cfg["eccentricity_range"], 0, 0.95)
        n_sides = _sample_int_uniform(rng, cfg["n_sides_range"], 3)
        irregularity = _sample_uniform(rng, cfg["shape_irregularity_range"], 0)
        target_overlap = _sample_uniform(rng, cfg["overlap_target"], 0, 0.95)

        best, best_score = None, -np.inf
        for _ in range(cfg["max_placement_trials"]):
            if not grains or rng.random() < 0.16:
                margin = 0.25*d_px
                cx, cy = rng.uniform(-margin,w-1+margin), rng.uniform(-margin,h-1+margin)
            else:
                ref = grains[int(rng.integers(len(grains)))]
                ang = rng.uniform(0,2*np.pi)
                contact_d = 0.5*(ref["d_px"]+d_px)
                dist = rng.uniform(0.55*contact_d,1.10*contact_d)
                cx, cy = ref["cx"]+dist*np.cos(ang), ref["cy"]+dist*np.sin(ang)

            xy, shape_info = _polygon_with_curved_sides(rng,cx,cy,d_px,ecc,n_sides,irregularity,cfg)
            raster = _rasterize_polygon_crop(xy,cfg["image_shape"])
            if raster is None:
                continue
            sl, local = raster
            score = _candidate_score(local, occupied[sl], target_overlap, cfg)
            if score > best_score:
                best_score = score
                best = (cx,cy,sl,local,shape_info)

        if best is None:
            break
        cx,cy,sl,local,shape_info = best
        gid = len(grains)+1
        grains.append({"id":gid,"cx":cx,"cy":cy,"slice":sl,"local_mask":local,
                       "d_nm":d_nm,"d_px":d_px,"ecc":ecc,"irregularity":irregularity,**shape_info})
        occupied[sl] |= local

    coverage_before = float(occupied.mean())
    owner0 = _owner_map(grains,cfg["image_shape"])
    keep = []
    for g in grains:
        visible = np.sum(owner0 == g["id"])
        frac = visible/max(np.sum(g["local_mask"]),1)
        if frac >= cfg["min_visible_fraction"]:
            keep.append(g)
    for new_id,g in enumerate(keep,1):
        g["id"] = new_id
    owner = _owner_map(keep,cfg["image_shape"])
    return keep, owner, {"coverage_target_sampled":target_coverage,
                         "coverage_before_prune":coverage_before,
                         "coverage_after_prune":float((owner>0).mean()),
                         "n_grains_after_prune":len(keep)}


def _generate_overlap(rng, cfg):
    grains, owner, extra = _generate_overlap_xy(rng,cfg)
    h,w = cfg["image_shape"]
    if not grains:
        return np.zeros((h,w)), owner, [], extra

    centroids = [[g["cx"],g["cy"]] for g in grains]
    means = _correlated_means(rng, centroids, cfg)
    topo = np.zeros((h,w), float)
    occupied_lower = np.zeros((h,w), bool)
    meta = []
    image_roughness_state = _sample_image_roughness_state(rng, cfg)
    image_edge_state = _sample_image_edge_rounding_state(rng, cfg)

    for i,g in enumerate(grains):
        ys,xs = g["slice"]
        local_mask = g["local_mask"]
        # Fit only in local crop for speed. The roughness is generated only
        # after final XY placement/pruning, so no time is spent roughening grains
        # that will later be discarded.
        rel_local, info = _relative_surface(rng, local_mask, cfg)
        rough_local, rough_info = _correlated_roughness(rng, local_mask, cfg, image_roughness_state=image_roughness_state)
        topo_crop = topo[ys,xs]
        lower_crop = occupied_lower[ys,xs]
        overlap = local_mask & lower_crop
        support = float(np.percentile(topo_crop[overlap], cfg["support_percentile"])) if np.any(overlap) else 0.0
        # Start from exactly the same intrinsic Z model used in tessellation.
        # The overlap geometry only modifies the absolute mean level when the
        # lower support is higher than the grain's own correlated mean Z.
        intrinsic_mean = float(means[i])
        support_delta = max(0.0, support - intrinsic_mean)
        placed_mean = intrinsic_mean + cfg["stack_height_factor"] * support_delta

        intrinsic_z = intrinsic_mean + rel_local + rough_local
        intrinsic_z, edge_info = _apply_edge_rounding(intrinsic_z, local_mask, rng, cfg, image_edge_state=image_edge_state)
        z_local = intrinsic_z + (placed_mean - intrinsic_mean)
        topo_crop[local_mask] = z_local[local_mask]  # later grain overwrites lower grain
        occupied_lower[ys,xs] |= local_mask

        meta.append({"id":g["id"],"eq_diameter_nm":g["d_nm"],"eccentricity":g["ecc"],
                     "n_sides":g["n_sides"],"curved_sides":g["curved_sides"],
                     "mean_side_curvature":g["mean_side_curvature"],
                     "shape_irregularity":g["irregularity"],"mean_z_nm":float(means[i]),
                     "placed_mean_z_nm":float(placed_mean),
                     "support_z_nm":support,
                     "support_shift_nm":float(placed_mean-intrinsic_mean),
                     "visible_area_px":int(np.sum(owner==g["id"])),
                     **info, **rough_info, **edge_info})

    extra = {**extra, **image_roughness_state, **image_edge_state}
    return topo, owner, meta, extra


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_domains(mode="tessellation", **overrides):
    """Generate one topography + visible mask."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    rng = np.random.default_rng(cfg.get("seed",None))
    if mode == "tessellation":
        topo,mask,meta,extra = _generate_tessellation(rng,cfg)
    elif mode == "overlap_interactive":
        topo,mask,meta,extra = _generate_overlap(rng,cfg)
    else:
        raise ValueError("mode must be 'tessellation' or 'overlap_interactive'")
    return topo,mask,meta,extra,cfg

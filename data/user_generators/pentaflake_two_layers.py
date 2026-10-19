"""
pentaflake_two_layers.py
────────────────────────
Synthetic AFM topography generator: pentagonal flake (5 corners)
with two layers (inner step) OFFSET for a more realistic look.

Returns (image, mask) as uint8 (HxW):
- image: grayscale topography on a black background
- mask : 255 over the full flake footprint (base layer), 0 background

Parameters (params dict, all optional)
-----------------------------------------
size                 : int   (default 256)  image size (square)
radius_frac          : float (default 0.35) pentagon radius (fraction of size/2)
rotation_deg         : float (default 0.0)  pentagon rotation in degrees

# Step (second layer)
step_radius_frac     : float (default 0.60) step pentagon size relative to the outer radius
step_offset_frac     : float (default 0.18) step centre offset relative to the outer radius
step_offset_angle_deg: float (default 25.0) offset direction (degrees, in the flake's rotated frame)

# Heights (relative units 0..1)
height_base          : float (default 0.5)  base layer height
height_step          : float (default 1.0)  step height (>= height_base)

# Realismo AFM
edge_blur_sigma      : float (default 1.2)  gaussian smoothing (px). 0 disables
noise_std            : float (default 0.02) gaussian noise (units 0..1)
bg_level             : float (default 0.0)  background level (0..1)
seed                 : int|None (default None) RNG seed for reproducibility

Notes
-----
- The mask corresponds to the full flake footprint (base layer),
  not just the step.
- If scipy is not available, an approximate local-average blur is used.
"""

from __future__ import annotations
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


def _regular_polygon_vertices(n: int, radius: float, rotation_rad: float, cx: float, cy: float):
    angles = rotation_rad + np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.stack([xs, ys], axis=1)  # (n,2)


def _points_in_poly_mask(h: int, w: int, verts: np.ndarray) -> np.ndarray:
    """Ray casting algorithm, vectorised. verts: (N,2) with (x,y)."""
    x = np.arange(w)[None, :].repeat(h, axis=0).astype(np.float32)
    y = np.arange(h)[:, None].repeat(w, axis=1).astype(np.float32)

    xv = verts[:, 0].astype(np.float32)
    yv = verts[:, 1].astype(np.float32)
    n = verts.shape[0]

    inside = np.zeros((h, w), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = xv[i], yv[i]
        xj, yj = xv[j], yv[j]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= intersect
        j = i
    return inside


def _fast_box_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Fallback blur: local average (2k+1)^2. k>=1"""
    h, w = img.shape
    pad = k
    t = np.pad(img, pad, mode="edge")
    acc = np.zeros_like(img, dtype=np.float32)
    cnt = 0
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            acc += t[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            cnt += 1
    return acc / float(cnt)


def generate(params: dict) -> tuple[np.ndarray, np.ndarray]:
    params = {} if params is None else dict(params)

    size = int(params.get("size", 256))
    radius_frac = float(params.get("radius_frac", 0.35))
    rotation_deg = float(params.get("rotation_deg", 0.0))

    step_radius_frac = float(params.get("step_radius_frac", 0.60))
    step_offset_frac = float(params.get("step_offset_frac", 0.18))
    step_offset_angle_deg = float(params.get("step_offset_angle_deg", 25.0))

    height_base = float(params.get("height_base", 0.5))
    height_step = float(params.get("height_step", 1.0))

    edge_blur_sigma = float(params.get("edge_blur_sigma", 1.2))
    noise_std = float(params.get("noise_std", 0.02))
    bg_level = float(params.get("bg_level", 0.0))
    seed = params.get("seed", None)

    rng = np.random.default_rng(seed)

    h = w = size
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    outer_r = radius_frac * (size / 2.0)
    inner_r = step_radius_frac * outer_r
    rot = np.deg2rad(rotation_deg)

    # Flake centre (base layer)
    outer_cx, outer_cy = cx, cy

    # Step centre: offset in the flake's frame
    offset = step_offset_frac * outer_r
    ang = rot + np.deg2rad(step_offset_angle_deg)
    step_cx = outer_cx + offset * np.cos(ang)
    step_cy = outer_cy + offset * np.sin(ang)

    outer_verts = _regular_polygon_vertices(5, outer_r, rot, outer_cx, outer_cy)
    inner_verts = _regular_polygon_vertices(5, inner_r, rot, step_cx, step_cy)

    outer_mask = _points_in_poly_mask(h, w, outer_verts)
    inner_mask = _points_in_poly_mask(h, w, inner_verts)

    # Topography in 0..1 units
    topo = np.full((h, w), bg_level, dtype=np.float32)
    topo[outer_mask] = height_base
    # step only inside the flake (intersection)
    step_region = inner_mask & outer_mask
    topo[step_region] = np.maximum(topo[step_region], height_step)

    # Smoothing (simulates AFM tip convolution)
    if edge_blur_sigma > 0:
        if gaussian_filter is not None:
            topo = gaussian_filter(topo, sigma=edge_blur_sigma)
        else:
            k = int(max(1, round(edge_blur_sigma * 2)))
            topo = _fast_box_blur(topo, k=k)

    # Noise (in 0..1), only over the flake
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=(h, w)).astype(np.float32)
        topo = np.clip(topo + noise * outer_mask.astype(np.float32), 0.0, 1.0)

    image = np.clip(topo * 255.0, 0, 255).astype(np.uint8)
    mask = (outer_mask.astype(np.uint8) * 255)

    return image, mask

"""
data/user_generators/benzene.py
────────────────────────────────
AFM topography generator — benzene molecule on a flat substrate.

Benzene (C₆H₆) is a flat aromatic ring.  In non-contact AFM it appears
as a hexagonal pattern with six peripheral lobes (carbon positions) and
a slight central depression.  The simulation uses a sum of Gaussians
at the six carbon sites plus a weak negative Gaussian at the centre to
reproduce this characteristic appearance.

Params
------
size          int   Output image side in pixels.          Default 256
radius_frac   float Ring radius as fraction of size/2.    Default 0.28
lobe_sigma    float Gaussian width of each carbon lobe
              (fraction of size).                         Default 0.055
center_depth  float Relative depth of central depression. Default 0.25
height_scale  float Peak intensity (0–255).               Default 220
rotation_deg  float Rotation angle of the ring in degrees.Default 0  (random if -1)
noise_sigma   float Gaussian noise standard deviation.    Default 3.0
seed          int   Random seed (-1 = random).            Default -1
"""

from __future__ import annotations
import math
import numpy as np
import cv2


def generate(params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (image, mask) as uint8 grayscale arrays (HxW).
    image : AFM topography of benzene on black substrate.
    mask  : binary mask — 255 inside the molecule footprint, 0 elsewhere.
    """
    p = params or {}

    size         = int(p.get("size",         256))
    radius_frac  = float(p.get("radius_frac",  0.28))
    lobe_sigma   = float(p.get("lobe_sigma",   0.055))
    center_depth = float(p.get("center_depth", 0.25))
    height_scale = float(p.get("height_scale", 220))
    rotation_deg = float(p.get("rotation_deg", 0))
    noise_sigma  = float(p.get("noise_sigma",  3.0))
    seed         = int(p.get("seed", -1))

    rng = np.random.default_rng(None if seed == -1 else seed)

    if rotation_deg == -1:
        rotation_deg = rng.uniform(0, 60)   # benzene has 60° symmetry

    # ── coordinate grid ───────────────────────────────────────────────────────
    cx = cy = size / 2.0
    ys, xs  = np.mgrid[0:size, 0:size].astype(np.float32)

    radius_px  = radius_frac * size
    sigma_px   = lobe_sigma  * size

    # ── six carbon lobe positions ─────────────────────────────────────────────
    angles = [math.radians(rotation_deg + 60 * k) for k in range(6)]
    topo   = np.zeros((size, size), dtype=np.float32)

    for ang in angles:
        ax = cx + radius_px * math.cos(ang)
        ay = cy + radius_px * math.sin(ang)
        r2 = (xs - ax) ** 2 + (ys - ay) ** 2
        topo += np.exp(-r2 / (2 * sigma_px ** 2))

    # ── central depression ────────────────────────────────────────────────────
    r2_center = (xs - cx) ** 2 + (ys - cy) ** 2
    topo -= center_depth * np.exp(-r2_center / (2 * (sigma_px * 0.8) ** 2))

    # ── normalise to [0, height_scale] ───────────────────────────────────────
    topo = np.clip(topo, 0, None)
    t_max = topo.max()
    if t_max > 0:
        topo = topo / t_max * height_scale

    # ── noise ─────────────────────────────────────────────────────────────────
    if noise_sigma > 0:
        topo += rng.normal(0, noise_sigma, topo.shape).astype(np.float32)

    topo = np.clip(topo, 0, 255).astype(np.uint8)

    # ── mask: ellipse enclosing the full molecule ─────────────────────────────
    mask = np.zeros((size, size), dtype=np.uint8)
    mol_radius = int(radius_px * 1.6)
    cv2.circle(mask, (size // 2, size // 2), mol_radius, 255, -1)
    # Keep only pixels above threshold in the topo
    _, obj_mask = cv2.threshold(topo, max(1, int(height_scale * 0.08)), 255,
                                cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, obj_mask)

    return topo, mask

"""
data/user_generators/ic_pattern.py
────────────────────────────────────
AFM topography generator — integrated circuit interconnect pattern.

Simulates a fragment of a metal interconnect layer as seen by AFM:
horizontal and vertical metal lines (traces) at different heights,
with via pads at some intersections.  The substrate is the reference
(height = 0).  Metal lines sit above it; pads are higher still.

The image is rendered as a float heightmap then converted to uint8.
Soft edges are added by Gaussian blurring to simulate AFM tip convolution.

Params
------
size          int   Output image side in pixels.             Default 256
n_h           int   Number of horizontal lines.              Default 3
n_v           int   Number of vertical lines.                Default 3
line_width    float Line width as fraction of size.          Default 0.06
metal_height  float Intensity of metal lines (0–255).        Default 180
pad_height    float Intensity of via pads (0–255).           Default 240
pad_radius    float Via pad radius as fraction of size.      Default 0.045
tip_sigma     float Tip convolution blur (fraction of size). Default 0.012
noise_sigma   float Gaussian noise std dev.                  Default 2.5
add_pads      bool  Whether to add via pads at intersections.Default True
jitter        float Random position jitter (fraction of size).Default 0.04
seed          int   Random seed (-1 = random).               Default -1
"""

from __future__ import annotations
import numpy as np
import cv2


def generate(params: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (image, mask) as uint8 grayscale arrays (HxW).
    image : AFM topography of an IC interconnect fragment on black substrate.
    mask  : binary mask — 255 over metal features, 0 on substrate.
    """
    p = params or {}

    size         = int(p.get("size",         256))
    n_h          = int(p.get("n_h",          3))
    n_v          = int(p.get("n_v",          3))
    line_width   = float(p.get("line_width",   0.06))
    metal_height = float(p.get("metal_height", 180))
    pad_height   = float(p.get("pad_height",   240))
    pad_radius   = float(p.get("pad_radius",   0.045))
    tip_sigma    = float(p.get("tip_sigma",    0.012))
    noise_sigma  = float(p.get("noise_sigma",  2.5))
    add_pads     = bool(p.get("add_pads",      True))
    jitter       = float(p.get("jitter",       0.04))
    seed         = int(p.get("seed",          -1))

    rng = np.random.default_rng(None if seed == -1 else seed)

    lw_px  = max(2, int(line_width  * size))
    pr_px  = max(2, int(pad_radius  * size))
    margin = int(size * 0.12)

    topo = np.zeros((size, size), dtype=np.float32)

    # ── horizontal line positions ─────────────────────────────────────────────
    h_positions = []
    if n_h > 0:
        step = (size - 2 * margin) / (n_h + 1)
        for i in range(1, n_h + 1):
            y = int(margin + i * step + rng.uniform(-jitter, jitter) * size)
            y = int(np.clip(y, margin, size - margin))
            h_positions.append(y)
            y1, y2 = max(0, y - lw_px // 2), min(size, y + lw_px // 2)
            topo[y1:y2, :] = metal_height

    # ── vertical line positions ───────────────────────────────────────────────
    v_positions = []
    if n_v > 0:
        step = (size - 2 * margin) / (n_v + 1)
        for i in range(1, n_v + 1):
            x = int(margin + i * step + rng.uniform(-jitter, jitter) * size)
            x = int(np.clip(x, margin, size - margin))
            v_positions.append(x)
            x1, x2 = max(0, x - lw_px // 2), min(size, x + lw_px // 2)
            topo[:, x1:x2] = metal_height

    # ── via pads at intersections ─────────────────────────────────────────────
    pad_centers = []
    if add_pads:
        for yc in h_positions:
            for xc in v_positions:
                # Only some intersections get a pad (random ~70 %)
                if rng.random() < 0.70:
                    cv2.circle(topo, (xc, yc), pr_px,
                               float(pad_height), -1)
                    pad_centers.append((xc, yc))

    # ── tip convolution (Gaussian blur) ──────────────────────────────────────
    sigma_px = tip_sigma * size
    if sigma_px > 0.5:
        k = int(sigma_px * 4) | 1   # odd kernel
        topo = cv2.GaussianBlur(topo, (k, k), sigma_px)

    # ── noise ─────────────────────────────────────────────────────────────────
    if noise_sigma > 0:
        topo += rng.normal(0, noise_sigma, topo.shape).astype(np.float32)

    topo = np.clip(topo, 0, 255).astype(np.uint8)

    # ── mask: all metal features ──────────────────────────────────────────────
    _, mask = cv2.threshold(topo, max(1, int(metal_height * 0.25)),
                            255, cv2.THRESH_BINARY)

    return topo, mask

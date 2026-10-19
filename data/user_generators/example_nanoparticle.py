"""
example_nanoparticle.py — Circular nanoparticle generator (example).

Generates a synthetic AFM image of a single disk-shaped nanoparticle
on a flat substrate, with a Gaussian height profile and Gaussian noise.

Functions
---------
generate(params) -> (image, mask)

    params (all optional):
        size      (int,   default 128) : image side in pixels
        radius    (float, default 0.2) : disk radius as fraction of image size
        noise_std (float, default 5.0) : Gaussian noise standard deviation (0–255 scale)
        height    (float, default 200) : peak pixel intensity of the nanoparticle

    Returns:
        image : uint8 2D array (HxW) — grayscale height map, black background
        mask  : uint8 2D array (HxW) — binary mask (255 inside disk, 0 outside)
"""

import numpy as np


def generate(params: dict) -> tuple:
    size      = int(params.get("size",      128))
    radius    = float(params.get("radius",  0.2)) * size
    noise_std = float(params.get("noise_std", 5.0))
    peak      = float(params.get("height",  200.0))

    y, x   = np.ogrid[:size, :size]
    cx, cy = size / 2.0, size / 2.0
    dist   = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Gaussian height profile centred on the disk
    sigma  = radius / 2.0
    height = np.exp(-dist ** 2 / (2.0 * sigma ** 2))

    # Add Gaussian noise
    rng   = np.random.default_rng()
    image = height * peak + rng.normal(0.0, noise_std, (size, size))
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Binary mask: pixels within radius
    mask = (dist <= radius).astype(np.uint8) * 255

    return image, mask

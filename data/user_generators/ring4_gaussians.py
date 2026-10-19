import numpy as np


def generate(params: dict) -> tuple:
    """Generate AFM-like nanoparticle ring made of 4 Gaussians.

    Geometry: 4 Gaussian peaks positioned at 90-degree intervals on a circle.
    Heights alternate between peak_height_a and peak_height_b.

    Returns
    -------
    (image, mask)
        image: uint8 grayscale topography (HxW) on black background.
        mask : uint8 binary mask (255 on object footprint, else 0).
    """
    # ---- Parameters (with defaults) ----
    size = int(params.get("size", 128))

    # Ring radius as fraction of (size/2)
    radius_frac = float(params.get("radius_frac", 0.25))

    # Gaussian spread as fraction of size (controls ring thickness)
    sigma_frac = float(params.get("sigma_frac", 0.06))

    # Heights: alternate high/low
    peak_height_a = float(params.get("peak_height_a", 1.0))
    peak_height_b = float(params.get("peak_height_b", 0.8))

    # Optional overall intensity scaling to 0-255 range
    peak_intensity = float(params.get("peak_intensity", 200.0))

    # Background-ish noise
    noise_std = float(params.get("noise_std", 5.0))

    # Mask threshold as fraction of peak_intensity
    # (object footprint around the ring)
    mask_threshold_frac = float(params.get("mask_threshold_frac", 0.15))

    # ---- Grid ----
    h = w = size
    y, x = np.ogrid[:h, :w]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Ring center radius in pixels
    R = radius_frac * (size / 2.0)

    # Sigma in pixels
    sigma = sigma_frac * size
    sigma = max(sigma, 1e-6)

    # ---- Four gaussian peaks ----
    # Angles at 0, 90, 180, 270 degrees (90-degree increments)
    angles = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]

    # Alternate heights: A, B, A, B
    heights = [peak_height_a, peak_height_b, peak_height_a, peak_height_b]

    height_map = np.zeros((h, w), dtype=np.float32)

    for theta, ph in zip(angles, heights):
        px = cx + R * np.cos(theta)
        py = cy + R * np.sin(theta)

        d2 = (x - px) ** 2 + (y - py) ** 2
        g = np.exp(-d2 / (2.0 * sigma ** 2))

        height_map += ph * g

    # Normalize so that max ~ peak_intensity (optional)
    maxv = float(height_map.max())
    if maxv > 0:
        height_map = height_map / maxv

    height_map = height_map * peak_intensity

    # Add noise
    if noise_std > 0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(h, w)).astype(np.float32)
        height_map = height_map + noise

    # Clip and convert
    image = np.clip(height_map, 0, 255).astype(np.uint8)

    # ---- Mask ----
    thr = mask_threshold_frac * peak_intensity
    mask = (height_map >= thr).astype(np.uint8) * 255

    return image, mask

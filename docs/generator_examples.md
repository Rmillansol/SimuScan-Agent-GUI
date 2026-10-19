# Custom Python Generator Examples

Generators let you define arbitrary shapes for AFM image synthesis.
The agent can create them, show them to the user for review, and save them in
`data/user_generators/`.

## Required interface

```python
def generate(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a topography image and its binary mask.

    Parameters
    ----------
    params : dict   Configurable parameters (size, shape, noise, etc.)

    Returns
    -------
    image : np.ndarray float32 [H, W]   Normalized topography (0–1)
    mask  : np.ndarray float32 [H, W]   Binary mask (0.0 or 1.0)
    """
```

- `image` and `mask` must have the same size.
- `image` values must be in [0, 1] (the physical scale is applied by SimuScan).
- `mask` must be binary: 1.0 inside the object, 0.0 outside.
- Use only NumPy and standard libraries (no PyTorch, no OpenCV in global imports).

---

## Example 1 — Spherical nanoparticle (Gaussian dome)

```python
import numpy as np

def generate(params: dict):
    size      = params.get("size", 128)
    radius    = params.get("radius", 20)
    height    = params.get("height", 1.0)
    noise_std = params.get("noise_std", 0.02)
    sharpness = params.get("sharpness", 1.0)   # 1=Gaussian, >1 flatter

    image = np.zeros((size, size), dtype=np.float32)
    mask  = np.zeros((size, size), dtype=np.float32)

    cx = cy = size // 2
    Y, X = np.ogrid[:size, :size]
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    r  = np.sqrt(r2)

    inside = r <= radius
    mask[inside] = 1.0

    # Gaussian profile
    sigma = radius / (2.0 * sharpness)
    image[inside] = height * np.exp(-r2[inside] / (2 * sigma ** 2))

    if noise_std > 0:
        rng = np.random.default_rng()
        image += rng.normal(0, noise_std, image.shape).astype(np.float32)
        image = np.clip(image, 0, None)

    if image.max() > 0:
        image /= image.max()

    return image, mask
```

**Typical parameters:**
```json
{"size": 128, "radius": 18, "height": 1.0, "noise_std": 0.02, "sharpness": 1.0}
```

---

## Example 2 — Cylindrical pillar with soft edges

```python
import numpy as np

def generate(params: dict):
    size        = params.get("size", 128)
    radius      = params.get("radius", 25)
    height      = params.get("height", 1.0)
    edge_width  = params.get("edge_width", 3)   # edge smoothing in px
    noise_std   = params.get("noise_std", 0.01)

    cx = cy = size // 2
    Y, X = np.ogrid[:size, :size]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)

    # Sigmoid profile for a soft edge
    image = height / (1.0 + np.exp((r - radius) / max(edge_width, 0.1)))
    mask  = (r <= radius).astype(np.float32)

    if noise_std > 0:
        rng = np.random.default_rng()
        image += rng.normal(0, noise_std, image.shape).astype(np.float32)

    image = np.clip(image, 0, None)
    if image.max() > 0:
        image /= image.max()

    return image, mask
```

**Typical parameters:**
```json
{"size": 128, "radius": 22, "height": 1.0, "edge_width": 4, "noise_std": 0.01}
```

---

## Example 3 — Rectangular structure (flat DNA origami)

```python
import numpy as np

def generate(params: dict):
    size        = params.get("size", 128)
    width_px    = params.get("width_px", 60)    # rectangle width
    height_px   = params.get("height_px", 40)   # rectangle height
    thickness   = params.get("thickness", 1.0)  # structure height
    roughness   = params.get("roughness", 0.05) # surface texture
    angle_deg   = params.get("angle_deg", 0.0)  # rotation

    image = np.zeros((size, size), dtype=np.float32)
    mask  = np.zeros((size, size), dtype=np.float32)

    cx = cy = size // 2
    Y, X = np.ogrid[:size, :size]
    Xc = (X - cx).astype(np.float32)
    Yc = (Y - cy).astype(np.float32)

    # Rotation
    ang = np.deg2rad(angle_deg)
    Xr =  Xc * np.cos(ang) + Yc * np.sin(ang)
    Yr = -Xc * np.sin(ang) + Yc * np.cos(ang)

    inside = (np.abs(Xr) <= width_px / 2) & (np.abs(Yr) <= height_px / 2)
    mask[inside] = 1.0
    image[inside] = thickness

    if roughness > 0:
        rng = np.random.default_rng()
        noise = rng.normal(0, roughness, image.shape).astype(np.float32)
        image += noise * mask

    image = np.clip(image, 0, None)
    if image.max() > 0:
        image /= image.max()

    return image, mask
```

**Typical parameters:**
```json
{"size": 128, "width_px": 55, "height_px": 38, "thickness": 1.0,
 "roughness": 0.04, "angle_deg": 15.0}
```

---

## Example 4 — Vesicle (hollow ring)

```python
import numpy as np

def generate(params: dict):
    size        = params.get("size", 128)
    radius      = params.get("radius", 28)    # outer radius
    wall_width  = params.get("wall_width", 6) # wall thickness
    height      = params.get("height", 1.0)
    noise_std   = params.get("noise_std", 0.02)

    cx = cy = size // 2
    Y, X = np.ogrid[:size, :size]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)

    inner = radius - wall_width
    ring  = (r <= radius) & (r >= max(inner, 0))
    disk  = r <= radius   # full outer mask

    image = np.zeros((size, size), dtype=np.float32)
    mask  = disk.astype(np.float32)   # mask includes the hollow interior

    # Gaussian profile over the ring
    sigma = wall_width / 2.5
    r_wall = np.abs(r - (inner + wall_width / 2))
    image[ring] = height * np.exp(-(r_wall[ring] ** 2) / (2 * sigma ** 2))

    if noise_std > 0:
        rng = np.random.default_rng()
        image += rng.normal(0, noise_std, image.shape).astype(np.float32) * mask
        image = np.clip(image, 0, None)

    if image.max() > 0:
        image /= image.max()

    return image, mask
```

**Typical parameters:**
```json
{"size": 128, "radius": 30, "wall_width": 5, "height": 1.0, "noise_std": 0.02}
```

---

## Full workflow to create a generator

```
1. The user describes the object:
   "I want nanoparticles with a hole in the center, vesicle-like"

2. The agent proposes the code:
   propose_generator("vesicle", <Python code>)
   → The agent shows the complete code so the user can review it

3. The user approves or asks for changes:
   "Perfect, save it"    → save_generator("vesicle")
   "Make it rougher"     → modifies and proposes again

4. The agent saves to data/user_generators/vesicle.py

5. To test it:
   run_generator("vesicle", {"radius": 30, "wall_width": 5})
   → generates a preview image

6. To use it in the dataset:
   Add a reference in default.yaml → objects section (label: vesicle)
```

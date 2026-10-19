"""
src/simuscan/image_shape.py
────────────────────────────
Renders image-based reference objects as synthetic AFM heightmaps.

Each PNG in ref_objects/image_files/ is used as a height-map template.
Its companion mask (mask_<name>.png) defines the binary footprint.

Return format matches generate_rand_image_from_obj_sections:
  topo_t     : list[np.ndarray]  — (image_size, image_size) float32
  masks_t    : list[np.ndarray]  — (image_size, image_size) uint8
  full_mask_t: np.ndarray        — (image_size, image_size) uint8
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from .SyntheticParams import ObjectConfig
from . import mask_tools


# ══════════════════════════════════════════════════════════════════════════════
# Mask generation (called when saving a new reference image from OpenAI)
# ══════════════════════════════════════════════════════════════════════════════

def generate_mask_from_image(
    image_path: str | Path,
    output_path: str | Path | None = None,
    threshold_method: str = "otsu",
    invert: bool = False,
) -> np.ndarray:
    """
    Generate a binary mask from a reference image.

    The default (invert=False) assumes BRIGHT object on DARK background,
    which matches the AFM image generation constraints (black substrate).

    Args:
        image_path:       Path to the source PNG/JPG.
        output_path:      If given, saves the mask as PNG there.
        threshold_method: "otsu" (recommended) or a float 0-255 for a manual
                          threshold.
        invert:           False (default) → object is brighter than background.
                          True  → object is darker than background (legacy).

    Returns:
        Binary uint8 array: 255 = object, 0 = background.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if threshold_method == "otsu":
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    else:
        thr  = float(threshold_method)
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, thr, 255, flag)

    # Slight dilation before close to recover edge pixels lost by Otsu
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, k3, iterations=1)

    # Close holes inside the object, remove isolated noise outside
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5, iterations=1)

    if output_path is not None:
        cv2.imwrite(str(output_path), mask)

    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_resize(
    image_path: Path,
    mask_path: Path,
    size_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image + mask, resize to (H, W).

    Returns:
        height_map : float32 in [0, 1], normalised using only object pixels.
                     Background is zeroed regardless of its raw intensity.
        binary_mask: uint8 {0, 1}.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read ref image: {image_path}")

    msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if msk is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")

    # ── Crop to the mask bounding box so size_shape maps to the OBJECT, not the
    #    full PNG canvas. Reference files on disk are never modified. ──────────
    ys, xs = np.where(msk > 127)
    if ys.size > 0:
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        img = img[y0:y1, x0:x1]
        msk = msk[y0:y1, x0:x1]

    h, w = size_hw

    # ── Aspect-preserving fit into the (h, w) box: scale by a single factor so
    #    the object keeps its native proportions (no stretching), then centre it
    #    in a zero-padded canvas. size_shape acts as a bounding box. ───────────
    oh, ow = img.shape[:2]
    if oh > 0 and ow > 0:
        scale = min(w / ow, h / oh)
        nw = max(1, int(round(ow * scale)))
        nh = max(1, int(round(oh * scale)))
        img_fit = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        msk_fit = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
        img_r = np.zeros((h, w), dtype=np.float32)
        msk_r = np.zeros((h, w), dtype=msk.dtype)
        oy, ox = (h - nh) // 2, (w - nw) // 2
        img_r[oy:oy + nh, ox:ox + nw] = img_fit
        msk_r[oy:oy + nh, ox:ox + nw] = msk_fit
    else:
        img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        msk_r = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
    msk_bin = (msk_r > 127).astype(np.uint8)   # {0, 1}

    # Normalise using only object-pixel range so the gray background
    # does not compress the object contrast.
    obj_pixels = img_r[msk_bin > 0]
    if obj_pixels.size > 0:
        lo, hi = float(obj_pixels.min()), float(obj_pixels.max())
        if hi > lo:
            img_r = (img_r - lo) / (hi - lo)
        else:
            img_r[:] = 1.0
        img_r = np.clip(img_r, 0.0, 1.0)
    else:
        img_r[:] = 1.0

    # Zero out background — critical for diffusion images where the
    # "black" background is actually a low-level gray.
    img_r[msk_bin == 0] = 0.0

    return img_r, msk_bin


def _rotate_patch(
    patch: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate patch and mask around their centre, EXPANDING the output canvas so the
    rotated object is never clipped at the corners. Only background (zeros) grows;
    the object keeps its size.
    """
    h, w = patch.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    # Shift so the rotated content is centred in the larger canvas
    M[0, 2] += (nW / 2.0) - (w / 2.0)
    M[1, 2] += (nH / 2.0) - (h / 2.0)
    patch_r = cv2.warpAffine(patch, M, (nW, nH), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_r  = cv2.warpAffine(mask,  M, (nW, nH), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return patch_r, mask_r


def _apply_bending_to_patch(
    patch: np.ndarray,
    mask: np.ndarray,
    axis: str,
    amplitude_px: float,
    frequency: float,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a geometric warp (bending) to the patch and mask using cv2.remap.

    Equivalent to apply_bending_to_points for section objects, but operating
    on a 2D image patch via a displacement field.

    axis: 'x' → rows are shifted horizontally
          'y' → columns are shifted vertically
    amplitude_px: max displacement in pixels
    frequency: number of sinusoidal cycles across the patch
    kind: 'sin', 'parabolic', 'gaussian', 'linear'
    """
    h, w = patch.shape

    # Base map: coordinates along the bending axis, normalised to [0, 1]
    if axis == 'y':
        base = np.linspace(0, 1, w)                       # (w,)
        base_2d = np.tile(base, (h, 1))                   # (h, w)
    else:
        base = np.linspace(0, 1, h)                       # (h,)
        base_2d = np.tile(base[:, None], (1, w))          # (h, w)

    if kind == 'sin':
        disp = amplitude_px * np.sin(2 * np.pi * base_2d * frequency)
    elif kind == 'parabolic':
        disp = -amplitude_px * (2 * base_2d - 1) ** 2 + amplitude_px
    elif kind == 'gaussian':
        sigma = 0.15
        disp = amplitude_px * np.exp(-((base_2d - 0.5) ** 2) / (2 * sigma ** 2))
    elif kind == 'linear':
        disp = amplitude_px * (base_2d - 0.5)
    else:
        raise ValueError(f"Unsupported bending kind: {kind}")

    # Centre displacement so mean offset is zero
    disp -= disp.mean()

    # Build remap grids
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    if axis == 'y':
        map_x = grid_x
        map_y = (grid_y + disp).astype(np.float32)
    else:
        map_x = (grid_x + disp).astype(np.float32)
        map_y = grid_y

    patch_w = cv2.remap(patch, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_w  = cv2.remap(mask.astype(np.float32), map_x, map_y,
                        interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_w  = (mask_w > 0.5).astype(np.uint8)

    return patch_w, mask_w


# ══════════════════════════════════════════════════════════════════════════════
# Public renderer
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_image_pairs(
    obj: ObjectConfig,
    ref_objects_dir: Path,
) -> tuple[list[Path], list[Path], list[float] | None]:
    """
    Resolve (image_path, mask_path) pairs and optional weights from ObjectConfig.

    Priority:
      1. image_folder + image_files → use the explicit file list with weights.
      2. image_folder only          → glob ALL *.png (excl. mask_*), uniform weights.
      3. image_file + mask_file     → legacy single-file fallback.

    Returns (img_paths, mask_paths, weights_or_None).
    """
    base = ref_objects_dir

    if obj.image_folder:
        folder = base / obj.image_folder
        if not folder.exists():
            raise ValueError(f"Object '{obj.label}': image_folder '{folder}' not found.")

        if obj.image_files:
            # Explicit list — use only those files
            imgs, masks = [], []
            for fname in obj.image_files:
                stem = Path(fname).stem
                ext  = Path(fname).suffix or ".png"
                imgs.append(folder / fname)
                masks.append(folder / f"mask_{stem}{ext}")
            weights = obj.image_weights if obj.image_weights else None
            return imgs, masks, weights
        else:
            # No explicit list — use ALL images in the folder (sorted, uniform weights)
            all_imgs = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() == ".png" and not p.name.startswith("mask_")
            )
            if not all_imgs:
                raise ValueError(
                    f"Object '{obj.label}': image_folder '{folder}' contains no PNG images."
                )
            masks = [folder / f"mask_{p.name}" for p in all_imgs]
            return all_imgs, masks, None  # uniform weights

    # Legacy fallback — single file
    if obj.image_file and obj.mask_file:
        return [base / obj.image_file], [base / obj.mask_file], None

    raise ValueError(
        f"Object '{obj.label}': no image_folder or image_file/mask_file defined."
    )


def generate_image_shape(
    obj: ObjectConfig,
    image_size: int,
    ref_objects_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Render obj.n_objects instances of an image-based reference object.

    Applies the same set of random transformations as
    generate_rand_image_from_obj_sections:
      - image sampled from image_files list with image_weights probabilities
      - uniform scale variation  (obj.scale_pct)
      - eccentricity / non-uniform scale  (obj.eccentricity_variation_*_pct)
      - bending  (obj.apply_bending, obj.bending_*)
      - random rotation  (obj.rotation_range_deg)
      - height randomisation  (obj.height) — preserves relative pixel intensities
      - ellipsoid texture  (obj.apply_ellipses_texture)
      - correlated / Perlin texture  (obj.apply_perlin)
      - noise  (obj.apply_noise)

    Returns topo_t, masks_t, full_mask_t in the same format as the other
    SimuScan renderers.
    """
    img_paths, mask_paths, weights = _resolve_image_pairs(obj, ref_objects_dir)

    topo_t:  list[np.ndarray] = []
    masks_t: list[np.ndarray] = []
    full_mask_t = np.zeros((image_size, image_size), dtype=np.uint8)

    count   = 0
    n_iter  = 0

    while count < obj.n_objects and n_iter < obj.n_objects * 4:
        n_iter += 1

        # ── 0. Pick image variant for this instance ───────────────────────────
        idx      = random.choices(range(len(img_paths)), weights=weights, k=1)[0]
        img_path  = img_paths[idx]
        mask_path = mask_paths[idx]

        # ── 1. Scale + eccentricity (mirror of mask_by_sections_transformation) ──
        scale_pct = random.uniform(0, obj.scale_pct)
        ecc       = random.uniform(obj.eccentricity_variation_down_pct,
                                   obj.eccentricity_variation_up_pct)
        sh_w = max(4, int(obj.size_shape[0] * (1 + scale_pct)))
        sh_h = max(4, int(obj.size_shape[1] * (1 + scale_pct) * (1 + ecc)))

        # ── 2. Load and resize with scaled dimensions ──────────────────────────
        patch, pmask = _load_and_resize(img_path, mask_path, (sh_h, sh_w))

        # ── 3. Bending — geometric warp of the patch ──────────────────────────
        if obj.apply_bending:
            amp   = random.uniform(-obj.bending_amplitud_pct, obj.bending_amplitud_pct)
            freq  = random.uniform(0, obj.bending_freq)
            dctr  = random.uniform(0, obj.bending_diff_center)
            # amplitude in pixels relative to the smaller patch dimension
            amp_px = abs(amp) * min(sh_w, sh_h)
            patch, pmask = _apply_bending_to_patch(
                patch, pmask,
                axis=obj.bending_axis,
                amplitude_px=amp_px,
                frequency=freq,
                kind=obj.bending_kind,
            )

        # ── 4. Random rotation ────────────────────────────────────────────────
        angle = random.uniform(obj.rotation_range_deg[0], obj.rotation_range_deg[1])
        patch, pmask = _rotate_patch(patch, pmask, angle)

        # ── 5. Height: single random value, preserving relative intensities ───
        height = random.uniform(obj.height[0], obj.height[1])
        patch  = patch * height          # patch ∈ [0,1] → [0, height]
        patch[pmask == 0] = 0.0          # background stays exactly at zero

        # ── 6. Ellipsoid texture (same as sections) ───────────────────────────
        if obj.apply_ellipses_texture:
            if random.random() < obj.ellipses_probability:
                text_img = mask_tools.fill_with_ellipsoids(
                    pmask,
                    relative_radios_range=obj.ellipses_relative_radios_range,
                    n_por_radio=obj.ellipses_num,
                    altura_range=[height * x for x in obj.ellipses_amplitud_range],
                    forma=obj.ellipses_shape,
                    e_range=obj.ellipses_eccentriciy_range,
                    overlap=obj.ellipses_overlap,
                )
                patch += text_img

        # ── 7. Correlated / Perlin texture ────────────────────────────────────
        if obj.apply_perlin:
            scale_px = random.choice(obj.perlin_scale_px)
            text_img = mask_tools.fill_with_correlated_texture(
                pmask,
                correlation_length=scale_px,
                amplitude=obj.perlin_amplitud_pct,
                seed=None,
            )
            patch += text_img

        # ── 8. Noise ──────────────────────────────────────────────────────────
        if obj.apply_noise:
            if random.random() < obj.noise_probability:
                ampl  = np.random.uniform(0, obj.noise_amplitud)
                ruido = mask_tools.fill_with_noise(pmask, ampl, obj.noise_sigma)
                patch += ruido

        # ── 9. Place at a random position inside the canvas ───────────────────
        ph, pw = patch.shape
        margin_y = ph // 2 + 2
        margin_x = pw // 2 + 2
        y_max = max(margin_y + 1, image_size - margin_y - ph)
        x_max = max(margin_x + 1, image_size - margin_x - pw)
        y = random.randint(margin_y, y_max)
        x = random.randint(margin_x, x_max)

        canvas_topo = np.zeros((image_size, image_size), dtype=np.float32)
        canvas_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        canvas_topo[y:y + ph, x:x + pw] = patch
        canvas_mask[y:y + ph, x:x + pw] = pmask

        topo_t.append(canvas_topo)
        masks_t.append(canvas_mask)
        full_mask_t = np.maximum(full_mask_t, canvas_mask)
        count += 1

    return topo_t, masks_t, full_mask_t

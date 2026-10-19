"""
src/automation/automation.py
─────────────────────────────
AFM automated exploration module -- simplified version.

Connects to Nanosurf Studio, performs large-area scans, runs YOLO detection,
and zooms onto each detected object with a frame size proportional to its
bounding box. No Triplet embeddings, no object library.

Public API
──────────
ExplorationConfig        : dataclass with all session parameters
ExplorationSession       : main controller -- run in a background thread
load_gwy_channel()       : load a .gwy file channel → numpy array + metadata
flatten_line_by_line()   : AFM line-flattening (vectorised OLS, lowest perc%)
array_to_png()           : save normalised numpy array as PNG
connect_microscope()     : connect to Studio and read tip position (call first)
run_exploration()        : start a session from a config dict
get_exploration_status() : thread-safe status query
stop_exploration()       : graceful abort signal
"""

from __future__ import annotations

import csv
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from src.bridge.ui_bridge import ui

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import gwyfile
    _HAS_GWYFILE = True
except ImportError:
    _HAS_GWYFILE = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ── Module-level session state (one session at a time) ────────────────────────
_SESSION_LOCK   : threading.Lock                 = threading.Lock()
_ACTIVE_SESSION : Optional["ExplorationSession"] = None


# ══════════════════════════════════════════════════════════════════════════════
# Result helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ok(data: Any = None, message: str = "OK") -> dict:
    r: dict = {"status": "ok", "message": message}
    if data is not None:
        r["data"] = data
    return r


def _err(message: str, detail: Any = None) -> dict:
    r: dict = {"status": "error", "message": message}
    if detail is not None:
        r["detail"] = detail
    return r


# ══════════════════════════════════════════════════════════════════════════════
# GWY I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

class ChannelType(Enum):        # Channel types used (Nanosurf titles)
    FORWARD_TOPO  = 'Forward - Topography'
    BACKWARD_TOPO = 'Backward - Topography'
    FORWARD_DEFL  = 'Forward - Deflection'
    BACKWARD_DEFL = 'Backward - Deflection'
    FORWARD_AMPL  = 'Forward - Analyzer 1 Amplitude'
    BACKWARD_AMPL = 'Backward - Analyzer 1 Amplitude'
    FORWARD_PHASE = 'Forward - Analyzer 1 Phase'
    BACKWARD_PHASE = 'Backward - Analyzer 1 Phase'


def topo_channel_for(scan_direction: str) -> str:
    """
    Map a scan direction to the Nanosurf topography channel title.

    'forward'  → 'Forward - Topography'
    'backward' → 'Backward - Topography'

    This is the single source of truth linking the exploration scan direction
    to the .gwy channel that is read, so it stays consistent with the direction
    the synthetic dataset was generated with (pid_conf.scan_direction).
    """
    sd = str(scan_direction).strip().lower()
    if sd == "forward":
        return ChannelType.FORWARD_TOPO.value
    if sd == "backward":
        return ChannelType.BACKWARD_TOPO.value
    raise ValueError(
        f"scan_direction must be 'forward' or 'backward', got {scan_direction!r}"
    )


def load_gwy_channel(
    filepath: str,
    channel: str | None = None,
    *,
    scan_direction: str = "forward",
) -> Tuple[np.ndarray, dict]:
    """
    Load a single channel from a .gwy file.

    The channel is selected by *name* (Nanosurf channel title). When ``channel``
    is None it is derived from ``scan_direction`` via ``topo_channel_for`` so the
    whole pipeline reads exactly one direction (forward *or* backward topography),
    never both.

    Parameters
    ----------
    filepath       : path to the .gwy file.
    channel        : explicit channel title. If None, derived from scan_direction.
    scan_direction : 'forward' | 'backward' (used only when channel is None).

    Returns
    -------
    (data_array, meta)
      data_array : 2-D numpy float64 array (rows = slow axis, cols = fast axis)
      meta       : dict with xreal, yreal, xoff, yoff, si_unit_xy, si_unit_z
    """
    if not _HAS_GWYFILE:
        raise ImportError("gwyfile is not installed. Run: pip install gwyfile")

    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"GWY file not found: '{filepath}'")

    if channel is None:
        channel = topo_channel_for(scan_direction)

    obj      = gwyfile.load(str(path))
    channels = gwyfile.util.get_datafields(obj)

    if channel not in channels:
        available = list(channels.keys())
        if not available:
            raise ValueError(f"No data channels found in '{filepath}'.")
        # Fallback: requested channel not present → use the first available one.
        channel = available[0]

    df   = channels[channel]
    data = np.asarray(df.data, dtype=np.float64)

    meta = {
        "channel"    : channel,
        "xreal"      : float(df.xreal),
        "yreal"      : float(df.yreal),
        "xoff"       : float(df.xoff),
        "yoff"       : float(df.yoff),
        "si_unit_xy" : df["si_unit_xy"].unitstr if "si_unit_xy" in df else "m",
        "si_unit_z"  : df["si_unit_z"].unitstr  if "si_unit_z"  in df else "m",
        "cols"       : data.shape[1],
        "rows"       : data.shape[0],
    }
    return data, meta


def flatten_line_by_line(
    data : np.ndarray,
    perc : float = 40.0,
) -> tuple:
    """
    Line-by-line AFM flattening (vectorised, ~9× faster than a curve_fit loop).

    For each scan line:
      1. Select the lowest ``perc``% pixels (background / substrate).
      2. Fit a straight line to those points using the closed-form OLS solution.
      3. Subtract the fitted line from the entire row.
      4. Subtract the median of the selected (corrected) points.

    Parameters
    ----------
    data : 2-D numpy array   (rows = scan lines, cols = fast axis)
    perc : float             percentage of lowest-value points used for fit
                             (default 40 -- same as the original FlattenPlus)

    Returns
    -------
    out  : float64 array, same shape as data -- flattened image
    mask : bool array, True where points were used for the linear fit
    """
    out   = data.astype(np.float64).copy()
    H, W  = out.shape
    x_all = np.arange(W, dtype=np.float64)
    n_pts = max(2, int(W * perc / 100.0))

    # Sort each row and take the n_pts lowest indices
    sorted_idx = np.argsort(out, axis=1)          # (H, W)
    sel_idx    = sorted_idx[:, :n_pts]             # (H, n_pts)

    mask = np.zeros((H, W), dtype=bool)
    np.put_along_axis(mask, sel_idx, True, axis=1)

    # Gather selected x and y values for all rows at once
    x_sel = x_all[sel_idx]                                       # (H, n_pts)
    y_sel = np.take_along_axis(out, sel_idx, axis=1)             # (H, n_pts)

    # Closed-form OLS: a = Σ(dx·dy) / Σ(dx²),  b = ȳ − a·x̄
    x_m = x_sel.mean(axis=1, keepdims=True)
    y_m = y_sel.mean(axis=1, keepdims=True)
    dx  = x_sel - x_m
    dxs = (dx ** 2).sum(axis=1)
    safe = dxs > 0
    a = np.where(safe, (dx * (y_sel - y_m)).sum(axis=1) / np.where(safe, dxs, 1.0), 0.0)
    b = y_m[:, 0] - a * x_m[:, 0]

    # Subtract fitted line from each row
    out -= a[:, None] * x_all[None, :] + b[:, None]

    # Subtract median of the selected (now corrected) points
    corrected_sel = np.take_along_axis(out, sel_idx, axis=1)
    out          -= np.median(corrected_sel, axis=1, keepdims=True)

    return out, mask


def array_to_png(
    data     : np.ndarray,
    out_path : str,
    colormap : str             = "afmhot",
    vmin     : Optional[float] = None,
    vmax     : Optional[float] = None,
) -> str:
    """Save a 2-D numpy array as a PNG. Returns the absolute path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if vmin is None:
        vmin = float(np.percentile(data, 1))
    if vmax is None:
        vmax = float(np.percentile(data, 99))

    fig, ax = plt.subplots(
        figsize=(data.shape[1] / 100, data.shape[0] / 100), dpi=100
    )
    ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out), dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return str(out.resolve())


def array_to_gray_png(data: np.ndarray, out_path: str) -> str:
    """
    Save a 2-D array as a GRAYSCALE, min-max normalised PNG — EXACTLY the format
    the training dataset uses (see COCOBuilder.save_image):
        img_norm = (a - a.min()) / (a.max() - a.min()) * 255  ->  uint8, 1 channel.
    YOLO was trained on these images, so inference MUST use the same representation
    (a colour-mapped / percentile-clipped image is a train/inference mismatch).
    Writes the exact HxW pixels (no matplotlib tight-bbox resizing), so detection
    bboxes line up 1:1 with the array coordinates. Returns the absolute path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(data, dtype=np.float64)
    mn, mx = float(a.min()), float(a.max())
    g = (((a - mn) / (mx - mn)) * 255.0).astype(np.uint8) if mx > mn else np.zeros(a.shape, np.uint8)
    if _HAS_CV2:
        cv2.imwrite(str(out), g)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(a.shape[1] / 100, a.shape[0] / 100), dpi=100)
        ax.imshow(g, cmap="gray", vmin=0, vmax=255, origin="upper", aspect="equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(str(out), dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    return str(out.resolve())


# ══════════════════════════════════════════════════════════════════════════════
# Object-masked flatten and crop export
# ══════════════════════════════════════════════════════════════════════════════

def flatten_with_object_mask(
    zoom_flat : np.ndarray,
    obj_mask  : np.ndarray,
    perc      : float = 40.0,
) -> np.ndarray:
    """
    Line-by-line AFM flattening using the region OUTSIDE the object mask
    as the background reference, then zeros all non-object pixels.

    For each scan line:
      1. Take background pixels (obj_mask == 0) as the reference set.
         Fall back to all pixels if fewer than 2 background pixels on the line.
      2. Among those, take the lowest perc% by value (robustness against
         residual contamination).
      3. Fit a straight line (OLS closed form) to those points.
      4. Subtract the fitted line from the entire row.
      5. Set all non-object pixels to zero.

    Parameters
    ----------
    zoom_flat : 2-D float64 array
    obj_mask  : 2-D bool or uint8  (True/255 = object, 0 = background)
    perc      : % of background pixels used for the linear fit

    Returns
    -------
    float64 array, same shape -- flattened with background zeroed
    """
    obj   = (obj_mask > 0).astype(bool)
    bg    = ~obj
    out   = zoom_flat.astype(np.float64).copy()
    H, W  = out.shape
    x_all = np.arange(W, dtype=np.float64)

    for i in range(H):
        row    = out[i]
        bg_idx = np.where(bg[i])[0]
        if len(bg_idx) < 2:
            bg_idx = np.arange(W)

        n_pts = max(2, int(len(bg_idx) * perc / 100.0))
        order = np.argsort(row[bg_idx])
        sel   = bg_idx[order[:n_pts]]

        x_sel = x_all[sel]
        y_sel = row[sel]
        x_m   = x_sel.mean()
        y_m   = y_sel.mean()
        dx    = x_sel - x_m
        dxs   = (dx ** 2).sum()
        if dxs > 0:
            a = (dx * (y_sel - y_m)).sum() / dxs
            b = y_m - a * x_m
        else:
            a, b = 0.0, y_m

        row -= a * x_all + b

    out[~obj] = 0.0
    return out


def make_crop_pair(
    zoom_raw        : np.ndarray,
    zoom_flat_masked: np.ndarray,
    obj_mask        : np.ndarray,
    zoom_name       : str,
    crops_dir       : Path,
    colormap        : str = "afmhot",
) -> None:
    """
    Save two PNG images for one detected object into crops_dir/<zoom_name>_*.png:

      _raw.png  : original zoom (afmhot) with the Otsu object contour in green (display)
      _flat.png : mask-flattened zoom, LINEAR GRAYSCALE intensity (background = 0).
                  This is the TripletEncoder input, matched to how training crops
                  are made (raw gray intensity) — NOT a colormap.

    Parameters
    ----------
    zoom_raw         : 2-D float64 -- full flattened zoom image
    zoom_flat_masked : 2-D float64 -- output of flatten_with_object_mask()
    obj_mask         : 2-D bool/uint8 -- Otsu object mask
    zoom_name        : base filename without extension
    crops_dir        : destination directory (created if needed)
    colormap         : matplotlib colormap (default 'afmhot')
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(crops_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Raw image with contour
        vmin_r = float(np.percentile(zoom_raw, 1))
        vmax_r = float(np.percentile(zoom_raw, 99))
        fig, ax = plt.subplots(
            figsize=(zoom_raw.shape[1]/100, zoom_raw.shape[0]/100), dpi=100
        )
        ax.imshow(zoom_raw, cmap=colormap, vmin=vmin_r, vmax=vmax_r,
                  origin="upper", aspect="equal")
        if _HAS_CV2:
            mask_u8 = (obj_mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                pts = cnt[:, 0, :]
                ax.plot(np.append(pts[:,0], pts[0,0]),
                        np.append(pts[:,1], pts[0,1]),
                        color="lime", linewidth=1.2, alpha=0.9)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(str(crops_dir / f"{zoom_name}_raw.png"),
                    dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Flat image (background zeroed) — saved as LINEAR GRAYSCALE intensity,
        # NOT a colormap. This is what the TripletEncoder consumes (it reads
        # _flat.png with IMREAD_GRAYSCALE), and it must match how training crops
        # are made (raw gray intensity, triplet_trainer._crop_and_normalise), so
        # the embeddings are comparable. An afmhot PNG read back as luminance
        # would warp the intensity non-linearly. Object pixels -> 1..255 (linear),
        # background stays exactly 0 so _prepare_crop can separate them.
        flat = np.asarray(zoom_flat_masked, dtype=np.float64)
        nz   = flat != 0
        g    = np.zeros(flat.shape, dtype=np.uint8)
        if nz.any():
            obj = flat[nz]
            lo, hi = float(obj.min()), float(obj.max())
            if hi > lo:
                s = np.clip((flat - lo) / (hi - lo), 0.0, 1.0)
                g[nz] = (1.0 + s[nz] * 254.0).astype(np.uint8)
            else:
                g[nz] = 255
        flat_path = str(crops_dir / f"{zoom_name}_flat.png")
        if _HAS_CV2:
            cv2.imwrite(flat_path, g)
        else:
            fig2, ax2 = plt.subplots(figsize=(flat.shape[1]/100, flat.shape[0]/100), dpi=100)
            ax2.imshow(g, cmap="gray", vmin=0, vmax=255, origin="upper", aspect="equal")
            ax2.axis("off"); fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig2.savefig(flat_path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig2)

    except Exception as exc:
        warnings.warn(f"make_crop_pair failed for '{zoom_name}': {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Position generators
# ══════════════════════════════════════════════════════════════════════════════

def _generate_raster(
    frame_size_m : float,
    overlap      : float = 0.1,
    n_x          : int   = 0,
    n_y          : int   = 0,
    area_m       : float = 0.0,
) -> Tuple[List[Tuple[float, float]], int, int]:
    """
    Grid/raster positions in metres, centred at (0, 0).
    overlap can be negative (gap between frames).
    If n_x > 0 and n_y > 0: exactly n_x × n_y positions.
    Otherwise derives a SQUARE grid from area_m so that the *bounding square*
    (the outer square that contains every frame, edge-to-edge) ≈ area_m — this
    is NOT the summed area of the individual frames.
    Bounding-square side of an n×n grid = (n-1)*step + frame_size.
    """
    step = frame_size_m * (1.0 - overlap)
    if step <= 0:
        step = frame_size_m * 0.01
    if n_x > 0 and n_y > 0:
        nx, ny = n_x, n_y
    else:
        # Pick the square n×n grid whose bounding square is closest to area_m.
        # Solve (n-1)*step + frame_size = sqrt(area_m)  ->  n = (side-frame)/step + 1
        side = math.sqrt(max(area_m, frame_size_m ** 2))
        n    = max(1, int(round((side - frame_size_m) / step + 1.0)))
        nx = ny = n
    half_x = (nx - 1) * step / 2.0
    half_y = (ny - 1) * step / 2.0
    positions = []
    for row in range(ny):
        for col in range(nx):
            x = -half_x + col * step
            y = -half_y + row * step
            positions.append((x, y))
    return positions, nx, ny


def _clip_grid_rect(positions_rel, nx, ny, origin_x_m, origin_y_m, bounds_m):
    """
    Keep only the rectangular sub-grid whose ABSOLUTE cell centres
    (origin + relative offset) fall inside bounds_m = (xmin, xmax, ymin, ymax),
    all in metres. positions_rel is the row-major list from _generate_raster
    (cell i = row*nx + col). Returns (new_positions_rel, new_nx, new_ny, n_removed).

    The grid stays rectangular (columns/rows are dropped as whole strips) so the
    probability-map reshaping downstream keeps working.
    """
    xmin, xmax, ymin, ymax = bounds_m
    cols_x = [positions_rel[c][0]       for c in range(nx)]   # x-offset per column (row 0)
    rows_y = [positions_rel[r * nx][1]  for r in range(ny)]   # y-offset per row  (col 0)
    good_c = [c for c in range(nx) if xmin <= origin_x_m + cols_x[c] <= xmax]
    good_r = [r for r in range(ny) if ymin <= origin_y_m + rows_y[r] <= ymax]
    new = [(cols_x[c], rows_y[r]) for r in good_r for c in good_c]
    return new, len(good_c), len(good_r), (nx * ny - len(new))


# ══════════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExplorationConfig:
    """All parameters for one automated AFM exploration session."""

    # ── Output / microscope ───────────────────────────────────────────────────
    gallery_dir        : str   = ""
    gwy_base_name      : str   = "scan"
    scan_direction     : str   = "forward"   # "forward" | "backward" — reads that single topo channel
    # Wait timeout is computed per-frame from scan_resolution_px & scan_rate_hz
    # (see _gwy_timeout_for); only the poll interval is configurable.
    gwy_poll_interval_s: float = 2.0

    # ── Exploration scan ──────────────────────────────────────────────────────
    scan_size_um       : float = 50.0     # frame size in µm
    scan_rate_hz       : float = 1.0
    scan_resolution_px : int   = 256
    overlap_fraction   : float = 0.1     # negative = gap between frames
    strategy           : str   = "raster"  # "raster" | "probabilistic"

    # ── Grid / area ───────────────────────────────────────────────────────────
    grid_n_x           : int   = 0        # 0 = auto from area_to_cover_um
    grid_n_y           : int   = 0
    area_to_cover_um   : float = 200.0   # µm² — ignored if grid_n_x/y > 0

    # ── Probabilistic strategy ────────────────────────────────────────────────
    # Defaults tuned to match the validated simulation. Movement penalty is applied
    # to distance in grid-cell units (dist / cell_step), so prob_lambda_move uses
    # the same value as the sim. µm values assume ~45 µm frame pitch (50 µm × 0.9).
    max_jump_um        : float = 270.0   # max distance to next frame (µm) ≈ 6 cells

    # Probabilistic — Beta-distribution active search
    prob_alpha_init              : float = 1.0
    prob_beta_init               : float = 1.0
    prob_spatial_sigma_um        : float = 90.0    # spatial propagation radius (µm) ≈ 2 cells
    prob_kappa_uncertainty       : float = 0.6     # uncertainty weight (sim ~0.55–0.70)
    prob_lambda_move             : float = 0.018   # movement penalty per cell (sim baseline)
    prob_epsilon_unknown         : float = 0.10    # probability of choosing uncertain cell (sim)
    prob_negative_update_scale   : float = 1.4     # weight of negative evidence (sim)
    prob_initial_random_n        : int   = 5       # initial random scans near center (sim)
    prob_initial_center_radius_um: float = 200.0   # radius for initial random scans (µm) ≈ 4.5 cells

    # Isolation filter
    isolation_max_bbox_overlap   : float = 0.3     # max bbox area overlap fraction
    isolation_min_distance_um    : float = 0.0     # min distance between centers (µm), 0=off

    # ── YOLO model ────────────────────────────────────────────────────────────
    yolo_enabled           : bool  = False
    yolo_model_path        : str   = ""
    yolo_conf_threshold    : float = 0.7
    yolo_overlap_threshold : float = 0.3
    border_filter_enabled  : bool  = True   # edge-touching detections: map-only (no count/library/zoom)
    # target_classes: dict of {class_name: {"n": int, "enabled": bool}}
    # empty dict means all classes with n=10
    target_classes         : Dict[str, Any] = field(default_factory=dict)

    # ── Zoom scan ─────────────────────────────────────────────────────────────
    zoom_enabled       : bool  = False
    zoom_resolution_px : int   = 512
    zoom_max_rate_hz   : float = 2.0      # scan rate never exceeds this
    zoom_max_size_um   : float = 25.0     # hard cap on zoom frame size
    # zoom_bbox_factors: list of {"min_bbox_um": float, "factor": float}
    # evaluated from largest min_bbox_um downward
    zoom_bbox_factors  : List[Dict[str, float]] = field(default_factory=lambda: [
        {"min_bbox_um": 5.0, "factor": 1.5},
        {"min_bbox_um": 2.0, "factor": 2.0},
        {"min_bbox_um": 0.0, "factor": 3.0},
    ])

    # ── Stopping ──────────────────────────────────────────────────────────────
    # raster        : full square grid is always scanned; this does NOT cut it.
    # probabilistic : stop condition — end after this many frames (0 or negative
    #                 = no limit; only target n / manual stop end the run).
    max_large_frames   : int   = 1000

    # ── Image processing ─────────────────────────────────────────────────────
    flatten_perc       : float = 40.0

    # ── Triplet analysis (optional) ───────────────────────────────────────────
    triplet_model_path : str   = ""

    # ── Stage travel limits (same coordinate frame as origin, in µm) ──────────
    # The positioner stalls / never arrives at the very edge of its travel, so
    # frames whose centre falls outside the reachable rectangle time out. At
    # planning time the grid is clipped to the reachable rectangle (minus a
    # safety margin) so no unreachable frames are ever queued.
    #   stage_auto_range = True → read the range from the hardware at connect.
    #   If that fails (or is False) the explicit stage_*_um bounds are used.
    #   Leave the bounds at 0 (xmax<=xmin) to disable clipping when the range
    #   is unknown.
    stage_auto_range    : bool  = True
    stage_x_min_um      : float = 0.0
    stage_x_max_um      : float = 0.0
    stage_y_min_um      : float = 0.0
    stage_y_max_um      : float = 0.0
    stage_edge_margin_um: float = 300.0   # keep this far from the hard limit

    # ── Session origin (set automatically on connect) ─────────────────────────
    origin_x_m        : float = 0.0
    origin_y_m        : float = 0.0

    # ── Source config file reference ──────────────────────────────────────────
    config_yaml_path   : str   = ""

    # ── Computed properties (not in YAML) ────────────────────────────────────
    @property
    def scan_size_m(self) -> float:
        return self.scan_size_um * 1e-6

    @property
    def area_to_cover_m2(self) -> float:
        return self.area_to_cover_um * 1e-12  # µm² → m²

    @property
    def area_side_m(self) -> float:
        return math.sqrt(self.area_to_cover_m2)  # side length in metres


# ══════════════════════════════════════════════════════════════════════════════
# Main session class
# ══════════════════════════════════════════════════════════════════════════════

class ExplorationSession:
    """Controls one full automated AFM exploration run."""

    def __init__(self, config: ExplorationConfig) -> None:
        self.config       = config
        self._stop        = threading.Event()
        self._status_lock = threading.Lock()
        self._status: dict = {
            "phase"            : "idle",
            "large_frames"     : 0,
            "zoom_frames"      : 0,
            "counts_per_class" : {},
            "last_message"     : "Session initialised.",
            "started_at"       : None,
            "finished_at"      : None,
            "output_dir"       : None,
            "error"            : None,
        }
        self._records: List[dict] = []   # one entry per successful zoom

        # Connection state -- populated by connect()
        self._studio     = None
        self._imaging    = None
        self._workspace  = None
        self._automation = None

        # YOLO model loaded once at session start (None until _load_yolo_model)
        self._yolo_model = None

    # ── Status ────────────────────────────────────────────────────────────────

    def _update(self, **kwargs) -> None:
        with self._status_lock:
            self._status.update(kwargs)

    def get_status(self) -> dict:
        with self._status_lock:
            return dict(self._status)

    def stop(self) -> None:
        self._stop.set()
        self._update(last_message="Stop signal received -- aborting after current frame.")
        ui.log('INFO', f"  [STOP]  Stop signal received -- will abort after current frame.")

    # ── Stopping criterion ────────────────────────────────────────────────────

    def _criterion_met(self) -> bool:
        """True when all enabled target classes have reached their n target."""
        cfg = self.config
        if not cfg.yolo_enabled or not cfg.target_classes:
            return False
        counts = self._status["counts_per_class"]
        for cls, info in cfg.target_classes.items():
            if not info.get("enabled", True):
                continue
            if counts.get(cls, 0) < info.get("n", 10):
                return False
        return True

    # ── GWY wait ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_on_border(det: dict, img_w: int, img_h: int,
                      enabled: bool, tol_px: float = 1.0) -> bool:
        """
        True if the detection bbox touches any frame edge (within tol_px pixels).
        Objects cut off by the frame border have unreliable bbox/shape, so they
        are dropped from counting/zoom. enabled=False disables the filter.
        """
        if not enabled:
            return False
        return (det["x1"] <= tol_px or det["y1"] <= tol_px or
                det["x2"] >= img_w - tol_px or det["y2"] >= img_h - tol_px)

    @staticmethod
    def _border_map_valid(det: dict, img_w: int, img_h: int,
                          min_visible: float = 0.5, tol_px: float = 1.0) -> bool:
        """
        For a detection that touches a frame edge, decide whether enough of the
        object is inside the frame to trust it as *presence evidence* for the
        probability map. A cut object's visible bbox is short PERPENDICULAR to the
        cut edge; a thin sliver (perpendicular << parallel) means the object is
        mostly outside and unreliable. Accept only if, for every touched edge, the
        visible depth perpendicular to that edge is at least `min_visible` times the
        bbox extent parallel to it (≈ the object is at least ~50% inside for
        roughly-equant shapes). Corners (two edges) must satisfy both directions.
        """
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
        frac = 1.0
        if x1 <= tol_px or x2 >= img_w - tol_px:   # cut on a vertical edge
            frac = min(frac, w / h)                # visible width vs full (~height)
        if y1 <= tol_px or y2 >= img_h - tol_px:   # cut on a horizontal edge
            frac = min(frac, h / w)                # visible height vs full (~width)
        return frac >= min_visible

    @staticmethod
    def _scan_gwy_index(gallery_dir: str) -> int:
        """
        Scan the gallery directory and return the highest numeric index
        found in any .gwy filename of the form  <anything>_<N>.gwy
        (e.g. scan_0.gwy, image_12.gwy, data_003.gwy).
        Returns -1 if the directory is empty or contains no matching files.
        """
        gallery = Path(gallery_dir)
        if not gallery.is_dir():
            return -1

        max_idx = -1
        for p in gallery.glob("*.gwy"):
            # Extract the last numeric segment before .gwy
            stem   = p.stem            # e.g. "scan_7" or "my_data_042"
            parts  = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx > max_idx:
                    max_idx = idx
        return max_idx

    @staticmethod
    def _gwy_timeout_for(resolution: int, line_rate_hz: float) -> float:
        """
        Seconds to allow for a single frame's .gwy to appear, derived from the
        scan geometry instead of a fixed config value.

        A frame scans `resolution` lines forward and back at `line_rate_hz`
        lines/s, so the raw scan costs ~2*resolution/line_rate seconds. We add a
        15% margin plus a fixed slice for file flush / stage settle so the wait
        never gives up on a frame that is merely slow to save. Floored at 30 s
        for very small/fast scans.
        """
        rate   = max(float(line_rate_hz), 0.01)
        scan_s = 2.0 * float(resolution) / rate
        return max(30.0, scan_s * 1.15 + 15.0)

    def _wait_for_next_gwy(self, after_index: int, timeout_s: float) -> Optional[Path]:
        """
        Wait until a NEW .gwy file appears in the gallery directory with a
        numeric index strictly greater than `after_index`, regardless of its
        base name.  Waits up to `timeout_s` seconds (see _gwy_timeout_for).

        Returns the Path of the new file once it is stable (size unchanged
        across two consecutive polls), or None on timeout / stop signal.
        """
        gallery  = Path(self.config.gallery_dir)
        deadline = time.monotonic() + timeout_s
        size_cache: Dict[Path, int] = {}

        while time.monotonic() < deadline:
            if self._stop.is_set():
                return None

            # Find all .gwy files with index > after_index
            candidates: List[Path] = []
            for p in gallery.glob("*.gwy"):
                stem  = p.stem
                parts = stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    if int(parts[1]) > after_index:
                        candidates.append(p)

            # Check each candidate for size stability
            for p in candidates:
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size > 0 and size_cache.get(p) == size:
                    return p          # file exists and size is stable
                size_cache[p] = size  # record size for next poll

            time.sleep(self.config.gwy_poll_interval_s)

        return None

    # ── Nanosurf connection ───────────────────────────────────────────────────

    def connect(self) -> dict:
        """
        Connect to Nanosurf Studio, read the current tip position, and
        store everything in instance attributes.

        Call this BEFORE run().  The method is safe to call multiple times;
        if already connected it returns the current state without reconnecting.

        Returns
        -------
        dict  {"status": "ok"|"error", "message": ..., "data": {...}}
          No data returned -- connection state is stored in the session.
        """
        # Already connected?
        if self._studio is not None:
            return _ok(message="Already connected to Nanosurf Studio.")

        try:
            import nanosurf
            studio = nanosurf.Studio()
            if not studio.connect():
                raise RuntimeError("studio.connect() returned False.")

            imaging    = studio.spm.workflow.imaging
            workspace  = studio.spm.workflow.workspace
            automation = studio.spm.workflow.automation

        except Exception as exc:
            self._update(
                phase        = "error",
                error        = str(exc),
                last_message = f"Connection failed: {exc}",
            )
            ui.log('ERR', f"  [ERR ]  Connection failed: {exc}")
            return _err(f"Connection failed: {exc}")

        # Store on instance
        self._studio     = studio
        self._imaging    = imaging
        self._workspace  = workspace
        self._automation = automation

        self._update(
            phase        = "connected",
            last_message = "Connected to Nanosurf Studio.",
        )
        ui.log('INFO', f"  [CONN]  Connected to Nanosurf Studio.")

        return _ok(message="Connected to Nanosurf Studio.")

    def _read_stage_bounds_m(self):
        """
        Best-effort read of the sample-stage reachable rectangle, in metres,
        in the same coordinate frame as the scan positions. Returns
        (x_min, x_max, y_min, y_max) or None if the range cannot be read.

        Uses the Nanosurf stage API (GetAxisRange / GetCurrentAxisZeroPosition,
        axis 0 = X, 1 = Y). The accessor differs between the legacy SPM and
        Studio APIs, so a few paths are tried; any failure → None, and the
        caller falls back to the explicit stage_*_um config bounds.
        """
        stage = None
        for getter in (
            lambda: self._studio.spm.application.Stage,
            lambda: self._studio.application.Stage,
            lambda: self._studio.spm.workflow.stage,
        ):
            try:
                cand = getter()
                if cand is not None:
                    stage = cand
                    break
            except Exception:
                continue
        if stage is None:
            return None
        try:
            rx = float(stage.GetAxisRange(0)); ry = float(stage.GetAxisRange(1))
            zx = float(stage.GetCurrentAxisZeroPosition(0))
            zy = float(stage.GetCurrentAxisZeroPosition(1))
            if rx <= 0 or ry <= 0:
                return None
            return (zx - rx / 2.0, zx + rx / 2.0, zy - ry / 2.0, zy + ry / 2.0)
        except Exception as exc:
            ui.log('INFO', f"  [STAGE]  Could not read stage range ({exc}); "
                           f"using config bounds if set.")
            return None

    @staticmethod
    def _capture_frame(
        workspace, automation,
        name: str, pos_x_m: float, pos_y_m: float,
        size_m: float, resolution: int, line_rate: float,
    ) -> str:
        """Queue a single frame, capture it, wait until idle, return item ID."""
        automation.clear_queue()
        item_id = workspace.add_frame(name, pos_x_m, pos_y_m)
        workspace.set_item_size(item_id, size_m, size_m)
        workspace.set_item_resolution(item_id, resolution, resolution)
        workspace.set_item_line_rate(item_id, line_rate, False)
        automation.add_to_queue(item_id)
        automation.start()
        while not automation.is_idle():
            time.sleep(0.5)
        return item_id

    # ── YOLO ──────────────────────────────────────────────────────────────────

    def _load_yolo_model(self) -> None:
        """Load the YOLO model once and cache it in self._yolo_model."""
        from ultralytics import YOLO
        self._yolo_model = YOLO(self.config.yolo_model_path)
        names = self._yolo_model.names
        self._yolo_names = list(names.values()) if isinstance(names, dict) else names
        ui.log('INFO', f"  [YOLO]  Model loaded: {len(self._yolo_names)} classes — {self._yolo_names}")

    def _run_yolo(self, png_path: str) -> tuple:
        """
        Run YOLO and return (accepted, all_raw, pred).

        accepted : List[dict]  -- detections that pass the overlap filter
        all_raw  : List[dict]  -- every detection; 'accepted' key marks status
        pred     : ultralytics result object (for mask extraction)
        """
        model = self._yolo_model
        preds = model.predict(png_path, conf=self.config.yolo_conf_threshold, verbose=False)
        if not preds or preds[0].boxes is None:
            return [], [], None

        pred  = preds[0]
        names = self._yolo_names

        raw = []
        for box in pred.boxes:
            cls_id = int(box.cls[0].item())
            label  = names[cls_id] if cls_id < len(names) else str(cls_id)
            if self.config.target_classes and label not in self.config.target_classes:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            raw.append({
                "label"   : label,
                "cls_id"  : cls_id,
                "conf"    : float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "accepted": True,
            })

        accepted = self._filter_by_overlap(raw, self.config.yolo_overlap_threshold)
        accepted_set = {id(d) for d in accepted}
        for d in raw:
            d["accepted"] = id(d) in accepted_set

        return accepted, raw, pred

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        ab = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        return inter / (aa + ab - inter + 1e-9)

    @staticmethod
    def _filter_by_overlap(dets: List[dict], max_iou: float) -> List[dict]:
        dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
        kept: List[dict] = []
        for d in dets:
            if all(ExplorationSession._iou(d, k) <= max_iou for k in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _max_iou_with_others(det: dict, others: List[dict]) -> float:
        """Return the maximum IoU between det and every other detection in others."""
        return max(
            (ExplorationSession._iou(det, o) for o in others if o is not det),
            default=0.0,
        )

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _zoom_size_for_bbox(self, bbox_size_m: float) -> float:
        """Compute zoom frame size from bbox using configurable factor table."""
        cfg = self.config
        bbox_um = bbox_size_m * 1e6
        # Sort factors by min_bbox_um descending, pick first that fits
        factors = sorted(cfg.zoom_bbox_factors,
                         key=lambda f: f["min_bbox_um"], reverse=True)
        factor = 3.0  # default
        for entry in factors:
            if bbox_um >= entry["min_bbox_um"]:
                factor = entry["factor"]
                break
        zoom_m = bbox_size_m * factor
        zoom_m = min(zoom_m, cfg.zoom_max_size_um * 1e-6)
        zoom_m = max(zoom_m, cfg.scan_size_m * 0.05)  # at least 5% of scan size
        return zoom_m

    def _det_to_physical(
        self,
        det      : dict,
        img_w    : int,
        img_h    : int,
        frame_cx : float,
        frame_cy : float,
    ) -> Tuple[float, float, float]:
        cx_px     = (det["x1"] + det["x2"]) / 2.0
        cy_px     = (det["y1"] + det["y2"]) / 2.0
        bbox_w_px = det["x2"] - det["x1"]
        bbox_h_px = det["y2"] - det["y1"]
        bbox_px   = max(bbox_w_px, bbox_h_px)

        px_per_m  = img_w / self.config.scan_size_m
        x_m = frame_cx + (cx_px - img_w / 2.0) / px_per_m
        y_m = frame_cy - (cy_px - img_h / 2.0) / px_per_m

        bbox_m      = bbox_px / px_per_m
        zoom_size_m = self._zoom_size_for_bbox(bbox_m)

        return x_m, y_m, zoom_size_m

    # ── Shape metrics from zoom image ────────────────────────────────────────

    @staticmethod
    def _compute_shape_metrics(
        zoom_flat: np.ndarray,
    ) -> tuple:
        """
        Compute morphological and AFM height metrics from a flattened zoom array.

        Uses the largest contour found by Otsu thresholding (no YOLO mask).

        Returns
        -------
        (metrics_dict, mask)
          metrics_dict : flat dict of metrics -- all None if computation fails
          mask         : uint8 binary array (255 = object, 0 = background),
                         or None if computation fails
        """
        null = {
            "area_px": None, "perimeter_px": None,
            "equivalent_diameter_px": None,
            "major_axis_px": None, "minor_axis_px": None,
            "aspect_ratio": None, "orientation_deg": None,
            "solidity": None, "compactness": None,
            "extent": None, "eccentricity": None,
            "mean_height_nm": None, "max_height_nm": None,
            "std_height_nm": None, "roughness_rms_nm": None,
        }
        if not _HAS_CV2:
            return null
        try:
            # Normalise to uint8 only for thresholding / contour detection;
            # height metrics use the original float64 values (metres).
            data64 = zoom_flat.astype(np.float64)
            norm   = zoom_flat.astype(np.float32)
            lo, hi = norm.min(), norm.max()
            if hi <= lo:
                return null, None
            gray = ((norm - lo) / (hi - lo) * 255).astype(np.uint8)
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return null, None

            # Largest contour by area
            cnt  = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(cnt))
            if area == 0:
                return null, None

            peri      = float(cv2.arcLength(cnt, True))
            hull      = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            x, y, w, h = cv2.boundingRect(cnt)

            equiv_diam  = float(np.sqrt(4.0 * area / math.pi))
            solidity    = round(area / hull_area, 4)          if hull_area > 0 else None
            compactness = round(peri**2 / (4 * math.pi * area), 4)
            extent      = round(area / (w * h), 4)            if w * h > 0 else None

            # Ellipse fit (requires ≥ 5 points)
            major = minor = orientation = eccentricity = None
            if len(cnt) >= 5:
                (_, _), (ma, mi), angle = cv2.fitEllipse(cnt)
                major       = round(float(max(ma, mi)), 4)
                minor       = round(float(min(ma, mi)), 4)
                orientation = round(float(angle), 4)
                if major and major > 0:
                    eccentricity = round(
                        float(np.sqrt(1.0 - (min(ma, mi) / max(ma, mi))**2)), 4
                    )
                aspect_ratio = round(major / minor, 4) if minor and minor > 0 else None
            else:
                aspect_ratio = round(w / h, 4) if h > 0 else None

            # AFM height metrics -- use original float64 values (metres)
            mask   = thresh > 0
            region = data64[mask]
            # Convert to nm for storage (AFM heights are ~1–100 nm)
            mean_h_nm = float(np.mean(region))   * 1e9
            max_h_nm  = float(np.max(region))    * 1e9
            std_h_nm  = float(np.std(region))    * 1e9
            mean_h    = float(np.mean(region))
            rms_nm    = float(np.sqrt(np.mean((region - mean_h) ** 2))) * 1e9

            metrics = {
                "area_px"               : round(area, 2),
                "perimeter_px"          : round(peri, 4),
                "equivalent_diameter_px": round(equiv_diam, 4),
                "major_axis_px"         : major,
                "minor_axis_px"         : minor,
                "aspect_ratio"          : aspect_ratio,
                "orientation_deg"       : orientation,
                "solidity"              : solidity,
                "compactness"           : compactness,
                "extent"                : extent,
                "eccentricity"          : eccentricity,
                "mean_height_nm"        : round(mean_h_nm, 4),
                "max_height_nm"         : round(max_h_nm,  4),
                "std_height_nm"         : round(std_h_nm,  4),
                "roughness_rms_nm"      : round(rms_nm,    4),
            }
            return metrics, thresh   # thresh is already uint8 (0/255)
        except Exception:
            return null, None

    def _export_large_crop(
        self, large_flat: np.ndarray, det: dict, det_i: int, label: str,
        frame_name: str, out_dir: Path, cfg: "ExplorationConfig",
        margin_pct: float = 0.15,
    ) -> None:
        """
        Crop one detected object out of the large frame and write a
        library/crops/<label>/<name>_flat.png + _raw.png pair, exactly the
        format the TripletAnalyzer consumes. This lets the triplet library be
        built in Mode 2 (YOLO, no zoom) when a triplet model is assigned --
        mirroring how training crops are cut from YOLO detections on synthetic
        images, just sourced from the exploration frame instead of a re-scan.
        """
        try:
            H, W = large_flat.shape
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            mx, my = (x2 - x1) * margin_pct, (y2 - y1) * margin_pct
            x1c = max(0, int(round(x1 - mx))); y1c = max(0, int(round(y1 - my)))
            x2c = min(W, int(round(x2 + mx))); y2c = min(H, int(round(y2 + my)))
            sub = large_flat[y1c:y2c, x1c:x2c]
            if sub.size == 0 or min(sub.shape) < 4:
                return

            metrics, mask = self._compute_shape_metrics(sub)
            if mask is None:
                return
            sub_masked = flatten_with_object_mask(sub, mask, perc=cfg.flatten_perc)
            crop_name  = f"{frame_name}_obj{det_i:02d}_{label}"
            make_crop_pair(sub, sub_masked, mask, crop_name,
                           out_dir / "library" / "crops" / label)

            record = {
                "zoom_index"   : len(self._records),
                "large_file"   : f"{frame_name}.png",
                "zoom_file"    : "",
                "class"        : label,
                "score"        : round(det["conf"], 4),
                "centroid_x_um": "",
                "centroid_y_um": "",
                "bbox_w_px"    : round(x2 - x1, 1),
                "bbox_h_px"    : round(y2 - y1, 1),
                "zoom_size_um" : "",
                "zoom_rate_hz" : "",
            }
            record.update(metrics)
            self._records.append(record)
        except Exception as exc:
            ui.log('WARN', f"  [WARN]  Crop export failed for '{label}': {exc}")

    @staticmethod
    def _beta_uncertainty(a: float, b: float) -> float:
        denom = (a + b) ** 2 * (a + b + 1)
        return math.sqrt(a * b / denom) if denom > 0 else 0.0

    @staticmethod
    def _compute_deficit(found: Dict[str, float], target_classes: Dict[str, Any]) -> Dict[str, float]:
        deficit: Dict[str, float] = {}
        for cls, info in target_classes.items():
            if not info.get("enabled", True):
                deficit[cls] = 0.0
                continue
            n_t = max(1, info.get("n", 10))
            deficit[cls] = max(0.0, n_t - found.get(cls, 0.0)) / n_t
        return deficit

    def _update_prob_maps_beta(
        self,
        alpha        : Dict[str, np.ndarray],
        beta         : Dict[str, np.ndarray],
        cell_idx     : int,
        counts       : Dict[str, float],
        dist_col     : np.ndarray,
        target_classes: Dict[str, Any],
    ) -> None:
        cfg = self.config
        for cls, info in target_classes.items():
            if not info.get("enabled", True):
                continue
            sigma_m = info.get("sigma_um", cfg.prob_spatial_sigma_um) * 1e-6
            w = np.exp(-dist_col ** 2 / (2 * sigma_m ** 2))
            obs = counts.get(cls, 0.0)
            if obs > 0:
                alpha[cls] += obs * w
            else:
                beta[cls] += cfg.prob_negative_update_scale * w

    def _select_best_utility_cell(
        self,
        alpha         : Dict[str, np.ndarray],
        beta          : Dict[str, np.ndarray],
        found         : Dict[str, float],
        target_classes: Dict[str, Any],
        unvisited     : List[int],
        dist_row      : np.ndarray,
        max_step_m    : float,
    ) -> Optional[int]:
        cfg     = self.config
        deficit = self._compute_deficit(found, target_classes)
        # Movement distance must be expressed in GRID-CELL units so prob_lambda_move
        # matches the validated simulation (which uses cell distances). Otherwise
        # the penalty is applied to metres (~1e-4) and vanishes → the search jumps
        # across the whole grid ("X" pattern). cell_step = frame pitch in metres.
        cell_step_m = max(cfg.scan_size_m * (1.0 - cfg.overlap_fraction), 1e-9)
        best_score, best_idx = -math.inf, None
        for multiplier in (1.0, 2.0):
            for i in unvisited:
                if dist_row[i] > max_step_m * multiplier:
                    continue
                score = 0.0
                for cls, info in target_classes.items():
                    if not info.get("enabled", True):
                        continue
                    a, b   = float(alpha[cls][i]), float(beta[cls][i])
                    mu     = a / (a + b)
                    unc    = self._beta_uncertainty(a, b)
                    kappa  = info.get("kappa", cfg.prob_kappa_uncertainty)
                    score += deficit[cls] * (mu + kappa * unc)
                score -= cfg.prob_lambda_move * (dist_row[i] / cell_step_m)
                if score > best_score:
                    best_score, best_idx = score, i
            if best_idx is not None:
                break
        return best_idx

    def _select_most_uncertain_cell(
        self,
        alpha         : Dict[str, np.ndarray],
        beta          : Dict[str, np.ndarray],
        found         : Dict[str, float],
        target_classes: Dict[str, Any],
        unvisited     : List[int],
        dist_row      : np.ndarray,
        max_step_m    : float,
    ) -> Optional[int]:
        cfg     = self.config
        deficit = self._compute_deficit(found, target_classes)
        best_score, best_idx = -math.inf, None
        for multiplier in (1.0, 2.0):
            for i in unvisited:
                if dist_row[i] > max_step_m * multiplier:
                    continue
                score = sum(
                    deficit[cls] * self._beta_uncertainty(float(alpha[cls][i]), float(beta[cls][i]))
                    for cls, info in target_classes.items()
                    if info.get("enabled", True)
                )
                if score > best_score:
                    best_score, best_idx = score, i
            if best_idx is not None:
                break
        return best_idx

    def _determine_search_focus(
        self,
        cell_idx      : int,
        alpha         : Dict[str, np.ndarray],
        beta          : Dict[str, np.ndarray],
        found         : Dict[str, float],
        target_classes: Dict[str, Any],
    ) -> str:
        cfg     = self.config
        deficit = self._compute_deficit(found, target_classes)
        contrib: Dict[str, float] = {}
        for cls, info in target_classes.items():
            if not info.get("enabled", True):
                continue
            a, b   = float(alpha[cls][cell_idx]), float(beta[cls][cell_idx])
            mu     = a / (a + b)
            unc    = self._beta_uncertainty(a, b)
            kappa  = info.get("kappa", cfg.prob_kappa_uncertainty)
            contrib[cls] = deficit[cls] * (mu + kappa * unc)
        if not contrib:
            return "none"
        srt = sorted(contrib, key=contrib.__getitem__, reverse=True)
        if len(srt) == 1 or contrib[srt[0]] == 0:
            return srt[0] if contrib[srt[0]] > 0 else "none"
        top, sec = contrib[srt[0]], contrib[srt[1]]
        # "multi" = the top classes are ~tied (within 10%) at this cell — the
        # utility is driven by several classes, not one. (Was "both", a leftover
        # from the 2-class validated simulation; misleading with >2 classes.)
        return "multi" if top > 0 and abs(top - sec) / top < 0.1 else srt[0]

    def _apply_isolation_filter(self, detections: List[dict], img_w: int) -> List[bool]:
        cfg        = self.config
        max_olap   = cfg.isolation_max_bbox_overlap
        min_dist_m = cfg.isolation_min_distance_um * 1e-6
        px_per_m   = img_w / cfg.scan_size_m
        n          = len(detections)
        isolated   = [True] * n
        for i in range(n):
            d  = detections[i]
            cx_i  = (d["x1"] + d["x2"]) / 2
            cy_i  = (d["y1"] + d["y2"]) / 2
            area_i = max(1.0, (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]))
            for j in range(n):
                if i == j:
                    continue
                d2    = detections[j]
                ix1   = max(d["x1"], d2["x1"]); iy1 = max(d["y1"], d2["y1"])
                ix2   = min(d["x2"], d2["x2"]); iy2 = min(d["y2"], d2["y2"])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_j = max(1.0, (d2["x2"] - d2["x1"]) * (d2["y2"] - d2["y1"]))
                if inter / min(area_i, area_j) > max_olap:
                    isolated[i] = False
                    break
                if min_dist_m > 0:
                    cx_j  = (d2["x1"] + d2["x2"]) / 2
                    cy_j  = (d2["y1"] + d2["y2"]) / 2
                    dist_m = math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2) / px_per_m
                    if dist_m < min_dist_m:
                        isolated[i] = False
                        break
        return isolated

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN RUN LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self.config
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = Path(cfg.gallery_dir) / f"exploracion_{ts}"
        for sub in ("large_area", "zooms", "library"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

        if cfg.config_yaml_path:
            try:
                import shutil as _shutil
                src = Path(cfg.config_yaml_path)
                if src.is_file():
                    _shutil.copy2(src, out_dir / src.name)
            except Exception:
                pass

        self._update(started_at=datetime.now().isoformat(), output_dir=str(out_dir))

        # ── Verify connection ─────────────────────────────────────────────
        if self._studio is None:
            msg = "No microscope connection. Call automation_connect_microscope() before running."
            self._update(phase="error", error=msg, last_message=msg)
            ui.log('ERR', f"  [ERR ]  {msg}")
            return

        workspace  = self._workspace
        automation = self._automation

        # ── Load YOLO model once ──────────────────────────────────────────
        if cfg.yolo_enabled and cfg.yolo_model_path:
            try:
                self._load_yolo_model()
            except Exception as exc:
                ui.log('ERR', f"  [ERR ]  Could not load YOLO model: {exc}")
                self._update(phase="error", error=str(exc),
                             last_message=f"YOLO model load failed: {exc}")
                return

        # ── Resolve target classes ────────────────────────────────────────
        target_classes = dict(cfg.target_classes)
        if cfg.yolo_enabled and not target_classes and self._yolo_model is not None:
            target_classes = {n: {"n": 10, "enabled": True} for n in self._yolo_names}
            ui.log('INFO', f"  [YOLO]  Auto-filled target_classes: {list(target_classes.keys())}")

        # ── Build position queue ──────────────────────────────────────────
        origin_x = cfg.origin_x_m
        origin_y = cfg.origin_y_m
        scan_m   = cfg.scan_size_m
        area_m   = cfg.area_to_cover_m2

        self._update(phase="planning",
                     last_message=f"Planning scan at ({origin_x*1e6:.3f}, {origin_y*1e6:.3f}) µm...")
        ui.log('INFO', f"  [PLAN]  Strategy '{cfg.strategy}', origin ({origin_x*1e6:.2f}, {origin_y*1e6:.2f}) µm")

        # ── Stage travel limits ───────────────────────────────────────────
        # Reachable rectangle in metres (same frame as origin). Read from the
        # hardware when possible, else from the config bounds. Frames outside it
        # are clipped below (the positioner stalls at the edge → gwy timeouts).
        stage_bounds_m = None
        if cfg.stage_auto_range:
            stage_bounds_m = self._read_stage_bounds_m()
            if stage_bounds_m is not None:
                ui.log('INFO', "  [STAGE]  Range read from hardware.")
        if stage_bounds_m is None and cfg.stage_x_max_um > cfg.stage_x_min_um \
                and cfg.stage_y_max_um > cfg.stage_y_min_um:
            stage_bounds_m = (cfg.stage_x_min_um * 1e-6, cfg.stage_x_max_um * 1e-6,
                              cfg.stage_y_min_um * 1e-6, cfg.stage_y_max_um * 1e-6)
        if stage_bounds_m is not None:
            _mg = cfg.stage_edge_margin_um * 1e-6
            _bx0, _bx1, _by0, _by1 = stage_bounds_m
            stage_bounds_m = (_bx0 + _mg, _bx1 - _mg, _by0 + _mg, _by1 - _mg)
            ui.log('INFO', "  [STAGE]  Reachable rect (µm): "
                   f"X[{stage_bounds_m[0]*1e6:.0f},{stage_bounds_m[1]*1e6:.0f}] "
                   f"Y[{stage_bounds_m[2]*1e6:.0f},{stage_bounds_m[3]*1e6:.0f}] "
                   f"(margin {cfg.stage_edge_margin_um:.0f} µm)")
            if not (stage_bounds_m[0] <= origin_x <= stage_bounds_m[1]
                    and stage_bounds_m[2] <= origin_y <= stage_bounds_m[3]):
                msg = (f"Origin ({origin_x*1e6:.0f}, {origin_y*1e6:.0f}) µm is outside "
                       f"the stage reachable range — nothing safe to scan. "
                       f"Move the tip/origin inside the stage travel and retry.")
                ui.log('ERR', f"  [STAGE]  {msg}")
                self._update(phase="error", error=msg, last_message=msg)
                ui.inject_to_agent(f"[AUTOMATION ABORTED] {msg}")
                return
        else:
            ui.log('INFO', "  [STAGE]  Stage limits unknown — no clipping "
                           "(enable stage_auto_range or set stage_*_um).")

        if cfg.strategy == "raster":
            positions_rel, _raster_nx, _raster_ny = _generate_raster(
                frame_size_m = scan_m,
                overlap      = cfg.overlap_fraction,
                n_x          = cfg.grid_n_x,
                n_y          = cfg.grid_n_y,
                area_m       = area_m,
            )
            if stage_bounds_m is not None:
                positions_rel, _raster_nx, _raster_ny, _nrem = _clip_grid_rect(
                    positions_rel, _raster_nx, _raster_ny, origin_x, origin_y, stage_bounds_m)
                if _nrem:
                    _m = (f"Clipped {_nrem} frame(s) outside stage range "
                          f"→ {_raster_nx}×{_raster_ny} = {len(positions_rel)} frames.")
                    ui.log('WARN', f"  [STAGE]  {_m}")
                    ui.inject_to_agent(f"[AUTOMATION] {_m}")
            # Raster scans the FULL square grid; max_large_frames does NOT cut it.
            pos_queue      = [(x + origin_x, y + origin_y) for x, y in positions_rel]
            grid_positions = []   # not used in raster
            unvisited      = []   # not used in raster
            prob_unvisited = []   # not used in raster (kept defined for _prob_refill)
            current_cell_idx: int = -1  # not used in raster; _prob_refill is a no-op
            _step_m = scan_m * (1.0 - cfg.overlap_fraction)
            _cov_m  = (_raster_nx - 1) * (_step_m if _step_m > 0 else scan_m * 0.01) + scan_m
            ui.log('INFO', f"  [PLAN]  Raster {_raster_nx}×{_raster_ny} = {len(pos_queue)} frames, "
                           f"bounding square ≈{_cov_m*1e3:.3f} mm/side ({(_cov_m*1e3)**2:.3f} mm²).")
        else:
            positions_rel, grid_nx, grid_ny = _generate_raster(
                frame_size_m=scan_m, overlap=cfg.overlap_fraction,
                n_x=cfg.grid_n_x, n_y=cfg.grid_n_y, area_m=area_m,
            )
            if stage_bounds_m is not None:
                positions_rel, grid_nx, grid_ny, _nrem = _clip_grid_rect(
                    positions_rel, grid_nx, grid_ny, origin_x, origin_y, stage_bounds_m)
                if _nrem:
                    _m = (f"Clipped {_nrem} cell(s) outside stage range "
                          f"→ {grid_nx}×{grid_ny} = {len(positions_rel)} cells.")
                    ui.log('WARN', f"  [STAGE]  {_m}")
                    ui.inject_to_agent(f"[AUTOMATION] {_m}")
            # Probabilistic considers the WHOLE grid; max_large_frames is only the
            # loop stop condition (below), it does NOT shrink the grid.
            grid_positions = [(x + origin_x, y + origin_y) for x, y in positions_rel]
            n_cells = len(grid_positions)

            gx = np.array([p[0] for p in grid_positions])
            gy = np.array([p[1] for p in grid_positions])
            dist_matrix = np.sqrt((gx[:, None] - gx[None, :])**2 +
                                  (gy[:, None] - gy[None, :])**2)

            center_idx = int(np.argmin((gx - origin_x)**2 + (gy - origin_y)**2))
            radius_m   = cfg.prob_initial_center_radius_um * 1e-6
            near_center = [i for i in range(n_cells)
                           if i != center_idx and dist_matrix[center_idx, i] <= radius_m]
            rng_init  = np.random.default_rng()
            n_rand    = min(cfg.prob_initial_random_n, len(near_center))
            rand_idxs = rng_init.choice(len(near_center), size=n_rand, replace=False) if near_center else []
            initial_cells = [center_idx] + [near_center[int(k)] for k in rand_idxs]

            alpha = {cls: np.full(n_cells, cfg.prob_alpha_init) for cls in target_classes}
            beta  = {cls: np.full(n_cells, cfg.prob_beta_init)  for cls in target_classes}
            found_weighted: Dict[str, float] = {cls: 0.0 for cls in target_classes}

            prob_queue: List[int] = list(initial_cells)
            prob_unvisited: List[int] = [i for i in range(n_cells) if i not in set(initial_cells)]
            current_cell_idx: int = center_idx

            pos_queue = []
            unvisited = []
            _step_m = scan_m * (1.0 - cfg.overlap_fraction)
            _cov_m  = (grid_nx - 1) * (_step_m if _step_m > 0 else scan_m * 0.01) + scan_m
            _stop_txt = (f"Stop after {cfg.max_large_frames} scans."
                         if cfg.max_large_frames > 0
                         else "No frame limit (stops on target/manual).")
            ui.log('INFO', f"  [PLAN]  Probabilistic Beta-search: {grid_nx}×{grid_ny} grid "
                           f"({n_cells} cells, bounding square ≈{_cov_m*1e3:.3f} mm/side, "
                           f"{(_cov_m*1e3)**2:.3f} mm²), {len(initial_cells)} initial scans "
                           f"(center + {n_rand} random). {_stop_txt}")

        # ── Per-class state ───────────────────────────────────────────────
        counts       : Dict[str, int] = {cls: 0 for cls in target_classes}
        _area_side_m = cfg.area_side_m

        large_frame_count : int        = 0
        zoom_frame_count  : int        = 0
        last_gwy_index    : int        = self._scan_gwy_index(cfg.gallery_dir)
        large_frames_log  : List[dict] = []
        current_x, current_y           = origin_x, origin_y

        self._update(
            last_message=f"Last .gwy index: {last_gwy_index}. Starting main loop.",
            counts_per_class=dict(counts),
        )

        # ══════════════════════════════════════════════════════════════════════
        # Main loop
        # ══════════════════════════════════════════════════════════════════════
        is_prob = cfg.strategy == "probabilistic"
        active_queue = prob_queue if is_prob else pos_queue

        def _prob_refill(from_idx: int) -> None:
            # Select and enqueue the next probabilistic cell. Must run after
            # EVERY scan attempt -- including skipped/timed-out frames -- or a
            # lost .gwy permanently shrinks the work queue and the run dies
            # early (queue starvation).
            #
            # Only refill when the queue is EMPTY. Otherwise the queue (seeded
            # with the N initial cells) stays size N forever: each cell picked
            # here (within max_jump of the CURRENT frame) would not be scanned
            # until N frames later, from a far-away position, so max_jump /
            # lambda_move would be enforced against a stale cell → N interleaved
            # walkers, huge inter-frame jumps and the crossing "X" path. Keeping
            # the working queue at size 1 during active search makes every hop
            # measured from the true current position, matching the simulation.
            if not (is_prob and prob_unvisited and not prob_queue
                    and not self._stop.is_set()):
                return
            if np.random.default_rng().random() < cfg.prob_epsilon_unknown:
                nxt = self._select_most_uncertain_cell(
                    alpha, beta, found_weighted, target_classes,
                    prob_unvisited, dist_matrix[from_idx], cfg.max_jump_um * 1e-6,
                )
            else:
                nxt = self._select_best_utility_cell(
                    alpha, beta, found_weighted, target_classes,
                    prob_unvisited, dist_matrix[from_idx], cfg.max_jump_um * 1e-6,
                )
            if nxt is not None:
                prob_unvisited.remove(nxt)
                prob_queue.append(nxt)

        while active_queue and not self._stop.is_set():
            # max_large_frames is a stop criterion for the probabilistic search
            # only (raster ends when its full grid queue is exhausted). 0 or
            # negative = no frame limit.
            if is_prob and cfg.max_large_frames > 0 and large_frame_count >= cfg.max_large_frames:
                ui.log('INFO', f"  [STOP]  Max large frames ({cfg.max_large_frames}) reached.")
                break
            if cfg.yolo_enabled and self._criterion_met():
                ui.log('OK', "  [DONE]  All target classes complete.")
                break

            if is_prob:
                cell_idx = prob_queue.pop(0)
                cx, cy   = grid_positions[cell_idx]
                current_cell_idx = cell_idx
                choice_type = "initial" if large_frame_count < len(initial_cells) else "active"
            else:
                cx, cy = pos_queue.pop(0)
                cell_idx = -1
                choice_type = "raster"
            current_x, current_y = cx, cy
            frame_name = f"large_{large_frame_count:04d}"
            frame_ts   = datetime.now().isoformat()
            focus          = ""
            weighted_counts: Dict[str, float] = {}

            self._update(
                phase            = "scanning_large",
                large_frames     = large_frame_count,
                zoom_frames      = zoom_frame_count,
                counts_per_class = dict(counts),
                last_message     = f"[{large_frame_count+1}] Scan at ({cx*1e6:.2f}, {cy*1e6:.2f}) µm",
            )
            ui.log('INFO', f"  [SCAN]  [{large_frame_count+1}] ({cx*1e6:.2f}, {cy*1e6:.2f}) µm")

            # ── Capture large frame ───────────────────────────────────────
            try:
                self._capture_frame(
                    workspace, automation,
                    name       = frame_name,
                    pos_x_m    = cx,
                    pos_y_m    = cy,
                    size_m     = scan_m,
                    resolution = cfg.scan_resolution_px,
                    line_rate  = min(cfg.scan_rate_hz, 100.0),
                )
            except Exception as exc:
                ui.log('ERR', f"  [ERR ]  Capture error: {exc}")
                large_frame_count += 1
                large_frames_log.append({
                    "frame_name": frame_name, "filename": f"{frame_name}.png",
                    "x_um": round(cx*1e6, 4), "y_um": round(cy*1e6, 4),
                    "n_detections": 0, "timestamp": frame_ts,
                    "detections_by_class": "", "error": str(exc),
                })
                _prob_refill(current_cell_idx)
                continue

            # ── Wait for .gwy ─────────────────────────────────────────────
            gwy_path = self._wait_for_next_gwy(
                last_gwy_index,
                self._gwy_timeout_for(cfg.scan_resolution_px, cfg.scan_rate_hz),
            )
            if gwy_path is None:
                ui.log('WARN', "  [WARN]  Timeout waiting for .gwy -- skipping.")
                large_frame_count += 1
                large_frames_log.append({
                    "frame_name": frame_name, "filename": f"{frame_name}.png",
                    "x_um": round(cx*1e6, 4), "y_um": round(cy*1e6, 4),
                    "n_detections": 0, "timestamp": frame_ts,
                    "detections_by_class": "", "error": "gwy_timeout",
                })
                _prob_refill(current_cell_idx)
                continue

            # ── Load → flatten → PNG ──────────────────────────────────────
            try:
                large_data, _ = load_gwy_channel(str(gwy_path), scan_direction=cfg.scan_direction)
                large_flat, _ = flatten_line_by_line(large_data, perc=cfg.flatten_perc)
                png_large     = str(out_dir / "large_area" / f"{frame_name}.png")
                # Grayscale min-max — SAME representation YOLO was trained on
                # (COCOBuilder), so exploration inference matches training.
                array_to_gray_png(large_flat, png_large)
                ui.plot_path(png_large)
                last_gwy_index = self._scan_gwy_index(cfg.gallery_dir)
                img_h, img_w   = large_flat.shape

            except Exception as exc:
                ui.log('ERR', f"  [ERR ]  GWY load error: {exc}")
                large_frame_count += 1
                last_gwy_index = self._scan_gwy_index(cfg.gallery_dir)
                large_frames_log.append({
                    "frame_name": frame_name, "filename": f"{frame_name}.png",
                    "x_um": round(cx*1e6, 4), "y_um": round(cy*1e6, 4),
                    "n_detections": 0, "timestamp": frame_ts,
                    "detections_by_class": "", "error": str(exc),
                })
                _prob_refill(current_cell_idx)
                continue

            # ── YOLO detection ────────────────────────────────────────────
            detections       = []   # reliable (fully inside): count/library/zoom/crops
            detections_raw   = []
            border_detections = []  # touch a frame edge but detected well: MAP only
            n_det_by_class : Dict[str, int] = {}

            if cfg.yolo_enabled and cfg.yolo_model_path:
                try:
                    detections, detections_raw, _pred = self._run_yolo(png_large)
                    # Filter to only target classes that are enabled
                    detections = [
                        d for d in detections
                        if d["label"] in target_classes
                        and target_classes[d["label"]].get("enabled", True)
                    ]
                    # Border filter: detections whose bbox touches a frame edge are
                    # cut-off → their shape/bbox is unreliable, so they are NOT
                    # counted toward targets, NOT added to the library and NOT
                    # zoomed/cropped. BUT if YOLO still detected them (they passed
                    # the confidence threshold, i.e. enough of the object is visible)
                    # they DO indicate an object is present, so they still feed the
                    # probabilistic map (border_detections → weighted_counts below).
                    # This avoids biasing the map toward "empty" at frame edges.
                    # Rejected ones are marked so the overlay draws them in red.
                    _bf = cfg.border_filter_enabled
                    if _bf:
                        # Edge-touching detections that are still mostly inside
                        # (>=~50% visible, judged from the bbox aspect vs the cut
                        # edge) feed the map only; thin slivers are dropped entirely.
                        border_detections = [
                            d for d in detections
                            if self._is_on_border(d, img_w, img_h, _bf)
                            and self._border_map_valid(d, img_w, img_h)]
                        detections = [d for d in detections
                                      if not self._is_on_border(d, img_w, img_h, _bf)]
                        _kept = {id(d) for d in detections}
                        for d in detections_raw:
                            if self._is_on_border(d, img_w, img_h, _bf) and id(d) not in _kept:
                                d["accepted"] = False
                    n_det_by_class = {}
                    for d in detections:
                        n_det_by_class[d["label"]] = n_det_by_class.get(d["label"], 0) + 1
                except Exception as exc:
                    ui.log('ERR', f"  [ERR ]  YOLO error: {exc}")

                # Draw overlay — upscale the frame first (nearest-neighbour, no
                # blur) so the class/score labels render at a legible resolution
                # instead of tiny aliased text on the raw 256/512-px image.
                if _HAS_CV2 and detections_raw:
                    try:
                        base   = cv2.imread(png_large)
                        h0, w0 = base.shape[:2]
                        scale  = max(2, int(round(1024 / max(w0, 1))))   # ~1024 px canvas
                        overlay = cv2.resize(base, (w0 * scale, h0 * scale),
                                             interpolation=cv2.INTER_NEAREST)
                        Hc     = h0 * scale
                        fs     = max(0.5, Hc / 900.0)                    # font scale
                        th_t   = max(1, int(round(fs * 1.6)))            # text thickness
                        th_b   = max(1, int(round(fs)))                  # box thickness
                        for d in detections_raw:
                            col = (0, 200, 0) if d.get("accepted", True) else (0, 0, 220)
                            x1, y1 = int(d["x1"] * scale), int(d["y1"] * scale)
                            x2, y2 = int(d["x2"] * scale), int(d["y2"] * scale)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, th_b)
                            lbl = f"{d['label']} {d['conf']:.2f}"
                            (tw, tht), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, th_t)
                            ty = max(tht + 3, y1 - 3)
                            cv2.rectangle(overlay, (x1, ty - tht - bl - 2),
                                          (x1 + tw + 2, ty + bl), col, -1)     # label background
                            cv2.putText(overlay, lbl, (x1 + 1, ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th_t, cv2.LINE_AA)
                        pred_path = str(out_dir / "large_area" / f"{frame_name}_pred.png")
                        cv2.imwrite(pred_path, overlay)
                        ui.plot_path(pred_path)
                    except Exception:
                        pass

                # ── Probabilistic: update Beta maps ───────────────────────────
                if is_prob and cfg.yolo_enabled:
                    # Confidence-weighted counts for the MAP: reliable detections
                    # PLUS border-touching ones (they still mark object presence).
                    # Pre-isolation.
                    weighted_counts: Dict[str, float] = {}
                    for d in detections + border_detections:
                        weighted_counts[d["label"]] = weighted_counts.get(d["label"], 0.0) + d["conf"]

                    # Isolation filter (reliable detections only)
                    isolation_flags = self._apply_isolation_filter(detections, img_w) if detections else []

                    # Count only isolated detections toward found_weighted
                    for det_i2, d in enumerate(detections):
                        if isolation_flags[det_i2]:
                            lbl = d["label"]
                            found_weighted[lbl] = found_weighted.get(lbl, 0.0) + d["conf"]

                    # Update Beta maps with spatial propagation
                    self._update_prob_maps_beta(
                        alpha, beta, cell_idx, weighted_counts,
                        dist_matrix[cell_idx], target_classes,
                    )

                    # Search focus for logging
                    focus = self._determine_search_focus(cell_idx, alpha, beta, found_weighted, target_classes)
                    ui.log('INFO', f"  [PROB]  focus={focus}  found={dict(found_weighted)}")

                # Log detections
                det_str = ", ".join(f"{k}:{v}" for k, v in n_det_by_class.items()) if n_det_by_class else "0"
                ui.log('INFO', f"  [DET ]  {frame_name}: {len(detections)} detections — {det_str}")

            # ── Large frames log entry ────────────────────────────────────
            _lf_entry: dict = {
                "frame_name"        : frame_name,
                "filename"          : f"{frame_name}.png",
                "x_um"              : round(cx * 1e6, 4),
                "y_um"              : round(cy * 1e6, 4),
                "n_detections"      : len(detections),
                "timestamp"         : frame_ts,
                "detections_by_class": json.dumps(n_det_by_class),
                "choice_type"       : choice_type,
                "search_focus"      : focus if is_prob else "",
                "error"             : "",
            }
            if is_prob:
                for _cls in target_classes:
                    _lf_entry[f"weighted_count_{_cls}"] = weighted_counts.get(_cls, 0.0) if is_prob and cfg.yolo_enabled else 0
            large_frames_log.append(_lf_entry)

            large_frame_count += 1
            self._update(large_frames=large_frame_count)

            # ── Probabilistic: select next cell ───────────────────────────
            _prob_refill(current_cell_idx)

            if is_prob:
                self._update(counts_per_class={cls: round(v, 2) for cls, v in found_weighted.items()})

            # ── Mode 1: no YOLO ───────────────────────────────────────────
            if not cfg.yolo_enabled:
                continue

            # ── Mode 2: detect only, no zoom ─────────────────────────────
            # When a triplet model is assigned we still build the object
            # library here by cropping each counted detection out of the large
            # frame (no zoom re-scan), so per-detection iteration is needed.
            if not cfg.zoom_enabled:
                make_crops = bool(cfg.triplet_model_path)
                for det_i, det in enumerate(detections):
                    cls = det["label"]
                    if not target_classes.get(cls, {}).get("enabled", True):
                        continue
                    n_target = target_classes.get(cls, {}).get("n", 10)
                    if counts.get(cls, 0) >= n_target:
                        continue
                    if make_crops:
                        self._export_large_crop(
                            large_flat, det, det_i, cls, frame_name, out_dir, cfg)
                    counts[cls] = counts.get(cls, 0) + 1
                self._update(counts_per_class=dict(counts))
                if self._criterion_met():
                    ui.log('OK', "  [DONE]  Stopping criterion met.")
                    break
                continue

            # ── Mode 3: zoom on detections ────────────────────────────────
            for det_i, det in enumerate(detections):
                if self._stop.is_set() or self._criterion_met():
                    break

                label = det["label"]
                cls_info = target_classes.get(label, {})
                if not cls_info.get("enabled", True):
                    continue
                n_target = cls_info.get("n", 10)
                if counts.get(label, 0) >= n_target:
                    continue

                x_m, y_m, zoom_size_m = self._det_to_physical(det, img_w, img_h, cx, cy)
                zoom_rate = min(cfg.scan_rate_hz, cfg.zoom_max_rate_hz)
                zoom_name = f"zoom_L{large_frame_count-1:04d}_Z{det_i:02d}_{label}"

                ui.log('INFO',
                    f"  [ZOOM]  #{counts.get(label,0)+1}/{n_target} '{label}'  "
                    f"({x_m*1e6:.2f},{y_m*1e6:.2f})µm  frame={zoom_size_m*1e6:.2f}µm")
                self._update(phase="scanning_zoom",
                             last_message=f"Zoom on '{label}' at ({x_m*1e6:.2f},{y_m*1e6:.2f})µm")

                try:
                    self._capture_frame(
                        workspace, automation,
                        name=zoom_name, pos_x_m=x_m, pos_y_m=y_m,
                        size_m=zoom_size_m, resolution=cfg.zoom_resolution_px,
                        line_rate=zoom_rate,
                    )
                except Exception as exc:
                    ui.log('ERR', f"  [ERR ]  Zoom capture error: {exc}")
                    zoom_frame_count += 1
                    continue

                gwy_zoom = self._wait_for_next_gwy(
                    last_gwy_index,
                    self._gwy_timeout_for(cfg.zoom_resolution_px, zoom_rate),
                )
                if gwy_zoom is None:
                    ui.log('WARN', "  [WARN]  Timeout waiting for zoom .gwy.")
                    zoom_frame_count += 1
                    continue

                try:
                    zoom_data, _ = load_gwy_channel(str(gwy_zoom), scan_direction=cfg.scan_direction)
                    zoom_flat, _ = flatten_line_by_line(zoom_data, perc=cfg.flatten_perc)
                    png_zoom     = str(out_dir / "zooms" / f"{zoom_name}.png")
                    array_to_png(zoom_flat, png_zoom)
                    ui.plot_path(png_zoom)
                except Exception as exc:
                    ui.log('ERR', f"  [ERR ]  Zoom GWY error: {exc}")
                    zoom_frame_count += 1
                    last_gwy_index = self._scan_gwy_index(cfg.gallery_dir)
                    continue

                last_gwy_index = self._scan_gwy_index(cfg.gallery_dir)
                zoom_frame_count += 1

                # Shape metrics + crop pair
                shape_metrics, otsu_mask = self._compute_shape_metrics(zoom_flat)
                if otsu_mask is not None:
                    try:
                        zoom_flat_masked = flatten_with_object_mask(
                            zoom_flat, otsu_mask, perc=cfg.flatten_perc)
                        class_crops_dir = out_dir / "library" / "crops" / label
                        make_crop_pair(zoom_flat, zoom_flat_masked, otsu_mask,
                                       zoom_name, class_crops_dir)
                        if _HAS_CV2:
                            mask_path = str(out_dir / "zooms" / f"{zoom_name}_mask.png")
                            cv2.imwrite(mask_path, otsu_mask)
                    except Exception:
                        pass

                # Build record
                bbox_w_px = det["x2"] - det["x1"]
                bbox_h_px = det["y2"] - det["y1"]
                record: dict = {
                    "zoom_index"    : zoom_frame_count - 1,
                    "large_file"    : f"{frame_name}.png",
                    "zoom_file"     : f"{zoom_name}.png",
                    "class"         : label,
                    "score"         : round(det["conf"], 4),
                    "centroid_x_um" : round(x_m * 1e6, 4),
                    "centroid_y_um" : round(y_m * 1e6, 4),
                    "bbox_w_px"     : round(bbox_w_px, 1),
                    "bbox_h_px"     : round(bbox_h_px, 1),
                    "zoom_size_um"  : round(zoom_size_m * 1e6, 4),
                    "zoom_rate_hz"  : zoom_rate,
                }
                record.update(shape_metrics)
                self._records.append(record)

                counts[label] = counts.get(label, 0) + 1
                self._update(counts_per_class=dict(counts))
                ui.log('OK',
                    f"  [ OK ]  '{label}' {counts[label]}/{n_target}  "
                    f"totals: {dict(counts)}")

        # ── Finalise ──────────────────────────────────────────────────────────
        # Save large_frames.csv
        if large_frames_log:
            lf_csv = out_dir / "large_frames.csv"
            with open(lf_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(large_frames_log[0].keys()))
                writer.writeheader()
                writer.writerows(large_frames_log)

        # Save detections.csv (zoom records)
        if self._records:
            csv_path = out_dir / "detections.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._records[0].keys()))
                writer.writeheader()
                writer.writerows(self._records)

        # Triplet analysis (skip when nothing was captured)
        if cfg.triplet_model_path and Path(cfg.triplet_model_path).is_file():
            if large_frame_count == 0:
                ui.log('INFO', "  [TRIP]  Skipped — no frames captured.")
            else:
                self._update(phase="finalising", last_message="Running Triplet analysis...")
                ui.log('INFO', "  [TRIP]  Running Triplet analysis...")
                try:
                    from automation.triplet_analyzer import TripletAnalyzer
                    ta_result = TripletAnalyzer(
                        out_dir=str(out_dir),
                        triplet_model_path=cfg.triplet_model_path,
                        class_names=list(target_classes.keys()),
                    ).run()
                    if ta_result.get("status") == "ok":
                        ui.log('OK', f"  [ OK ]  Triplet: {ta_result.get('message','done.')}")
                    else:
                        ui.log('WARN', f"  [WARN]  Triplet skipped: {ta_result.get('message','')}")
                except Exception as _te:
                    ui.log('ERR', f"  [WARN]  TripletAnalyzer failed: {_te}")

        # ── Save probabilistic maps ───────────────────────────────────────
        if cfg.strategy == "probabilistic" and 'alpha' in dir():
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            maps_dir = out_dir / "prob_maps"
            maps_dir.mkdir(exist_ok=True)

            _nx = grid_nx if 'grid_nx' in dir() else int(math.sqrt(len(alpha[next(iter(alpha))])))
            _ny = grid_ny if 'grid_ny' in dir() else _nx

            def _to_grid(arr1d):
                # Row-major: cell index i maps to (row=i//_nx, col=i%_nx),
                # matching _generate_raster. (Previously assumed snake order,
                # which mirrored odd rows and scrambled the maps.)
                grid2d = np.zeros((_ny, _nx))
                n = min(len(arr1d), _nx * _ny)
                grid2d.reshape(-1)[:n] = np.asarray(arr1d, dtype=float)[:n]
                return grid2d

            npz_data: dict = {}
            for cls in alpha:
                a_arr = alpha[cls]
                b_arr = beta[cls]
                mu_arr  = a_arr / (a_arr + b_arr)
                unc_arr = np.array([self._beta_uncertainty(float(a_arr[i]), float(b_arr[i]))
                                    for i in range(len(a_arr))])
                npz_data[f"alpha_{cls}"] = a_arr
                npz_data[f"beta_{cls}"]  = b_arr

                for name, arr, cmap, title in [
                    ("mu",          mu_arr,  "hot",    f"P(object) — {cls}"),
                    ("uncertainty", unc_arr, "plasma", f"Uncertainty — {cls}"),
                ]:
                    g = _to_grid(arr)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    # origin="lower": row 0 = min-Y at the bottom, so the map
                    # matches the microscope viewport (Y grows upward).
                    im = ax.imshow(g, cmap=cmap, origin="lower", aspect="equal",
                                   vmin=0, vmax=g.max() or 1)
                    n_found = round(found_weighted.get(cls, 0), 1)
                    n_tgt   = target_classes.get(cls, {}).get("n", "?")
                    ax.set_title(f"{title}  ({n_found}/{n_tgt})")
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    fig.savefig(str(maps_dir / f"{name}_{cls}.png"), dpi=100, bbox_inches="tight")
                    plt.close(fig)

            visited_arr = np.zeros(n_cells)
            for order_i, scan_entry in enumerate(large_frames_log):
                pass

            np.savez_compressed(str(maps_dir / "maps.npz"), **npz_data)

            visited_order = np.full((_ny, _nx), np.nan)
            if 'initial_cells' in dir():
                all_scanned_order = initial_cells[:]
                visited_order = _to_grid(np.array([
                    1.0 if i not in prob_unvisited and i not in set(prob_queue) else 0.0
                    for i in range(n_cells)
                ]))
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(visited_order, cmap="Blues", origin="lower", aspect="equal")
            ax.set_title("Visited cells")
            plt.colorbar(im, ax=ax, fraction=0.046)
            fig.savefig(str(maps_dir / "visited.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)
            ui.log('INFO', f"  [MAP ]  Probability maps saved → {maps_dir}")

        # session_summary.json
        summary = self.get_status()
        summary["finished_at"]     = datetime.now().isoformat()
        summary["large_frames_csv"]= str(out_dir / "large_frames.csv")
        summary["detections_csv"]  = str(out_dir / "detections.csv")
        summary["config"] = {
            "strategy"          : cfg.strategy,
            "scan_size_um"      : cfg.scan_size_um,
            "scan_rate_hz"      : cfg.scan_rate_hz,
            "scan_resolution_px": cfg.scan_resolution_px,
            "overlap_fraction"  : cfg.overlap_fraction,
            "grid_n_x"          : cfg.grid_n_x,
            "grid_n_y"          : cfg.grid_n_y,
            "area_to_cover_um"  : cfg.area_to_cover_um,
            "max_large_frames"  : cfg.max_large_frames,
            "yolo_enabled"      : cfg.yolo_enabled,
            "zoom_enabled"      : cfg.zoom_enabled,
            "target_classes"    : target_classes,
            "max_jump_um"       : cfg.max_jump_um,
            "zoom_bbox_factors" : cfg.zoom_bbox_factors,
            "zoom_max_size_um"  : cfg.zoom_max_size_um,
        }
        if is_prob:
            summary["prob_params"] = {
                "alpha_init"              : cfg.prob_alpha_init,
                "beta_init"               : cfg.prob_beta_init,
                "spatial_sigma_um"        : cfg.prob_spatial_sigma_um,
                "kappa_uncertainty"       : cfg.prob_kappa_uncertainty,
                "lambda_move"             : cfg.prob_lambda_move,
                "epsilon_unknown"         : cfg.prob_epsilon_unknown,
                "negative_update_scale"   : cfg.prob_negative_update_scale,
                "initial_random_n"        : cfg.prob_initial_random_n,
                "initial_center_radius_um": cfg.prob_initial_center_radius_um,
            }
            summary["final_deficit"] = self._compute_deficit(found_weighted, target_classes)
            total_found = sum(found_weighted.values())
            summary["scan_efficiency"] = round(total_found / max(1, large_frame_count), 3)
            summary["isolation_params"] = {
                "max_bbox_overlap"  : cfg.isolation_max_bbox_overlap,
                "min_distance_um"   : cfg.isolation_min_distance_um,
            }
        with open(out_dir / "session_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        _done_msg = (
            f"Done. {large_frame_count} large frames, "
            f"{zoom_frame_count} zooms. Output: {out_dir}"
        )
        self._update(phase="done", finished_at=datetime.now().isoformat(),
                     last_message=_done_msg)
        ui.inject_to_agent(
            f"[AUTOMATION FINISHED] AFM exploration ended. "
            f"{large_frame_count} large frames, {zoom_frame_count} zooms. "
            f"Counts: {dict(counts)}. Output: {out_dir}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Module-level API
# ══════════════════════════════════════════════════════════════════════════════

def connect_microscope() -> dict:
    """
    Connect to Nanosurf Studio.
    If a session is already in phase 'connected', returns immediately.
    Otherwise always creates a fresh session and connects.
    Call this BEFORE run_exploration().

    Returns
    -------
    dict  {"status": "ok"|"error", "message": ...}
    """
    global _ACTIVE_SESSION
    with _SESSION_LOCK:
        current = _ACTIVE_SESSION
        if current is not None and current.get_status()["phase"] == "connected":
            return _ok(message="Already connected to Nanosurf Studio.")
        session         = ExplorationSession(ExplorationConfig())
        _ACTIVE_SESSION = session
    return session.connect()


def run_exploration(config_dict: dict) -> dict:
    """
    Start an exploration session in a background thread.

    If connect_microscope() was called first, the existing Studio connection
    is reused -- the active session gets the new config applied to it.
    If not yet connected, a new session is created and connect() is called
    automatically at the start of run().
    """
    global _ACTIVE_SESSION

    # Build and validate config first, outside the lock
    valid    = set(ExplorationConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in valid}
    try:
        cfg = ExplorationConfig(**filtered)
    except Exception as exc:
        return _err(f"Invalid configuration: {exc}")

    if not cfg.gallery_dir:
        return _err("'gallery_dir' must be provided.")
    if cfg.yolo_enabled and not cfg.yolo_model_path:
        return _err("'yolo_model_path' must be set when yolo_enabled=True.")
    if cfg.zoom_enabled and not cfg.yolo_enabled:
        return _err("'zoom_enabled=True' requires 'yolo_enabled=True'.")

    with _SESSION_LOCK:
        current = _ACTIVE_SESSION
        current_phase = current.get_status()["phase"] if current is not None else None

        # Block only if a real scan is actively running
        if current_phase in ("planning", "scanning_large", "detecting",
                             "scanning_zoom", "finalising"):
            return _err(
                f"A session is already running (phase='{current_phase}'). "
                "Call stop_exploration() first."
            )

        # Require an explicit prior call to connect_microscope()
        if current is None or current_phase != "connected" or current._studio is None:
            return _err(
                "Microscope not connected. "
                "Call automation_connect_microscope() first."
            )

        session        = current
        session.config = cfg
        session._stop.clear()

    threading.Thread(target=session.run, daemon=True, name="afm_exploration").start()

    return _ok(
        data={"gallery_dir": cfg.gallery_dir, "strategy": cfg.strategy},
        message=(
            f"Exploration started. Strategy: '{cfg.strategy}'. "
            f"Classes: {list(cfg.target_classes.keys())}."
        ),
    )


def get_exploration_status() -> dict:
    """Return the status of the active (or last) session."""
    with _SESSION_LOCK:
        if _ACTIVE_SESSION is None:
            return _ok(data={"phase": "no_session"},
                       message="No session started yet.")
        return _ok(data=_ACTIVE_SESSION.get_status())


def stop_exploration() -> dict:
    """Graceful abort -- finishes the current frame then stops."""
    with _SESSION_LOCK:
        if _ACTIVE_SESSION is None:
            return _err("No active session.")
        phase = _ACTIVE_SESSION.get_status()["phase"]
        if phase in ("done", "error"):
            return _err(f"Session already finished (phase='{phase}').")
        _ACTIVE_SESSION.stop()
    return _ok(message="Stop signal sent. Session will finish after the current frame.")


def load_automation_config(path: str) -> dict:
    """
    Load an ExplorationConfig from a YAML file (current schema only — old
    pre-refactor field names are not supported).
    """
    try:
        import yaml
    except ImportError:
        return _err("PyYAML is not installed. Run: pip install pyyaml")

    p = Path(path)
    if not p.is_file():
        return _err(f"Config file not found: '{path}'")

    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        return _err(f"Failed to parse YAML: {exc}")

    # Store config file reference
    raw.setdefault("config_yaml_path", str(p.resolve()))

    # Build config from known fields only
    valid    = set(ExplorationConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid}
    try:
        cfg = ExplorationConfig(**filtered)
    except Exception as exc:
        return _err(f"Invalid configuration values: {exc}")

    return _ok(data=cfg, message=f"Config loaded from '{path}'.")


def save_automation_config(cfg: ExplorationConfig, path: str) -> dict:
    """Save an ExplorationConfig to a YAML file."""
    try:
        import yaml
    except ImportError:
        return _err("PyYAML is not installed. Run: pip install pyyaml")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Collect all dataclass fields (skip computed properties)
    fields = ExplorationConfig.__dataclass_fields__
    data   = {name: getattr(cfg, name) for name in fields}

    try:
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as exc:
        return _err(f"Failed to write config: {exc}")

    return _ok(message=f"Config saved to '{path}'.")


def list_automation_configs(directory: str) -> dict:
    """List all YAML config files in a directory."""
    d = Path(directory)
    if not d.is_dir():
        return _err(f"Directory not found: '{directory}'")
    configs = [str(p) for p in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml"))]
    return _ok(data=configs, message=f"{len(configs)} config file(s) found.")
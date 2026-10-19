from dataclasses import dataclass, field, asdict,is_dataclass, fields
#from math import cos, sin, pi
from typing import List, Tuple, Dict, Any, Set, Optional
from copy import deepcopy
import numpy as np
import random
import trimesh
import json
import os
import yaml
import ast

def to_json_dict(obj):
    """
    Converts dataclasses/lists/dicts to a JSON-safe dict,
    skipping fields with metadata {'serialize': False} and converting NumPy.
    """
    if is_dataclass(obj):
        out = {}
        for f in fields(obj):
            if f.metadata.get('serialize', True) is False:
                continue
            out[f.name] = to_json_dict(getattr(obj, f.name))
        return out
    elif isinstance(obj, dict):
        return {k: to_json_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_dict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):  # np.float32, np.int64, etc.
        return obj.item()
    else:
        return obj

# =========================
# 1) Shape generators
# =========================

def regular_polygon(*, n_sides: int) -> list:
    """
    Generates a closed regular polygon with n_sides sides.
    Coordinates centred at (0,0) and normalised.
    Returns one variant: a list of segments {'type': 'line', 'points': [p1, p2]}.
    """
    if n_sides < 3:
        raise ValueError("The number of sides must be at least 3")

    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    points = [(float(np.cos(a)), float(np.sin(a))) for a in angles]

    max_abs = max(max(abs(x), abs(y)) for x, y in points) or 1.0
    points = [(x / max_abs, y / max_abs) for x, y in points]

    sections = []
    for i in range(n_sides):
        p1 = points[i]
        p2 = points[(i + 1) % n_sides]
        sections.append({"type": "line", "points": [p1, p2]})
    return sections

def ellipse_spline(*, n_points: int = 8) -> list:
    """
    Generates a closed ellipse using n_points points (single spline).
    Centred at (0,0), within [-1,1] in x and y.
    Devuelve: [{'type': 'spline', 'points': [[x,y], ...]}]
    """
    if n_points < 3:
        raise ValueError("The ellipse must have at least 3 control points")

    angles = np.linspace(0, 2 * np.pi, n_points + 1, endpoint=True)
    a, b = 1.0, 1.0
    points = [(float(a * np.cos(t)), float(b * np.sin(t))) for t in angles]
    return [{"type": "spline", "points": points}]
# (optional) bacillus-style capsule; use it if you call it from the YAML
def capsule(*, length: float = 2.0, width: float = 1.0) -> list:
    r = width / 2.0
    yt, yb = length / 2.0, -length / 2.0
    return [
        {"type": "line",   "points": [( r, yt), (-r, yt)]},
        {"type": "spline", "points": [(-r, yt), (-2*r, 0.0), (-r, yb)]},
        {"type": "line",   "points": [(-r, yb), ( r, yb)]},
        {"type": "spline", "points": [( r, yb), ( 2*r, 0.0), ( r, yt)]},
    ]


# Registry of functions available for the YAML 'type:function'
SHAPE_FUNC_REGISTRY = {
    "regular_polygon": regular_polygon,
    "ellipse_spline":  ellipse_spline, 
    "capsule":         capsule,
}


# ==========================================
# 2) Helpers to resolve type: "function"
# ==========================================

def _parse_func_call(expr: str):
    """
    Safely parses 'name(arg=..., ...)' -> (name, kwargs).
    Only literals are allowed (ast.literal_eval), no eval().
    """
    node = ast.parse(expr, mode="eval")
    if not isinstance(node.body, ast.Call) or not isinstance(node.body.func, ast.Name):
        raise ValueError(f"Invalid function call: {expr}")
    fname = node.body.func.id
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.body.keywords}
    return fname, kwargs


def _resolve_function_segment(seg: dict):
    """
    Expands a {'type':'function', ...} segment into a list of line/spline segments.
    """
    if "call" in seg:
        fname, kwargs = _parse_func_call(seg["call"])
    else:
        fname = seg.get("name")
        kwargs = seg.get("kwargs", {}) or {}
    if fname not in SHAPE_FUNC_REGISTRY:
        raise ValueError(f"Unknown shape function: {fname}")
    variante = SHAPE_FUNC_REGISTRY[fname](**kwargs)
    if not isinstance(variante, list) or not variante or not isinstance(variante[0], dict):
        raise ValueError("The function must return a list of segments (dicts).")
    return variante


def resolve_sections(raw_sections):
    """
    raw_sections: List[Variant]; Variant = List[Segment].
    For each variant, any {'type':'function',...} entries are expanded
    and mixed with line/spline within the same variant.
    """
    resolved = []
    for variant in raw_sections:
        if not isinstance(variant, list):
            raise ValueError("Each variant must be a list of segments.")
        segs = []
        for seg in variant:
            if isinstance(seg, dict) and seg.get("type") == "function":
                segs.extend(_resolve_function_segment(seg))
            else:
                segs.append(seg)
        resolved.append(segs)
    return resolved
from dataclasses import fields as dc_fields, MISSING

def _get_field_default(cls, name):
    """Returns the default (or default_factory()) of a dataclass field."""
    for f in dc_fields(cls):
        if f.name == name:
            if f.default is not MISSING:
                return f.default
            if f.default_factory is not MISSING:  # type: ignore[attr-defined]
                return f.default_factory()
    raise AttributeError(f"No default found for field '{name}'")


# ==========================================
# 3) Objects loader from YAML (catalogue)
# ==========================================

# DEFAULT_YAML = "./synthetic/config_objects.yaml"
#DEFAULT_YAML = "./data/config/default.yaml"
DEFAULT_YAML = "/data/config/triangles_only.yaml"
def _shape_catalog_resuelto(data: dict) -> Dict[str, list]:
    """
    Builds a catalogue {name: resolved_sections} resolving type:function
    only once per shape.
    """
    catalog = {}
    for s in (data.get("shapes") or []):
        name = s.get("name")
        secs = s.get("sections")
        if name and secs:
            catalog[name] = resolve_sections(secs)
    return catalog


def default_object_list(yaml_path: str = DEFAULT_YAML) -> list:
    """
    Reads the full configuration YAML and returns the list of ObjectConfig,
    resolving 'shape' -> 'sections' from the catalogue and expanding functions.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    shape_catalog = _shape_catalog_resuelto(data)

    objetos = []
    for conf in (data.get("objects") or []):
        sections = conf.get("sections")
        if sections is None:
            shape_key = conf.get("shape")
            if shape_key:
                sections = shape_catalog.get(shape_key)

        # image, STL or domains objects do not need sections (they have their own renderer)
        shape_key = conf.get("shape", "")
        has_stl   = bool(conf.get("stl_files"))
        is_self_rendered = (shape_key in ("image", "STL", "domains")) or has_stl

        # if disabled and it has no sections/shape nor its own renderer, ignore it
        if sections is None and not is_self_rendered and not conf.get("enable", True):
            continue

        if sections is None and not is_self_rendered:
            raise ValueError(
                f"Object '{conf.get('label','<no label>')}' has no 'sections' "
                f"nor a 'shape' resolvable in the YAML 'shapes:'."
            )

        # Empty placeholder for objects that do not use sections
        if sections is None:
            sections = [[{"type": "line", "points": [(0.0, 0.0), (1.0, 0.0)]}]]

        # if they came inline with functions, expand them
        sections = resolve_sections(sections)

        merged = dict(conf)
        merged["sections"] = sections
        objetos.append(ObjectConfig(**merged))

    return objetos


# =========================
# 4) Config dataclasses
# =========================

# Parameter subgroups
@dataclass
class ObjectConfig:
    label: str = 'class_1'
    category: str = ''
    shape: str = ''
    enable: bool = True
    n_objects: int = 4
    probability: float = 1.0
    max_overlap: float = 0.3

    # shape
    size_shape: List[int] = field(default_factory=lambda: [30, 30])
    height:  List[float] = field(default_factory=lambda: [0.5, 1.0])
    orden_z:  List[float] = field(default_factory=lambda: [0.1, 4])
    step_perfil_z: List[float] = field(default_factory=lambda: [0.0, 0.02])
    is_fill: bool = True
    variation_border_pct: float = 0.0
    num_points_border: Optional[int] = None
    dmax_for_DNA: float = 2
    max_angel_for_DNA: float = 30

    # object sections (list of variants; each variant = list of segments)
    sections: List[List[Dict[str, List[Tuple[float, float]]]]] = field(default_factory=lambda: [[
        {"type": "line", "points": [(0.5, 0.5), (-0.5, 0.5)]},
        {"type": "spline", "points": [(-0.5, 0.5), (-1, 0), (-0.5, -0.5)]},
        {"type": "line", "points": [(-0.5, -0.5), (0.5, -0.5)]},
        {"type": "spline", "points": [(0.5, -0.5), (1, 0), (0.5, 0.5)]},
    ],])
    closed: bool = True

    # Optional STL
    stl_files: Optional[List[str]] = None          # list of STL paths
    stl_weights: Optional[List[float]] = None      # same indices as stl_files (optional)

    # Grain/domain field (shape == "domains")
    # All geometry/Z parameters live in the dataset-wide `domains_conf:` section
    # (DomainsConfig) so they are not duplicated on every object. Only the mode
    # can be overridden per object, which is what lets two domains objects share
    # one config while rendering a tessellation and an island film respectively.
    domains_mode: Optional[str] = None   # None → use domains_conf.mode

    # Image-based object (shape == "image")
    # Files live in ref_objects/image_files/<image_folder>/
    image_folder:  Optional[str]        = None   # subfolder name = superclass_class
    image_files:   Optional[List[str]]  = None   # list of PNG filenames inside image_folder
    image_weights: Optional[List[float]]= None   # sampling weights per file (uniform if None)
    # Legacy single-file fields (kept for backward compatibility)
    image_file: Optional[str] = None
    mask_file:  Optional[str] = None
    # cache y estado
    mesh: Optional[trimesh.Trimesh] = field(default=None, repr=False, compare=False, metadata={'serialize': False})
    _last_stl_file: Optional[str] = field(default=None, repr=False, compare=False, metadata={'serialize': False})
    _mesh_cache: Dict[str, trimesh.Trimesh] = field(default_factory=dict, repr=False, compare=False, metadata={'serialize': False})
 
    def pick_stl_file(self) -> Optional[str]:
        """
        Returns an STL chosen at random from self.stl_files.
        - If 'stl_weights' is invalid, a uniform distribution is used.
        """
        files: Optional[List[str]] = self.stl_files
        if not files:
            return None

        weights = self.stl_weights

        def valid_weights(ws, n) -> bool:
            if ws is None or len(ws) != n:
                return False
            for w in ws:
                if not isinstance(w, (int, float)) or  w < 0:
                    return False

            # sums to exactly 1 (slight imprecision allowed)
            return abs(sum(ws) - 1.0) < 1e-3

        w = weights if valid_weights(weights, len(files)) else None  # None => uniform
        return random.choices(files, weights=w, k=1)[0]

    def load_mesh(self, stl_folder: str, filename: str):
        """Loads from cache or disk."""
        from .prosess_stl import load_stl
        path = f'{stl_folder}/{filename}'
        if filename in self._mesh_cache:
            self.mesh = self._mesh_cache[filename]
        else:
            m = load_stl(path)           
            self._mesh_cache[filename] = m
            self.mesh = m
        self._last_stl_file = filename

    # Transformations
    scale_pct: float = 0.5
    eccentricity_variation_up_pct: float = 0.3
    eccentricity_variation_down_pct: float = -0.6
    rotation_range_deg: List[float] = field(default_factory=lambda: [0, 360])
    apply_bending: bool = True
    bending_axis: str = 'y'
    bending_amplitud_pct: float = 0.1
    bending_freq: float = 2
    bending_kind: str = 'sin'
    bending_diff_center: float = 0.1

    # textura
    apply_perlin: bool = True
    perlin_scale_px: List[int] = field(default_factory=lambda: [2, 5, 10, 20])
    perlin_amplitud_pct: float = 0.05

    apply_ellipses_texture: bool = True
    ellipses_probability: float = 0.5
    ellipses_relative_radios_range: List[float] = field(default_factory=lambda: [0.05, 0.5])
    ellipses_num: int = 10
    ellipses_amplitud_range:  List[float] = field(default_factory=lambda: [0.05, 0.1])
    ellipses_eccentriciy_range: List[float] = field(default_factory=lambda: [0.5, 1.0])
    ellipses_overlap: float = 0.2
    ellipses_shape: str = 'dome'

    apply_noise: bool = True
    noise_probability: float = 0.2
    noise_sigma: float = 1.0
    noise_amplitud: float = 0.01

    # colony
    apply_colony: bool = False
    colony_probability: float = 0.25
    colony_n_objects: List[int] = field(default_factory=lambda: [2, 3, 4, 12, 24, 50, 100])
    colony_realtive_height: float = 0.1
    colony_diff_rotation: float = 10.0
    colony_relative_center: float = 0.0
    colony_relative_scale_pct: float = 0.0
    colony_relative_eccentricity_variation_pct: float = 0.0
    colony_min_overlap: float = 0.05
    colony_max_overlap: float = 0.2

    def show_params(self):
        print(f"ObjectConfig for label='{self.label}'")
        for label, value in asdict(self).items():
            if label == "sections":
                print(f"  {label}: {len(value)} variants")
            else:
                print(f"  {label}: {value}")

@dataclass
class SubstratumConfig:
    # -------- TILT / BOW (scanner_tilt + scanner_bow) --------
    apply_tilt: bool = False
    tilt_probability: float = 0.5
    tilt_range: float = 0.2            # ±tilt range (max peak-to-peak Δz_lin)
    second_order_range: float = 0.2    # Δz_quad ~ second_order_range * tilt_range

    # -------- ESCALONES (STEPS / STAIRS) --------
    apply_stair: bool = False
    stair_kind: str = 'spline'         # 'spline' o 'polygon'
    stair_probability: float = 0.1     # note: keeping your original default
    stair_height: List[float] = field(default_factory=lambda: [0.2])

    # -------- PERIODIC PATTERNS (PATTERN: pillars/holes/grating) --------
    apply_pattern: bool = False
    pattern_probability: float = 0.5
    pattern_type: str = 'pillars'      # 'pillars', 'holes', 'grating'
    pattern_shape: str = 'circle'      # 'circle', 'square', 'triangle', 'hexagon'

    # ranges → always [min, max]
    pattern_shape_angle_deg: List[float] = field(default_factory=lambda: [0.0, 45.0])
    pattern_period_x_px_range: List[int] = field(default_factory=lambda: [50, 75])
    pattern_period_y_px_range: List[int] = field(default_factory=lambda: [50, 75])
    pattern_duty_cycle: List[float] = field(default_factory=lambda: [0.45, 0.55])
    pattern_slope: List[float] = field(default_factory=lambda: [0.1, 0.3])
    pattern_height: List[float] = field(default_factory=lambda: [0.5, 1.0])
    pattern_orientation_deg: List[float] = field(default_factory=lambda: [0.0, 90.0])
    pattern_phase_x: List[float] = field(default_factory=lambda: [0.0, 60.0])  # in px
    pattern_phase_y: List[float] = field(default_factory=lambda: [0.0, 60.0])  # in px

    # -------- SHAPES  --------
    apply_shapes: bool = True
    shapes_probability: float = 0.5
    shapes_n: int = 1
    shapes_size: List[int] = field(default_factory=lambda: [75, 45, 25, 10])
    shapes_height: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.1, 0.05])

    # -------- DIRT  --------
    apply_dirt: bool = True
    dirt_probability: float = 0.5
    dirt_n: int = 10
    dirt_size: List[int] = field(default_factory=lambda: [25, 10, 5])
    dirt_height: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.1])
    dirt_intensity: float = 0.5
    dirt_delta_n: int = 250
    dirt_delta_height: float = 0.2
    dirt_delta_elongation_max: int = 0        # extra pixels in X per spike (0 = no elongation)
    dirt_delta_elongation_decay: float = 0.5  # height decay per extra pixel (0–1)
    dirt_delta_elongation_prob: float = 1.0   # probability that each spike is elongated (0–1)

    # -------- PERLIN NOISE  --------
    apply_perlin: bool = True
    perlin_scale_px: List[int] = field(default_factory=lambda: [10, 5, 2])
    perlin_amplitud: float = 0.3

    # -------- GAUSSIAN NOISE  --------
    apply_gaussian: bool = True        # keeping your original default
    gaussian_amplitud: float = 0.1
    gaussian_mean: float = 0.0
    gaussian_scale: float = 0.5

    # -------- GAUSSIAN FILTER  --------
    apply_gaussian_filter: bool = False
    filter_sigma: float = 1.0


@dataclass
class TipConfig:
    is_double: bool = False
    double_probability: float = 0.5
    radi_range: List[int] = field(default_factory=lambda: [5, 30])
    eccentrisity_range: List[float] = field(default_factory=lambda: [0.9, 1.0])
    theta_range: List[int] = field(default_factory=lambda: [10, 30])
    distance_min_pr1: float = 0.5
    distance_max_pr1:float = 1.5


@dataclass
class PIDConfig:
    apply_pid       : bool  = True
    pid_probability : float = 0.7          # probability of applying per image
    scan_direction  : str   = "forward"    # "forward" | "backward" — fixed for the whole dataset

    # Physical parameters of tapping mode — fixed for the whole dataset
    A0   : float = 1.0   # normalised free amplitude
    beta : float = 1.0   # interaction decay rate
    z0   : float = 2.0   # interaction onset distance
    ASP  : float = 0.5   # amplitude set-point (fraction of A0)

    # Controller parameters — random [min, max] range per image
    Kp            : List[float] = field(default_factory=lambda: [0.3, 0.8])
    Ki            : List[float] = field(default_factory=lambda: [0.05, 0.3])
    Kd            : List[float] = field(default_factory=lambda: [0.01, 0.1])
    dt            : List[float] = field(default_factory=lambda: [0.5, 1.5])
    max_speed     : List[float] = field(default_factory=lambda: [0.5, 2.0])
    stickiness_prob: List[float] = field(default_factory=lambda: [0.0, 0.05])
    jump_prob     : List[float] = field(default_factory=lambda: [0.0, 0.02])
    jump_magnitude: List[float] = field(default_factory=lambda: [0.5, 3.0])


@dataclass
class DomainsConfig:
    """
    Grain / domain topography generator (shape: "domains").

    Flat, dataset-wide section `domains_conf:` in the YAML — a sibling of
    tip_conf / substrate_conf / artifacts_conf / pid_conf, so the agent can edit
    it with the existing set_param / set_params / describe_section tools.

    UNITS — SimuScan native, like every other section:
      * lateral sizes are in PIXELS (`*_px`), as in size_shape, perlin_scale_px,
        pattern_period_x_px_range, dirt_size, ...
      * heights are in RELATIVE HEIGHT UNITS (~0-1), as in an object's `height`,
        dirt_height, pattern_height, stair_height, ...

    The vendored generator in simuscan.domains_core is written in nm; the
    translation happens once, in core_overrides(), by feeding it the pixel values
    with pixel_size_nm = 1. That is exact: the generator uses pixel_size_nm only
    for lateral nm->px conversions and never mixes it into Z.

    GRAIN COUNT is not a parameter — it follows from the frame:
        grains ~= (config.image_size / mean_grain_size_px)^2
    """
    # -- SimuScan bridge ------------------------------------------------------
    mode: str = "tessellation"       # "tessellation" | "overlap_interactive"
    seed: Optional[int] = None       # None = different grains on every image

    # How the grain relief lands in SimuScan height units:
    #   "direct"  : use the z_* values below as they are, so grain heights stay
    #               comparable from image to image.
    #   "relative": rescale each image so the z_norm_percentile of the relief
    #               equals a value drawn from the object's own `height` range.
    z_mode: str = "direct"
    z_norm_percentile: float = 99.5

    # Instances -> COCO. Grains come out of the generator as a label map and are
    # used AS IS: no re-thresholding against the convolved topography (that rule
    # only makes sense for isolated objects on a black background).
    min_grain_area_px: int = 0       # 0 = keep every visible grain
    max_instances: int = 0           # 0 = no cap; else keep the N largest
    mask_dilate_px: int = 0          # COCO-time dilation; 0 keeps shared borders sharp

    # Objects sitting on the film take the pixel, so grain masks are clipped
    # against them and a grain that loses more than this fraction of its area is
    # dropped — a sliver peeking out from under a particle is not a useful
    # training instance.
    min_visible_after_objects: float = 0.5

    # Whether the grain field blocks other objects. False (default) lets the
    # grains act as a textured film that particles/bacteria can sit on top of,
    # and keeps the substrate texture from being smoothed away underneath them.
    block_other_objects: bool = False

    # -- Grain-size distribution (px) -----------------------------------------
    size_distribution: str = "gaussian"          # gaussian | lognormal | uniform
    mean_grain_size_px: float = 25.0
    grain_size_sigma_px: float = 7.5
    grain_size_range_px: List[float] = field(default_factory=lambda: [15.0, 35.0])

    # -- XY shape -------------------------------------------------------------
    eccentricity_range: List[float] = field(default_factory=lambda: [0.0, 0.4])
    n_sides_range: List[int] = field(default_factory=lambda: [4, 8])
    curved_grain_probability: float = 0.30
    curved_side_probability: float = 0.30
    side_curvature_range: List[float] = field(default_factory=lambda: [0.05, 0.50])
    shape_irregularity_range: List[float] = field(default_factory=lambda: [0.03, 0.12])

    # -- Mean height and its spatial correlation ------------------------------
    z_mean: List[float] = field(default_factory=lambda: [0.5, 1.2])
    neighbor_z_correlation: float = 0.60
    correlation_length_px: float = 50.0
    correlation_cutoff_factor: float = 2.0

    # -- Periodic height modulation around the perimeter ----------------------
    boundary_z_amplitude: List[float] = field(default_factory=lambda: [0.05, 0.30])
    boundary_n_ctrl: int = 12
    boundary_samples: int = 24

    # -- Internal extrema -----------------------------------------------------
    inner_extrema_count_range: List[int] = field(default_factory=lambda: [0, 2])
    inner_extrema_amplitude: List[float] = field(default_factory=lambda: [0.10, 0.35])
    inner_extrema_edge_exclusion_px: int = 2
    inner_extrema_min_distance_px: int = 5

    # -- Surface planarity / smoothness ---------------------------------------
    surface_smoothing: float = 0.15
    flatness_range: List[float] = field(default_factory=lambda: [0.4, 0.8])

    # -- Correlated roughness (per grain) -------------------------------------
    roughness_enabled: bool = True
    roughness_rms: List[float] = field(default_factory=lambda: [0.04, 0.10])
    roughness_corr_length_px: List[float] = field(default_factory=lambda: [2.5, 6.0])
    roughness_edge_taper_px: int = 0

    # -- Directional roughness anisotropy (sampled once per image) ------------
    angular_correlation_probability: float = 0.50
    angular_correlation_angle_range_deg: List[float] = field(default_factory=lambda: [0.0, 180.0])
    roughness_anisotropy_ratio: List[float] = field(default_factory=lambda: [3.0, 8.0])

    # -- Edge rounding in Z ---------------------------------------------------
    edge_rounding_probability: float = 0.40
    edge_rounding_border_fraction: List[float] = field(default_factory=lambda: [0.65, 0.90])
    edge_rounding_width_px: List[float] = field(default_factory=lambda: [2.0, 6.0])
    edge_rounding_profile: str = "cosine"        # cosine | smoothstep

    # -- Tessellation only ----------------------------------------------------
    tessellation_relax: int = 1
    tessellation_side_penalty_strength: float = 0.15

    # -- overlap_interactive only ---------------------------------------------
    overlap_target: List[float] = field(default_factory=lambda: [0.05, 0.20])
    contact_preference: float = 1.0
    gap_preference: float = 0.60
    max_placement_trials: int = 50
    coverage_target: List[float] = field(default_factory=lambda: [0.90, 0.98])
    min_visible_fraction: float = 0.01

    # -- Stacking in Z (overlap_interactive) ----------------------------------
    support_percentile: float = 75.0
    stack_height_factor: float = 0.80

    # SimuScan name -> vendored-generator name. The generator speaks nm; we feed
    # it pixels with pixel_size_nm = 1 (lateral) and height units verbatim (Z,
    # which the generator never scales).
    _CORE_NAME = {
        "mean_grain_size_px":       "mean_grain_size_nm",
        "grain_size_sigma_px":      "grain_size_sigma_nm",
        "grain_size_range_px":      "grain_size_range_nm",
        "correlation_length_px":    "correlation_length_nm",
        "roughness_corr_length_px": "roughness_corr_length_nm",
        "edge_rounding_width_px":   "edge_rounding_width_nm",
        "z_mean":                   "z_mean_nm",
        "boundary_z_amplitude":     "boundary_z_amplitude_nm",
        "inner_extrema_amplitude":  "inner_extrema_amplitude_nm",
        "roughness_rms":            "roughness_rms_nm",
    }

    # Fields that configure SimuScan, not the generator.
    _SIMUSCAN_ONLY = (
        "mode", "z_mode", "z_norm_percentile",
        "min_grain_area_px", "max_instances", "mask_dilate_px",
        "min_visible_after_objects", "block_other_objects",
    )

    def core_overrides(self) -> Dict[str, Any]:
        """Return this config translated into domains_core.DEFAULT_CONFIG keys."""
        out: Dict[str, Any] = {"pixel_size_nm": 1.0}   # px in, px out
        for f in fields(self):
            if f.name in self._SIMUSCAN_ONLY or f.name.startswith("_"):
                continue
            v = getattr(self, f.name)
            out[self._CORE_NAME.get(f.name, f.name)] = tuple(v) if isinstance(v, list) else v
        return out


@dataclass
class ArtifactsConfig:
    # delta noise
    apply_delta: bool = True
    delta_probability: float = 0.5
    delta_intensity: float = 0.1
    delta_n: int = 850

    # gaussian noise
    apply_gaussian: bool = True
    gaussian_probability: float = 0.9
    gaussian_scale: float = 0.01

    # lines with loss of contact
    apply_loss_contact: bool = True
    loss_contact_probaility: float = 0.3
    loss_contact_n: int = 8
    loss_contact_max_offset: float = 2
    loss_contact_max_slope: float = 0.02

    # local adhesion
    apply_adhesion: bool = True
    local_adhesion_probability: float = 0.2
    local_adhesion_intensity: float = 0.01
    local_adhesion_pixel_probability: float = 0.1

    # sticky zones
    apply_sticky_zones: bool = True
    sticky_zones_probability: float = 0.2
    sticky_zones_n: int = 10
    sticky_zones_size_range: List[int] = field(default_factory=lambda: [10, 50])
    sticky_zones_intensity: float = 0.1

    # line noise
    apply_line_noise: bool = True
    line_noise_probability: float =  0.2
    line_noise_intesity: float =  0.01
    line_noise_line_probability: float = 0.01

    # periodic noise
    apply_periodic_noise: bool = True
    periodic_probability: float = 0.2
    periodic_freq: float = 100
    periodic_amplitud: float = 0.05
    periodic_angle_deg_range: List[float] = field(default_factory=lambda: [0.0, 90.0])

    # line shift
    apply_shift_lines_noise: bool = True
    shift_lines_probability: float = 0.25
    shift_lines_roe_probability: float = 1.0
    shift_lines_max: int = 2

    # flatten
    apply_flatten: bool = True
    flatten_perc: float = 40


# =========================
# 5) Main class
# =========================

@dataclass
class SyntheticParams:
    file_name: str = 'synthetic_params.json'
    dataset_path: str = './data/Dataset'
    training_split_ratios: List[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])
    image_size: int = 512
    n_images: int = 10
    sigma_smooth_inside_mask: float = 2
    # Write a per-image instance label map next to each saved image
    # (instances/<split>/<name>.png, uint16, 0 = background). COCO polygons are a
    # re-derivation of the masks; this is the exact pixel-level ground truth, and
    # it is the only place where individual grains survive for the semantic UNet
    # path. Each annotation carries the matching `instance_index`.
    save_instance_map: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    # substrate/tip/artifacts/pid parameters
    substratum: SubstratumConfig = field(default_factory=SubstratumConfig)
    tip: TipConfig = field(default_factory=TipConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    domains: DomainsConfig = field(default_factory=DomainsConfig)

    # objects
    objects: List[ObjectConfig] = field(default_factory=lambda: deepcopy(default_object_list(DEFAULT_YAML)))

    # low-quality saving
    save_low_quality_probability: float = 0.7
    simu_size_low_quality: List[int] = field(default_factory=lambda: [412, 256, 128])

    # For reference
    yaml_path: str = DEFAULT_YAML

    @classmethod
    def from_yaml(cls, yaml_path: str = DEFAULT_YAML) -> "SyntheticParams":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        cfg = data.get("config", {}) or {}
        sub = SubstratumConfig(**(data.get("substrate_conf") or {}))
        tip = TipConfig(**(data.get("tip_conf") or {}))
        art = ArtifactsConfig(**(data.get("artifacts_conf") or {}))
        pid = PIDConfig(**(data.get("pid_conf") or {}))
        dom = DomainsConfig(**(data.get("domains_conf") or {}))
        objs = default_object_list(yaml_path)

        return cls(
            file_name=cfg.get("file_name", _get_field_default(cls, "file_name")),
            dataset_path=cfg.get("dataset_path", _get_field_default(cls, "dataset_path")),
            training_split_ratios=cfg.get("training_split_ratios", _get_field_default(cls, "training_split_ratios")),
            image_size=cfg.get("image_size", _get_field_default(cls, "image_size")),
            n_images=cfg.get("n_images", _get_field_default(cls, "n_images")),
            sigma_smooth_inside_mask=cfg.get("sigma_smooth_inside_mask", _get_field_default(cls, "sigma_smooth_inside_mask")),
            save_instance_map=cfg.get("save_instance_map", _get_field_default(cls, "save_instance_map")),
            save_low_quality_probability=cfg.get("save_low_quality_probability", _get_field_default(cls, "save_low_quality_probability")),
            simu_size_low_quality=cfg.get("simu_size_low_quality", _get_field_default(cls, "simu_size_low_quality")),
            substratum=sub,
            tip=tip,
            artifacts=art,
            pid=pid,
            domains=dom,
            objects=objs,
            yaml_path=yaml_path,
        )

    def get_object_by_label(self, label: str) -> ObjectConfig:
        for obj in self.objects:
            if obj.label == label:
                return obj
        raise ValueError(f"No object with label was found='{label}'")

    def get_objects_by_enabled(self, enabled: bool = True) -> List[ObjectConfig]:
        return [obj for obj in self.objects if obj.enable == enabled]

    def enable_objects_by_label(self, label: str):
        encontrados = [obj for obj in self.objects if obj.label == label]
        if not encontrados:
            raise ValueError(f"No object found with label='{label}'")
        for obj in encontrados:
            obj.enable = True

    def show_params(self):
        def pretty_print(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    pretty_print(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")

        print("== SyntheticParams ==")
        data = asdict(self)
        for i, obj in enumerate(self.objects):
            if obj.mesh is not None:
                data["objects"][i]["mesh"] = f"Triangles: {len(obj.mesh.faces)}"
            else:
                data["objects"][i]["mesh"] = "None"
        pretty_print(data)
    
    def save_json(self, ruta: str = None):
        if ruta is None:
            ruta = os.path.join(self.dataset_path, self.file_name)
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        data = to_json_dict(self)
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_json(cls, ruta: str = None):
        if ruta is None:
            ruta = os.path.join(cls.dataset_path, cls.file_name)  # use cls to avoid self
        if os.path.exists(ruta):
            with open(ruta, 'r') as f:
                data = json.load(f)
            data["substratum"] = SubstratumConfig(**data["substratum"])
            data["tip"] = TipConfig(**data["tip"])
            data["artifacts"] = ArtifactsConfig(**data["artifacts"])
            if "pid" in data and isinstance(data["pid"], dict):
                data["pid"] = PIDConfig(**data["pid"])
            # Older synthetic_params.json files predate domains_conf
            data["domains"] = DomainsConfig(**data.get("domains", {}))
            data["objects"] = [ObjectConfig(**o) for o in data["objects"]]
            return cls(**data)
        else:
            print('file not found')
            return None

    def edit(self, grupo: str, clave: str, valor: Any):
        obj = getattr(self, grupo, None)
        if hasattr(obj, clave):
            setattr(obj, clave, valor)
        else:
            self.metadata[f"{grupo}.{clave}"] = valor

    def get_class_info(self) -> List[Tuple[str, str]]:
        seen: Set[Tuple[str, str]] = set()
        for obj in self.objects:
            if obj.enable:
                seen.add((obj.label, obj.category))
        return sorted(seen)


if __name__ == '__main__':
    # Usage example reading EVERYTHING from the YAML:
    # Adjust the path if your file has a different name/location.
    yaml_path = DEFAULT_YAML
    params = SyntheticParams.from_yaml( yaml_path)
    print(f"Loaded {len(params.objects)} objects from YAML.")
    params.show_params()

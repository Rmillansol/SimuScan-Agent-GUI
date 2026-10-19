# Parameter Reference — SimuScanAgent

All parameters can be edited with `set_param(section, param, value)` or
`set_params(section, {param: value, ...})`. Objects are edited with
`set_object_param(label, param, value)`.

---

## Section `config`

| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `file_name` | str | `"synthetic_params.json"` | Name of the reproducibility JSON |
| `dataset_path` | str | `"../Dataset"` | Dataset output path |
| `training_split_ratios` | list | `[0.7, 0.2, 0.1]` | Train/Val/Test ratios |
| `image_size` | int | `512` | Image resolution (px, square) |
| `n_images` | int | `10` | Number of images to generate |
| `sigma_smooth_inside_mask` | float | `2.0` | Smoothing at the substrate-object junction |
| `save_low_quality_probability` | float | `0.0` | Probability of saving low-resolution replicas |
| `simu_size_low_quality` | list | `[412,256,128,64]` | Alternative resolutions (px) |

---

## Section `tip_conf`

| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `is_double` | bool | `true` | Enables the double-tip effect |
| `double_probability` | float | `1.0` | Double-tip probability per image |
| `radi_range` | list | `[2, 5]` | Tip radius range [min, max] |
| `eccentrisity_range` | list | `[0.9, 1.0]` | Tip eccentricity |
| `theta_range` | list | `[10, 30]` | Cone half-angle (degrees) |
| `distance_min_pr1` | float | `0.5` | Minimum distance between primary and secondary tips |
| `distance_max_pr1` | float | `1.5` | Maximum distance between the two tips |

---

## Section `substrate_conf`

### Plane and tilt
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `apply_tilt` | bool | `true` | Enables a tilted background plane |
| `tilt_probability` | float | `1.0` | Probability of applying tilt |
| `tilt_range` | float | `0.9` | Tilt range (± fraction) |
| `second_order_range` | float | `0.1` | Second-order curvature |

### Steps (stair)
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `apply_stair` | bool | `true` | Enables steps in the substrate |
| `stair_probability` | float | `0.99` | Step probability |
| `stair_kind` | str | `"spline"` | Type: `polygon` or `spline` |
| `stair_height` | list | `[0.3, 0.6]` | Step height [min, max] |

### Periodic pattern
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `apply_pattern` | bool | `false` | Enables a periodic pattern in the substrate |
| `pattern_probability` | float | `0.9` | Pattern probability |
| `pattern_type` | str | `"pillars"` | Type: `pillars`, `holes`, `grating` |
| `pattern_shape` | str | `"square"` | Shape: `circle`, `square`, `triangle`, `hexagon` |
| `pattern_period_x_px_range` | list | `[120,121]` | Period in X (px) |
| `pattern_period_y_px_range` | list | `[120,121]` | Period in Y (px) |
| `pattern_duty_cycle` | list | `[0.2, 0.4]` | Duty cycle (0–1) |
| `pattern_height` | list | `[0.5, 0.9]` | Pattern height |

### Substrate dirt and noise
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `apply_shapes` | bool | `true` | Dirt in the form of random shapes |
| `shapes_n` | int | `4` | Number of dirt shapes |
| `apply_dirt` | bool | `true` | Dirt made of random particles |
| `dirt_n` | int | `10` | Number of dirt particles |
| `dirt_delta_n` | int | `250` | Number of delta spikes in the substrate |
| `dirt_delta_height` | float | `0.3` | Maximum height of each delta (absolute, range 0–1 nm) |
| `dirt_delta_elongation_max` | int | `0` | Extra pixels in X per spike (0 = no elongation) |
| `dirt_delta_elongation_decay` | float | `0.5` | Height decay per elongated pixel (0–1) |
| `dirt_delta_elongation_prob` | float | `1.0` | Probability that each spike is elongated (0–1) |
| `apply_perlin` | bool | `true` | Correlated Perlin texture |
| `perlin_amplitud` | float | `0.4` | Perlin texture amplitude |
| `apply_gaussian` | bool | `true` | Gaussian noise in the substrate |
| `gaussian_amplitud` | float | `0.1` | Gaussian noise amplitude |

---

## Section `artifacts_conf`

Artifacts applied after tip convolution and PID. Delta amplitudes are relative to the image range.

| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `apply_delta` | bool | `true` | Delta noise (random spikes) |
| `delta_probability` | float | `0.5` | Probability of applying delta noise |
| `delta_intensity` | float | `0.9` | Maximum intensity = `U(0, delta_intensity) × image_range` |
| `delta_n` | int | `850` | Number of affected pixels |
| `apply_gaussian` | bool | `true` | Additive Gaussian noise |
| `gaussian_probability` | float | `0.9` | Gaussian noise probability |
| `gaussian_scale` | float | `0.02` | Gaussian noise amplitude |
| `apply_loss_contact` | bool | `true` | Loss-of-contact lines |
| `loss_contact_probaility` | float | `0.5` | Loss-of-contact probability |
| `loss_contact_n` | int | `20` | Number of affected lines |
| `apply_adhesion` | bool | `true` | Local tip adhesion |
| `local_adhesion_probability` | float | `0.4` | Adhesion probability |
| `apply_sticky_zones` | bool | `true` | Sticky zones on the surface |
| `apply_line_noise` | bool | `true` | Per-scan-line noise |
| `apply_periodic_noise` | bool | `true` | Periodic noise (interference) |
| `periodic_freq` | int | `100` | Periodic noise frequency |
| `periodic_amplitud` | float | `0.05` | Periodic noise amplitude |
| `apply_shift_lines_noise` | bool | `true` | Random line shifting |
| `apply_flatten` | bool | `true` | Final image flattening |
| `flatten_perc` | float | `40` | Percentage of lines used for flattening |

---

## Section `pid_conf`

Simulation of the feedback PID controller of AFM tapping mode.
Applied after tip convolution and before artifacts.

### Physical parameters (fixed for the whole dataset)
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `apply_pid` | bool | `true` | Enables PID simulation |
| `pid_probability` | float | `0.7` | Probability of applying PID per image |
| `scan_direction` | str | `"forward"` | Scan direction: `forward` or `backward` |
| `A0` | float | `1.0` | Free oscillation amplitude (a.u.) |
| `beta` | float | `1.0` | Sensitivity of the amplitude to the tip-sample distance |
| `z0` | float | `2.0` | Characteristic decay distance (a.u.) |
| `ASP` | float | `0.5` | Setpoint amplitude (fraction of A0) |

### Control parameters (range [min, max], random per image)
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `Kp` | list | `[0.3, 0.8]` | Proportional gain |
| `Ki` | list | `[0.05, 0.3]` | Integral gain |
| `Kd` | list | `[0.01, 0.1]` | Derivative gain |
| `dt` | list | `[0.5, 1.5]` | Time step per pixel (a.u.) |
| `max_speed` | list | `[0.5, 2.0]` | Maximum Z actuator speed |
| `stickiness_prob` | list | `[0.0, 0.05]` | Probability that the tip gets stuck |
| `jump_prob` | list | `[0.0, 0.02]` | Probability of a sudden position jump |
| `jump_magnitude` | list | `[0.5, 3.0]` | Jump magnitude (a.u.) |

> To see oscillations/peaks on the way up: increase `Kp`, decrease `Kd`, decrease `max_speed`.
> Use `tool_pid_tuner()` for interactive tuning.

---

## Section `objects` — per-object fields

| Field | Tipo | Description |
|---|---|---|
| `label` | str | Class name (used in COCO and YOLO) |
| `shape` | str | Reference shape (must exist in `shapes`) |
| `category` | str | Informational category |
| `enable` | bool | Includes/excludes this class from generation |
| `n_objects` | int | Number of objects per image |
| `probability` | float | Probability of inserting this class |
| `max_overlap` | float | Maximum allowed overlap between objects |
| `size_shape` | list | Reference size [width, height] in px |
| `height` | list | Height range [min, max] |
| `orden_z` | list | Range of the object's Z profile |
| `is_fill` | bool | Filled object (true) or outline only (false) |
| `variation_border_pct` | float | Stochastic border variation (fraction) |
| `scale_pct` | float | Symmetric scale variation |
| `rotation_range_deg` | list | Rotation range [min, max] in degrees |
| `apply_bending` | bool | Enables bending-type deformation |
| `bending_amplitud_pct` | float | Bending amplitude |
| `apply_perlin` | bool | Correlated texture on the object |
| `perlin_amplitud_pct` | float | Perlin texture amplitude on the object |
| `apply_ellipses_texture` | bool | Texture with embedded ellipsoids |
| `apply_colony` | bool | Allows the object to form colonies |
| `colony_probability` | float | Colony probability per image |
| `stl_files` | list | STL files in `data/STL_files/` |
| `stl_weights` | list | Weight of each STL (must sum to 1.0) |

---

## Exploration Configuration (`automation_default.yaml`)

### Microscope and output
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `gallery_dir` | str | (Nanosurf path) | Folder where Nanosurf saves the .gwy files |
| `gwy_base_name` | str | `"scan"` | Prefix of the .gwy files |
| `scan_direction` | str | `"forward"` | Scan direction to read: `forward`\|`backward`. Selects the single topography channel (`Forward - Topography` / `Backward - Topography`); must match the dataset's `pid_conf.scan_direction` |
| `origin_x_um` | float | `-300.0` | Initial tip X position (µm) |
| `origin_y_um` | float | `-300.0` | Initial tip Y position (µm) |

### Scanning
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `scan_size_um` | float | `50.0` | Scan frame size (µm) |
| `scan_rate_hz` | float | `100.0` | Line rate (Hz) |
| `scan_resolution_px` | int | `256` | Pixels per side |
| `strategy` | str | `"raster"` | `raster` or `probabilistic` |

### Grid
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `grid_n_x` | int | `0` | Grid columns (0 = auto from area) |
| `grid_n_y` | int | `0` | Grid rows (0 = auto from area) |
| `area_to_cover_um` | float | `10000.0` | Total area to cover (µm²) |
| `overlap_fraction` | float | `0.1` | Overlap between frames (negative = gap) |

### Probabilistic strategy (Beta Active Search)
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `prob_alpha_init` | float | `1.0` | Initial alpha (prior belief of presence) |
| `prob_beta_init` | float | `1.0` | Initial beta (prior belief of absence) |
| `prob_spatial_sigma_um` | float | `100.0` | Gaussian evidence propagation radius (µm) |
| `prob_kappa_uncertainty` | float | `1.0` | Weight of the uncertainty bonus in the utility |
| `prob_lambda_move` | float | `0.01` | Distance penalty in the utility |
| `prob_epsilon_unknown` | float | `0.05` | Probability of exploring the most uncertain cell (ε-greedy) |
| `prob_negative_update_scale` | float | `0.5` | Weight of negative evidence (no detection) |
| `prob_initial_random_n` | int | `3` | Initial random scans near the center |
| `prob_initial_center_radius_um` | float | `200.0` | Radius for those initial scans (µm) |
| `isolation_max_bbox_overlap` | float | `0.3` | Maximum bbox overlap for counting an isolated object |
| `isolation_min_distance_um` | float | `0.0` | Minimum center-to-center distance (µm) |

### YOLO and target classes
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `yolo_enabled` | bool | `false` | Enables YOLO inference |
| `yolo_model_path` | str | `""` | Path to best.pt |
| `yolo_conf_threshold` | float | `0.7` | YOLO confidence threshold |
| `yolo_overlap_threshold` | float | `0.3` | NMS overlap threshold |
| `border_filter_enabled` | bool | `true` | Detections touching the border: do not count toward target/library/zoom/crops, but if >~50% visible they feed the probability map (avoids an "empty" bias at the borders). `false` = they count for everything |
| `target_classes` | dict | `{}` | `{clase: {n: 10, enabled: true}}` |

### Zoom
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `zoom_enabled` | bool | `false` | Enables zoom on detected objects |
| `zoom_resolution_px` | int | `512` | Zoom resolution |
| `zoom_max_rate_hz` | float | `2.0` | Maximum zoom rate |
| `zoom_max_size_um` | float | `25.0` | Maximum zoom frame size (µm) |
| `zoom_bbox_factors` | list | see YAML | Zoom factors by bbox size |

### Stopping and others
| Parameter | Tipo | Default | Description |
|---|---|---|---|
| `max_large_frames` | int | `200` | Maximum number of exploration frames |
| `triplet_model_path` | str | `""` | Path to the triplet encoder (optional) |
| `flatten_perc` | float | `40.0` | GWY image flattening percentage |
| `gwy_poll_interval_s` | float | `2.0` | GWY polling interval (s) |
| `stage_auto_range` | bool | `false` | Try to read the stage range through the API (not exposed in Nanosurf Studio → use manual bounds) |
| `stage_x_min_um` / `stage_x_max_um` | float | `-10600` / `10600` | Stage travel limit in X (µm), same coordinate system as `origin_x_um` |
| `stage_y_min_um` / `stage_y_max_um` | float | `-10600` / `10600` | Stage travel limit in Y (µm) |
| `stage_edge_margin_um` | float | `500.0` | Safety margin from the hard edge (the positioner is slow/does not reach at the edge) |

> The `.gwy` wait timeout is **not a parameter**: it is derived automatically from `scan_resolution_px` and `scan_rate_hz` (~2×lines/rate + 15% margin), so it scales with each scan.

> **Stage limits.** When planning, the grid is **clipped** to the reachable stage area (`stage_*_um` minus `stage_edge_margin_um`); cells outside are discarded (warning in the log) so that positions where the positioner gets stuck and the `.gwy` times out are not queued. Leave the bounds at 0 (max ≤ min) to disable clipping. Bounds use the same stage coordinates as `origin_x_um/origin_y_um` (read them in Nanosurf: this instrument is symmetric ±10.6 mm centered on 0).

# SimuScanAgent — User Manual

SimuScanAgent is a system for generating synthetic AFM datasets, training detection/segmentation models and autonomously exploring the microscope, controlled through a natural-language agent.

---

## General Workflow

```
1. Define reference objects (images, STL, Python generators)
        ↓
2. Configure and generate synthetic dataset
        ↓
3. Train models (YOLO / UNet / Triplet)
        ↓
4. Configure and launch autonomous AFM exploration
        ↓
5. Analyze results (PNG folders or experimental GWY)
```

---

## 1. Reference Objects

Before generating data, the agent needs to know what the objects of interest look like.

### Reference image (gpt-image-1)
The agent can generate a reference image from a text description:
> "Generate a reference image of spherical nanoparticles in AFM"

Tool: `tool_generate_ref_image` → saves to `data/ref_images/`
Tool: `tool_show_ref_images` → shows the saved references

**AFM height profile.** Images from diffusion models have unrealistic lighting and
shadows (the topography does not grow with distance). After generating them, a
**height profile based on distance to the edge** (`mask_to_height_map`) is applied
automatically: a dome that grows from the edge toward the center, modulated by the model's texture.
- On generation: `apply_height_profile=True` (default), `height_profile_texture=0.4`
  (0 = pure dome, 1 = keeps all the texture). Disable with `apply_height_profile=False`.
- On an existing image (without generating): `tool_apply_height_profile(image_path)` →
  writes `<name>_height.png` without overwriting the original.

### Geometric definition by segments and splines
Objects are defined in `default.yaml` → `shapes` section. Each shape has sections that describe its outline as a combination of:
- `type: line` — straight segment between two points
- `type: spline` — smooth curve through control points
- `type: function` — call to a predefined function (`regular_polygon`, `ellipse_spline`)

Points are in normalized coordinates [−1, 1]. Available examples: `triangle`, `square`, `hexagon`, `ellipse`, `oval`, `bacillus`, `cocos`, `linearDNA`, `circularDNA`, `star5`, `STL`.

### STL files
3D objects imported from `data/STL_files/`. They are assigned to an object through the `stl_files` and `stl_weights` fields.

### Custom Python generators
The agent can propose, review and save Python functions to generate arbitrary shapes:
1. `propose_generator(module_name, code)` — shows the code to the user for review
2. `save_generator(module_name)` — saves to `data/user_generators/`
3. `discard_generator(module_name)` — discards
4. `run_generator(module_name, params)` — runs and previews

Each generator must implement `generate(params) -> (image, mask)` where both are numpy float32 arrays of the same size.

---

## 2. Synthetic Dataset Generation

### Configuration
```
load_config()                          # loads default.yaml into memory
load_config("mi_config.yaml")          # loads an existing user config
set_param("config", "n_images", 500)   # edits one parameter
set_params("substrate_conf", {...})    # edits several at once
set_object_param("hexagon", "enable", true)
save_config("mi_experimento.yaml")     # saves (never overwrites default.yaml)
```

**State model (important):** there is **a single** configuration in memory.
All edits (`set_param`, `set_object_param`, etc.) modify it directly.
`save_config("name.yaml")` dumps that in-memory config to that file — it is the only
"real" file. `temp.yaml` is **not** a config that you manage: `tool_generate_temp_dataset`
automatically writes a temporary snapshot of the in-memory config just for the preview.
The final dataset is never loaded, edited, saved or generated from `temp.yaml`.

### YAML sections
- `config` — global parameters (n_images, image_size, etc.)
- `tip_conf` — AFM tip geometry (radius, double tip, angle)
- `substrate_conf` — substrate (tilt, steps, dirt, delta, Perlin, Gaussian)
- `artifacts_conf` — image artifacts (relative delta, Gaussian, loss_contact, adhesion, lines, periodic, flattening)
- `pid_conf` — PID feedback simulation in tapping mode
- `shapes` — geometric definition of outlines (read-only from the agent)
- `objects` — list of classes with their properties

### Synthesis pipeline (per image)
1. Generates the substrate (tilt + steps + dirt + noise)
2. Places objects according to their properties
3. Applies tip convolution → **saves clean image**
4. Applies PID simulation (if `apply_pid: true`) → **saves PID image**
5. Applies artifacts → **saves final image**

### Generation
```
tool_generate_temp_dataset()                        # quick preview (~10 imgs)
tool_generate_final_dataset("mi_experimento.yaml")  # full dataset → COCO
tool_convert_coco_to_yolo("20260101_COCO")          # converts to YOLO format
tool_visualize_dataset("20260101_COCO")             # visualizes annotations
```

### Interactive PID tuner
```
tool_pid_tuner()   # opens a window with a synthetic pillar + real-time controls
```

---

## 3. Model Training

### YOLO (detection + segmentation)
```
tool_train_yolo(
    coco_folder_name="20260101_COCO",
    task="segment",          # "detect" or "segment"
    epochs=100,
    imgsz=512,
    batch=16,
    model_name="yolo11n-seg.pt"
)
```
Output: `data/out_yolo/<run>/weights/best.pt`
Saves: CSV metrics, training curves, confusion matrix.

### UNet (alternative segmentation)
```
tool_train_unet(
    coco_folder_name="20260101_COCO",
    epochs=50,
    batch_size=8,
    lr=1e-4
)
```
Output: `data/out_unet/<run>/`

### Triplet Encoder (similarity embeddings)
Requires a trained YOLO. Learns a metric space where similar objects are close together.
```
tool_train_triplet(
    coco_folder_name="20260101_COCO",
    yolo_model_run_name=None,    # None = uses the most recent
    latent_dim=32,
    crop_size=64,
    triplet_epochs=40,
    triplet_margin=0.5
)
```
Output: `data/out_triplet/<run>/triplet_encoder.pt` + metrics + UMAP + `library.json`.

Each training run also builds a **library** (`library.json`) from the synthetic
crops (train+val+test splits, with ground-truth labels). This makes it possible to
explore the training embeddings with the same interactive explorer used for
experimental libraries:
```
tool_show_triplet_library_explorer("time_20260101_120000")   # one training run
tool_show_triplet_library_explorer()                          # the most recent (any type)
```

---

## 4. Autonomous AFM Exploration

### Configuration
```
load_automation_config()                     # loads automation_default.yaml
load_automation_config("sesion1.yaml")       # loads a user config
set_automation_param("scan_size_um", 25.0)
set_automation_param("yolo_enabled", true)
set_automation_param("yolo_model_path", "data/out_yolo/run/weights/best.pt")
save_automation_config("sesion1.yaml")
```

### Operating modes
| yolo_enabled | zoom_enabled | Description |
|---|---|---|
| false | false | Scan only, no analysis |
| true | false | Scan + YOLO detection |
| true | true | Scan + detection + zoom on objects |

**Triplet library without zoom.** If `triplet_model_path` is set in the
`true / false` mode (detection without zoom), the exploration still builds the
object library: each detection is **cropped directly from the large frame**
(with a small margin) and saved as a `_flat.png` / `_raw.png` pair in
`library/crops/<class>/`, just as objects are cropped when training the
triplets from YOLO detections on synthetic images — only the source is the
exploration frame instead of a re-scan. When finished, the `TripletAnalyzer`
is run on those crops. Note: the crops will have the resolution of the large
frame; for higher-quality crops, enable zoom.

**Border filter (`border_filter_enabled`).** Detections whose bbox touches a
frame edge are cut off, so their shape is unreliable: they **do not count**
toward the target (`target_classes`), nor the library, nor do they trigger zoom/crops.
But if YOLO detects them and they are **>~50% visible** (judged by the aspect of the
bbox relative to the cut edge), they **do feed the probability map** — so the
map is not biased toward "empty" at the borders. Thin strips stuck to the edge are
discarded. With `border_filter_enabled: false`, border detections count
for everything.

### Strategies
- `raster` — deterministic grid in row-major order (row by row, each row left to right)
- `probabilistic` — adaptive Beta-distribution Active Search

### Launch
```
automation_connect_microscope()   # connects to Nanosurf Studio
tool_run_exploration()            # starts the exploration
automation_get_status()           # queries real-time status
automation_stop()                 # stops the exploration
```

The exploration ends when all classes in `target_classes` reach their `n`, or when `max_large_frames` is reached (only in the probabilistic strategy; in raster the full grid is always scanned). If `max_large_frames` is 0 or negative, **there is no frame limit**: the search continues until the `target_classes` criterion is met or the grid is exhausted — in probabilistic when the queue of cells to visit empties (the whole grid explored), and in raster when the grid is completed.

The `.gwy` file wait timeout is **not a parameter**: it is computed automatically from `scan_resolution_px` and `scan_rate_hz` (~2×lines/rate + 15% margin), so it scales with each scan and does not fall short on high-resolution scans.

**Stage limits.** The stage positioner gets stuck / is very slow at the edge of its travel, and frames placed there time out waiting for the `.gwy`. So, when planning, the grid is **clipped** to the reachable stage area (`stage_x_min_um`/`stage_x_max_um`/`stage_y_min_um`/`stage_y_max_um`, minus `stage_edge_margin_um`); cells that fall outside are discarded and a warning is logged, and the exploration continues. The bounds use the same stage coordinates as `origin_x_um/origin_y_um` (on this instrument, symmetric **±10.6 mm** centered on 0). Nanosurf Studio does not expose the stage range to Python, so they are set by hand (`stage_auto_range: false`); leave them at 0 to disable clipping. If the `origin` falls outside the range, the exploration aborts with a warning.

---

## 5. Results Analysis

### Prediction on folders
```
tool_predict_folder(image_folder, yolo_run_name)         # PNGs with YOLO
tool_predict_folder_unet(image_folder, unet_run_name)    # PNGs with UNet
tool_predict_folder_gwy(gwy_folder)                      # experimental GWY
```
GWY files are loaded, flattened (line-by-line flatten) and processed the same way as PNGs.

### Triplet embedding library
```
tool_build_triplet_library(
    image_folder="/ruta/a/imagenes",   # accepts PNG/JPG or GWY (auto-detected)
    pixel_size_nm=1.0,
    n_clusters=None,                   # None = auto silhouette
    yolo_conf=0.5
)
tool_show_triplet_library_explorer()   # interactive UMAP explorer
```
The explorer works with both experimental libraries (`out_triplet_lib/`) and
training runs (`out_triplet/`) — both contain a compatible `library.json`.

Shows 3 UMAP views:
- **Combined** — embedding (32-d) + shape features
- **Shape features only** — aspect ratio, area, perimeter, compactness, etc.
- **Embedding only** — space learned by the triplet network

### Additional visualizations
```
tool_show_crop_grid()           # grid of crops per class
tool_show_embedding_umap()      # UMAP of the triplet training
tool_get_model_info()           # info on the active YOLO model
tool_get_triplet_model_info()   # info on the active triplet encoder
```

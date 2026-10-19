# Usage Examples — SimuScanAgent

---

## Example 1 — Quick preview dataset

```
User: Generate a quick preview with the default configuration.

Agent:
  load_config()                          # loads default.yaml into memory
  tool_generate_temp_dataset()           # generates ~10 images in temp_COCO/
  tool_visualize_dataset("temp_COCO")    # shows annotations
```

---

## Example 2 — Hexagon dataset with PID enabled

```
User: Create a dataset of 300 images with only hexagons,
      with PID enabled and quite a lot of delta noise.

Agent:
  load_config()
  set_param("config", "n_images", 300)
  set_object_param("hexagon", "enable", true)
  # disable the rest of the classes
  set_object_param("triangle", "enable", false)
  set_object_param("bacillus",  "enable", false)
  ...
  set_params("pid_conf", {
      "apply_pid": true,
      "pid_probability": 0.8,
      "Kp": [0.5, 1.5],
      "Kd": [0.0, 0.05]
  })
  set_params("artifacts_conf", {
      "apply_delta": true,
      "delta_intensity": 0.8,
      "delta_n": 1200
  })
  save_config("hexagons_pid.yaml")
  tool_generate_final_dataset("hexagons_pid.yaml")
  tool_convert_coco_to_yolo("20260101_COCO")
```

---

## Example 3 — Full YOLO and Triplet training

```
User: Train YOLO and then the triplet encoder.

Agent:
  # 1. YOLO
  tool_train_yolo(
      coco_folder_name="20260101_COCO",
      task="segment",
      epochs=100,
      imgsz=512,
      batch=16,
      model_name="yolo11n-seg.pt"
  )

  # 2. Triplet (uses the most recent YOLO automatically)
  tool_train_triplet(
      coco_folder_name="20260101_COCO",
      latent_dim=32,
      crop_size=64,
      triplet_epochs=50,
      triplet_margin=0.5
  )

  # 3. Visualize embeddings
  tool_show_embedding_umap()
  tool_show_crop_grid()
```

---

## Example 4 — Basic raster exploration

```
User: Launch a 100×100 µm raster exploration
      with YOLO detection of hexagons, 10 per class.

Agent:
  load_automation_config()
  set_automation_param("scan_size_um", 50.0)
  set_automation_param("area_to_cover_um", 10000.0)
  set_automation_param("overlap_fraction", 0.1)
  set_automation_param("strategy", "raster")
  set_automation_param("yolo_enabled", true)
  set_automation_param("yolo_model_path",
      "data/out_yolo/20260101_seg/weights/best.pt")
  set_automation_param("target_classes",
      {"hexagon": {"n": 10, "enabled": true}})
  set_automation_param("max_large_frames", 100)
  save_automation_config("exploracion_hex.yaml")

  automation_connect_microscope()
  tool_run_exploration("exploracion_hex.yaml")
```

---

## Example 5 — Probabilistic exploration with zoom

```
User: Smart exploration looking for bacillus with zoom.

Agent:
  load_automation_config()
  set_automation_param("strategy", "probabilistic")
  set_automation_param("yolo_enabled", true)
  set_automation_param("zoom_enabled", true)
  set_automation_param("zoom_max_size_um", 20.0)
  set_automation_param("target_classes",
      {"bacillus": {"n": 15, "enabled": true}})
  set_automation_param("prob_kappa_uncertainty", 1.5)
  set_automation_param("prob_lambda_move", 0.005)
  save_automation_config("busqueda_bacillus.yaml")

  automation_connect_microscope()
  tool_run_exploration("busqueda_bacillus.yaml")
```

---

## Example 6 — Analysis of an experimental GWY folder

```
User: Analyze the GWY files from yesterday's session.

Agent:
  # YOLO on GWY (flattens automatically)
  tool_predict_folder_gwy(
      gwy_folder="C:/AFM_Data/sesion_20260622",
      conf_threshold=0.6
  )

  # Triplet library from GWY (auto-detects the type)
  tool_build_triplet_library(
      image_folder="C:/AFM_Data/sesion_20260622",
      pixel_size_nm=2.5,
      n_clusters=None,
      yolo_conf=0.5
  )

  # Interactive UMAP explorer
  tool_show_triplet_library_explorer()
```

---

## Example 7 — Interactive PID tuner

```
User: I want to see what happens with oscillations at the edges.

Agent:
  tool_pid_tuner()
  # → opens a window with a synthetic 128×128 pillar
  #   left panel: original image
  #   right panel: image with PID
  #   bottom plot: central profile (ideal vs PID)
  #   controls: Kp, Ki, Kd, dt, max_speed, A0, beta, z0, ASP, etc.
  #
  # For oscillations: increase Kp, decrease Kd, decrease max_speed
  # When you are happy → set_params("pid_conf", {...}) with those values
```

---

## Example 8 — Shape definition by segments and splines

Custom shape defined in `default.yaml` → `shapes` section.
Coordinates are normalized in [−1, 1]. The agent reads this section
but cannot edit it directly (it is read-only).

```yaml
shapes:
- name: capsula
  sections:
  - - type: line
      points:
        - [0.5, 0.5]
        - [-0.5, 0.5]
    - type: spline
      points:
        - [-0.5, 0.5]
        - [-1.0, 0.0]
        - [-0.5, -0.5]
    - type: line
      points:
        - [-0.5, -0.5]
        - [0.5, -0.5]
    - type: spline
      points:
        - [0.5, -0.5]
        - [1.0, 0.0]
        - [0.5, 0.5]
```

Using predefined functions (simpler):

```yaml
- name: hexagono_regular
  sections:
  - - type: function
      call: "regular_polygon(n_sides = 6)"

- name: elipse_suave
  sections:
  - - type: function
      call: "ellipse_spline(n_points = 16)"
```

Once the shape is defined, it is referenced from an object:
```yaml
objects:
- label: capsula
  shape: capsula
  enable: true
  n_objects: 5
  size_shape: [60, 25]
  ...
```

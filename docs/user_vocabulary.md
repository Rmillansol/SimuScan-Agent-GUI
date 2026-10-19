# User Vocabulary — Synonyms and Concepts

This file helps the agent map the user's natural language to the real
parameters and tools of the system. Users may write in English or Spanish:
each phrase below is given in English followed by its Spanish equivalent
(the Spanish terms are kept on purpose so they can still be matched).

---

## Objects and classes

| The user says | Refers to |
|---|---|
| "enable/disable a class" / "activar/desactivar una clase" | `set_object_param(label, "enable", true/false)` |
| "rename class / shorter name / change the label" / "renombrar clase / nombre más corto / cambiar el label" | `rename_object(old_label, new_label)` (NEVER `set_object_param(...,'label',...)`) |
| "more/fewer objects per image" / "más/menos objetos por imagen" | `set_object_param(label, "n_objects", N)` |
| "bigger/smaller objects" / "objetos más grandes/pequeños" | `set_object_param(label, "size_shape", [w, h])` |
| "taller/shorter objects" / "objetos más altos/bajos" | `set_object_param(label, "height", [min, max])` |
| "more shape variation" / "más variación de forma" | `set_object_param(label, "variation_border_pct", X)` |
| "let them rotate" / "que puedan rotar" | `set_object_param(label, "rotation_range_deg", [0, 360])` |
| "bent / curved" / "que se doblen / que estén curvados" | `set_object_param(label, "apply_bending", true)` |
| "form colonies / groups" / "que formen colonias / grupos" | `set_object_param(label, "apply_colony", true)` |
| "texture on the object surface" / "textura en la superficie del objeto" | `set_object_param(label, "apply_perlin", true)` |
| "overlap / do not overlap" / "que se solapen / no se solapen" | `set_object_param(label, "max_overlap", 0.0–1.0)` |

---

## Dataset and generation

| The user says | Refers to |
|---|---|
| "preview / quick test / few images" / "preview / prueba rápida / pocas imágenes" | `tool_generate_temp_dataset()` |
| "final dataset / full dataset" / "dataset final / dataset completo" | `tool_generate_final_dataset("config.yaml")` |
| "how many images" / "cuántas imágenes" | `set_param("config", "n_images", N)` |
| "resolution / image size" / "resolución / tamaño de imagen" | `set_param("config", "image_size", 512)` |
| "convert to YOLO / YOLO format" / "convertir a YOLO / formato YOLO" | `tool_convert_coco_to_yolo("folder_COCO")` |
| "see the annotations / see the dataset" / "ver las anotaciones / ver el dataset" | `tool_visualize_dataset("folder_COCO")` |
| "load config / open config" / "cargar configuración / abrir config" | `load_config("archivo.yaml")` |
| "save config" / "guardar configuración / salvar config" | `save_config("archivo.yaml")` |
| "default configuration" / "configuración por defecto" | `load_config()` (no argument → copy of the default) |
| "config from the previous session" / "config de la sesión anterior" | `load_config("archivo_existente.yaml")` |

---

## Substrate and background noise

| The user says | Refers to |
|---|---|
| "substrate tilt / tilted plane" / "inclinación del sustrato / tilt / plano inclinado" | `substrate_conf.apply_tilt`, `tilt_range` |
| "steps / terraces" / "escalones / steps / terrazas" | `substrate_conf.apply_stair`, `stair_height` |
| "dirt / background particles" / "suciedad / dirt / partículas de fondo" | `substrate_conf.apply_dirt`, `dirt_n`, `dirt_height` |
| "substrate spikes / substrate delta" / "spikes del sustrato / delta del sustrato" | `substrate_conf.dirt_delta_n`, `dirt_delta_height` |
| "elongated spikes / particle dragging" / "spikes alargados / arrastre de partículas" | `dirt_delta_elongation_max`, `dirt_delta_elongation_prob` |
| "substrate texture / correlated roughness" / "textura del sustrato / rugosidad correlacionada" | `substrate_conf.apply_perlin`, `perlin_amplitud` |
| "substrate white noise" / "ruido blanco del sustrato" | `substrate_conf.apply_gaussian`, `gaussian_amplitud` |

---

## Image artifacts

| The user says | Refers to |
|---|---|
| "spikes / bright dots / isolated peaks" / "spikes / puntos brillantes / picos aislados" | `artifacts_conf.apply_delta`, `delta_intensity`, `delta_n` |
| "noise / grainy" / "ruido / granulado / noise" | `artifacts_conf.apply_gaussian`, `gaussian_scale` |
| "horizontal lines / scan lines" / "líneas horizontales / líneas de escaneo" | `artifacts_conf.apply_loss_contact` or `apply_line_noise` |
| "loss of contact / line jumps" / "pérdida de contacto / saltos de línea" | `artifacts_conf.apply_loss_contact` |
| "periodic stripes / interference" / "rayas periódicas / interferencias" | `artifacts_conf.apply_periodic_noise`, `periodic_freq` |
| "shifted lines / sudden jumps" / "líneas desplazadas / saltos bruscos" | `artifacts_conf.apply_shift_lines_noise` |
| "the tip sticks / adhesion" / "la punta se pega / adhesión" | `artifacts_conf.apply_adhesion` or `apply_sticky_zones` |
| "flattening / flatten / plane correction" / "aplanado / flatten / corrección de plano" | `artifacts_conf.apply_flatten`, `flatten_perc` |
| "wide tip / wide objects / broadening" / "punta ancha / objetos anchos / broadening" | `tip_conf.radi_range` (higher radius) |
| "double tip / ghost" / "doble punta / fantasma" | `tip_conf.is_double`, `double_probability` |

---

## PID simulation

| The user says | Refers to |
|---|---|
| "feedback effects / PID / controller" / "efectos del feedback / PID / controlador" | `pid_conf` |
| "oscillations on the way up / ringing / overshoot" / "oscilaciones al subir / ringing / overshoot" | high `Kp` + low `Kd` + low `max_speed` |
| "blurry image at the edges / tracking error" / "imagen borrosa en los bordes / tracking error" | low `max_speed`, moderate `Kp` |
| "the tip does not follow well / lag" / "la punta no sigue bien / lag" | low `max_speed` |
| "smoothed image / no oscillations" / "imagen suavizada / sin oscilaciones" | low `Kp` + high `Kd` |
| "tune the PID visually" / "ajustar el PID visualmente" | `tool_pid_tuner()` |
| "scan direction" / "dirección de escaneo" | `pid_conf.scan_direction` (dataset) and `scan_direction` (exploration) — `"forward"` or `"backward"`; they must match so that the channel read (`Forward`/`Backward - Topography`) corresponds to the direction used to generate the dataset |

---

## Tip

| The user says | Refers to |
|---|---|
| "sharp tip / thin tip" / "punta fina / tip fino" | `tip_conf.radi_range` (small values, e.g. [1, 2]) |
| "thick tip / wide tip" / "punta gruesa / tip ancho" | `tip_conf.radi_range` (large values, e.g. [5, 10]) |
| "tip angle / cone" / "ángulo del tip / cono" | `tip_conf.theta_range` |
| "double tip / second tip" / "doble tip / segundo tip" | `tip_conf.is_double: true`, `double_probability` |

---

## Training

| The user says | Refers to |
|---|---|
| "train YOLO / detector / segmenter" / "entrenar YOLO / detector / segmentador" | `tool_train_yolo(...)` |
| "train UNet" / "entrenar UNet" | `tool_train_unet(...)` |
| "train triplet / embeddings / similarity" / "entrenar triplet / embeddings / similitud" | `tool_train_triplet(...)` |
| "detection" / "detección" | `task="detect"` in `tool_train_yolo` |
| "segmentation / masks" / "segmentación / máscaras" | `task="segment"` in `tool_train_yolo` |
| "epochs" / "épocas / epochs" | `epochs` parameter |
| "small/large/fast model" / "modelo pequeño/grande/rápido" | `model_name`: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` |
| "see the training results / curves" / "ver los resultados del entrenamiento / curvas" | the agent shows them automatically when finished |

---

## Autonomous exploration

| The user says | Refers to |
|---|---|
| "launch exploration / start scanning" / "lanzar exploración / empezar a escanear" | `tool_run_exploration()` |
| "connect to the microscope" / "conectar al microscopio" | `automation_connect_microscope()` |
| "stop exploration" / "parar / detener exploración" | `automation_stop()` |
| "exploration status" / "estado de la exploración" | `automation_get_status()` |
| "grid scan / raster" / "escaneo en grid / raster" | `strategy: "raster"` |
| "smart / adaptive / probabilistic search" / "búsqueda inteligente / adaptativa / probabilística" | `strategy: "probabilistic"` |
| "area to cover" / "área a cubrir" | `area_to_cover_um` (µm²) |
| "scan / frame size" / "tamaño del scan / frame" | `scan_size_um` (µm) |
| "overlap between frames" / "solapamiento entre frames" | `overlap_fraction` (fraction, negative = gap) |
| "how many of each class" / "cuántos de cada clase" | `target_classes: {class: {n: X, enabled: true}}` |
| "zoom on the objects" / "hacer zoom sobre los objetos" | `zoom_enabled: true` |
| "maximum zoom size" / "tamaño máximo del zoom" | `zoom_max_size_um` (µm) |
| "maximum number of scans" / "cuántos scans máximo" | `max_large_frames` |
| "YOLO confidence threshold" / "umbral de confianza YOLO" | `yolo_conf_threshold` |
| "isolate objects / isolated objects" / "aislar objetos / objetos aislados" | `isolation_min_distance_um`, `isolation_max_bbox_overlap` |

---

## Results analysis

| The user says | Refers to |
|---|---|
| "analyze folder / predict folder" / "analizar carpeta / predecir carpeta" | `tool_predict_folder(folder)` |
| "analyze GWY / experimental files" / "analizar archivos GWY / experimentales" | `tool_predict_folder_gwy(folder)` |
| "prediction with UNet" / "predicción con UNet" | `tool_predict_folder_unet(folder)` |
| "embedding library / build library" / "librería de embeddings / construir librería" | `tool_build_triplet_library(folder)` |
| "UMAP explorer / visualize embeddings" / "explorador UMAP / visualizar embeddings" | `tool_show_triplet_library_explorer()` |
| "see object crops" / "ver recortes de objetos / crops" | `tool_show_crop_grid()` |
| "clustering / group objects" / "clustering / agrupar objetos" | `n_clusters` in `tool_build_triplet_library` |
| "rare objects / anomalies / novelty" / "objetos raros / anomalías / novelty" | novelty score computed automatically in the library |
| "physical pixel size / calibration" / "tamaño físico del píxel / calibración" | `pixel_size_nm` in `tool_build_triplet_library` |

---

## Custom generators

| The user says | Refers to |
|---|---|
| "create a custom function / shape" / "crear una función / shape personalizada" | `propose_generator(name, code)` |
| "save the generator" / "guardar el generador" | `save_generator(name)` |
| "discard / reject the generator" / "descartar / rechazar el generador" | `discard_generator(name)` |
| "test / run the generator" / "probar / ejecutar el generador" | `run_generator(name, params)` |
| "reference image / example photo" / "imagen de referencia / foto de ejemplo" | `tool_generate_ref_image(prompt)` |
| "see reference images" / "ver imágenes de referencia" | `tool_show_ref_images()` |
| "height profile / dome topography / let it grow from the edge" / "perfil de altura / topografía de cúpula / que crezca desde el borde" | distance profile: `apply_height_profile`, `height_profile_*` |
| "remove the shadow effect / make it topographic" / "quítale el efecto de sombras / hazla topográfica" | `tool_apply_height_profile(image_path)` (existing image) |
| "leave it as the model generates it / no height profile" / "déjala como la genera el modelo / sin perfil de altura" | `apply_height_profile=false` in `tool_generate_ref_image` |
| "apply the profile to this image of mine / this file" / "aplica el perfil a esta imagen mía / a este archivo" | `tool_apply_height_profile(image_path)` |

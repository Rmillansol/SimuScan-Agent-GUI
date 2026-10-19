# Grain / domain fields (`shape: domains`)

SimuScan can generate images of **topographic grains or domains** — a
polycrystalline mosaic, or a film of stacked islands — with one instance mask
per visible grain, and then push that scene through the whole AFM simulation
chain (tip convolution, PID feedback, scanner noise, line artifacts, drift,
flattening) exactly like any other object type.

The ideal-topography engine is vendored in `src/simuscan/domains_core.py` from
an external reference implementation. It produces **only** the ground truth;
every AFM artifact is added afterwards by the SimuScan pipeline.

---

## 1. Quick start

```
load_config("domains_grains.yaml")
tool_preview_image()
tool_generate_final_dataset("domains_grains.yaml")
```

`domains_grains.yaml` is a ready-to-run preset: only the `grain` object is
enabled, grains are ~60 px across (~70 per 512 px image), and the substrate and
artifact amplitudes are sized against the grain relief (§7) so the grains stay
readable. Every substrate feature and every artifact is still switched ON.

To add grains to an existing config instead:

```
set_object_param("grain", "enable", True)
set_param("domains_conf", "mean_grain_size_px", 40.0)
```

---

## 2. The two geometry modes

| `domains_conf.mode` | What it produces |
|---|---|
| `tessellation` | A complete touching mosaic. No gaps, grains share boundaries, every pixel belongs to a grain. Side count is emergent but biased toward `n_sides_range`; shared boundaries can be curved. Use for polycrystalline films and domain maps. |
| `overlap_interactive` | Grains deposited one by one, like a growth/deposition process. They touch, overlap and leave gaps; later grains sit on top of earlier ones and the mask keeps only the **top visible** grain. Use for island films and partial coverage. |

A single object can override the section-wide mode with `domains_mode:`, so you
can keep several domains objects defined in the YAML and enable whichever one you
want. Only one may be enabled at a time — see §5.

---

## 3. Units — pixels and height units, like the rest of SimuScan

All parameters are in SimuScan's native units:

- **lateral sizes in pixels** (`*_px`), like `size_shape`, `perlin_scale_px`,
  `pattern_period_x_px_range`, `dirt_size`;
- **heights in relative height units** (~0–1), the same scale as an object's
  `height`, `dirt_height`, `pattern_height`, `stair_height`.

The vendored engine is written in nm internally. `DomainsConfig.core_overrides()`
translates once, by handing it the pixel values with `pixel_size_nm = 1`. That
substitution is exact — the engine uses `pixel_size_nm` only for lateral nm→px
conversions and never applies it to Z.

### How many grains per image?

There is no grain-count parameter. The count follows from the frame:

```
grains ≈ (config.image_size / domains_conf.mean_grain_size_px)²
```

| `image_size` | `mean_grain_size_px` | grains | time per image |
|---|---|---|---|
| 512 | 25 | ~400–500 | ~20 s |
| 512 | 60 | ~70–95 | ~5 s |
| 256 | 25 | ~100–130 | ~1 s |

This is the knob to reach for when generation is too slow or the COCO file is
too heavy: **bigger grains, fewer instances, much faster**.

---

## 4. Masks are used verbatim, then clipped under the objects

Isolated objects (`sections`, `image`, `STL`) get their mask re-derived from the
convolved topography with `mask = topo_conv > 0.15 · topo.max()`. That rule
assumes a bright object on a black background — a tessellation has no
background, so applying it would dissolve the grain boundaries.

Grain masks therefore come **straight from the generator's label map**, with no
re-thresholding. The COCO-time dilation that compensates for tip broadening is
also off by default (`mask_dilate_px: 0`), because dilating touching grains would
make every one of them bleed into its neighbours.

Then, because objects land on top of the film (§6), grain masks are **clipped**
against them: the object owns the pixel, so nothing is ever labelled both
`grain` and `hexagon`. A grain that keeps less than `min_visible_after_objects`
(default 0.5) of its area is dropped — a sliver peeking out from under a particle
is not a useful training instance, and you will see those as holes in the
instance map.

A grain that an object cuts into several pieces stays **one instance**: its
contours are folded into a single annotation with several `segmentation`
polygons, which is what a list of polygons is for. Splitting it would produce two
annotations sharing one `grain_id`.

---

## 5. Classes and per-grain identity

In COCO the **class** is `category_id`, and categories are dataset-global — so a
grain's own ID cannot be a class. Every grain of a `shape: domains` object gets
that object's `label` as its class; the grain's identity lives in the
annotation, which carries these extra fields:

| Field | Meaning |
|---|---|
| `grain_id` | The generator's own ID for that grain within its image |
| `eq_diameter_px` | Equivalent grain diameter, in pixels |
| `mean_z` | Mean grain height, on the same scale as the saved image |
| `eccentricity` | Grain elongation |
| `n_sides` | Polygon side count (`overlap_interactive` only) |
| `flatness` | How flat the fitted surface was |
| `roughness_rms_actual` | Realised RMS roughness inside the grain |
| `edge_rounding_used` | Whether the border was rounded in Z |

That makes grain-size and grain-height distributions readable straight out of
`instances_train.json`, without re-measuring the masks.

Only **one** `shape: domains` object may be enabled at a time. Two fields would
both fill the frame and their masks would contradict each other on almost every
pixel, so the pipeline raises instead of writing a dataset whose labels disagree
with themselves. Keep as many domains objects defined in the YAML as you like and
enable the one you want, or switch `domains_mode` on it.

### The instance map — where individual grains actually survive

COCO already holds one annotation per grain. The **semantic** UNet converter does
not: it rasterises every polygon with `fill=cls_idx`, so all grains of a class
collapse into a single region. And for a tessellation that region covers the
whole frame — a mask that is 100% foreground, which no segmentation model can
learn anything from.

So every saved image is accompanied by an instance label map:

| | |
|---|---|
| Path | `instances/<split>/<same name as the image>.png` |
| Type | **uint16** — a 512 px frame of 25 px grains holds ~530 of them, well past what uint8 can index |
| Value | instance index within that image; 0 = background |
| Link | every COCO annotation carries the matching `instance_index` |
| Switch | `config.save_instance_map` (default `true`) |

It covers **all** instances, grains and ordinary objects alike, and it is the
exact pixel-level ground truth: the COCO polygons are a re-derivation of the same
masks and drop contours below 1 px².

From it you derive, without storing anything else:

```python
inst = cv2.imread("instances/train/000000.png", cv2.IMREAD_UNCHANGED)

one_grain = inst == 7                      # a single instance
semantic  = (inst > 0).astype(np.uint8)    # foreground / background
boundaries = cv2.morphologyEx(             # the grain-boundary network
    inst.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3))) > 0
```

Cost, measured: ~15 KB per 512 px frame at 60 px grains, ~38 KB at 25 px grains —
20 to 52 MB for a 350-image dataset (the pipeline saves four variants per source
image, and each gets its own map).

---

## 6. Interaction with the rest of the scene

A grain field is not one more object in the bag — it is a **base layer**. It is
rendered before the object loop and folded into the substrate
(`_render_domain_layer` in `syntheticAFM_COCO.py`).

That placement is what makes the composition physically right. Ordinary objects
are combined with each other by `np.maximum`, but the substrate is **added**:

```
topo = max(object topographies) + (substrate + grain film)
```

So a particle of height 0.9 on a film that is locally 0.7 high ends up at 1.6 —
sitting on the film at its full height, with the grain texture still visible
around it. Max-compositing the grains instead would sink the object into the
film and flatten the grain relief underneath it.

`block_other_objects: false` (the default) is therefore the useful setting:
particles, bacteria or DNA on a granular film. Set it to `true` and the film
claims the frame, so the overlap filter rejects every other object — the
explicit way to say "grains only".

The exclusivity mask is kept separate from the occupancy mask that drives the
substrate smoothing, otherwise a full-frame film would have its own texture
gaussian-smoothed away.

---

## 7. Keeping the grains readable through the substrate and the artifacts

The grain field rides with the SUBSTRATE (§6), so substrate texture and
artifacts land straight on top of the grains rather than beside them. With
`default.yaml`'s amplitudes they bury the signal.

Measured as *grain contrast* — the spread of the per-grain mean height divided by
the typical spread inside a grain, so higher means the grains read as distinct
plateaux:

| Configuration | Grain contrast |
|---|---|
| `default.yaml` amplitudes, everything on | 0.72 |
| `domains_grains.yaml` amplitudes, everything still on | **1.01** |
| substrate and artifacts switched off entirely (the ceiling) | 1.12 |

So the preset recovers ~90% of what is achievable without giving up a single
AFM effect. The reference scale for choosing amplitudes: grains span
`z_mean` 0.5–1.2 with ~0.1–0.3 of relief inside each, so **anything in the
substrate with an amplitude of 0.3–0.9 is comparable to the whole grain signal**.

### The PID is the one that really matters

A tessellation has a height step at *every* grain boundary. With
`default.yaml`'s `max_speed: [1.4, 1.5]` the z-actuator cannot keep up and digs
deep negative spikes at each step. Those few pixels then dominate the min-max
normalisation `COCOBuilder` applies when saving, and the grains are squeezed into
a narrow band of the 0–255 output:

| `pid_conf` | Fraction of the saved range used by the middle 98% of pixels |
|---|---|
| `max_speed: [1.4, 1.5]` (default.yaml) | 24% |
| `max_speed: [3.0, 4.0]` (this preset) | 50% |
| PID disabled | 55% |

Raising `max_speed` recovers almost everything while keeping the feedback
simulation. **Do not raise `Ki` to compensate** — integral windup sends the
maximum to ~12 and makes it far worse.

### Ranked, one feature at a time

Measured from a clean base, so these are the ones to raise last:

| Feature | Cost to grain contrast | Note |
|---|---|---|
| `artifacts.loss_contact_*` | −42% | The long horizontal streaks. `loss_contact_n: 3` and probability 0.2 keep them visible without dominating. |
| `substrate.dirt_delta_n` | −20% | `default.yaml` fires 10500 spikes into a 512×512 frame — 4% of the pixels. 200 reads as dirt. |
| `artifacts.delta_n` | −6% | |
| everything else | <4% | tilt, perlin, gaussian, line noise, periodic noise are cheap. |

The masks are unaffected by any of this: they come from the ideal topography,
before the substrate or a single artifact is applied.

---

## 8. The Z model, in one page

For every visible grain:

1. **Correlated mean height.** Each grain draws a mean from `z_mean`, coupled to
   its neighbours with `w(d) = exp(-(d/L)²)` where `L = correlation_length_px`.
   `neighbor_z_correlation` sets the coupling strength.
2. **Perimeter modulation.** A periodic cubic spline of amplitude
   `boundary_z_amplitude` runs around the border, so grains are not featureless
   domes.
3. **Internal extrema.** `inner_extrema_count_range` hills/dips of amplitude
   `inner_extrema_amplitude` are placed inside.
4. **Smooth surface fit.** A thin-plate RBF fits perimeter + extrema;
   `flatness_range` and `surface_smoothing` control how flat the result is.
5. **Correlated roughness.** An independent realisation per grain, with RMS
   `roughness_rms` and lateral scale `roughness_corr_length_px`. Once per image,
   with probability `angular_correlation_probability`, a single preferred
   direction is sampled and shared by every grain — that is what produces the
   streaked, oriented texture.
6. **Edge rounding.** With probability `edge_rounding_probability`, heights decay
   smoothly toward the border down to `edge_rounding_border_fraction` over
   `edge_rounding_width_px`.

In `overlap_interactive` a grain is additionally lifted toward the support level
of whatever it overlaps: `support_percentile` picks the support height and
`stack_height_factor` how strongly the grain follows it (0 = keep its own mean,
1 = rise fully to the support).

### `z_mode`

- `direct` (default) — the `z_*` values are used as they are, so grain heights
  stay comparable from image to image.
- `relative` — each image is rescaled so that `z_norm_percentile` of the relief
  equals a value drawn from the object's own `height` range. Use when you want
  every image to show the same relief regardless of the sampled grain heights.

In both cases the pedestal is removed first: the baseline is the background
level when the mode leaves background, and the lowest grain level when it does
not. Background pixels are set to 0 so the substrate shows through the gaps.

---

## 9. Parameter cheat sheet

| Want | Change |
|---|---|
| Fewer, bigger grains (and much faster) | ↑ `mean_grain_size_px` |
| More irregular outlines | ↑ `shape_irregularity_range`, `side_curvature_range`, `curved_grain_probability`, `curved_side_probability` |
| Flatter, smoother grains | ↑ `flatness_range`, `surface_smoothing`; ↓ `boundary_z_amplitude`, `inner_extrema_amplitude` |
| Rougher grains | ↑ `roughness_rms` |
| Coarser roughness texture | ↑ `roughness_corr_length_px` |
| Clearly oriented texture | `angular_correlation_probability: 1.0`, a narrow `angular_correlation_angle_range_deg`, high `roughness_anisotropy_ratio` |
| More rounded borders | ↓ `edge_rounding_border_fraction`, ↑ `edge_rounding_width_px`, `edge_rounding_probability: 1.0` |
| More regular mosaic | ↑ `tessellation_relax` |
| More gaps / less coverage (overlap) | ↓ `coverage_target`, ↑ `gap_preference` |
| Reproducible grains across the dataset | set `seed` to an integer |

---

## 10. Files

| File | Role |
|---|---|
| `src/simuscan/domains_core.py` | Vendored ideal-topography engine. Do not edit — re-vendor from the reference to pick up upstream fixes. |
| `src/simuscan/domains_shape.py` | SimuScan renderer: unit bridge, Z baseline/scaling, label map → instance masks + metadata. |
| `src/simuscan/SyntheticParams.py` | `DomainsConfig` dataclass and the `domains_conf` → engine name mapping. |
| `src/simuscan/syntheticAFM_COCO.py` | `_render_domain_layer` plus the base-layer wiring in the dataset loop and the preview renderer. |
| `src/simuscan/COCOBuilder.py` | Instance-map writer, `instance_index`, per-mask `dilate_px` and `merge_contours`. |
| `data/config/conf_simuscan/domains_grains.yaml` | Ready-to-run preset. |

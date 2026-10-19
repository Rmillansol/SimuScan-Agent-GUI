# AFM Artifacts in SimuScan

SimuScan models the most common tapping-mode AFM artifacts, applied in an
ordered pipeline. All parameters can be edited with
`set_params("artifacts_conf", {...})` or `set_params("pid_conf", {...})`.

---

## Artifact pipeline (order of application)

```
Ideal topography
    ↓  tip_convolution_roi_fast    — convolution with the tip → saves clean image
    ↓  simulate_pid_topography     — PID feedback             → saves PID image
    ↓  add_gaussian_noise          — additive Gaussian noise
    ↓  add_loss_contact            — loss-of-contact lines
    ↓  add_local_adhesion          — local tip adhesion
    ↓  add_sticky_zones            — sticky zones
    ↓  add_line_noise              — per-line noise
    ↓  add_periodic_noise          — periodic interference
    ↓  add_shift_lines             — line shifting
    ↓  flatten                     — final flattening
    ↓  add_delta_noise             — delta spikes (LAST, amplitude relative to the range)
    → final image with artifacts
```

---

## Tip convolution

The dominant AFM artifact. The measured topography is the morphological dilation of
the real surface with the tip shape.

- Modeled as a dilation using the tip geometry (cone + hemisphere).
- Parameters in `tip_conf`: `radi_range`, `theta_range`, `is_double`, `double_probability`.
- **Effect:** Objects appear wider. Edges are rounded. With a double tip, a "ghost"
  shifted laterally appears.

---

## PID simulation (tapping mode)

Simulates the behavior of the feedback controller that adjusts the piezo height
to keep the oscillation amplitude constant.

- **Amplitude model:** `A(Zts) = A0 × (1 − exp(−β × (Zts − z0)))`
- **PID error:** `e = ASP − A(Zts)`
- **Adjustment:** `Δz = Kp×e + Ki×∫e dt + Kd×de/dt`

**Visible effects:**
- **Tracking error on walls:** the actuator does not follow fast enough → image compressed on the way up
- **Overshoot/oscillations:** underdamped PID (high Kp, low Kd) → peaks when reaching the top
- **Stickiness:** the tip gets stuck → flat lines in adhesion zones
- **Jumps:** sudden position jumps → vertical discontinuities

**Parameters in `pid_conf`:** `Kp`, `Ki`, `Kd`, `dt`, `max_speed`, `A0`, `beta`, `z0`,
`ASP`, `stickiness_prob`, `jump_prob`, `jump_magnitude`, `scan_direction`.

> For more oscillations: increase `Kp` ([0.8, 2.5]), decrease `Kd` ([0.0, 0.02]),
> decrease `max_speed` ([0.2, 0.8]).
> Use `tool_pid_tuner()` for interactive visual tuning.

---

## Gaussian noise (`apply_gaussian`)

Additive white noise that models thermal noise, electronic noise and vibrations.

- `gaussian_scale`: noise amplitude (std).
- `gaussian_probability`: fraction of images that receive it.
- It is applied **first** in the artifact pipeline so that later effects
  can act on top of it.

---

## Loss of contact (`apply_loss_contact`)

Horizontal lines with an offset and/or slope, caused by a momentary loss of
tip-sample contact or feedback noise.

- `loss_contact_n`: number of lines affected per image.
- `loss_contact_max_offset`: maximum vertical offset of the line.
- `loss_contact_max_slope`: maximum slope of the affected line.

---

## Local adhesion (`apply_adhesion`)

Zones where the tip momentarily sticks to the sample, creating local
disturbances in the image.

- `local_adhesion_intensity`: adhesion strength.
- `local_adhesion_pixel_probability`: per-pixel probability that adhesion occurs.

---

## Sticky zones (`apply_sticky_zones`)

Circular regions where the tip is more likely to get stuck.

- `sticky_zones_n`: number of zones per image.
- `sticky_zones_size_range`: zone size range [min, max] px.
- `sticky_zones_intensity`: disturbance intensity.

---

## Line noise (`apply_line_noise`)

Random variation in the height level of individual scan lines,
caused by feedback instability.

- `line_noise_intesity`: per-line noise amplitude.
- `line_noise_line_probability`: probability that a line is affected.

---

## Periodic noise (`apply_periodic_noise`)

High-frequency interference in the form of periodic waves, typical of
mechanical resonances of the system or electrical interference.

- `periodic_freq`: spatial frequency of the pattern.
- `periodic_amplitud`: amplitude of the waves.
- `periodic_angle_deg_range`: orientation range of the pattern.

---

## Line shifting (`apply_shift_lines_noise`)

Random lateral displacement of scan lines, caused by vibrations
or sudden disturbances of the scanner.

- `shift_lines_max`: maximum displacement in pixels.
- `shift_lines_roe_probability`: probability that a line is shifted.

---

## Flattening (`apply_flatten`)

Plane correction applied to the whole image to remove background tilt,
just like the real flatten of the AFM software.

- `flatten_perc`: percentage of lines used to compute the plane reference.
- Applied **before** the delta noise.

---

## Delta noise — artifacts (`apply_delta`)

Random vertical spikes caused by suspended particles, momentary vibration
or impulsive noise.

- **Relative amplitude:** `h = U(0, delta_intensity) × image_range`
  (relative to the image range, not absolute → always visible)
- `delta_n`: number of affected pixels.
- `delta_intensity`: maximum fraction of the image range.
- Applied **last** in the pipeline so it is not erased by flattening.

> **Difference from `dirt_delta`:** The substrate delta (`substrate_conf.dirt_delta_*`)
> uses an **absolute** amplitude (the substrate is stable). The artifact delta uses an
> amplitude **relative** to the range of the final image.

---

## Substrate delta noise (`dirt_delta`)

Spikes associated with dirt on the substrate surface, present before
objects are placed.

- `dirt_delta_n`: number of spikes.
- `dirt_delta_height`: maximum height (absolute, 0–1 nm scale).
- `dirt_delta_elongation_max`: extra pixels in the X direction (scan direction).
- `dirt_delta_elongation_decay`: height decay per elongated pixel (0–1).
- `dirt_delta_elongation_prob`: probability that each individual spike is elongated.

The elongation simulates particles being dragged by the tip during scanning.

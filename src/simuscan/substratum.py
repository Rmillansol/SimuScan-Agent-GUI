# Contains functions that generate substrates
import numpy as np
from scipy.ndimage import gaussian_filter
from . import shapes
from . import noisegenerator as gnoise
import matplotlib.pyplot as plt
import random
import os
from .SyntheticParams import SubstratumConfig as subs
import matplotlib.pyplot as plt

def show_image(
    img,
    title=None,
    cmap="cividis",
    vmin=None,
    vmax=None,
    figsize=(5, 4),
    show_axis=False,
):
    """
    Shows a 2D image with a colour bar using matplotlib.

    Parameters
    ----------
    img : ndarray 2D
        2D image as a NumPy array.
    title : str, optional
        Figure title.
    cmap : str, optional
        Matplotlib colormap (default 'cividis').
    vmin, vmax : float, optional
        Colour scale limits. If None, the image limits are used.
    figsize : tuple, optional
        Figure size in inches.
    show_axis : bool, optional
        If False, hides the axes.
    """
    img = np.asarray(img)

    plt.figure(figsize=figsize)
    im = plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    cbar = plt.colorbar(im)
    cbar.set_label("Height (a.u.)")

    if title is not None:
        plt.title(title)

    if not show_axis:
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def _smooth_edge(d, edge):
    """
    Smooth transition from 1 (d << 0) to 0 (d >> 0).
    d: signed distance to the edge (negative inside).
    edge: width of the transition zone (in the same units as d).
    """
    if edge <= 0:
        return (d <= 0).astype(float)

    t = np.clip((d + edge) / (2.0 * edge), 0.0, 1.0)  # 0 at d=-edge, 1 at d=+edge
    s = t * t * (3.0 - 2.0 * t)                       # smoothstep
    return 1.0 - s                                    # 1 at d=-edge, 0 at d=+edge


def _polygon_sdf(u, v, vx, vy):
    """
    Signed distance to a convex CCW polygon.
    d <= 0  -> inside
    d >  0  -> outside

    u, v: arrays of points
    vx, vy: vertices (1D) in CCW order
    """
    n = len(vx)
    d = None

    for i in range(n):
        x0, y0 = vx[i],         vy[i]
        x1, y1 = vx[(i+1) % n], vy[(i+1) % n]

        ex, ey = x1 - x0, y1 - y0
        # CCW polygon: outward normal = (ey, -ex)
        nx, ny = ey, -ex
        norm = np.hypot(nx, ny) + 1e-12
        nx /= norm
        ny /= norm

        d_i = (u - x0) * nx + (v - y0) * ny
        if d is None:
            d = d_i
        else:
            d = np.maximum(d, d_i)

    return d


def _shape_mask(u, v, shape, duty_cycle, slope, shape_angle_deg=0.0):
    """
    u, v: local coords in units of p_min (approx. [-0.5, 0.5]).
    shape: 'circle', 'square', 'triangle', 'hexagon'
    duty_cycle: fraction of p_min occupied by the shape (0-1).
    slope: edge smoothing (0 = hard).
    shape_angle_deg: rotation of the shape (not of the lattice) in degrees.
    """
    # Shape rotation
    phi = np.deg2rad(shape_angle_deg)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    ur = cos_p * u - sin_p * v
    vr = sin_p * u + cos_p * v

    size = 0.5 * duty_cycle
    edge = slope * size

    shape = shape.lower()

    if shape == "circle":
        r = np.sqrt(ur**2 + vr**2)
        d = r - size
        mask = _smooth_edge(d, edge)

    elif shape == "square":
        dx = np.abs(ur) - size
        dy = np.abs(vr) - size
        d = np.maximum(dx, dy)
        mask = _smooth_edge(d, edge)

    elif shape == "triangle":
        angles = np.deg2rad([-90.0, 30.0, 150.0])  # centered equilateral
        vx = size * np.cos(angles)
        vy = size * np.sin(angles)
        d = _polygon_sdf(ur, vr, vx, vy)
        mask = _smooth_edge(d, edge)

    elif shape == "hexagon":
        angles = np.deg2rad(np.arange(0.0, 360.0, 60.0))
        vx = size * np.cos(angles)
        vy = size * np.sin(angles)
        d = _polygon_sdf(ur, vr, vx, vy)
        mask = _smooth_edge(d, edge)

    else:
        raise ValueError(f"Unrecognized shape: {shape}")

    return mask

def generate_pillar_hole_pattern(
    image_size,
    pattern_type="pillars",
    shape="circle",
    period_x_px=50.0,
    period_y_px=50.0,
    duty_cycle=0.5,
    slope=0.1,
    height=1.0,
    orientation_deg=0.0,
    phase_x=0.0,
    phase_y=0.0,
    shape_angle_deg=0.0,
    dtype=np.float32,
):
    """
    Pillar/hole pattern on a rectangular lattice with shape rotation.

    period_x_px, period_y_px: lattice periods (x, y) in pixels.
    duty_cycle: shape size relative to p_min = min(period_x_px, period_y_px).
    shape_angle_deg: shape angle (triangle, square, hexagon, etc.).
    """
    ny, nx = image_size
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)

    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    shift_x = phase_x * period_x_px
    shift_y = phase_y * period_y_px

    X = xx - cx - shift_x
    Y = yy - cy - shift_y

    # Global lattice rotation
    theta = np.deg2rad(orientation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    xr = cos_t * X + sin_t * Y
    yr = -sin_t * X + cos_t * Y

    # Cells according to the x/y periods
    ix = np.floor(xr / period_x_px)
    iy = np.floor(yr / period_y_px)

    x_cell_center = (ix + 0.5) * period_x_px
    y_cell_center = (iy + 0.5) * period_y_px

    dx = xr - x_cell_center
    dy = yr - y_cell_center

    p_min = min(period_x_px, period_y_px)
    if p_min <= 0:
        raise ValueError("Periods must be positive.")

    u = dx / p_min
    v = dy / p_min

    mask = _shape_mask(u, v, shape, duty_cycle, slope, shape_angle_deg=shape_angle_deg)

    if pattern_type.lower() == "pillars":
        img = height * mask
    elif pattern_type.lower() == "holes":
        img = -height * mask
    else:
        raise ValueError("pattern_type must be 'pillars' or 'holes'")

    return img.astype(dtype)

def generate_grating_pattern(
    image_size,
    pattern_type="pillars",
    period_px=50.0,
    duty_cycle=0.5,
    slope=0.1,
    height=1.0,
    orientation_deg=0.0,
    phase_x=0.0,
    phase_y=0.0,
    dtype=np.float32,
):
    """
    Generates a grating-type substrate (1D stripes).

    Parameters
    ----------
    image_size : tuple (ny, nx)
        Output image size in pixels.
    pattern_type : {'pillars', 'holes'}
        'pillars' = positive ridges, 'holes' = negative grooves.
    period_px : float
        Period of the stripes in pixels.
    duty_cycle : float
            Fraction of the period in which the grating is "high".
    slope : float
        Edge smoothing control (0 = hard edge).
    height : float
        Height of the ridges or depth of the grooves (absolute value).
    orientation_deg : float
        Orientation of the stripes (angle of the normal).
    phase_x : float
        Phase in x in units of the period (shifts the grating in x).
    phase_y : float
        Phase in y in units of the period (shifts the grating in y).
    dtype : NumPy type
        Data type of the returned image.

    Returns
    --------
    img : ndarray (ny, nx)
        2D image with the generated grating.
    """
    ny, nx = image_size
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)

    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    shift_x = phase_x * period_px
    shift_y = phase_y * period_px

    X = xx - cx - shift_x
    Y = yy - cy - shift_y

    theta = np.deg2rad(orientation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Coordinate along the grating normal
    s = cos_t * X + sin_t * Y

    # Phase within the period [0,1) -> centred at 0
    phase = (s / period_px) - np.floor(s / period_px)
    phase = phase - 0.5

    # Duty cycle
    half_on = 0.5 * duty_cycle

    # Signed distance to the edge of the "on" band
    d = np.abs(phase) - half_on

    # Edge smoothing
    edge = slope * half_on
    mask = _smooth_edge(d, edge)

    if pattern_type.lower() == "pillars":
        img = height * mask
    elif pattern_type.lower() == "holes":
        img = -height * mask
    else:
        raise ValueError("pattern_type must be 'pillars' or 'holes'")

    return img.astype(dtype)

def generate_poly_plane(
    image_size,
    lin_range_x=1.0,
    lin_range_y=0.0,
    quad_ratio_x=0.2,
    quad_ratio_y=0.2,
    center_mean=True,
    dtype=np.float32,
):
    """
    Generates a polynomial plane of the form:
        z(x,y) = a*x + b*y + c*x^2 + d*y^2
    but defined intuitively from peak-to-peak variations.

    Parameters
    ----------
    image_size : tuple (ny, nx)
        Output image size in pixels.
    lin_range_x : float
        Peak-to-peak variation due to the linear term in x over the whole image.
        If x goes from -0.5 to 0.5, Delta z_lin_x = lin_range_x.
    lin_range_y : float
        Peak-to-peak variation due to the linear term in y.
    quad_ratio_x : float
        Fraction of the linear variation in x that the quadratic term in x will have.
        That is, Delta z_quad_x = quad_ratio_x * |lin_range_x|.
    quad_ratio_y : float
        Analogous for y.
    center_mean : bool
        If True, subtracts the mean of z so the plane has mean value ~0.
    dtype : NumPy type
        Data type of the returned image.

    Returns
    --------
    z : ndarray (ny, nx)
        2D image with the polynomial plane.
    """
    ny, nx = image_size

    # Normalized coordinates centered in [-0.5, 0.5]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    x = (xx - cx) / max(nx - 1, 1)
    y = (yy - cy) / max(ny - 1, 1)
    # x,y ~ [-0.5, 0.5]

    # -------------------------
    # 1) LINEAR TERMS
    # We want:
    #   Delta z_lin_x = a * (0.5 - (-0.5)) = a = lin_range_x
    #   Delta z_lin_y = b * (0.5 - (-0.5)) = b = lin_range_y
    # -------------------------
    a = lin_range_x
    b = lin_range_y
    z_lin = a * x + b * y

    # -------------------------
    # 2) QUADRATIC TERMS
    # We use (x^2 - <x^2>) so the term has zero mean.
    # For x in [-0.5, 0.5], <x^2> = 1/12 ~= 0.08333
    # The value of (x^2 - 1/12) goes from -1/12 (at x=0) to 1/6 (at x=+-0.5).
    # Peak-to-peak = (1/6 - (-1/12)) = 1/4.
    #
    # We want: Delta z_quad_x = |c_x| * 1/4 = quad_ratio_x * |lin_range_x|
    # -> |c_x| = 4 * quad_ratio_x * |lin_range_x|
    # (sign = sign of lin_range_x by default, to keep things simple).
    # If lin_range_x ~= 0 but we want curvature, we take a base scale
    # from lin_range_y.
    # -------------------------
    x2_centered = x**2 - (1.0 / 12.0)
    y2_centered = y**2 - (1.0 / 12.0)

    # Base scales to define the relative curvature
    base_x = abs(lin_range_x)
    base_y = abs(lin_range_y)

    # If one direction has no linear component, we use the other
    base_total = max(base_x, base_y, 1e-12)

    # Quadratic coefficients
    if base_x > 0:
        c_sign = np.sign(lin_range_x)
        c = c_sign * 4.0 * quad_ratio_x * base_x
    else:
        c = 4.0 * quad_ratio_x * base_total  # positive sign by default

    if base_y > 0:
        d_sign = np.sign(lin_range_y)
        d = d_sign * 4.0 * quad_ratio_y * base_y
    else:
        d = 4.0 * quad_ratio_y * base_total

    z_quad = c * x2_centered + d * y2_centered

    # -------------------------
    # 3) Total sum
    # -------------------------
    z = z_lin + z_quad

    if center_mean:
        z = z - np.mean(z)

    return z.astype(dtype)

def rand_substrate_old(substratum: subs , size = 512):
   
    """
    Generates a substrate image with different height levels defined
    by the list v_steps. Also adds n random shapes with heights given by the list r_shapes_px
    and height within h-shapes. Then adds Perlin noise at different scales with noise value
    defined by v_noise.
    It also allows adding delta-type noise.
    """
    masks=[]
    img = np.zeros((size,size), dtype=np.float32)
    img_stairs = img.copy()
    #generate steps
    if substratum.apply_stair:
        if random.random() < substratum.stair_probability: 
            for h_step in substratum.stair_height:
                sp1, sp2 = np.random.randint(round(size),4*size, size=2)
                cx, cy = np.random.randint(round(size*0.25), round(size*0.75), size=2)
                if substratum.stair_kind == 'spline':
                    num_puntos = np.random.randint(4, 12, size=1)[0]
                    imgt,mask = shapes.generate_spline_image(size=size, size_shape=(sp1, sp2), 
                                                    height=h_step, cx=cx, cy=cy, rotation=0, num_points=num_puntos, variation=0.2, smoothness=0)
                else:
                    n_sides = np.random.randint(3, 6, size=1)[0]
                    rot = np.random.uniform(0.0,90, size=1)[0]
                    imgt,mask = shapes.generate_polygon_image(size = size, size_shape=(sp1,sp2),
                                                  height=h_step,cx = cx, cy = cy, rotation=rot,num_sides= n_sides)
                masks.append(mask)
        
        for h, mask in zip(substratum.shapes_height, masks):
            h0 = np.random.uniform(h/2,h,size = 1)[0]
            img[mask > 0] = h0
    #generate shapes
    if substratum.apply_shapes:
        if random.random() < substratum.shapes_probability: 
            for r,h_shp in zip(substratum.shapes_size,substratum.stair_height):
                count =0
                while count < substratum.shapes_n:
                    sp1, sp2 = np.random.randint(r-r*0.15,r, size=2)
                    cx, cy = np.random.randint(round(5), size-5, size=2)
                    num_puntos = np.random.randint(4, 8, size=1)[0]
                    h = np.random.uniform(h_shp/2,h_shp,size = 1)[0]
                    order = np.random.uniform(0.1,0.5,size = 1)[0]
                    imgt,mask = shapes.generate_spline_image(size=size, size_shape=(sp1, sp2), 
                                                    height=h, cx=cx, cy=cy, rotation=0, num_points=num_puntos, variation=0.2, smoothness=0)

                    imgt = shapes.adjust_height_map(imgt, mask, dh=1.0, order=order)
                    v_max = img[mask > 1].max()
                    img[mask > 1] = np.maximum(imgt[mask > 1]+v_max, img[mask > 1])
                    #img[mask>1] += imgt[mask > 1] + v_max
                    count +=1

    #add dirt
    if substratum.apply_dirt:
        if random.random() <substratum.dirt_probability:
            for r,h_di in zip(substratum.dirt_size, substratum.dirt_height):
                count =0
                while count < substratum.dirt_n:
                    if random.random() < substratum.dirt_intensity:
                        sp1, sp2 = np.random.randint(r-r*0.15,r, size=2)
                        cx, cy = np.random.randint(5, size-5, size=2)
                        num_puntos = np.random.randint(4, 10, size=1)[0]
                        h = np.random.uniform(h_di/2,h_di,size = 1)[0]
                        imgt, mask = shapes.generate_surface_image(size=size, size_shape=(sp1, sp2), 
                                            height_min=0, height_max=h, cx=cx, cy=cy, num_puntos=num_puntos)
                        img+= imgt
                    count +=1
        if substratum.dirt_delta_n>0:
            n_d = np.random.randint(0, substratum.dirt_delta_n, size=1)[0]
            img = gnoise.add_delta_random_noise(img, num_pixeles = n_d, valor=substratum.dirt_delta_height)     
    #add noise
    if substratum.apply_perlin:
        ampl = np.random.uniform(0,substratum.perlin_amplitud, size=1)[0]
        for scale in substratum.perlin_scale_px:
            img = gnoise.add_perlin_noise(img,scale_px=scale, value=ampl)
    # noise
    if substratum.apply_gaussian:
        ampl = np.random.uniform(0,substratum.gaussian_amplitud, size=1)[0]
        ruido = np.random.normal(loc=substratum.gaussian_mean, scale=substratum.gaussian_scale, size=(size,size))
        img += ampl * ruido
    if substratum.apply_gaussian_filter:
        # Apply smoothing to the image
        img = gaussian_filter(img, sigma=substratum.filter_sigma)
    return img

def rand_substrate(substratum: subs, size: int = 512):
    """
    Generates a substrate image with:
        - Tilt + curvature (scanner tilt / bow) if apply_tilt
        - Periodic patterns (pillars, holes, grating) if apply_pattern
        - Steps (stairs)
        - Large shapes (shapes)
        - Dirt
        - Perlin noise
        - Gaussian noise
        - Optional Gaussian filter
    Using ranges defined in SubstratumConfig (YAML).
    """
    masks = []
    img = np.zeros((size, size), dtype=np.float32)
    img_stairs = img.copy()

    # ----------------------------------------------------
    # 1) TILT + BOW (scanner_tilt + scanner_bow)
    # ----------------------------------------------------
    if substratum.apply_tilt and random.random() < substratum.tilt_probability:
        # Random tilt magnitude between 0 and tilt_range
        tilt_mag = np.random.uniform(0.0, substratum.tilt_range)
        # Random direction in the plane (angle in [0, 2*pi))
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        lin_range_x = tilt_mag * np.cos(phi)
        lin_range_y = tilt_mag * np.sin(phi)

        # The quadratic part is scaled as a fraction of the tilt (second_order_range)
        quad_ratio = substratum.second_order_range

        tilt_plane = generate_poly_plane(
            image_size=(size, size),
            lin_range_x=lin_range_x,
            lin_range_y=lin_range_y,
            quad_ratio_x=quad_ratio,
            quad_ratio_y=quad_ratio,
            center_mean=True,
            dtype=np.float32,
        )
        
        img += tilt_plane
        
    # ----------------------------------------------------
    # 2) PERIODIC PATTERNS (pillars / holes / grating)
    # ----------------------------------------------------
    if substratum.apply_pattern and random.random() < substratum.pattern_probability:
        # Pick random parameters within the ranges
        period_x = np.random.uniform(
            substratum.pattern_period_x_px_range[0],
            substratum.pattern_period_x_px_range[1],
        )
        period_y = np.random.uniform(
            substratum.pattern_period_y_px_range[0],
            substratum.pattern_period_y_px_range[1],
        )
        duty_cycle = np.random.uniform(
            substratum.pattern_duty_cycle[0],
            substratum.pattern_duty_cycle[1],
        )
        slope = np.random.uniform(
            substratum.pattern_slope[0],
            substratum.pattern_slope[1],
        )
        height = np.random.uniform(
            substratum.pattern_height[0],
            substratum.pattern_height[1],
        )
        orientation = np.random.uniform(
            substratum.pattern_orientation_deg[0],
            substratum.pattern_orientation_deg[1],
        )
        shape_angle = np.random.uniform(
            substratum.pattern_shape_angle_deg[0],
            substratum.pattern_shape_angle_deg[1],
        )

        # phase_x, phase_y are given in pixels in the YAML -> convert them to a fraction of the period
        phase_x_px = np.random.uniform(
            substratum.pattern_phase_x[0],
            substratum.pattern_phase_x[1],
        )
        phase_y_px = np.random.uniform(
            substratum.pattern_phase_y[0],
            substratum.pattern_phase_y[1],
        )
        # Avoid division by zero if any of the periods comes out odd
        phase_x = phase_x_px / max(period_x, 1e-9)
        phase_y = phase_y_px / max(period_y, 1e-9)

        pattern_type = substratum.pattern_type.lower()
        pattern_shape = substratum.pattern_shape.lower()

        if pattern_type in ("pillars", "holes"):
            pattern_img = generate_pillar_hole_pattern(
                image_size=(size, size),
                pattern_type=pattern_type,             # 'pillars' or 'holes'
                shape=pattern_shape,                   # circle, square, triangle, hexagon
                period_x_px=period_x,
                period_y_px=period_y,
                duty_cycle=duty_cycle,
                slope=slope,
                height=height,
                orientation_deg=orientation,
                phase_x=phase_x,
                phase_y=phase_y,
                shape_angle_deg=shape_angle,
                dtype=np.float32,
            )
        elif pattern_type == "grating":
            # For grating we use only one period (e.g. period_x)
            pattern_img = generate_grating_pattern(
                image_size=(size, size),
                pattern_type="pillars",                # positive ridges
                period_px=period_x,
                duty_cycle=duty_cycle,
                slope=slope,
                height=height,
                orientation_deg=orientation,
                phase_x=phase_x,
                phase_y=phase_y,
                dtype=np.float32,
            )
        else:
            pattern_img = np.zeros_like(img)

        img += pattern_img

    # ----------------------------------------------------
    # 3) STAIRS
    # ----------------------------------------------------
    if substratum.apply_stair:
        if random.random() < substratum.stair_probability:
            for h_step in substratum.stair_height:
                sp1, sp2 = np.random.randint(round(size) * 0.25, 4 * size, size=2)
                cx, cy = np.random.randint(-1 * round(size * 0.9), round(size * 0.9), size=2)
                if substratum.stair_kind == 'spline':
                    num_puntos = np.random.randint(4, 12, size=1)[0]
                    imgt, mask = shapes.generate_spline_image(
                        size=size,
                        size_shape=(sp1, sp2),
                        height=h_step,
                        cx=cx,
                        cy=cy,
                        rotation=0,
                        num_points=num_puntos,
                        variation=0.2,
                        smoothness=0,
                    )
                else:
                    n_sides = np.random.randint(3, 6, size=1)[0]
                    rot = np.random.uniform(0.0, 90, size=1)[0]
                    imgt, mask = shapes.generate_polygon_image(
                        size=size,
                        size_shape=(sp1, sp2),
                        height=h_step,
                        cx=cx,
                        cy=cy,
                        rotation=rot,
                        num_sides=n_sides,
                    )
                masks.append(mask)

            for h, mask in zip(substratum.stair_height, masks):
                h0 = np.random.uniform(h / 2, h, size=1)[0]
                img[mask > 0] += h0

    # ----------------------------------------------------
    # 4) LARGE SHAPES
    # ----------------------------------------------------
    if substratum.apply_shapes:
        if random.random() < substratum.shapes_probability:
            for r, h_shp in zip(substratum.shapes_size, substratum.shapes_height):
                count = 0
                while count < substratum.shapes_n:
                    sp1, sp2 = np.random.randint(r - r * 0.15, r, size=2)
                    cx, cy = np.random.randint(round(5), size - 5, size=2)
                    num_puntos = np.random.randint(4, 8, size=1)[0]
                    h = np.random.uniform(h_shp / 2, h_shp, size=1)[0]
                    order = np.random.uniform(0.1, 0.5, size=1)[0]
                    imgt, mask = shapes.generate_spline_image(
                        size=size,
                        size_shape=(sp1, sp2),
                        height=h,
                        cx=cx,
                        cy=cy,
                        rotation=0,
                        num_points=num_puntos,
                        variation=0.2,
                        smoothness=0,
                    )

                    imgt = shapes.adjust_height_map(imgt, mask, dh=1.0, order=order)
                    v_max = img[mask > 1].max() if np.any(mask > 1) else 0.0
                    img[mask > 1] = np.maximum(imgt[mask > 1] + v_max, img[mask > 1])
                    count += 1

    # ----------------------------------------------------
    # 5) DIRT
    # ----------------------------------------------------
    if substratum.apply_dirt:
        if random.random() < substratum.dirt_probability:
            for r, h_di in zip(substratum.dirt_size, substratum.dirt_height):
                count = 0
                while count < substratum.dirt_n:
                    if random.random() < substratum.dirt_intensity:
                        sp1, sp2 = np.random.randint(r - r * 0.15, r, size=2)
                        cx, cy = np.random.randint(5, size - 5, size=2)
                        num_puntos = np.random.randint(4, 10, size=1)[0]
                        h = np.random.uniform(h_di / 2, h_di, size=1)[0]
                        imgt, mask = shapes.generate_surface_image(
                            size=size,
                            size_shape=(sp1, sp2),
                            height_min=0,
                            height_max=h,
                            cx=cx,
                            cy=cy,
                            num_puntos=num_puntos,
                        )
                        img += imgt
                    count += 1

        if substratum.dirt_delta_n > 0:
            n_d = np.random.randint(0, substratum.dirt_delta_n, size=1)[0]
            img = gnoise.add_delta_random_noise(
                img,
                num_pixeles=n_d,
                valor=substratum.dirt_delta_height,
                elongation_max=substratum.dirt_delta_elongation_max,
                elongation_decay=substratum.dirt_delta_elongation_decay,
                elongation_prob=substratum.dirt_delta_elongation_prob,
            )

    # ----------------------------------------------------
    # 6) PERLIN NOISE
    # ----------------------------------------------------
    if substratum.apply_perlin:
        ampl = np.random.uniform(0, substratum.perlin_amplitud, size=1)[0]
        for scale in substratum.perlin_scale_px:
            img = gnoise.add_perlin_noise(img, scale_px=scale, value=ampl)

    # ----------------------------------------------------
    # 7) GAUSSIAN NOISE
    # ----------------------------------------------------
    if substratum.apply_gaussian:
        ampl = np.random.uniform(0, substratum.gaussian_amplitud, size=1)[0]
        ruido = np.random.normal(
            loc=substratum.gaussian_mean,
            scale=substratum.gaussian_scale,
            size=(size, size),
        )
        img += ampl * ruido

    # ----------------------------------------------------
    # 8) GAUSSIAN FILTER
    # ----------------------------------------------------
    if substratum.apply_gaussian_filter:
        img = gaussian_filter(img, sigma=substratum.filter_sigma)


    return img


def substrate_base(size = 256, height=1, scale= 0.1, sigma_smooth=1.0, steps = 100, seed=None):
    """
    Generates a substrate-like image with a Gaussian error at different scales,
    smooths each one and sums them to form a single image.

    Parameters:
    - matriz: numpy matrix containing the image to add noise to
    - escalas : Standard deviations for the Gaussian noise
    - sigma_smooth (float): Sigma for the pixel-to-pixel smoothing.
    - range(int): Number of times the noise summation is repeated
    - seed (int): Seed for reproducible noise generation.

    Returns:
    - np.ndarray: Matriz final combinada.
    """

    if seed is not None:
        np.random.seed(seed)

    img = np.zeros((size,size), dtype=np.float32)
    tamanio =img.shape
    rng = max(steps,2)
    for i in range(1,rng,5):
        # Generate Gaussian noise with the given standard deviation
        ruido = np.random.normal(loc=0, scale=scale*i, size=tamanio)
        # Apply smoothing to the image
        ruido_suavizado = gaussian_filter(ruido, sigma=sigma_smooth*i)
        #ruido_suavizado = uniform_filter(ruido, size=i)
        # Add the smoothed image to the final image
        img  += ruido_suavizado

    img = img / np.max(img) if np.max(img) != 0 else img

    return height*img

def SaveMatriz(img, fileName, path, size_x=None,unixy='nm',uniz='nm'):
    dir_path = os.path.join(path, fileName)
    
    if size_x is None:
        size_x = img.shape[0]     
            
    # Compute the steps along the X and Y axes
    step_x = size_x / img.shape[0]  # Total size in X divided by the number of pixels in X
    step_y = size_x / img.shape[1]  # Total size in Y divided by the number of pixels in Y
   

    with open(dir_path, 'w') as file:
        file.write("WSxM file copyright UAM\n")
        file.write("WSxM ASCII XYZ file\n")
        file.write(f"X[{unixy}]\tY[{unixy}]\tZ[{uniz}]\n")
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # Convert the indices to physical coordinates
                x_real = i * step_x
                y_real = j * step_y
                file.write(f"{x_real} {y_real} {img[i, j]}\n")


if __name__ == '__main__':

    # 512x512 image with circular pillars
    img = generate_pillar_hole_pattern(
    image_size=(512, 512),
    pattern_type="pillars",
    shape="square",
    period_x_px=80,
    period_y_px=120,
    duty_cycle=0.4,
    slope=0.2,
    height=8.0,
    orientation_deg=0.0,   # lattice orientation
    phase_x=0.0,
    phase_y=0.0,
    shape_angle_deg=45,  # triangle rotation
    )
    show_image(img)
    
  
    # Grating 1D tilted 45 deg, with phase
    img_grating = generate_grating_pattern(
        image_size=(512, 512),
        pattern_type="pillars",
        period_px=40,
        duty_cycle=0.3,
        slope=0.15,
        height=3.0,
        orientation_deg=45.0,
        phase_x=0.1,
        phase_y=0.0,
    )
    show_image(img_grating)


    # Plane with a strong slope in x and a bit of curvature
    z1 = generate_poly_plane(
        image_size=(512, 512),
        lin_range_x=1.0,   # the image rises 10 units from one side to the other in x
        lin_range_y=0.1,
        quad_ratio_x=0.2,   # curvature with Δz ~ 0.2 * 10 = 2
        quad_ratio_y=0.5,
    )
    show_image(z1)
    # Plane with tilt in x and y, and gentle curvatures
    z2 = generate_poly_plane(
        image_size=(512, 512),
        lin_range_x=1.0,
        lin_range_y=-1.0,
        quad_ratio_x=0.1,
        quad_ratio_y=0.1,
    )
    show_image(z2)
    # #img= rand_substrate()

      
    # # Visualization
    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='viridis')
    # plt.title('Matriz Original')
    # plt.colorbar()

    # plt.subplot(1, 2, 2)
    # plt.imshow(mask, cmap='viridis')
    # plt.title('Matriz Original')
    # plt.colorbar()


    # plt.show()
    
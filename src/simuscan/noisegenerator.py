import numpy as np
from matplotlib.path import Path
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from skimage.draw import polygon

# ---------- AFM image noise generator functions --------------
def add_perlin_noise(matriz, scale_px=10, value=1.0, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    matriz_modificada = np.copy(matriz)
    h,w = matriz.shape

    base = 2*np.random.rand(h, w)-1.0
    base = gaussian_filter(base, sigma=scale_px)
    
    matriz_modificada+=value*base

    return  matriz_modificada

def add_perline_noise_inside_mask(matriz,mask,scale_px=10, value=1.0):

    texture_image = add_perlin_noise(matriz, scale_px=scale_px, value=value, seed=None)
    # Keep the original values outside the mask
    result = np.where(mask, texture_image, matriz)

    return result
    
# Function that returns a matrix with Gaussian noise at different scales
def generator_gaussian_noise(matriz, scale= 0.1, sigma_smooth=1.0, seed=None):
    """
    Generates a matrix with Gaussian noise at different scales,
    smooths each one and sums them to form a single image.

    Parameters:
    - matriz: numpy matrix containing the image to add noise to
    - escalas : Standard deviations for the Gaussian noise
    - sigma_smooth (float): Sigma for the pixel-to-pixel smoothing.
    - seed (int): Seed for reproducible noise generation.

    Returns:
    - np.ndarray: Matriz final combinada.
    """
    if seed is not None:
        np.random.seed(seed)

    matriz_modificada = np.copy(matriz)
    tamanio = matriz.shape

    for i in range(1,100,5):
        # Generate Gaussian noise with the given standard deviation
        ruido = np.random.normal(loc=0, scale=scale, size=tamanio)
        # Apply smoothing to the image
        ruido_suavizado = gaussian_filter(ruido, sigma=sigma_smooth*i)
        # Add the smoothed image to the final image
        matriz_modificada  += ruido_suavizado

    return matriz_modificada 
# Function that adds Gaussian noise
def add_normal_noise(matriz,scale = 0.01,seed = None):
    
    if seed is not None:
        np.random.seed(seed)

    matriz_modificada = np.copy(matriz)
    tamanio = matriz.shape

    ruido = np.random.normal(loc=0, scale=scale, size=tamanio)
    matriz_modificada  += ruido

    return matriz_modificada 

def add_delta_random_noise(
    matriz,
    num_pixeles,
    valor=1,
    seed=None,
    elongation_max=0,
    elongation_decay=0.5,
    elongation_prob=1.0,
):
    """
    Adds num_pixeles impulsive spikes (delta) with uniform height U(0, valor).

  Optionally stretches each spike along the X direction (columns):
      - elongation_prob  : probability that a given spike is stretched.
      - elongation_max   : maximum stretch length in pixels (uniform [1, max]).
        - elongation_decay : height decay factor per extra pixel (0-1).

    This simulates dirt dragged by the AFM tip along the scan
    direction, where not all particles are dragged, nor by the same amount.

    Parameters
    ----------
    matriz          : np.ndarray  Input image (float, 2D).
    num_pixeles     : int         Number of spikes to insert.
    valor           : float       Maximum spike height (same units as the image).
    seed            : int | None  Seed for reproducibility.
    elongation_max  : int         Maximum extra pixels in X (0 = no stretching).
    elongation_decay: float       Height decay per extra pixel (0-1).
    elongation_prob : float       Probability that each individual spike is stretched (0-1).

    Returns
    -------
    np.ndarray  Copy of the matrix with the spikes added.
    """
    rng = np.random.default_rng(seed)

    matriz_modificada = np.array(matriz, copy=True)
    filas, columnas = matriz_modificada.shape
    total_pixels = filas * columnas
    num_pixeles = int(min(max(num_pixeles, 0), total_pixels))

    if num_pixeles == 0:
        return matriz_modificada

    # Vectorised selection of unique positions - O(k), no while-loop
    flat_idx = rng.choice(total_pixels, size=num_pixeles, replace=False)
    xs = flat_idx // columnas
    ys = flat_idx  % columnas

    # Random heights for each spike
    alturas = rng.uniform(0, valor, size=num_pixeles)
    matriz_modificada[xs, ys] += alturas

    # Stretching in X - each spike is decided independently
    if elongation_max > 0 and elongation_prob > 0.0:
        for i in range(num_pixeles):
            if rng.random() > elongation_prob:
                continue                          # this spike is not stretched
            n_extra = int(rng.integers(1, elongation_max + 1))
            h = alturas[i]
            for k in range(1, n_extra + 1):
                yc = ys[i] + k
                if yc >= columnas:
                    break
                h *= elongation_decay
                matriz_modificada[xs[i], yc] += h

    return matriz_modificada

# Function that adds random shapes to the substrate
def add_random_shapes(matriz, num_splines, radio_medio, valor=1, num_puntos=12, seed=None):
    """
    Adds random structures (closed splines) with a random value to a matrix.

    Parameters:
    - matriz (np.ndarray): Original matrix to which the splines are added.
    - num_splines (int): Number of splines to add.
    - radio_medio (float): Mean radius of the splines.
    - valor (float): Maximum value for the spline height.
    - num_puntos (int): Number of points used to generate the splines.
    - seed (int): Seed for reproducible spline generation.

    Returns:
    - np.ndarray: Matrix with the splines added.
    """
    if seed is not None:
        np.random.seed(seed)

    matriz_modificada = np.copy(matriz)
    filas, columnas = matriz.shape

    for _ in range(num_splines):
        # Generate a random center for the spline
        centro_x = np.random.randint(radio_medio, filas - radio_medio)
        centro_y = np.random.randint(radio_medio, columnas - radio_medio)

        # Generate random points around the center
        angulos = np.linspace(0, 2 * np.pi, num_puntos, endpoint=False)
        radios = np.random.uniform(0.5 * radio_medio, 1.5 * radio_medio, num_puntos)
        x_puntos = centro_x + radios * np.cos(angulos)
        y_puntos = centro_y + radios * np.sin(angulos)

        # Close the curve by appending the first point at the end
        x_puntos = np.append(x_puntos, x_puntos[0])
        y_puntos = np.append(y_puntos, y_puntos[0])

        # Fit the closed spline
        tck, _ = splprep([x_puntos, y_puntos], s=0, per=True)
        spline_x, spline_y = splev(np.linspace(0, 1, 50), tck)

        # Create a Path to identify the closed area
        spline_path = Path(np.column_stack((spline_x, spline_y)))

        # Create a mask of the pixels inside the closed area
        xx, yy = np.meshgrid(np.arange(columnas), np.arange(filas))
        puntos = np.column_stack((xx.ravel(), yy.ravel()))
        dentro = spline_path.contains_points(puntos).reshape(filas, columnas)

        # Assign a random value to the closed area
        altura = np.random.uniform(0, valor)
        matriz_modificada[dentro] += altura

    return matriz_modificada

# Function that adds a straight line with random offset and slope to every line, or only to some lines, to simulate AFM error
def add_lines_random(matriz, max_pendiente, num_lineas=None, valor_max_offset=10, seed=None):
    """
    Adds straight lines with random slopes and offsets to a matrix.

    Parameters:
    - matriz (np.ndarray): Input matrix to which the lines are added.
    - max_pendiente (float): Maximum value for the random slopes.
    - num_lineas (int): Number of random lines to modify. If None, applies to all lines.
    - valor_max_offset (float): Maximum value for the random offset of each line.
    - seed (int): Seed for reproducible random values.

    Returns:
    - np.ndarray: Matrix with the straight lines added.
    """
    if seed is not None:
        np.random.seed(seed)

    matriz_modificada = np.copy(matriz)
    filas, columnas = matriz.shape

    # If no lines are specified, apply to all
    if num_lineas is None:
        lineas = np.arange(filas)
    else:
        lineas = np.random.choice(filas, size=num_lineas, replace=False)

    for linea in lineas:
        # Generate random slope and offset
        pendiente = np.random.uniform(-max_pendiente, max_pendiente)
        offset = np.random.uniform(-valor_max_offset, valor_max_offset)
        # Generate the straight line with slope and offset
        valores_linea = np.arange(columnas) * pendiente + offset
        # Add the generated values to the corresponding line
        matriz_modificada[linea, :] += valores_linea

    return matriz_modificada

# Function that adds straight-line noise only in some zones of the rows
def add_lines_segments_random(matriz, max_pendiente, num_lineas=None, valor_max_offset=10, seed=None):
    """
    Adds straight line segments with random slopes, offsets and starts to a matrix.

    Parameters:
    - matriz (np.ndarray): Input matrix to which the segments are added.
    - max_pendiente (float): Maximum value for the random slopes.
    - num_lineas (int): Number of random lines to modify. If None, applies to all lines.
    - valor_max_offset (float): Maximum value for the random offset of each line.
    - seed (int): Seed for reproducible random values.

    Returns:
    - np.ndarray: Matrix with the straight line segments added.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    matriz_modificada = np.copy(matriz)
    filas, columnas = matriz.shape

    # If no lines are specified, apply to all
    if num_lineas is None:
        lineas = np.arange(filas)
    else:
        lineas = np.random.choice(filas, size=num_lineas, replace=False)


    for linea in lineas:
        # Generate random slope and offset
        
        pendiente = np.random.uniform(- max_pendiente, max_pendiente)
        offset = np.random.uniform(-valor_max_offset, valor_max_offset)

        # Segment length
        L = rng.integers(columnas // 4, columnas // 2 + 1)

        # Random direction: +1 (right) or -1 (left)
        direccion = rng.choice([-1, 1])

        if direccion == 1:
            inicio = rng.integers(0, columnas - L + 1)
            fin = inicio + L
            x = np.arange(L)                    # 0..L-1
        else:
            fin = rng.integers(L, columnas + 1) # end in [L..columnas]
            inicio = fin - L
            x = np.arange(L)[::-1]              # L-1..0 (grows "towards the left")

        valores_segmento = x * pendiente + offset
        matriz_modificada[linea, inicio:fin] += valores_segmento


    return matriz_modificada

# ----------Noise in regions-------------------
# Local noise (increased roughness)
def add_local_adhesion_noise(image, intensity=0.01, probability=0.1):
    noisy_image = image.copy()
    mask = np.random.rand(*image.shape) < probability  # Mask of affected points
    noise = intensity * np.random.normal(size=image.shape) * mask
    noisy_image += noise
    return noisy_image

# Circular sticky zones
def add_circular_sticky_zones(image, num_zones=3, zone_size=20, intensity=0.01):
    noisy_image = image.copy()
    rows, cols = image.shape
    
    for _ in range(num_zones):
        # Select the center of the sticky zone
        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)
        
        # Create a circular mask for the sticky zone
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = distance <= zone_size
        
        # Add noise in the sticky zone
        noise = intensity * np.random.normal(size=image.shape)
        noisy_image[mask] += noise[mask]
    
    return noisy_image

# Adds sticky zones with random shapes defined by closed polygons.
def add_polygonal_sticky_zones(image, num_zones=3, num_points=8, intensity=0.01, rrange = (10,50)):
    """
    Adds sticky zones with random shapes defined by closed polygons.
    
    Args:
        image (np.ndarray): Input image.
        num_zones (int): Number of sticky zones to add.
        num_points (int): Number of vertices defining the random shape.
        intensity (float): Intensity of the noise inside the zones.
        rrange (int,int): Radius variation range
    Returns:
        np.ndarray: Image with noise added in the sticky zones.
    """
    noisy_image = image.copy()
    rows, cols = image.shape
    
    for _ in range(num_zones):
        # Generate random points for the shape
        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)
        radius = np.random.randint(rrange[0],rrange[1])  # Maximum radius of the shape
        
        # Create vertices around the center
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        random_offsets = radius * (0.5 + np.random.rand(num_points))  # Vary the radius
        x_vertices = (center_x + random_offsets * np.cos(angles)).astype(int)
        y_vertices = (center_y + random_offsets * np.sin(angles)).astype(int)
        
        # Make sure the vertices are within the image bounds
        x_vertices = np.clip(x_vertices, 0, cols - 1)
        y_vertices = np.clip(y_vertices, 0, rows - 1)
        
        # Create a mask for the random shape
        rr, cc = polygon(y_vertices, x_vertices, image.shape)
        
        # Add noise in the sticky zone
        noise = intensity * np.random.normal(size=image.shape)
        noisy_image[rr, cc] += noise[rr, cc]
    
    return noisy_image

# Adds sticky zones with random shapes defined by closed splines.
def add_spline_sticky_zones(image, num_zones=3, num_points=8, intensity=0.01, rrange=(10, 50)):
    """
    Adds sticky zones with random shapes defined by closed splines.
    
    Args:
        image (np.ndarray): Input image.
        num_zones (int): Number of sticky zones to add.
        num_points (int): Number of vertices defining the random shape.
        intensity (float): Intensity of the noise inside the zones.
        rrange (tuple): Radius range for the shapes.
    
    Returns:
        np.ndarray: Image with noise added in the sticky zones.
    """
    noisy_image = image.copy()
    rows, cols = image.shape
    
    for _ in range(num_zones):
        # Generate random points around a center
        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)
        radius = np.random.randint(rrange[0], rrange[1])  # Mean radius
        
        # Create vertices around the center
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        random_offsets = radius * (0.5 + np.random.rand(num_points))  # Vary the radius
        x_vertices = center_x + random_offsets * np.cos(angles)
        y_vertices = center_y + random_offsets * np.sin(angles)
        
        # Close the loop to create a closed spline
        x_vertices = np.append(x_vertices, x_vertices[0])
        y_vertices = np.append(y_vertices, y_vertices[0])
        
        # Fit a closed spline
        tck, _ = splprep([x_vertices, y_vertices], s=0, per=True)
        spline_points = splev(np.linspace(0, 1, 100), tck)  # Generate smooth points

        # Convert spline to pixels and create mask
        x_spline = np.clip(spline_points[0], 0, cols - 1).astype(int)
        y_spline = np.clip(spline_points[1], 0, rows - 1).astype(int)
        rr, cc = polygon(y_spline, x_spline, image.shape)
        
        # Add noise in the sticky zone
        noise = intensity * np.random.normal(size=image.shape)
        noisy_image[rr, cc] += noise[rr, cc]
    
    return noisy_image
# Line artifacts: higher noise on some lines
def add_line_artifacts(image, intensity=0.1, probability=0.1):
    noisy_image = image.copy()
    rows = image.shape[0]
    
    # Randomly select affected lines
    affected_lines = np.random.choice(rows, size=int(rows * probability), replace=False)
    
    for line in affected_lines:
        # Add a jerk to the whole line
        noise = intensity * np.random.normal(size=image.shape[1])
        noisy_image[line, :] += noise
    
    return noisy_image
# Periodic noise at an angle
def add_periodic_noise_angled(image, frequency=10, amplitude=0.05, angle=45):
    rows, cols = image.shape
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Create periodic noise at a specific angle
    periodic_noise = amplitude * np.sin(2 * np.pi * frequency * (x * np.cos(angle_rad) + y * np.sin(angle_rad)) / max(rows, cols))
    
    noisy_image = image + periodic_noise
    return noisy_image
# Shifts lines randomly
def shift_lines_randomly(image, max_shift=3):
    """
    Shifts each line of an image horizontally by a random number of pixels.
    
    Args:
        image (np.ndarray): Input image.
        max_shift (int): Maximum shift (in pixels) allowed per line.
    
    Returns:
        np.ndarray: Resulting image with shifted lines.
    """
    # Create a copy of the image so the original is not modified
    shifted_image = np.zeros_like(image)
    rows, cols = image.shape

    for i in range(rows):
        # Generate a random shift between -max_shift and max_shift
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        # Apply the shift circularly to the row
        shifted_image[i] = np.roll(image[i], shift)

    return shifted_image


def simulate_pid_topography(
    image,
    Kp=0.5, Ki=0.1, Kd=0.05, dt=0.8,
    max_speed=None,
    scan_direction="forward",
    stickiness_prob=0.0,
    jump_prob=0.0,
    jump_magnitude=2.0,
    A0=1.0, beta=1.0, z0=2.0, ASP=0.5,
):
    """
    Simulates the effect of the PID controller on an AFM topography image in
    tapping mode.

    The oscillation amplitude model is:
        A(Zts) = A0 * (1 - exp(-beta * (Zts - z0)))
    where Zts = height_pid - image[i,j] is the tip-sample distance.

    The PID tries to keep A = ASP by adjusting height_pid at every pixel.

    Corrections with respect to the original version:
    - offset and physical invariants computed once outside the loop.
    - integral and previous_error are reset at the start of each line.
    - Stickiness bug fixed: the stuck pixel keeps height_pid.
    - scan_direction fixed for the whole image (does not alternate between lines).

    Parameters
    ----------
    image          : np.ndarray  Ideal topography (2D float).
    Kp, Ki, Kd     : float       PID gains (proportional, integral, derivative).
    dt             : float       Simulated time step.
    max_speed      : float|None  Actuator speed limit (None = no limit).
    scan_direction : str         "forward" (left->right) or "backward" (right->left).
    stickiness_prob: float       Probability that the tip gets stuck.
    jump_prob      : float       Probability of an abrupt position jump.
    jump_magnitude : float       Maximum jump magnitude.
    A0             : float       Free oscillation amplitude (normalised).
    beta           : float       Interaction decay rate.
    z0             : float       Interaction onset distance.
    ASP            : float       Set-point amplitude (fraction of A0).

    Returns
    -------
    np.ndarray  Simulated image with the PID effect.
    """
    # Stationary operating point: z value where A(z) = ASP
    # Computed once, outside the loop
    arg = 1.0 - ASP / A0
    if arg <= 0:
        arg = 1e-9
    offset = z0 - np.log(arg) / beta

    def _amplitude(Zts):
        return A0 * (1.0 - np.exp(-beta * (Zts - z0)))

    rows, cols = image.shape
    simulated = np.zeros_like(image)

    if scan_direction not in ("forward", "backward"):
        raise ValueError("scan_direction must be 'forward' or 'backward'")

    col_indices = list(range(cols)) if scan_direction == "forward" else list(range(cols - 1, -1, -1))

    height_pid = 0.0  # initial tip height

    for i in range(rows):
        # Reset PID at the start of each line (physically correct behaviour)
        integral = 0.0
        previous_error = 0.0
        stuck_counter = 0

        for step, j in enumerate(col_indices):
            # ── Tip sticking ───────────────────────────────────────────
            if stuck_counter > 0:
                stuck_counter -= 1
                simulated[i, j] = height_pid   # keeps the last height
                continue

            if np.random.rand() < stickiness_prob:
                stuck_counter = np.random.randint(1, 5)
                simulated[i, j] = height_pid   # bug fixed: assigns height_pid
                continue

            # ── Abrupt jump ───────────────────────────────────────────────────
            if np.random.rand() < jump_prob:
                height_pid = max(
                    height_pid + np.random.uniform(-jump_magnitude, jump_magnitude),
                    image[i, j],
                )

            # ── Control PID ────────────────────────────────────────────────────
            Zts   = height_pid - image[i, j]
            A_val = _amplitude(Zts)
            error = ASP - A_val

            integral       += error * dt
            derivative      = (error - previous_error) / dt if step > 0 else 0.0
            adjustment      = Kp * error + Ki * integral + Kd * derivative

            if max_speed is not None:
                adjustment = np.clip(adjustment, -max_speed, max_speed)

            height_pid     += adjustment
            simulated[i, j] = height_pid
            previous_error  = error

    return simulated - offset
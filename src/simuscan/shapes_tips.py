
import numpy as np
import matplotlib.pyplot as plt
from . import shapes
import time
from functools import wraps
from scipy import ndimage as ndi
from .SyntheticParams import TipConfig

MAX_TIP_SIZE = 128  # or 64 to be more conservative

def limit_tip_size(tip, max_size=MAX_TIP_SIZE):
    h, w = tip.shape
    if h <= max_size and w <= max_size:
        return tip

    # centred crop
    cx, cy = h // 2, w // 2
    half = max_size // 2

    x0 = max(0, cx - half)
    x1 = x0 + max_size
    y0 = max(0, cy - half)
    y1 = y0 + max_size

    return tip[x0:x1, y0:y1]

def crop_zero_nan_border(tip, eps=1e-9):
    mask_valid = ~np.isnan(tip)
    # if everything is NaN, return as is
    if not mask_valid.any():
        return tip
    xs, ys = np.where(mask_valid)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return tip[x0:x1+1, y0:y1+1]

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"{fn.__name__} took {dt:.4f} s")
        return out
    return wrapper

def generate_elliptical_tip_image(rx=10, ry=5, rz=1, dx=0,dy=0,angle_deg=0, size = None):
    """
    Generates a 2D image of an ellipsoid rotated with the given radii.
    
    Args:
        rx, ry, rz: Ellipsoid radii in x, y, z.
        angle_deg: Rotation angle in degrees about the XY plane.
               
    Returns:
        A NumPy array representing the ellipsoid image.
    """
    # Make sure the image size is large enough
    if size == None:
        rmax = max(rx,ry)
        dmax = max(dx,dy)
        tam = dmax + 2*rmax + 1
    else:
        tam = size
        
    # Make sure tam is odd
    if tam % 2 == 0:
        tam += 1  # If even, add 1 to make it odd
    
    tam = int(tam)

    img,mask = shapes.generate_ellipsoid_image(size=tam, size_shape=(rx, ry),  height=rz, cx = int(tam/2)+dx, cy =int(tam/2)+dy, rotation=angle_deg)
    
    tip = abs(np.max(img)-img)
    return tip

def generate_double_elliptical_tip_image(rx1, ry1, rz1, rx2, ry2, rz2, angle_deg1,angle_deg2,dx=0,dy=0):
    """
    Generates a 2D image of two ellipsoids rotated with the given radii.
    
    Args:
        rx1, ry2, rz2: Ellipsoid radii in x, y, z.
        angle_deg: Rotation angle in degrees about the XY plane.
               
    Returns:
        A NumPy array representing the ellipsoid image.
    """
    rmax = max(rx1,ry1,rx2, ry2)
    dmax = max(dx,dy)
    
    tam = 2*dmax + 2*rmax + 1

    # Make sure tam is odd
    if tam % 2 == 0:
        tam += 1  # If even, add 1 to make it odd
    
    tam = int(tam)

    tip1 = generate_elliptical_tip_image(rx1, ry1, rz1,0,0, angle_deg1,size = tam)
    tip2 = generate_elliptical_tip_image(rx2, ry2, rz2,dx,dy, angle_deg2,size = tam)

    # Take the per-pixel maximum instead of summing
    tip = np.minimum(tip1, tip2)
    mask = np.ones_like(tip)
    tip2 = shapes.smooth_gaussian_inside_mask(tip, mask, sigma=1.0)
    tip = np.minimum(tip, tip2)
    return tip

def generate_double_esferical_tip_image(R1=20,R2=12,theta_deg=20,dx=10,dy=0):
    """
    Generates a 2D image of two ellipsoids rotated with the given radii.
    
    Args:
        rx1, ry2, rz2: Ellipsoid radii in x, y, z.
        angle_deg: Rotation angle in degrees about the XY plane.
               
    Returns:
        A NumPy array representing the ellipsoid image.
    """
    rmax = max(R1,R2)
    dmax = max(dx,dy)
    
    tam = 2*dmax + 2*rmax + 1

    # Make sure tam is odd
    if tam % 2 == 0:
        tam += 1  # If even, add 1 to make it odd
    
    tam = int(tam)

    tip1 = generate_esfericalcone_tip_image(R=R1,theta_deg=theta_deg,dx=0,dy=0,size = tam)
    tip2 = generate_esfericalcone_tip_image(R=R2,theta_deg=theta_deg,dx=dx,dy=dy,size = tam)

    # Take the minimum of the two images at each pixel
    tip = np.minimum(tip1, tip2)
    
    
    return tip

def generate_esfericalcone_tip_image(R=20,theta_deg=20,dx=0,dy=0,size = None):
    """
    Generates a 2D image, in polar coordinates, of a hemisphere joined to a cone.
    
    Parameters:
    - size: Image size in pixels (width and height).
    - R: Hemisphere radius.
    - theta_deg: Cone inclination angle in degrees.
    - dx, dy: Offset of the hemisphere centre from the origin.
    
    Returns:
    - img: NumPy array with intensity values normalised to [0,1].
    """
    # Make sure the image size is large enough
    if size == None:
        dmax = max(dx,dy)
        tam = 2*dmax + 4*R + 1
    else:
        tam = size

    tam = int(tam)  
    # Make sure tam is odd
    if tam % 2 == 0:
        tam += 1  # If even, add 1 to make it odd
    
    # Convert angle to radians
    theta = np.radians(theta_deg)
    phi = np.radians(90) - theta  # Angle of the line

    # Hemisphere centre
    y_c = R

    # Junction point between the hemisphere and the cone
    r0 = R * np.cos(theta)
    y0 = y_c - np.sqrt(R**2 - r0**2)

    # Create the coordinate grid
    scale = tam  # Image scale in real coordinates
    x_vals = np.linspace(-scale / 2, scale / 2,  tam)
    y_vals = np.linspace(-scale / 2, scale / 2,  tam)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Shift coordinates
    X_shifted = X - dx
    Y_shifted = Y - dy

    # Convert to polar coordinates relative to the new centre
    rho_shifted = np.sqrt(X_shifted**2 + Y_shifted**2)

    # Initialise the image with NaN
    tip = np.full(( tam,  tam), np.nan)

    # Assign values from the polar-coordinate function with offset
    for i in range(tam):
        for j in range(tam):
            r = rho_shifted[i, j]

            # If the point is inside the shifted hemisphere
            if r < R:
                tip[i, j] = y_c - np.sqrt(R**2 - r**2)
            # If the point is outside and in the shifted cone region
            elif r > r0:
                tip[i, j] = y0 + (r - r0) * np.tan(phi)

    # 1) crop NaN borders
    tip = crop_zero_nan_border(tip)
    # 2) limit the maximum size
    tip = limit_tip_size(tip, max_size=128)
    return tip

# Function that convolves the tip with a topography image
#@timed
def tip_convolution(topo,tip):
    # Matrix dimensions
    filas_topo, columnas_topo = topo.shape
    filas_punta, columnas_punta = tip.shape

    # Create the output matrix
    resultado = np.zeros_like(topo)

    # Offset to centre the tip matrix
    offset_x = filas_punta // 2
    offset_y = columnas_punta // 2

    # Loop over every pixel of the topography matrix
    for i in range(filas_topo):
        for j in range(columnas_topo):
            # Determine the overlapping regions
            x_inicio = max(0, i - offset_x)
            x_fin = min(filas_topo, i + offset_x + 1)
            y_inicio = max(0, j - offset_y)
            y_fin = min(columnas_topo, j + offset_y + 1)

            # Corresponding coordinates in the tip matrix
            px_inicio = max(0, offset_x - i)
            px_fin = px_inicio + (x_fin - x_inicio)
            py_inicio = max(0, offset_y - j)
            py_fin = py_inicio + (y_fin - y_inicio)

            # Get the overlapping regions
            region_topo = topo[x_inicio:x_fin, y_inicio:y_fin]
            region_punta =tip[px_inicio:px_fin, py_inicio:py_fin]

            # Compute the differences
            diferencias = region_topo - region_punta

            # Assign the minimum value to the output matrix
            resultado[i, j] = np.max(diferencias)

    return resultado

#@timed
def tip_convolution_roi_fast(topo, tip, thresh=None):
    topo = topo.astype(np.float32, copy=False)
    tip  = tip.astype(np.float32,  copy=False)

    H, W = topo.shape
    kh, kw = tip.shape
    ox, oy = kh // 2, kw // 2

    if thresh is None:
        # Use the WHOLE image as ROI
        x0, x1 = 0, H-1
        y0, y1 = 0, W-1
    else:
        # Threshold-based ROI, as before
        xs, ys = np.where(topo > thresh)
        if xs.size == 0:
            return np.zeros_like(topo, dtype=topo.dtype)
        x0, x1 = max(0, xs.min()-ox), min(H-1, xs.max()+ox)
        y0, y1 = max(0, ys.min()-oy), min(W-1, ys.max()+oy)

    roi = topo[x0:x1+1, y0:y1+1]

    out_roi = ndi.grey_dilation(roi, structure=-tip, mode='constant', cval=-np.inf)

    out = np.zeros_like(topo, dtype=topo.dtype)
    out[x0:x1+1, y0:y1+1] = out_roi
    return out

def rand_tip(tip: TipConfig, seed=None):
    """
    Generates a single or double tip.
    Return compatible: (tip, radius_tip, centre_r2, theta, is_double)
      - tip: generated tip image
      - radius_tip: [r1, r2] (r2=0 for a single tip)
      - centre_r2: [dx, dy] ([0,0] for a single tip)
      - theta: angle (deg) used to generate the tip(s)
      - is_double: bool
    """
    if seed is not None:
        np.random.seed(seed)

    # Angle for the tip geometry in degrees (float, not array)
    angle = float(np.random.uniform(tip.theta_range[0], tip.theta_range[1]))

    # Double tip only if enabled AND drawn by its probability.
    # (The 'and' short-circuits: if is_double is False no random number is consumed.)
    want_double = bool(tip.is_double) and (float(np.random.random()) < float(tip.double_probability))

    if not want_double:
        # ---- SINGLE TIP ----
        r1 = random_log_distribution(tip.radi_range)
        tip = generate_esfericalcone_tip_image(R=r1, theta_deg=angle)

        radius_tip = [float(r1), 0.0]
        centre_r2 = [0.0, 0.0]
        theta = angle
        is_double = False
        return tip, radius_tip, centre_r2, theta, is_double

    else:
        # ---- DOUBLE TIP ----
        r1 = float(random_log_distribution(tip.radi_range))

        # range for r2: between the minimum of the original range and r1
        rmin = float(min(tip.radi_range))
        
        r2 = float(random_log_distribution([rmin, r1]))

        # Distance d between r1/3 and 1.5*r1
        d_low = max(r1 *tip.distance_min_pr1, 0.0)
        d_high = tip.distance_max_pr1 * r1
        d = float(np.random.uniform(d_low, d_high))

        # Random direction of the second tip's offset
        theta_pos = float(np.random.uniform(0.0, 2.0 * np.pi))
        dx = d * np.cos(theta_pos)
        dy = d * np.sin(theta_pos)

        tip = generate_double_esferical_tip_image(R1=r1, R2=r2, theta_deg=angle, dx=dx, dy=dy)

        radius_tip = [r1, r2]
        centre_r2 = [dx, dy]
        theta = angle
        is_double = True
        return tip, radius_tip, centre_r2, theta, is_double

# def rand_tip(rrang = [5,35], thetarang = [10,25], seed=None):

#     if seed is not None:
#         np.random.seed(seed)

#     r= random_log_distribution(rrang)
#     angle = np.random.uniform(thetarang[0],thetarang[1],size = 1)[0]         
#     tip = generate_esfericalcone_tip_image(R=r,theta_deg=angle)
    
#     radius_tip = [r,0]
#     centre_r2 = [0,0] 
#     theta= angle
#     is_double=False

#     return tip, radius_tip,centre_r2,theta,is_double

# def rand_double_tip(rrang = [5,35],  thetarang = [10,25],  seed=None):

#     if seed is not None:
#         np.random.seed(seed)
#     # Generates an image of two elliptical tips, making sure they are distinguishable according to

#     r1 = random_log_distribution(rrang)
#     r2 = random_log_distribution(rrang)
#     angle = np.random.uniform(thetarang[0],thetarang[1],size = 1)     

#     # Generate a distance within the allowed range
#     d = np.random.uniform(5,5)

#     # Generate a random angle
#     theta = np.random.uniform(0, 2 * np.pi)  # En radianes

#     # Compute dx and dy from d and theta
#     dx = d * np.cos(theta)
#     dy = d * np.sin(theta)
   
#     tip= generate_double_esferical_tip_image(R1=r1,R2=r2,theta_deg=angle,dx=dx,dy=dy)
    
#     parm = (r1,r2,angle,dx,dy)

#     return tip, parm

def random_log_distribution(rrange=(1, 35)):
    log_min = np.log(rrange[0])
    log_max = np.log(rrange[1])
    r = np.random.uniform(log_min, log_max)
    return np.exp(r)

if __name__ == '__main__':


    rect,_ =rand_tip(rrang = [30,35], thetarang = [24,25], seed=None)

    # Extract the central profile
    profile_x = rect[ rect.shape[1] // 2 ,:]
   
    # Visualisation
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.imshow(rect, cmap='viridis')
    plt.title('Original matrix')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.plot( np.arange(rect.shape[0]), profile_x, color='r')
    #plt.gca().invert_yaxis()
    plt.title('Central profile')
    plt.xlabel('Altura')
    plt.ylabel('Fila')
    plt.show()
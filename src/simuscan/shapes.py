import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import cv2
import scipy.ndimage
from scipy.interpolate import splprep, splev, griddata
from skimage.draw import line

# Formas planas
def generate_rectangle_image(size = 256,size_shape=(50,50), height = 1.0, cx = 128, cy =128, rotation = 0):

    img = np.zeros((size,size), dtype=np.float32)
    h, w = size, size

    # Create a coordinate grid
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    # Coordinates relative to the centre
    Xr = X - cx
    Yr = Y - cy

    # Apply rotation if needed
    angle_rad = np.radians(rotation)
    X_rot = Xr * np.cos(angle_rad) - Yr * np.sin(angle_rad)
    Y_rot = Xr * np.sin(angle_rad) + Yr * np.cos(angle_rad)

    # Initialize mask
    mask = np.zeros_like(img, dtype=np.uint8)

    mask = (np.abs(X_rot) <= size_shape[0]/2) & (np.abs(Y_rot) <= size_shape[1]/2)
    
    img[mask] = height
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_ellipse_image(size = 256,size_shape=(50,25), height = 1.0, cx = 128, cy =128, rotation = 0):

    img = np.zeros((size,size), dtype=np.float32)
    h, w = size, size
    
    # Create a coordinate grid
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    # Coordinates relative to the centre
    Xr = X - cx
    Yr = Y - cy

    # Apply rotation if needed
    angle_rad = np.radians(rotation)
    X_rot = Xr * np.cos(angle_rad) - Yr * np.sin(angle_rad)
    Y_rot = Xr * np.sin(angle_rad) + Yr * np.cos(angle_rad)

    # Initialize mask
    mask = np.zeros_like(img, dtype=np.uint8)
    a = size_shape[0]/2
    b = size_shape[1]/2
    mask = ((X_rot / a)**2 + (Y_rot / b)**2) <= 1
    
    img[mask] = height
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_polygon_image(size = 256,size_shape=(50,25), height = 1.0, cx = 128, cy =128, rotation = 0, num_sides=5):

    img = np.zeros((size,size), dtype=np.float32)
    
    # Apply rotation if needed
    angle_rad = np.radians(rotation)
    
    # Initialize mask
    mask = np.zeros_like(img, dtype=np.uint8)

    angles = np.linspace(0, 2 * np.pi, num_sides + 1)[:-1]

    # Generate polygon points
    ax= size_shape[0]/2
    ay = size_shape[1]/2
    pts = np.array([[ax * np.cos(a), ay * np.sin(a)] for a in angles])
    pts = np.dot(pts, np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])) + np.array([cx, cy])

    # Create the polygon mask
    mask = cv2.fillPoly(np.zeros_like(img, dtype=np.uint8), [pts.astype(np.int32)], 1).astype(bool)

    img[mask] = height
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_spline_image(size=256, size_shape=(50, 25), height=1.0, cx=128, cy=128, rotation=0, num_points=10, variation=0.1, smoothness=0):
    """
    Generates an image with a contour defined by a random closed spline within the specified area.
    
    :param size: Image size (size x size)
    :param size_shape: Approximate contour size (width, height)
    :param height: Height value in the image inside the contour
    :param cx: x coordinate of the contour centre
    :param cy: y coordinate of the contour centre
    :param rotation: Contour rotation in degrees
    :param num_points: Number of base points for the spline
    :param variation: Percentage of variation in the spline points relative to the shape size
    :param smoothness: Spline smoothing parameter (0 to pass through the points)
    :return: Generated image and contour mask
    """
    img = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros_like(img, dtype=np.uint8)
    
    # Generate random points within the specified area
    ax= size_shape[0]/2
    ay = size_shape[1]/2
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    desprnd = variation * sum(size_shape) / len(size_shape)
    random_offsets = np.random.uniform(-desprnd, desprnd, (num_points, 2))  # Small variations to make the spline random
    pts = np.array([[ax * np.cos(a), ay * np.sin(a)] for a in angles]) + random_offsets
    
    # Apply rotation
    angle_rad = np.radians(rotation)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    pts = np.dot(pts, rotation_matrix) + np.array([cx, cy])
    
    # Closed spline interpolation
    tck, u = splprep(pts.T, per=True, s=smoothness)
    u_fine = np.linspace(0, 1, 100)
    spline_pts = np.array(splev(u_fine, tck)).T
    
    # Draw the contour
    mask = cv2.fillPoly(np.zeros_like(img, dtype=np.uint8), [spline_pts.astype(np.int32)], 1).astype(bool)
    img[mask] = height
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_oval_image(size=256, size_shape=(50, 25),  height=1.0, cx = 128, cy =128, rotation=0, p_width=0.1):
    """
    Generates an image with an extended oval (half-ellipses at the ends and a central rectangle).

    Parameters:
    - size: Size of the square image (pixels).
    - size_shape: Oval size (total width, total height).
    - p_width: Percentage of the total width taken by the central rectangular region.
    - height: Relief height inside the oval.
    - rotation: Rotation angle in degrees.

    Retorna:
    - img: Image with the generated shape and applied height.
    - mask: Binary mask of the generated shape.
    """

    # Create empty image and mask
    img = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Image centre
    rect_width = int(p_width * size_shape[0])  # Width of the central rectangle
    axis_x = (size_shape[0]-rect_width) // 2  # Semi-major axis (horizontal radius of the ends)
    axis_y = size_shape[1] // 2  # Semieje menor (radio vertical)
    
    # 1. Draw the central rectangle
    rect_x1, rect_x2 = cx - rect_width // 2, cx + rect_width // 2
    rect_y1, rect_y2 = cy - axis_y, cy + axis_y
    cv2.rectangle(mask, (rect_x1, rect_y1), (rect_x2, rect_y2), 1, -1)

    # 2. Draw the half-ellipses at the ends
    cv2.ellipse(mask, (rect_x1, cy), (axis_x, axis_y), 0, 90, 270, 1, -1)
    cv2.ellipse(mask, (rect_x2, cy), (axis_x, axis_y), 0, 270, 470, 1, -1)

    # 3. Apply rotation if needed
    if rotation != 0:
        M = cv2.getRotationMatrix2D((cx, cy), rotation, 1.0)
        mask = cv2.warpAffine(mask, M, (size, size),flags=cv2.INTER_NEAREST)

    # 4. Apply the height inside the mask
    img[mask > 0] = height
    mask = (mask > 0).astype(np.uint8) * 255
    return img, mask

# Formas 3D
def generate_ellipsoid_image(size=256, size_shape=(50, 25),  height=1.0, cx = 128, cy =128, rotation=0):
    """
    Generates an image with an ellipsoid of variable height.
    
    :param size: Image size (size x size)
    :param size_shape: Base ellipse size (width, height)
    :param height: Maximum ellipsoid height
    :param cx: x coordinate of the ellipse centre
    :param cy: y coordinate of the ellipse centre
    :param rotation: Ellipse rotation in degrees
    :return: Generated image with the ellipsoid and mask of the occupied region
    """
    # Create empty image and mask
    img = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Create coordinate grid
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # Coordinates relative to the centre
    Xr = X - cx
    Yr = Y - cy
    
    # Apply rotation
    angle_rad = np.radians(rotation)
    X_rot = Xr * np.cos(angle_rad) - Yr * np.sin(angle_rad)
    Y_rot = Xr * np.sin(angle_rad) + Yr * np.cos(angle_rad)
    
    # Define ellipsoid of variable height
    ax= size_shape[0]/2
    ay = size_shape[1]/2
    ellipse_eq = (X_rot / ax)**2 + (Y_rot / ay)**2
    mask = ellipse_eq <= 1
    
    # Assign height in ellipsoid form
    img[mask] = height * (1 - ellipse_eq[mask])
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_half_ellipsoid_image(size=256, size_shape=(50, 25), height=1.0, cx=128, cy=128, rotation=0):
    """
    Generates an image with a half-ellipsoid of variable height.

    :param size: Image size (size x size)
    :param size_shape: Base ellipse size (width, height)
    :param height: Maximum ellipsoid height
    :param cx: x coordinate of the ellipse centre
    :param cy: y coordinate of the ellipse centre
    :param rotation: Ellipse rotation in degrees
    :return: Generated image with the half-ellipsoid and mask of the occupied region
    """
    # Create empty image and mask
    img = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Create coordinate grid
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # Coordinates relative to the centre
    Xr = X - cx
    Yr = Y - cy

    # Apply rotation
    angle_rad = np.radians(rotation)
    X_rot = Xr * np.cos(angle_rad) - Yr * np.sin(angle_rad)
    Y_rot = Xr * np.sin(angle_rad) + Yr * np.cos(angle_rad)

    # Define the ellipse equation
    ax= size_shape[0]
    ay = size_shape[1]/2
    ellipse_eq = (X_rot / ax)**2 + (Y_rot / ay)**2
    mask = (ellipse_eq <= 1) & (X_rot >= 0)  # Keep only the upper half

    # Assign height in half-ellipsoid form
    img[mask] = height * (1 - ellipse_eq[mask])
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_elliptical_cylinder_image(size=256, size_shape=(100, 50), height=1.0, cx=128, cy=128, rotation=0):
    """
    Generates an image with an elliptical cylinder lying on its side.
    
    Parameters:
    - size: Size of the square image (size x size).
    - size_shape: (cylinder length, ellipse width).
    - height: Maximum ellipse height.
    - cx, cy: Cylinder centre in the image.
    - rotation: Cylinder rotation in degrees.
    
    Retorna:
    - NumPy array with the cylinder image.
    """
    # Create empty image and mask
    img = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Extract parameters
    longitud, ancho = size_shape
    
    # Create a coordinate grid
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    
    # Centre the coordinates at (cx, cy)
    x_rel = x - cx
    y_rel = y - cy
    
    # Apply rotation to the coordinates
    angle_rad = np.radians(rotation)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    x_rot = cos_theta * x_rel - sin_theta * y_rel
    y_rot = sin_theta * x_rel + cos_theta * y_rel
    
    # Define the ellipse in the new basis
    mask = (np.abs(x_rot) <= longitud / 2) & ((y_rot**2 / (ancho / 2)**2) <= 1)
    
    # Compute the intensity based on the ellipse
    elipse_valores = np.maximum(0, height * (1 - (y_rot**2 / (ancho / 2)**2)))
    
    # Apply the intensity only inside the mask
    img[mask] = elipse_valores[mask]
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_ovaloid_image(size=256, size_shape=(100, 100), height=1.0, cx=128, cy=128, rotation=0, p_width=0.3):
    """
    Generates an image with a three-dimensional ovaloid with a central cylindrical region and two half-ellipsoids at the ends.

    :param size: Size of the square image (pixels)
    :param size_shape: Ovaloid size (total width, total height)
    :param height: Maximum ovaloid height
    :param cx: x coordinate of the shape centre
    :param cy: y coordinate of the shape centre
    :param rotation: Rotation angle in degrees
    :param p_width: Percentage of the total width taken by the central cylindrical region
    :return: Image with the generated shape and binary mask of the generated shape
    """
    # Create empty image and mask
    img = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Image centre
    w = p_width * size_shape[0]  # Width of the central rectangle
    axis_x = (size_shape[0]-w)/2   # Semi-major axis (horizontal radius of the ends)
    axis_y = size_shape[1]   # Semieje menor (radio vertical)

    alfa = rotation
    alfa_r = np.radians(rotation)
    
    rect1, mask1 =generate_half_ellipsoid_image(size=size, size_shape=(axis_x, axis_y), 
                                                height=height, cx=cx+0.5*w*np.cos(alfa_r), cy=cy-0.5*w*np.sin(alfa_r), rotation=alfa)
    rect2, mask2 =generate_half_ellipsoid_image(size=size, size_shape=(axis_x, axis_y), 
                                                height=height, cx=cx-0.5*w*np.cos(alfa_r), cy=cy+0.5*w*np.sin(alfa_r), rotation=180+alfa)
    cil,mask3 = generate_elliptical_cylinder_image(size=size, size_shape=(w, axis_y), height=height, cx=cx, cy=cy, rotation=alfa)
    img = np.maximum(np.maximum(rect1, rect2), cil)
    mask = np.maximum(np.maximum(mask1, mask2), mask3)
    
    # not needed because it is already 0-255 mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_Lineal_DNA_image(size=256, size_shape=(100, 1), height=1.0, cx = 128, cy=128, max_angulo=30):
    """
    Generates a zero matrix and places a path defined by x, y points.
    It also marks the pixels between consecutive points using skimage.

    Parameters:
    - size=256 : Dimensions of the matrix.
    - size_shape : Length of the linear DNA (number of steps in px) and DNA width
    - height=1.0 : DNA height and radius
    - cx = 128 : x coordinate of the shape centre
    - cy = 128: y coordinate of the shape centre
    - max_angulo = 30 : Maximum deviation angle for the random walker
    
    """
    puntos_x, puntos_y = puntos_Lineal_DNA_random_walk(n_puntos=size_shape[0], paso=1.0, max_angulo=np.radians(max_angulo), cerrado=False)
    if len(puntos_x) != len(puntos_y):
        raise ValueError("The x and y point lists must have the same length.")
    # Create empty image and mask
    img = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Process the path points
    for i in range(len(puntos_x) - 1):
        # Start and end coordinates of the segment
        x0 = int(round(puntos_x[i] + cx))
        y0 = int(round(puntos_y[i] + cy))
        x1 = int(round(puntos_x[i + 1] + cx))
        y1 = int(round(puntos_y[i + 1] + cy))

        # Get the indices of the pixels that make up the line
        rr, cc = line(x0, y0, x1, y1)

        # Mark the points in the matrix, if they are within bounds
        valid = (0 <= rr) & (rr < size) & (0 <= cc) & (cc < size)
        mask[rr[valid], cc[valid]] = 1
    
    # if size_shape[1] > 1:      
    #     # Create a circular kernel
    #     radius = size_shape[1]-1  # Circle radius
    #     kernel_size = 2 * radius + 1
    #     kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    #     cv2.circle(kernel, (radius, radius), radius, 1, -1)  # Solid circle

    #     # Apply the dilation
    #     mask = cv2.dilate(mask, kernel, iterations=1)
    
    img[mask==1]=height
    img = adjust_height_map(img, mask, dh=1.0, order=0.1)

    #dilate the mask for better segmentation
    # Create a circular kernel
    # radius = 1  # Circle radius
    # kernel_size = 2 * radius + 1
    # kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    # cv2.circle(kernel, (radius, radius), radius, 1, -1)  # Solid circle

    # # Apply the dilation
    # mask = cv2.dilate(mask, kernel, iterations=1)
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_circular_DNA_image(size=256, size_shape=(100, 1), height=1.0, cx = 128, cy=128, p_points_to_move=0.2, dmax = 2):
    """
    Generates a zero matrix and places a path defined by x, y points.
    It also marks the pixels between consecutive points using skimage.

    Parameters:
    - size=256 : Dimensions of the matrix.
    - size_shape : Length of the linear DNA (number of steps in px) and DNA width
    - height=1.0 : DNA height and radius
    - cx = 128 : x coordinate of the shape centre
    - cy = 128: y coordinate of the shape centre
    - p_points_to_move (0-1): percentage of points that will be displaced perpendicularly and randomly (equidistant)
    - dmax: normal displacement distance
    
    """
    puntos_x, puntos_y = puntos_circular_DNA(n_points=5, rango=size_shape[0]/2, p_evaluacion=max(0.5, p_points_to_move), p_points_to_move = p_points_to_move, dmax = dmax)
    if len(puntos_x) != len(puntos_y):
        raise ValueError("The x and y point lists must have the same length.")
    # Create empty image and mask
    img = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Process the path points
    for i in range(len(puntos_x) - 1):
        # Start and end coordinates of the segment
        x0 = int(round(puntos_x[i] + cx))
        y0 = int(round(puntos_y[i] + cy))
        x1 = int(round(puntos_x[i + 1] + cx))
        y1 = int(round(puntos_y[i + 1] + cy))

        # Get the indices of the pixels that make up the line
        rr, cc = line(x0, y0, x1, y1)

        # Mark the points in the matrix, if they are within bounds
        valid = (0 <= rr) & (rr < size) & (0 <= cc) & (cc < size)
        mask[rr[valid], cc[valid]] = 1

    if size_shape[1] > 1:   
        # Create a circular kernel
        radius = size_shape[1]-1  # Circle radius
        kernel_size = 2 * radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        cv2.circle(kernel, (radius, radius), radius, 1, -1)  # Solid circle

        # Apply the dilation
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    img[mask==1]=height
    img = adjust_height_map(img, mask, dh=1.0, order=0.1)

    #dilate the mask for better segmentation
    # Create a circular kernel
    radius = 1  # Circle radius
    kernel_size = 2 * radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)  # Solid circle

    # Apply the dilation
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = (mask * 255).astype(np.uint8)
    return img, mask

def generate_surface_image(size=256, size_shape=(50, 25), height_min=0, height_max=1.0, cx=128, cy=128, num_puntos=10):
    """
    Generates an image with a shape bounded by a closed spline, interpolating a surface inside it.
    
    :param size: Image size (size x size)
    :param size_shape: Approximate shape size (width, height)
    :param height_min: Minimum height of the generated points
    :param height_max: Maximum height of the generated points
    :param cx: x coordinate of the shape centre
    :param cy: y coordinate of the shape centre
    :param num_puntos: Number of random points to generate inside the shape
    :return: Generated image with the interpolated surface and mask of the occupied region
    """
    # Create empty image and mask
    img = np.zeros((size, size), dtype=np.float32)

    # Generate base points for the spline within the range
    x_random = np.random.uniform(cx - size_shape[0], cx + size_shape[0], 4)
    y_random = np.random.uniform(cy - size_shape[1], cy + size_shape[1], 4)

    # Create the closed spline
    tck, u = splprep([x_random, y_random], s=0, per=True)

    # Evaluate the spline at more points to define the contour
    u_fine = np.linspace(0, 1,125)
    x_spline, y_spline = splev(u_fine, tck)

    # Create a polygon from the spline
    polygon = Path(np.column_stack((x_spline, y_spline)))
    grid_spacing= int(size_shape[0]/num_puntos)
    
    # Generate random points and keep only those inside the polygon
    dismin = min(size_shape) /(num_puntos-1)
    x, y, z = [], [], []
    
    max_attempts = num_puntos * 10  # Attempt limit to avoid infinite loops
    attempts = 0

    while len(x) < num_puntos and attempts < max_attempts:
        xt = np.random.uniform(cx - size_shape[0], cx + size_shape[0])
        yt = np.random.uniform(cy - size_shape[1], cy + size_shape[1])
        if polygon.contains_point((xt, yt)):  # Make sure it is inside the contour
            # Check distance to the already accepted points
            if all(np.sqrt((xt - xp)**2 + (yt - yp)**2) >= dismin for xp, yp in zip(x, y)):
                x.append(xt)
                y.append(yt)
                z.append(np.random.uniform(height_min, height_max))
        
        attempts += 1


    # Add the spline points to the interpolation for a better fit
    x = np.array(x + list(x_spline))
    y = np.array(y + list(y_spline))
    z = np.array(z + [height_min] * len(x_spline))  # Set the edges to height 0

    # Create the interpolation grid
    grid_x, grid_y = np.meshgrid(np.linspace(0, size - 1, size), np.linspace(0, size - 1, size))

    # Interpolate the values on the grid
    img = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Fill NaN values with zeros
    img = np.nan_to_num(img)

    # Create the binary mask of the shape
    mask = polygon.contains_points(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    mask = mask.reshape((size, size))

    # Apply the mask to the interpolated image
    img[~mask] = 0
    mask = (mask * 255).astype(np.uint8)
    img = height_max* normalize_matrix(img)
    return img, mask

# Funciones varias

#function that normalizes a matrix
def normalize_matrix(matrix):
    """
    Normalizes a matrix to the range [0, 1].
    
    Args:
        matrix (np.ndarray): Input matrix.
    
    Returns:
        np.ndarray: Matriz normalizada.
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val == 0:  # Avoid division by zero
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)
def combinar_imagenes_con_maximo(img1, img2, desplazamiento=(0, 0)):
    """
    Combines two images using the per-pixel maximum, placing img2 centred on img1 when the offset is (0, 0).
    
    :param img1: First image (numpy array, larger size)
    :param img2: Second image (numpy array, smaller or equal size)
    :param desplazamiento: Tuple (dx, dy) indicating how much to shift img2 relative to the centre of img1
    :return: Combined image taking the maximum value at each pixel
    """
    # Get the image dimensions
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # Compute the position where img2 will be placed in img1 when there is no offset
    cx1, cy1 = w1 // 2, h1 // 2  # Centre of img1
    cx2, cy2 = w2 // 2, h2 // 2  # Centre of img2

    # Apply offset
    x_min = cx1 - cx2 + desplazamiento[0]
    y_min = cy1 - cy2 + desplazamiento[1]
    x_max = x_min + w2
    y_max = y_min + h2

    # Adjust the indices to avoid overflow
    x_min_img2 = max(0, -x_min)
    y_min_img2 = max(0, -y_min)
    x_max_img2 = w2 - max(0, x_max - w1)
    y_max_img2 = h2 - max(0, y_max - h1)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w1, x_max)
    y_max = min(h1, y_max)

    # Create a zero image the size of img1 to insert img2
    img2_desplazada = np.zeros_like(img1)

    # Insert img2 only in the valid region
    img2_desplazada[y_min:y_max, x_min:x_max] = img2[y_min_img2:y_max_img2, x_min_img2:x_max_img2]

    # Take the maximum value per pixel instead of summing them
    img_resultante = np.maximum(img1, img2_desplazada)

    return img_resultante

def smooth_gaussian_inside_mask(image, mask, sigma=1.0):
    """
    Applies smoothing inside a mask without affecting values outside the mask.

    Parameters:
    - image: NumPy array with the image to smooth.
    - mask: Binary matrix (1 in the region to smooth, 0 outside).
    - sigma: Standard deviation of the Gaussian filter (larger sigma = more smoothing).

    Retorna:
    - Array with the smoothed image inside the mask.
    """

    # Smooth the whole image with a Gaussian filter
    smoothed_image = scipy.ndimage.gaussian_filter(image, sigma=sigma)

    # Keep the original values outside the mask
    result = np.where(mask, smoothed_image, image)

    return result

def smooth_media_inside_mask(image, mask, kernel_size=3):
    """
    Applies mean-filter smoothing inside a mask without affecting values outside the mask.

    Parameters:
    - image: NumPy array with the image to smooth.
    - mask: Binary matrix (1 in the region to smooth, 0 outside).
    - kernel_size: Mean filter size (number of pixels in each dimension).

    Retorna:
    - Array with the smoothed image inside the mask.
    """

    # Apply a mean filter (uniform_filter) over the whole image
    smoothed_image = scipy.ndimage.uniform_filter(image, size=kernel_size)

    # Keep the original values outside the mask
    result = np.where(mask, smoothed_image, image)

    return result

def puntos_Lineal_DNA_random_walk(n_puntos=100, paso=1.0, max_angulo=np.pi/4, cerrado=False):
    """
    Generates a random shape using a 2D random walk with a limited maximum angle.

    Args:
        n_puntos (int): Number of points of the random shape.
        paso (float): Average step length.
        max_angulo (float): Maximum allowed angle between consecutive steps (in radians).
        cerrado (bool): If True, connects the last point to the first to close the shape.

    Returns:
        np.ndarray, np.ndarray: x and y coordinates of the generated shape.
    """
    # Initialize coordinate arrays
    x = np.zeros(n_puntos)
    y = np.zeros(n_puntos)

    # Initialize the initial direction (random angle)
    angulo_actual = np.random.uniform(0, 2 * np.pi)

    for i in range(1, n_puntos):
        # Generate a new angle within the allowed range
        delta_angulo = np.random.uniform(-max_angulo, max_angulo)
        angulo_actual += delta_angulo

        # Compute displacement based on the current angle
        dx = paso * np.cos(angulo_actual)
        dy = paso * np.sin(angulo_actual)

        # Update coordinates based on the previous point
        x[i] = x[i - 1] + dx
        y[i] = y[i - 1] + dy

    # If the shape is closed, connect the last point to the first
    if cerrado:
        x[-1], y[-1] = x[0], y[0]

    return x, y

def puntos_circular_DNA(n_points=4, rango=100, p_evaluacion=0.3, p_points_to_move = 0.2, dmax = 10):
    """
    Generates a closed spline with random points that do not approximate a straight line,
    making sure they form "wide" shapes.

    Args:
        n_points (int): Number of random points for the spline.
        rango (fload): Range of the random points (minimum, maximum).
        p_evaluacion (0-1): percentage of the points that will be evaluated
        p_points_to_move (0-1): percentage of points that will be displaced perpendicularly and randomly (equidistant)
        dmax: normal displacement distance

    Returns:
        np.ndarray: x coordinates of the spline.
        np.ndarray: y coordinates of the spline.
    """
    # evaluated at a percentage of the points
    puntos_evaluacion = max(25,int(p_evaluacion*rango))
    n_points_to_move = max(5,int(p_points_to_move*rango))

    # Generate random points in the given range
    x_random = np.random.uniform(-rango, rango, n_points)
    y_random = np.random.uniform(-rango, rango, n_points)
    ## Create the closed spline
    tck, u = splprep([x_random, y_random], s=0, per=True)

    # Evaluate the spline at puntos_evaluacion
    u_fine = np.linspace(0, 1, puntos_evaluacion)
    x_spline, y_spline = splev(u_fine, tck)

    # Compute derivatives to obtain the spline direction
    dx, dy = splev(u_fine, tck, der=1)

    # Compute the perpendicular direction
    norms = np.sqrt(dx**2 + dy**2)
    perp_dx = -dy / norms
    perp_dy = dx / norms

    # Select equidistant indices
    #n_points_to_move = 50
    indices = np.linspace(0, len(x_spline) - 1, n_points_to_move, dtype=int)

    # Generate random offsets
    random_distances = np.random.uniform(-dmax, dmax, size=n_points_to_move)
    #random_distances = np.random.normal(0,1,size=num_points_to_move)
    # Apply offsets
    x_moved = np.copy(x_spline)
    y_moved = np.copy(y_spline)

    for i, idx in enumerate(indices):
        x_moved[idx] += random_distances[i] * perp_dx[idx]
        y_moved[idx] += random_distances[i] * perp_dy[idx]

    # Select only the modified points
    x_selected = x_moved[indices]
    y_selected = y_moved[indices]

    # Create a new spline from the modified points
    new_tck, new_u = splprep([x_selected, y_selected], s=0,per=True)
    x_new_spline, y_new_spline = splev(np.linspace(0, 1, 500), new_tck)

    return x_new_spline, y_new_spline

def adjust_height_map(height_map, mask, dh=1.0, order=1):
    """
    Adjusts the height at the edges of a mask, reducing it by a percentage dh with order control.

    Parameters:
    - height_map: NumPy array with the original heights inside the mask.
    - mask: Binary matrix (1 in the shape, 0 outside the shape).
    - dh: Height difference at the edges (0 = no change, 1 = zero height at the edges).
    - order: Decay order (1 = linear, 2 = quadratic, 3 = cubic, etc.).

    Retorna:
    - height_map adjusted with a smooth transition at the edges according to the specified order.
    """

    # Compute the distance to the edges inside the mask
    distance = scipy.ndimage.distance_transform_edt(mask).astype(np.float32)

    # Normalize the distance (0 at the edge, 1 at the centre)
    max_distance = np.max(distance)
    if max_distance > 0:
        normalized_distance = distance / max_distance
    else:
        normalized_distance = np.ones_like(distance)  # Avoid division by zero if there are no edges

    # Apply the decay function according to the specified order
    height_factor = (1 - dh * (1 - normalized_distance ** order))

    # Adjust the surface height
    adjusted_height_map = height_map * height_factor

    return adjusted_height_map


if __name__ == '__main__':


    # rect =generate_double_esferical_tip_image(R1=12,R2=12,theta_deg=20,dx=10,dy=10)

    # # Extract the central profile
    # profile_x = rect[ rect.shape[1] // 2 ,:]
   
    # # Visualization
    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 2, 1)
    # plt.imshow(rect, cmap='viridis')
    # plt.title('Original Matrix')
    # plt.colorbar()

    # plt.subplot(1, 2, 2)
    # plt.plot( np.arange(rect.shape[0]), profile_x, color='r')
    # #plt.gca().invert_yaxis()
    # plt.title('Central Profile')
    # plt.xlabel('Height')
    # plt.ylabel('Row')
    # plt.show()
    rect, mask =generate_ovaloid_image(size=256, size_shape=(500, 150), height=1.0, cx=128, cy=128, rotation=90, p_width=0.3)

    # tip = generate_elliptical_tip_image(rx=5, ry=5, rz=1, dx=0,dy=0,angle_deg=0, size = None)
    # #tip = generate_elliptical_tip_image(rx=10, ry=5, rz=1, dx=0,dy=0,angle_deg=0, size =25)
    # topo = tip_convolution(rect,tip)
    # #rect = adjust_height_map(rect, mask, dh=1.0, order=0.5)
    # #rect = smooth_gaussian_inside_mask(rect, mask, sigma=1)

    # Extract the central profile
    profile_x = rect[ -1+rect.shape[1] // 2 ,:]
     # Extract the central profile
    #profile_topox = tip[ -1+tip.shape[1] // 2 ,:]
    # Visualization
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 3, 1)
    plt.imshow(rect, cmap='viridis')
    plt.title('Original Matrix')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(mask, cmap='viridis')
    plt.title('mask')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.plot( np.arange(rect.shape[0]), profile_x, color='r')
    #plt.gca().invert_yaxis()
    plt.title('Central Profile')
    plt.xlabel('Height')
    plt.ylabel('Row')
    
    # plt.subplot(2, 3, 4)
    # plt.imshow(topo, cmap='viridis')
    # plt.title('topo')
    # plt.colorbar()

    # plt.subplot(2, 3, 5)
    # plt.imshow(tip, cmap='viridis')
    # plt.title('tip')
    # plt.colorbar()

    # plt.subplot(2, 3, 6)
    # plt.plot( np.arange(tip.shape[0]), profile_topox, color='r')
    # #plt.gca().invert_yaxis()
    # plt.title('Central Profile')
    # plt.xlabel('Height')
    # plt.ylabel('Row')

    plt.show()

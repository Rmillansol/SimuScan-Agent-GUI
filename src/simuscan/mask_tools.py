import numpy as np
import scipy.ndimage
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import rotate
import cv2

def mask_to_height_map(mask, height_max = 1.0,  order=2.0, dh0=0.0):
    """
    Adjusts the height at the edges of a mask, reducing its height by a percentage dh with order control.

    Parameters:
    - mask: mask to convert into a height image
    - height_max: Maximum height of the image.
    - dh: Height difference at the edges (0 = no change, 1 = zero height at the edges). (allows making a step)
    - order: Decay order (1 = linear, 2 = quadratic, 3 = cubic, etc.).

    Returns:
    - height_map: height image with the given decay
    """
    height_map = np.zeros_like(mask, dtype=np.float32)

    # Smooth the mask to avoid aliasing at the edge
    mask_smooth = scipy.ndimage.gaussian_filter(mask.astype(np.float32), sigma=1)

    # Create a smoothed but binary mask (optional)
    mask= (mask_smooth > 0.5).astype(np.uint8)

    # Compute the distance to the edges inside the mask
    distance = scipy.ndimage.distance_transform_edt(mask)
    # Smooth the distance 
    distance = scipy.ndimage.gaussian_filter(distance, sigma=1.0)
    # Normalize the distance (0 at the edge, 1 at the center)
    max_distance = np.max(distance)
    if max_distance > 0:
        normalized_distance = distance / max_distance
    else:
        normalized_distance = np.ones_like(distance)  # Avoid division by zero if there are no edges

    # Real dome profile
    height_factor = dh0 + (1 - dh0) * (1 - (1 - normalized_distance)**2) ** (order / 2.0)
   #  Apply only inside the mask
    height_map[mask > 0] = height_max * height_factor[mask > 0] 

    return height_map

def mask_overlap(mask1, mask2):
    interseccion = np.logical_and(mask1>0, mask2>0).sum()
    total_mask1 = (mask1>0).sum()
    if total_mask1 == 0:
        return 0.0
    
    p=interseccion / total_mask1
    #print(p)
    return p

def stack_masks(masks, method='union'):
    stack = np.stack(masks)

    if method == 'union':
        return (np.any(stack, axis=0).astype(np.uint8)) * 255
    elif method == 'intersection':
        return (np.all(stack, axis=0).astype(np.uint8)) * 255
    elif method == 'mean':
        return (np.mean(stack, axis=0) * 255).astype(np.uint8)

def dilate_circular_mask(mask, radius = 1):
    """
    Dilates a binary mask with a circular structuring element.

    Parameters:
    - mask (ndarray): Binary mask (values 0 and 1, or 0 and 255).
    - radius (int): Radius of the circular structuring element in pixels.

    Returns:
    - dilated_mask (ndarray): Dilated mask.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    kernel_size = 2 * radius + 1  # radius = 1 dilates one pixel on each side of every pixel with value 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    return (dilated_mask > 0).astype(np.uint8)
#Textures********************

def smooth_gaussian_inside_mask(image, mask, sigma=1.0):
    """
    Applies smoothing inside a mask without affecting the values outside the mask.

    Parameters:
    - image: NumPy array with the image to smooth.
    - mask: Binary array (1 in the region to smooth, 0 outside).
    - sigma: Standard deviation of the Gaussian filter (higher sigma = more smoothing).

    Returns:
    - Array with the image smoothed inside the mask.
    """

    # Smooth the whole image with a Gaussian filter
    smoothed_image = scipy.ndimage.gaussian_filter(image, sigma=sigma)

    # Keep the original values outside the mask
    result = np.where(mask, smoothed_image, image)

    return result

def fill_with_ellipsoids(mask, relative_radios_range=[0.05, 0.5],
                            n_por_radio=10,
                            altura_range=(0.0,1.0),
                            forma='dome',
                            e_range=(0.0, 0.9), 
                            overlap=0.2):
    
    H, W = mask.shape
    l_m,w_m = get_size_object_from_mask(mask)       #l_m is the maximum and w_m is the minimum
    full_mask = np.zeros_like(mask,dtype=np.uint8)
    masks = []
    image = np.zeros_like(mask,dtype=np.float32) 
    n_por_radio = np.random.randint(1,n_por_radio)
    radios = np.random.uniform(relative_radios_range[0],relative_radios_range[1],n_por_radio)
    alturas = np.random.uniform(altura_range[0],altura_range[1],n_por_radio)
    radios = np.sort(radios)[::-1]      #sort from largest to smallest
    indx = np.argsort(-np.abs(alturas))
    alturas = alturas[indx]      #sort from largest to smallest
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    for radi,h in zip(radios,alturas):
        radi *= w_m
        count = 0
        n_iter = 0
        while count<n_por_radio and n_iter<n_por_radio*4:
            n_iter +=1
            # look for a possible center
            posibles = np.argwhere(mask == 1)
            if posibles.shape[0] == 0:
                break
            cy, cx = posibles[np.random.randint(len(posibles))]
            # Random ellipse
            e = np.random.uniform(*e_range)
            b = radi *e
            angle = np.random.uniform(0, 2 * np.pi)

            # Rotated coordinates
            x_shift = xx - cx
            y_shift = yy - cy
            x_rot = x_shift * np.cos(angle) + y_shift * np.sin(angle)
            y_rot = -x_shift * np.sin(angle) + y_shift * np.cos(angle)
            
            elipse_mask = (x_rot / radi)**2 + (y_rot / b)**2 <= 1

            # Must be completely inside the original mask
            if not np.all(mask[elipse_mask] == 1):
                continue

            if mask_overlap(elipse_mask, full_mask) < overlap:
                count +=1
                # generate ellipse image
                if forma=='dome':
                    img =mask_to_height_map(elipse_mask, height_max = h,  order=4.0, dh0=0.0)
                else:
                    img =mask_to_height_map(elipse_mask, height_max = h,  order=0.1, dh0=0.0)
                # Create local height
                if h > 0:
                    # Only update by maximum inside the mask
                    image[elipse_mask] = np.maximum(image[elipse_mask], img[elipse_mask])
                else:
                    # Only update by minimum inside the mask
                    image[elipse_mask] = np.minimum(image[elipse_mask], img[elipse_mask])
                
                masks.append(elipse_mask)
                full_mask = stack_masks(masks)
                
          
    return image  
    
def fill_with_zebra_like_texture(mask, 
                                 freq=10,             # controls the periodicity
                                 thickness=1.0,       # controls the thickness (width of the bands)
                                 orientation=0,       # tilt angle (degrees)
                                 deformation_strength=20, # deformation intensity
                                 deformation_scale=10,    # deformation scale
                                 smooth_sigma=1):     # final smoothing

    H, W = mask.shape
    size = max(H, W)
    mapa_altura = np.zeros((H, W), dtype=np.float32)

    # Create a zebra-like base texture with horizontal sinusoids
    x = np.linspace(0, freq * 2 * np.pi, size)
    #base = np.tile(np.sin(x * thickness), (size, 1))
    base = np.tile(np.sin(x), (size, 1))
    # Apply smoothing
    base = gaussian_filter(base, sigma=smooth_sigma)

    # Deformation field
    coords = np.indices((size, size)).astype(float)
    coords[0] += deformation_strength * np.sin(coords[1] / deformation_scale)
    coords[1] += deformation_strength * 0.5 * np.sin(coords[0] / deformation_scale)

    # Apply deformation
    deformed = map_coordinates(base, coords, order=1, mode='reflect')

    # Rotate to apply the tilt
    rotated = rotate(deformed, orientation, resize=False, mode='reflect')

    # Normalize
    rotated -= rotated.min()
    rotated /= rotated.max()

    # Crop if necessary
    textura = rotated[:H, :W]

    # Apply to the mask
    mapa_altura[mask > 0] = textura[mask > 0]

    return mapa_altura

def fill_with_correlated_texture(mask, correlation_length=20, amplitude=1.0, seed=None):
    """
    Fills the mask region with a smooth random texture,
    with spatial correlation controlled by `correlation_length`.

    Parameters:
        mask (ndarray): binary mask (H x W)
        correlation_length (float): controls the size of the structures (in pixels)
        amplitude (float): scales the intensity of the pattern
        seed (int or None): seed for reproducibility

    Returns:
        mapa_altura (ndarray): output image with smooth texture inside the mask
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = mask.shape
    mapa_altura = np.zeros((H, W), dtype=np.float32)

    # Generate random noise and smooth it
    ruido = np.random.randn(H, W)
    textura = gaussian_filter(ruido, sigma=correlation_length)

    # Normalize and scale (-1, 1)
    textura -= np.mean(textura)
    textura /= np.max(np.abs(textura))
    textura *= amplitude/2.0

    # Apply inside the mask
    mapa_altura[mask > 0] = textura[mask > 0]

    return mapa_altura


def fill_with_noise(mask, amplitude=1.0, sigma=1.0, seed=None):
    """
    Fills the mask region with Gaussian white noise.

    Parameters:
        mask (ndarray): binary mask (H x W)
        sigma (float): noise standard deviation. Default 1.0 (68% falls between -1 and 1).
        amplitude (float): additional scale factor on the generated noise.
        seed (int or None): seed for reproducibility.

    Returns:
        mapa_altura (ndarray): output image with noise inside the mask.
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = mask.shape
    mapa_altura = np.zeros((H, W), dtype=np.float32)

    # Generate random (Gaussian) noise with deviation sigma
    ruido = np.random.normal(loc=0.0, scale=sigma, size=(H, W))

    # Apply noise only inside the mask
    mapa_altura[mask > 0] = ruido[mask > 0]

    # Scale the final amplitude
    return amplitude * mapa_altura

def get_size_object_from_mask(mask):
    """
    Returns (length, width) of the binary object in the mask.
    Uses the minimum rectangle that contains the contour.

    Parameters:
    - mask: binary mask (values 0 and 1, or 0 and 255)

    Returns:
    - (length, width): floats, in pixels
    """
    # Ensure uint8 type
    mask = (mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, 0  # no object

    # Take the largest contour
    cnt = max(contours, key=cv2.contourArea)

    # Minimum rotated rectangle
    rect = cv2.minAreaRect(cnt)  # ((cx, cy), (width, height), angle)
    w, h = rect[1]

    # Define length and width as the larger and smaller
    length = max(w, h)
    width = min(w, h)

    return length, width

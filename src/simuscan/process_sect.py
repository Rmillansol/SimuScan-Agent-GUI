
# file with functions to manipulate STL files and prepare the data

import numpy as np
import copy
from typing import  Tuple
import random
import matplotlib.pyplot as plt
from .SyntheticParams import ObjectConfig
from . import mask_tools
from scipy.interpolate import splprep, splev
import cv2
import os

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

def sample_random_value(base, rel_var=0.0, abs_var=0.0, integer=False):
    """
    Generates a random number around a base value, with relative and/or absolute variation.

    Parameters:
    - base: central value
    - rel_var: relative variation (fraction of base, e.g. 0.1 = +-10%)
    - abs_var: absolute variation (fixed value, e.g. 20 = +-20 units)
    - integer: if True, returns an integer

    Returns:
    - Random value (float or int), or base if there is no variation
    """
    # Compute limits
    low = base - abs_var - abs(base) * rel_var
    high = base + abs_var + abs(base) * rel_var

    if low == high:
        return int(round(base)) if integer else base

    if integer:
        low_i = int(np.floor(low))
        high_i = int(np.ceil(high)) + 1
        if low_i >= high_i:
            high_i = low_i + 2
        return np.random.randint(low_i, high_i)
    else:
        return np.random.uniform(low, high)

def compute_neighbor_center(center0, size_shape, rotation0,
                            rotation, direction='right', scale=0.9):
   
    """
    Computes the center (cx, cy) of a neighboring object according to the orientation and position of the original object.

    Parameters:
    - center0: center of the original object
    - size_shape: size of the original object (width, height)
    - rotation0: rotation of the original object (degrees)
    - rotation: rotation of the new object (degrees)
    - direction: 'right', 'left', 'up', 'down'
    - scale: contact factor (0.9 ~ close contact without overlap)

    Returns:
    - (cx, cy): center of the new object
    """
    scale=scale/2.0
    # Convert to radians
    rot0_rad = np.radians(rotation0)
    rot_rad  = np.radians(rotation)
    alfa = 90-rotation0         #perpendicular
    alfa_rad = np.radians(alfa)
    cx0, cy0 = center0
    sp10, sp20 = size_shape
    sp1, sp2 = size_shape
    
    if direction == 'right':
        #delta = scale * (sp10 * vec_main0 + sp1 * vec_main )
        cx = cx0+scale*(sp10*np.cos(rot0_rad) + sp1*np.cos(rot_rad))
        cy = cy0-scale*(sp10*np.sin(rot0_rad) + sp1*np.sin(rot_rad))
    elif direction == 'left':
        cx = cx0-scale*(sp10*np.cos(rot0_rad) + sp1*np.cos(rot_rad))
        cy = cy0+scale*(sp10*np.sin(rot0_rad) + sp1*np.sin(rot_rad))
    elif direction == 'down':
        cx = cx0+scale*(sp20*np.cos(alfa_rad) + sp2*np.cos(alfa_rad))
        cy = cy0+scale*(sp20*np.sin(alfa_rad) + sp2*np.sin(alfa_rad))
    elif direction == 'up':
        cx = cx0-scale*(sp20*np.sin(rot0_rad) + sp2*np.sin(rot_rad))
        cy = cy0-scale*(sp20*np.cos(rot0_rad) + sp2*np.cos(rot_rad))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return cx, cy


def object_fits_in_image(cx, cy, size_shape, image_shape, margin=0):
    """
    Checks whether an object fits within the image borders,
    taking size and center into account.

    Parameters:
    - cx, cy: center of the object
    - size_shape: size (width, height)
    - image_shape: image shape (H, W)
    - margin: extra margin (in pixels)

    Returns:
    - True if it fits, False if it falls outside
    """
    sp1,sp2 = size_shape
    H, W = image_shape

    # Conservative estimate of the maximum radius
    # Radius = distance from the center to the farthest corner of the rotated bounding box
    r = np.sqrt((sp1/2)**2 + (sp2/2)**2)

    return (
        (cx - r - margin) >= 0 and
        (cy - r - margin) >= 0 and
        (cx + r + margin) < W and
        (cy + r + margin) < H
    )

def interp_line(p1, p2, n_points=30):
    return np.linspace(p1, p2, n_points)

def remove_consecutive_duplicates(arr):
    # arr must be (N, 2)
    diff = np.diff(arr, axis=0)
    keep = np.any(diff != 0, axis=1)
    keep = np.concatenate([[True], keep])  # the first one is always kept
    return arr[keep]

def apply_bending_to_points(points, axis='y', amplitude=10, frequency=1, kind='sin', diff_center_px=0):
    """
    Applies bending to a curve represented by 2D points.

    - points: array (N, 2) with [x, y] coordinates.
    - axis: 'x' or 'y', direction along which the displacement is applied.
    - amplitude: in pixels.
    - frequency: number of cycles (if 'sin').
    - kind: 'sin', 'parabolic', 'gaussian', 'linear'.
    - diff_center_px: center offset.
    
    Returns: deformed points (N, 2)
    """
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    cx, cy = np.mean(x), np.mean(y)

    if axis == 'x':
        base = (y - cy - diff_center_px) / (np.ptp(y) + 1e-6) + 0.5
    elif axis == 'y':
        base = (x - cx - diff_center_px) / (np.ptp(x) + 1e-6) + 0.5
    else:
        raise ValueError("Axis must be 'x' or 'y'")

    if kind == 'sin':
        disp = amplitude * np.sin(2 * np.pi * base * frequency)
    elif kind == 'parabolic':
        disp = -amplitude * (2 * base - 1)**2 + amplitude
    elif kind == 'gaussian':
        center = 0.5
        sigma = 0.15
        disp = amplitude * np.exp(-((base - center) ** 2) / (2 * sigma ** 2))
    elif kind == 'linear':
        disp = amplitude * (base - 0.5)
    else:
        raise ValueError(f"Unsupported bending kind: {kind}")

    # Center the displacement
    disp -= np.mean(disp)

    if axis == 'x':
        x_new = x + disp
        y_new = y
    else:
        x_new = x
        y_new = y + disp

    return np.stack([x_new, y_new], axis=1)

def scale_points(points, scale_x=1.0, scale_y=1.0, center=None):
    """
    Scales the points in x and y with respect to a center.
    
    - points: (N, 2) array
    - scale_x, scale_y: scale factors
    - center: (x, y). If None, the centroid is used
    
    Returns: scaled points
    """
    points = np.array(points)
    if center is None:
        center = np.mean(points, axis=0)
    
    scaled = points* np.array([scale_x, scale_y])
   
    return scaled + center

def rotate_points(points, angle_degrees, center=None):
    """
    Rotates the points around a given center.
    
    - points: (N, 2) array
    - angle_degrees: angle in degrees
    - center: (x, y). If None, the centroid is used

    Returns: rotated points
    """
    points = np.array(points)
    if center is None:
        center = np.mean(points, axis=0)
    theta = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])
    shifted = points - center
    rotated = shifted @ rotation_matrix.T
    return rotated + center

def aplicar_variacion_perpendicular_con_cierre(sections, amplitud=0.05, cerrar=True):
    sections_mod = copy.deepcopy(sections)
    prev_last_point = None  # Stores the last point of the previous section

    for idx, sec in enumerate(sections_mod):
        pts = np.array(sec["points"], dtype=np.float64)
        pts_mod = []

        for i in range(len(pts)):
            if idx > 0 and i == 0:
                # First point: use the last point of the previous section
                p_mod = prev_last_point
            else:
                ruido = (np.random.rand(2) - 0.5) * 2 * amplitud
                desplazamiento = pts[i] * ruido
                p_mod = pts[i] + ruido

            pts_mod.append(tuple(p_mod))

        # Save the modified section
        sections_mod[idx]["points"] = pts_mod
        prev_last_point = pts_mod[-1]

    # Close the figure: first point = last point
    if cerrar:
        first_sec = sections_mod[0]
        first_sec["points"][0] = tuple(prev_last_point)

    return sections_mod

def show_image_perfils(img,pcx=0.5,pcy=0.5,title= 'title'):
    # # Central coordinates
    cy, cx = round(pcx * img.shape[0]), round(pcy * img.shape[1])

    # Extract profiles
    perfil_y = img[:, cx]  # vertical (central column)
    perfil_x = img[cy, :]  # horizontal (central row)

    # Show everything
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 3]})

    # Image
    axs[1, 0].imshow(img, cmap='gray', origin='upper')
    axs[1, 0].axhline(cy, color='r', linestyle='--')
    axs[1, 0].axvline(cx, color='b', linestyle='--')
    axs[1, 0].set_title("Image with profiles")
    axs[1, 0].axis('off')

    # Horizontal profile (X)
    axs[0, 0].plot(perfil_x, color='r')
    axs[0, 0].set_title("Horizontal profile (Y = center)")
    axs[0, 0].axis('tight')

    # Vertical profile (Y)
    axs[1, 1].plot(perfil_y, np.arange(len(perfil_y)), color='b')
    axs[1, 1].invert_xaxis()
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_title("Vertical profile (X = center)")
    axs[1, 1].axis('tight')

    # Remove the empty subplot
    axs[0, 1].axis('off')

    plt.tight_layout()
    plt.show()

def mask_by_sections_transformation(sections,
                                    size: int = 512,
                                    size_shape: Tuple[int,int] = [30, 30],
                                    scale_pct: float = 0.2,
                                    eccentricity_variation_pct:float = 0.05,
                                    rotation_deg: float = 0.0,
                                    center: Tuple[float,float] = (0.0, 0.0),
                                    apply_bending: bool = False,
                                    bending_axis: str = 'y',
                                    bending_amplitud_pct: float = 0.1,
                                    bending_freq: float = 5,
                                    bending_kind: str = 'parabolic',
                                    bending_diff_center: float = 0.1,
                                    border_rand_variation: float = 0.0,
                                    closed: bool = True,
                                    is_fill: bool = True
                                    ):
    
    # Apply percentage variation (e.g. +-5%)
    sections = aplicar_variacion_perpendicular_con_cierre(sections, 
                                                          amplitud=border_rand_variation,
                                                            cerrar=closed)

    full_path = []
    for sec in sections:
        pts = sec["points"]
        if sec["type"] == "line":
            seg = interp_line(*pts, n_points=30)
        elif sec["type"] == "spline":
            seg = np.array(pts)  
        seg = np.array(seg)      
        full_path.append(seg)

    trajectory = np.vstack(full_path)
    
    # Make sure the curve is closed
    # if not np.allclose(trajectory[0], trajectory[-1]):
    #     trajectory = np.vstack([trajectory, trajectory[0]])

    trajectory = remove_consecutive_duplicates(trajectory)
    if apply_bending:
        trajectory = apply_bending_to_points(trajectory, 
                                         axis=bending_axis, 
                                         amplitude=bending_amplitud_pct,
                                         frequency=bending_freq, 
                                         kind=bending_kind, diff_center_px=bending_diff_center)
     # scale and translate
    s_shx = size_shape[0]/2 * (1 + scale_pct)
    s_shy = size_shape[1]/2 * (1 + scale_pct)
    s_shy *= (1 + eccentricity_variation_pct)  # stretch or flatten in y
    trajectory = scale_points(trajectory, scale_x=s_shx, scale_y=s_shy, center=center)
   
    trajectory = rotate_points(trajectory, rotation_deg, center=center)
   
       # Fit a periodic B-spline
    tck, u = splprep(trajectory.T, s=0, per=True)

    u_fine = np.linspace(0, 1, 300)
    x_new, y_new = splev(u_fine, tck)

    # Result: smoothed trajectory with an adjusted number of points
    fine_trajectory = np.vstack([x_new, y_new]).T
    
    # Temporarily increase the resolution
    scale_factor = 8
    temp_size = size * scale_factor

    # Create the mask at a higher resolution
    mask_high_res = np.zeros((temp_size, temp_size), dtype=np.uint8)

    # Scale the contour
    contour_high_res = (fine_trajectory * scale_factor).round().astype(np.int32).reshape((-1, 1, 2))

    if is_fill:
        cv2.fillPoly(mask_high_res, [contour_high_res], color=1)                                    #to fill
    else:
        cv2.polylines(mask_high_res, [contour_high_res], isClosed=False, color=255, thickness=4*scale_factor)        #for outlines
    
    # Downscale with smoothing (implicit antialiasing)
    mask = cv2.resize(mask_high_res, (size, size), interpolation=cv2.INTER_AREA)
    
    return mask

def rand_image_from_sections(obj:ObjectConfig, size = 512 ):

    # randomly vary the object dimensions
    height=random.uniform(obj.height[0],obj.height[1])
    dh0=random.uniform(obj.step_perfil_z[0],obj.step_perfil_z[1])
    order_z = random.uniform(obj.orden_z[0],obj.orden_z[1])
    scale_pct=random.uniform(0,obj.scale_pct)
    eccentricity_variation_pct = random.uniform(obj.eccentricity_variation_down_pct,obj.eccentricity_variation_up_pct)
    rotation_deg = random.uniform(obj.rotation_range_deg[0],obj.rotation_range_deg[1])
    center = [random.uniform(0, size), random.uniform(0, size)]
    bending_amplitud_pct = random.uniform(-obj.bending_amplitud_pct,obj.bending_amplitud_pct)
    bending_freq = random.uniform(0,obj.bending_freq)
    bending_diff_center = random.uniform(0,obj.bending_diff_center)
    border_rand_variation = random.uniform(0,obj.variation_border_pct)
    #select among the possible shapes within the same object
    sections_form = random.choice(obj.sections)

    # Apply the transformation and generate the mask
    mask = mask_by_sections_transformation(sections = sections_form,
                                    size = size,
                                    size_shape = obj.size_shape,
                                    # height = height,
                                    # order_z = order_z,
                                    # dh0 = dh0,
                                    scale_pct = scale_pct ,
                                    eccentricity_variation_pct = eccentricity_variation_pct,
                                    rotation_deg = rotation_deg,
                                    center = center,
                                    apply_bending = obj.apply_bending,
                                    bending_axis = obj.bending_axis,
                                    bending_amplitud_pct = bending_amplitud_pct,
                                    bending_freq = bending_freq,
                                    bending_kind = obj.bending_kind,
                                    bending_diff_center = bending_diff_center,
                                    border_rand_variation = border_rand_variation,
                                    closed = obj.closed,
                                    is_fill = obj.is_fill
                                    )
   
    img = mask_tools.mask_to_height_map(mask,height,order_z,dh0*height)
    return img, mask, height

def generate_rand_image_from_obj_sections(obj: ObjectConfig, size: int = 512):
   
    full_mask = np.zeros((size, size), dtype=np.uint8)

    masks = []
    images =[]
    image = np.zeros((size,size), dtype=np.float32)
    if obj.enable:
        count = 0
        n_iter = 0
        while count<obj.n_objects and n_iter<obj.n_objects*4:
            n_iter +=1
            if random.random() < obj.probability:
                img, mask,height = rand_image_from_sections(obj,size)
                if mask_tools.mask_overlap(mask, full_mask) < obj.max_overlap:
                    count +=1
                    masks.append(mask)
                    full_mask = mask_tools.stack_masks(masks)
                    #ellipse texture
                    if obj.apply_ellipses_texture:
                        if random.random() < obj.ellipses_probability:
                            text_img = mask_tools.fill_with_ellipsoids(mask,
                                                            relative_radios_range=obj.ellipses_relative_radios_range, 
                                                            n_por_radio=obj.ellipses_num,
                                                            altura_range=[height * x for x in obj.ellipses_amplitud_range], 
                                                            forma=obj.ellipses_shape,
                                                            e_range=obj.ellipses_eccentriciy_range, 
                                                            overlap=obj.ellipses_overlap)
                            
                            img += text_img
                    # texture
                    if obj.apply_perlin:
                        scale_px = random.choice(obj.perlin_scale_px)
                        #print(scale_px)
                        text_img = mask_tools.fill_with_correlated_texture(mask,
                                                                        correlation_length=scale_px, 
                                                                        amplitude=obj.perlin_amplitud_pct, seed=None)
                    
                        #SaveMatriz(text_img, f'perlin_img{count}.xyz','./')
                        img += text_img
        
                    # noise
                    if obj.apply_noise:
                        if random.random() < obj.noise_probability:
                            ampl = np.random.uniform(0,obj.noise_amplitud, size=1)[0]
                            ruido = mask_tools.fill_with_noise(mask,ampl,obj.noise_sigma)
                            img += ruido
                    images.append(img)
       
    else: print("You must enable the object to be able to paint")


    return images, masks, full_mask

def generate_rand_colony_image_from_obj_sections(obj: ObjectConfig, size: int = 512):
    # mask with all the objects, list of individual masks and the final image
    full_mask = np.zeros((size, size), dtype=np.uint8)

    masks = []
    images =[]
   
    if obj.enable:
        #select the group
        n_obj = random.choice(obj.colony_n_objects)
        if n_obj> 4:
            n_obj = np.random.randint(4,n_obj)
        #lists for the neighbors
        ls_center = []
        ls_rot = []
        # Random initial conditions
        # randomly vary the object dimensions
        height0=random.uniform(obj.height[0],obj.height[1])
        dh00=random.uniform(obj.step_perfil_z[0],obj.step_perfil_z[1])
        order_z0 = random.uniform(obj.orden_z[0],obj.orden_z[1])
        scale_pct0=random.uniform(0,obj.scale_pct)
        eccentricity_variation_pct0 = random.uniform(obj.eccentricity_variation_down_pct,obj.eccentricity_variation_up_pct)
        rotation_deg0 = random.uniform(obj.rotation_range_deg[0],obj.rotation_range_deg[1])
        center0 = [random.uniform(0, size), random.uniform(0, size)]
        bending_amplitud_pct0 = random.uniform(0,obj.bending_amplitud_pct)
        bending_freq0 = random.uniform(0,obj.bending_freq)
        bending_diff_center0 = random.uniform(0,obj.bending_diff_center)
        border_rand_variation0 = random.uniform(0,obj.variation_border_pct)
        #select among the possible shapes within the same object
        sections_form0 = random.choice(obj.sections)
            
        count = 0
        n_iter = 0
        directions = ['right','left','up', 'down']
        
        while count<n_obj and n_iter<n_obj*4:
            n_iter +=1
            add_new = False
            #propose new parameters    
            scale_pct=sample_random_value(scale_pct0, rel_var=obj.colony_relative_scale_pct, abs_var=0.0, integer=False)
            eccentricity_variation_pct = sample_random_value(eccentricity_variation_pct0, rel_var=obj.colony_relative_eccentricity_variation_pct, abs_var=0.0, integer=False)
            bending_amplitud_pct = sample_random_value(bending_amplitud_pct0, rel_var=obj.bending_amplitud_pct, abs_var=0.0, integer=False)
            bending_freq  = sample_random_value(bending_freq0, rel_var=obj.bending_freq, abs_var=0.0, integer=False)
            bending_diff_center = sample_random_value(bending_diff_center0, rel_var=obj.bending_diff_center, abs_var=0.0, integer=False)

            #direction = np.random.choice(['up', 'down', 'left', 'right'])
            #np.random.shuffle(directions)
            for dir in directions:
                #try several times so the neighbor ends up touching
                for _ in range(20):
                    rotation = sample_random_value(rotation_deg0, rel_var=0.0, abs_var=obj.colony_diff_rotation, integer=False)
                    center = compute_neighbor_center(center0, obj.size_shape, rotation_deg0,
                                     rotation, direction=dir, scale=0.8)
                    cx,cy = center
                    # slightly shift the new center
                    cx =  sample_random_value(cx, rel_var=obj.colony_relative_center, abs_var=0.0, integer=True)
                    cy =  sample_random_value(cy, rel_var=obj.colony_relative_center, abs_var=0.0, integer=True)
                    
                    if object_fits_in_image(cx, cy, obj.size_shape, (size,size), margin=0):
                        # Apply the transformation and generate the mask
                        mask = mask_by_sections_transformation(sections = sections_form0,
                                                        size = size,
                                                        size_shape = obj.size_shape,
                                                        scale_pct = scale_pct ,
                                                        eccentricity_variation_pct = eccentricity_variation_pct,
                                                        rotation_deg = rotation,
                                                        center = center,
                                                        apply_bending = obj.apply_bending,
                                                        bending_axis = obj.bending_axis,
                                                        bending_amplitud_pct = bending_amplitud_pct,
                                                        bending_freq = bending_freq,
                                                        bending_kind = obj.bending_kind,
                                                        bending_diff_center = bending_diff_center,
                                                        border_rand_variation = border_rand_variation0,
                                                        closed = obj.closed,
                                                        is_fill = obj.is_fill
                                                        )
                        overlap = mask_tools.mask_overlap(mask, full_mask)
                        if (count <= 1) or (obj.colony_min_overlap <= overlap <=obj.colony_max_overlap):
                            # Save the mask and the parameters
                            masks.append(mask)
                            full_mask = mask_tools.stack_masks(masks)
                            # Save parameters in lists 
                            ls_center.append(center)
                            ls_rot.append(rotation)
                         
                            count +=1
                            add_new = True
                            center0 = cx,cy
                            rotation_deg0 = rotation
                            break
                if add_new:
                    break   #exit the directions loop
            if add_new==False and (count > 1):
                index= np.random.randint (0, count-1, size=1)[0]
                center0, rotation_deg0 = ls_center[index],ls_rot[index]
            else:
                center0 = [random.uniform(0, size), random.uniform(0, size)]
        # Now generate the image with textures
        for mask in masks:
            height = sample_random_value(height0, rel_var=obj.colony_realtive_height, abs_var=0.0, integer=False)
            img = mask_tools.mask_to_height_map(mask,height,order_z0,dh00*height)
            
            # ellipse texture
            if obj.apply_ellipses_texture:
                if random.random() < obj.ellipses_probability:
                    text_img = mask_tools.fill_with_ellipsoids(mask,
                                                    relative_radios_range=obj.ellipses_relative_radios_range, 
                                                    n_por_radio=obj.ellipses_num,
                                                    altura_range=[height0 * x for x in obj.ellipses_amplitud_range], 
                                                    forma=obj.ellipses_shape,
                                                    e_range=obj.ellipses_eccentriciy_range, 
                                                    overlap=obj.ellipses_overlap)
                    
                    #SaveMatriz(text_img, f'text_img{count}.xyz','./')
                    img += text_img     
            # texture
            if obj.apply_perlin:
                scale_px = random.choice(obj.perlin_scale_px)
                text_img = mask_tools.fill_with_correlated_texture(mask,
                                                                correlation_length=scale_px, 
                                                                amplitude=obj.perlin_amplitud_pct, seed=None)
                #SaveMatriz(text_img, f'perlin_img{count}.xyz','./')
                img += text_img
            
            # noise
            if obj.apply_noise:
                if random.random() < obj.noise_probability:
                    ampl = np.random.uniform(0,obj.noise_amplitud, size=1)[0]
                    ruido = mask_tools.fill_with_noise(mask,ampl,obj.noise_sigma)
                    img += ruido

            images.append(img)
       
    return images, masks, full_mask  
   
# file with functions to manipulate STL files and prepare the data

import numpy as np
from typing import Tuple
import random
import copy
import trimesh
import time
from .SyntheticParams import ObjectConfig
from . import mask_tools
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

def filter_occluded_triangles(mesh, grid_size=512, margin=0.0):
    """
    Removes triangles that are completely hidden from the top view (XY).
    Keeps only those that are visible at some point.
    """
    triangles = mesh.triangles
    faces = mesh.faces

    # Define the physical XY grid
    min_x, max_x = mesh.bounds[:, 0]
    min_y, max_y = mesh.bounds[:, 1]

    x_vals = np.linspace(min_x - margin, max_x + margin, grid_size)
    y_vals = np.linspace(min_y - margin, max_y + margin, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z_buffer = np.full_like(X, -np.inf)
    face_buffer = -np.ones_like(X, dtype=int)

    # Rasterise each triangle and fill the Z-buffer
    for tri_id, tri in enumerate(triangles):
        v0, v1, v2 = tri
        x_min, x_max = np.min(tri[:, 0]), np.max(tri[:, 0])
        y_min, y_max = np.min(tri[:, 1]), np.max(tri[:, 1])

        mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
        idx = np.argwhere(mask)

        def barycentric(p, v0, v1, v2):
            A = np.array([[v0[0], v1[0], v2[0]],
                          [v0[1], v1[1], v2[1]],
                          [1, 1, 1]])
            b = np.array([p[0], p[1], 1])
            try:
                return np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return np.array([-1, -1, -1])

        for i, j in idx:
            p = np.array([X[i, j], Y[i, j]])
            w = barycentric(p, v0[:2], v1[:2], v2[:2])
            if np.all(w >= 0):
                z = w[0]*v0[2] + w[1]*v1[2] + w[2]*v2[2]
                if z > Z_buffer[i, j]:
                    Z_buffer[i, j] = z
                    face_buffer[i, j] = tri_id

    # Determine which triangles were visible at least once
    visible_faces = np.unique(face_buffer[face_buffer >= 0])
    visible_faces = np.sort(visible_faces)

    # Create a new mesh with only the visible triangles
    new_faces = faces[visible_faces]
    new_mesh = mesh.submesh([visible_faces], append=True)

    return new_mesh

# def load_stl_old(stl_file):
#     # === Load and transform STL ===

#     mesh = trimesh.load_mesh(stl_file)
#     print("\n--- STL Mesh Loading --------------------------------")
#     print(f"File: {stl_file}")

#     print(f"Initial triangles: {len(mesh.faces):>8d}")

#     print("Processing: removing occluded (hidden) triangles...")
#     start = time.time()
#     mesh = filter_occluded_triangles(mesh)
#     elapsed = time.time() - start

#     print(f"Final triangles:   {len(mesh.faces):>8d}")
#     print(f"Processing time:   {elapsed:>6.2f} s")
#     print("----------------------------------------------------\n")

#     # === Centre at 0 and normalise to [-1, 1] in XY and [0, 1] in Z ===
#     bounds = mesh.bounds
#     scale_x = 2.0 / (bounds[1, 0] - bounds[0, 0])
#     scale_y = 2.0 / (bounds[1, 1] - bounds[0, 1])
#     scale_z = 1.0 / (bounds[1, 2] - bounds[0, 2] + 1e-6)
#     translate = -np.mean(bounds, axis=0)
#     mesh.apply_translation(translate)
#     mesh.apply_scale([scale_x, scale_y, scale_z])
#     #mesh.merge_vertices()  # merges vertices at the same position

#     return mesh


def drop_faces_below_zcut(mesh, z0=0.0, eps0=0.0, z_tol=1e-6,
                          require_all=True, in_place=False):
    """
    Removes faces below the plane z = z0 + eps0 (with tolerance).
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
    z0 : float | None
        Base plane. If None, the mesh z_min is used.
    eps0 : float
        Offset above z0 that defines the threshold: z_cut = z0 + eps0.
    z_tol : float
        Numerical tolerance. A vertex counts as "below" if z < z_cut - z_tol.
    require_all : bool
        - True: removes the face if all 3 vertices are below.
        - False: removes the face if any vertex is below (more aggressive).
    in_place : bool
        If True modifies the original mesh, if False works on a copy.

    Returns
    --------
    trimesh.Trimesh : mesh with filtered faces.
    """
    m = mesh if in_place else mesh.copy()

    # Determine z0 if not given
    if z0 is None:
        z0 = m.bounds[0, 2]  # z_min

    # Cut threshold
    z_cut = z0 + eps0
    z_thr = z_cut - z_tol

    # z of each vertex per face: shape (F, 3)
    tri_z = m.vertices[m.faces][:, :, 2]

    # "Below the threshold" mask with tolerance
    below = (tri_z < z_thr)

    # Rule for removing faces
    drop = below.all(axis=1) if require_all else below.any(axis=1)

    # Keep the rest
    m.update_faces(~drop)
    m.remove_unreferenced_vertices()
    return m

def load_stl(stl_file, filtrar_ocultos=True, grid_size=512, margin=0.0, eps=1e-4):
    mesh = trimesh.load_mesh(stl_file)
    print("\n--- STL Mesh Loading --------------------------------")
    print(f"File: {stl_file}")

    print(f"Initial triangles: {len(mesh.faces):>8d}")

    # === Normalise ===
    # 1) Translate: centre XY and bring Z_min to 0
    (min_x, min_y, min_z), (max_x, max_y, max_z) = mesh.bounds
    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z

    tx = -0.5 * (min_x + max_x)   # centre in X
    ty = -0.5 * (min_y + max_y)   # centre in Y
    tz = -min_z                   # Z_min -> 0
    # print(f"before min z: {min_z:.3e}  max z: {max_z:.3e}")
    mesh.apply_translation([tx, ty, tz])
   
    # 2) Scale: uniform XY by the longer side; Z to [0,1]
    major_xy = max(dx, dy)
    sxy = 2.0 / major_xy if major_xy > eps else 1.0
    sz  = 1.0 / dz if dz > eps else 1.0
    #sz = (1.0 /  max_z) if  max_z > eps else 1.0

    T = np.eye(4)
    T[0, 0] = sxy
    T[1, 1] = sxy
    T[2, 2] = sz
    mesh.apply_transform(T)

    (min_x, min_y, min_z), (max_x, max_y, max_z) = mesh.bounds
    # print(f"after min z: {min_z:.3e}  max z: {max_z:.3e}")
    if filtrar_ocultos:
        print("Processing: removing occluded (hidden) triangles...")
        start = time.time()
        
        #mesh = filter_occluded_triangles(mesh, grid_size=grid_size, margin=margin)  # your ray function/version
        mesh = drop_faces_below_zcut(mesh, z0=0.0, eps0=0.1, z_tol=0.0)
        elapsed = time.time() - start
        print(f"Final triangles:   {len(mesh.faces):>8d}")
        print(f"Processing time:   {elapsed:>6.2f} s")
        print("----------------------------------------------------\n")
   
    # (optional) mesh.merge_vertices()
    return mesh


def apply_random_vertex_displacement(mesh, amplitude=0.01):
    """
    Applies a random displacement to each unique vertex (by index),
    ensuring that vertices shared between triangles deform together.
    """
    mesh = mesh.copy()
    
    # Each row of mesh.vertices is a unique vertex by index
    displacement = (np.random.rand(*mesh.vertices.shape) - 0.5) * amplitude

    # Apply the displacement directly
    mesh.vertices += displacement

    return mesh

def apply_bending_to_points(points, axis='y', amplitude=10, frequency=1, kind='sin', diff_center_px=0):
    """
    Applies bending to a curve represented by 2D points.

    - points: array (N, 2) with [x, y] coordinates.
    - axis: 'x' or 'y', direction along which the displacement is applied.
    - amplitude: in pixels.
    - frequency: number of cycles (if 'sin').
    - kind: 'sin', 'parabolic', 'gaussian', 'linear'.
    - diff_center_px: centre offset.
    
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

    # Centring of the displacement
    disp -= np.mean(disp)

    if axis == 'x':
        x_new = x + disp
        y_new = y
    else:
        x_new = x
        y_new = y + disp

    return np.stack([x_new, y_new], axis=1)

def apply_bending_to_mesh(mesh:trimesh.Trimesh,
                          bending_axis: str = 'y',
                          bending_amplitud_pct: float = 0.1,
                          bending_freq: float = 5,
                          bending_kind: str = 'parabolic',
                          bending_diff_center: float = 0.1,
                          )-> trimesh.Trimesh:
        #mesh, axis='y', amplitude=10, frequency=1, kind='sin', diff_center_px=0):
    """
    Applies bending to a trimesh mesh, keeping Z constant.
    Only affects X or Y depending on the selected axis.
    """
    points_xy = mesh.vertices[:, :2]
    z = mesh.vertices[:, 2]

    # Use your function on the XY points
    bent_xy = apply_bending_to_points(points_xy, 
                                      axis=bending_axis,
                                        amplitude= bending_amplitud_pct,
                                        frequency= bending_freq,
                                        kind=bending_kind, 
                                        diff_center_px=bending_diff_center)

    # Rebuild vertices
    bent_vertices = np.hstack([bent_xy, z[:, np.newaxis]])
    mesh.vertices = bent_vertices
    return mesh  # optional, since it already modifies in place

def apply_transformations_to_mesh(mesh:trimesh.Trimesh,
                                    size_shape: Tuple[int,int] = [30, 30],
                                    height: float = 1.0,
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
                                    )-> trimesh.Trimesh:
    
    mesh = copy.deepcopy(mesh)
    mesh = apply_random_vertex_displacement(mesh, border_rand_variation)
    # === Apply bending ===
    if apply_bending:
        mesh = apply_bending_to_mesh(mesh,
                          bending_axis,
                          bending_amplitud_pct,
                          bending_freq,
                          bending_kind,
                          bending_diff_center,
                          )

    # === Apply rotation (in XY) ===
    theta = np.deg2rad(rotation_deg)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    mesh.apply_transform(np.eye(4))  # reset transform
    mesh.apply_transform(np.vstack([np.hstack([rot_mat, [[0],[0],[0]]]), [0,0,0,1]]))

    # === Scale to the desired size ===
    
    s_shx = size_shape[0] * (1 + scale_pct)
    s_shy = size_shape[1] * (1 + scale_pct)
    s_shy *= (1 + eccentricity_variation_pct)  # stretch or flatten in y
    sx, sy = s_shx / 2, s_shy / 2
    mesh.apply_scale([sx, sy, height])

    # === Translate to the centre ===
    mesh.apply_translation([center[0], center[1], 0.0])

    return mesh

def apply_rand_transformations(obj: ObjectConfig, size: int = 512):
    """
    function that applies a random transformation according to an object's configuration
    """
    height=random.uniform(obj.height[0],obj.height[1])
    scale_pct=random.uniform(0,obj.scale_pct)
    eccentricity_variation_pct = random.uniform(obj.eccentricity_variation_down_pct,obj.eccentricity_variation_up_pct)
    rotation_deg = random.uniform(obj.rotation_range_deg[0],obj.rotation_range_deg[1])
    center = [random.uniform(0, size), random.uniform(0, size)]
    bending_amplitud_pct = random.uniform(-obj.bending_amplitud_pct,obj.bending_amplitud_pct)
    bending_freq = random.uniform(0,obj.bending_freq)
    bending_diff_center = random.uniform(0,obj.bending_diff_center)
    border_rand_variation = random.uniform(0,obj.variation_border_pct)
    mesh = apply_transformations_to_mesh(obj.mesh,
                                    obj.size_shape,
                                    height = height,
                                    scale_pct = scale_pct,
                                    eccentricity_variation_pct = eccentricity_variation_pct,
                                    rotation_deg = rotation_deg,
                                    center = center,
                                    apply_bending = obj.apply_bending,
                                    bending_axis = obj.bending_axis,
                                    bending_amplitud_pct = bending_amplitud_pct,
                                    bending_freq = bending_freq,
                                    bending_kind = obj.bending_kind,
                                    bending_diff_center = bending_diff_center,
                                    border_rand_variation = border_rand_variation
                                    )
    return mesh,height

def mesh_to_image( mesh:trimesh.Trimesh, size: int = 512):
    # mesh, center=(100, 50), size_shape=(30, 10), grid_size=512, offset_z=0.0,
    # rotation_deg=0.0, bending_params=None):
    """
    Rasterises an STL mesh into a 2D numpy image using barycentric interpolation.
        - mesh: trimesh.Trimesh
        - size: image size
    Returns:
    - Z: np.ndarray (grid_size, grid_size) and mask
    """

    # === Create the pixel grid ===
    Z = np.zeros((size, size))
   
    def barycentric(p, v0, v1, v2):
        A = np.array([[v0[0], v1[0], v2[0]],
                      [v0[1], v1[1], v2[1]],
                      [1, 1, 1]])
        b = np.array([p[0], p[1], 1])
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.array([-1, -1, -1])

    for tri in mesh.triangles:
        v0, v1, v2 = tri
        x0, x1 = np.min(tri[:, 0]), np.max(tri[:, 0])
        y0, y1 = np.min(tri[:, 1]), np.max(tri[:, 1])

        i_min = int(np.clip(np.floor(x0), 0, size - 1))
        i_max = int(np.clip(np.ceil(x1), 0, size - 1))
        j_min = int(np.clip(np.floor(y0), 0, size - 1))
        j_max = int(np.clip(np.ceil(y1), 0, size - 1))

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                px, py = i, j
                w = barycentric((px, py), v0[:2], v1[:2], v2[:2])
                if np.all(w >= 0):
                    z = w[0] * v0[2] + w[1] * v1[2] + w[2] * v2[2]
                    Z[j, i] = max(Z[j, i], z)

    mask = (Z > 0).astype(np.uint8)
    return Z, mask

def generate_rand_image_from_objSTL(obj: ObjectConfig, size: int = 512):
   
    full_mask = np.zeros((size, size), dtype=np.uint8)

    masks = []
    images =[]
    #image = np.zeros((size,size), dtype=np.float32)
    if obj.enable:
        count = 0
        n_iter = 0
        while count < obj.n_objects and n_iter < obj.n_objects*4:
            n_iter +=1
            if random.random() < obj.probability:
                mesh, height = apply_rand_transformations(obj,size)
                img, mask = mesh_to_image(mesh,size)
                if mask_tools.mask_overlap(mask, full_mask) <= obj.max_overlap:
                    count +=1
                    masks.append(mask)
                    full_mask = mask_tools.stack_masks(masks)
                    # ellipses texture
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
                    # perlin texture
                    if obj.apply_perlin:
                        scale_px = random.choice(obj.perlin_scale_px)
                        #print(scale_px)
                        text_img = mask_tools.fill_with_correlated_texture(mask,
                                                                        correlation_length=scale_px, 
                                                                        amplitude=obj.perlin_amplitud_pct, seed=None)
                        #SaveMatriz(text_img, f'perlin_img{count}.xyz','./')
                        img += text_img
                    
                    images.append(img)

    else: print("You must enable the object to be able to paint")


    return images, masks, full_mask
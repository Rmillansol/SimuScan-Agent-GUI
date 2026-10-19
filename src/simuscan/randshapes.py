#Creates images with random shapes
import numpy as np
import matplotlib.pyplot as plt
from . import shapes as shapes
from skimage.measure import label
from . import noisegenerator as gnoise
from . import mask_tools
import os
from scipy.ndimage import label, distance_transform_edt

def is_valid_circular_dna(mask,
                          min_inner_area=20,
                          min_gap_width=3):
    """
    Checks whether the mask contains at least one closed loop with
    an inner 'hole' of minimum area and width.

    Parameters:
        mask : np.ndarray binaria (0 fondo, >0 DNA)
        min_inner_area : minimum area (in pixels) of the inner hole
        min_gap_width  : minimum hole radius (in pixels).
                         In practice it is about half the
                         minimum separation between strands.
    """
    dna = mask.astype(bool)
    inv = ~dna  # fondo = True

    # Label background components
    labeled, n = label(inv)
    if n == 0:
        return False

    H, W = mask.shape

    for lab_id in range(1, n + 1):
        comp = (labeled == lab_id)

        # If it touches the image border, it is not a closed hole
        touches_border = (
            comp[0, :].any() or comp[-1, :].any() or
            comp[:, 0].any() or comp[:, -1].any()
        )
        if touches_border:
            continue

        area = comp.sum()
        if area < min_inner_area:
            continue

        # Distance transform inside the hole:
        # dist[x,y] = distance to the boundary (DNA)
        dist = distance_transform_edt(comp)
        max_gap = dist.max()

        # If the maximum radius is large enough, accept it
        if max_gap >= min_gap_width:
            return True

    # If no hole passes the criteria:
    return False


def show_image_perfils(img,pcx=0.5,pcy=0.5,title= 'title'):
    # # Coordenadas centrales
    cy, cx = round(pcx * img.shape[0]), round(pcy * img.shape[1])

    # Extract profiles
    perfil_y = img[:, cx]  # vertical (columna central)
    perfil_x = img[cy, :]  # horizontal (fila central)

    # Show everything
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 3]})

    # Imagen
    axs[1, 0].imshow(img, cmap='gray', origin='upper')
    axs[1, 0].axhline(cy, color='r', linestyle='--')
    axs[1, 0].axvline(cx, color='b', linestyle='--')
    axs[1, 0].set_title("Image with profiles")
    axs[1, 0].axis('off')

    # Perfil horizontal (X)
    axs[0, 0].plot(perfil_x, color='r')
    axs[0, 0].set_title("Horizontal profile (Y = centre)")
    axs[0, 0].axis('tight')

    # Perfil vertical (Y)
    axs[1, 1].plot(perfil_y, np.arange(len(perfil_y)), color='b')
    axs[1, 1].invert_xaxis()
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_title("Vertical profile (X = centre)")
    axs[1, 1].axis('tight')

    # Remove empty subplot
    axs[0, 1].axis('off')

    plt.tight_layout()
    plt.show()


# Formas planas
def rand_rectangle_image(size=256, n_objects=10, rang_shape=50, p_eccentrisity=0.0, range_height=1.0):
    """
    Generates a square image with multiple random rectangles.

    Parameters:
        size (int, optional): Size of the square image in pixels. Default, 256.
        n_objects (int, optional): Number of rectangles to insert in the image. Default, 10.
        rang_shape (int, optional): Maximum range of rectangle sizes in pixels. Default, 50.
        range_height (float, optional): Maximum range of possible heights for the rectangles. Default, 1.0.
        p_eccentrisity=0(float 0-1): percentage of eccentricity variation, 0 for a square
    Retorna:
        img_final (numpy.ndarray): Final image with all rectangles superimposed, of size (size, size).
        masks (list of numpy.ndarray): List of individual binary masks, where each mask represents a rectangle.
    """

    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []

    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.uniform(rang_shape*0.5, rang_shape, size=1)[0]
        sp2 = np.random.uniform(sp1 - sp1 *  p_eccentrisity, sp1 + sp1 *  p_eccentrisity, size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.uniform(0, size, size=2)
        rotation = np.random.uniform(0, 180, size=1)[0]

        # Generate rectangle
        img, mask = shapes.generate_rectangle_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation)

        # Add the rectangle image to the final image
        img_final = np.maximum(img_final, img)

        # Store the mask and the parameters
        masks.append(mask)
    

    return img_final, masks

def rand_ellipse_image(size=256, n_objects=10, rang_shape=50, p_eccentrisity=0.0, range_height=1.0):
    """
    Generates a square image with multiple random ellipses.

    Parameters:
        size (int, optional): Size of the square image in pixels. Default, 256.
        n_objects (int, optional): Number of ellipses to insert in the image. Default, 10.
        rang_shape (int, optional): Maximum range of ellipse sizes in pixels. Default, 50.
        range_height (float, optional): Maximum range of possible heights for the ellipses. Default, 1.0.
        p_eccentrisity=0(float 0-1): percentage of eccentricity variation, 0 for a circle
    Retorna:
        img_final (numpy.ndarray): Final image with all ellipses superimposed, of size (size, size).
        masks (list of numpy.ndarray): List of individual binary masks, where each mask represents an ellipse.
    """
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []
   
    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.uniform(0.5*rang_shape, rang_shape, size=1)[0]
        sp2 = np.random.uniform(sp1 - sp1 * p_eccentrisity, sp1 + sp1 * p_eccentrisity, size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.uniform(0, size, size=2)
        rotation = np.random.uniform(0, 180, size=1)[0]

        # Generate rectangle
        img, mask = shapes.generate_ellipse_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation)

        # Add the rectangle image to the final image
        img_final = np.maximum(img_final, img)

        # Store the mask and the parameters
        masks.append(mask)
      
    return img_final, masks

def rand_polygon_image(size=256, n_objects=10, rang_shape=50, p_eccentrisity=0.0,range_height=1.0, num_sides=5):
    """
    Generates a square image with multiple random polygons.

    Parameters:
        size (int, optional): Size of the square image in pixels. Default, 256.
        n_objects (int, optional): Number of polygons to insert in the image. Default, 10.
        rang_shape (int, optional): Maximum range of polygon sizes in pixels. Default, 50.
        p_eccentrisity=0(float 0-1): percentage of eccentricity variation, 0 for a regular polygon
        range_height (float, optional): Maximum range of possible heights for the polygons. Default, 1.0.
        num_sides (int, optional): Number of sides of each polygon. Default, 5.

    Retorna:
        img_final (numpy.ndarray): Final image with all polygons superimposed, of size (size, size).
        masks (list of numpy.ndarray): List of individual binary masks, where each mask represents a polygon.
    """

    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []
   
    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.uniform(rang_shape*0.5, rang_shape, size=1)[0]
        sp2 = np.random.uniform(sp1 - sp1 * p_eccentrisity, sp1 + sp1 * p_eccentrisity, size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.uniform(0, size, size=2)
        rotation = np.random.uniform(0, 180, size=1)[0]

        # Generate rectangle
        img, mask = shapes.generate_polygon_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation,num_sides=num_sides)
                       
        # Add the rectangle image to the final image
        img_final = np.maximum(img_final, img)

        # Store the mask and the parameters
        masks.append(mask)

    return img_final, masks

def rand_spline_image(size=256, n_objects=10, rang_shape=50, range_height=1.0, num_points=4, variation=0.2):
    """
    Generates a square image with multiple random closed splines.

    Parameters:
        size (int, optional): Size of the square image in pixels. Default, 256.
        n_objects (int, optional): Number of splines to insert in the image. Default, 10.
        rang_shape (int, optional): Maximum range of spline sizes in pixels. Default, 50.
        range_height (float, optional): Maximum range of possible heights for the splines. Default, 1.0.
        num_points (int, optional): Number of control points of each spline. Default, 4.
        variation (float, optional): Percentage of variation in the spline points relative to the shape size. Default, 0.2.
       
    Retorna:
        img_final (numpy.ndarray): Final image with all splines superimposed, of size (size, size).
        masks (list of numpy.ndarray): List of individual binary masks, where each mask represents a spline.
    """

    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []
  
    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.uniform(rang_shape*0.5, rang_shape, size=1)[0]
        sp2 = np.random.uniform(sp1 - sp1 * 0.5, sp1 + sp1 * 0.5, size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.uniform(0, size, size=2)
        rotation = np.random.uniform(0, 180, size=1)[0]

        # Generate rectangle
        img, mask = shapes.generate_spline_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation,num_points=num_points, variation=variation, smoothness=0)
                     
        # Add the rectangle image to the final image
        img_final = np.maximum(img_final, img)

        # Store the mask and the parameters
        masks.append(mask)
      

    return img_final, masks

def rand_oval_image(size=256, n_objects=10, rang_shape=50, range_height=1.0, p_width_max=0.6):
    """
    Generates an image with multiple random closed ovals.

    Parameters:
        size (int): Size of the square image.
        n_objects (int): Number of objects to insert.
        rang_shape (int): Range of possible sizes for the objects.
        range_height (float): Range of possible heights for the objects.
        p_width: percentage of the central cylindrical part of the oval

    Retorna:
        img_final (numpy.ndarray): Image with all the rectangles.
        masks (list of numpy.ndarray): List of individual masks for each rectangle.
      
    """
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []
   
    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.randint(round(rang_shape/4), rang_shape, size=1)[0]
        pw = np.random.uniform(0, p_width_max, size=1)[0]
        sp2 = np.random.randint(int(max(sp1-sp1 * 0.8,1)), int(max(sp1-sp1*0.4,1)) , size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.randint(0, size, size=2)
        cx, cy = int(cx), int(cy)  # Convert to scalar integers
        rotation = np.random.uniform(0, 180, size=1)[0]
        
        # Generate rectangle
        img, mask = shapes.generate_oval_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation,p_width=pw)
        
        # Add the rectangle image to the final image
        img_final = np.maximum(img_final, img)

        # Store the mask and the parameters
        masks.append(mask)
    

    return img_final, masks

# Formas 3D

def rand_ellipsoid_image(size=256, n_objects=10, rang_shape=50, p_eccentrisity=0.0,  range_height=1.0):
    """
    Generates an image with multiple random 3D ellipsoids.

    Parameters:
        size (int): Size of the square image.
        n_objects (int): Number of objects to insert.
        rang_shape (int): Range of possible sizes for the objects.
        range_height (float): Range of possible heights for the objects.
        num_sides: number of sides of the polygon

    Retorna:
        img_final (numpy.ndarray): Image with all the rectangles.
        masks (list of numpy.ndarray): List of individual masks for each rectangle.
     
    """
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []
 
    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.uniform(1, rang_shape, size=1)[0]
        sp2 = np.random.uniform(sp1 - sp1 * p_eccentrisity, sp1 + sp1 * p_eccentrisity, size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.uniform(0, size, size=2)
        rotation = np.random.uniform(0, 180, size=1)[0]

        # Generate ellipse
        img, mask = shapes.generate_ellipsoid_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation)
                 
        # Add the ellipsoid image to the final image
        img_final = np.maximum(img_final, img)

        # Store the mask and the parameters
        masks.append(mask)
      
    return img_final, masks

def rand_elliptical_cylinder_image(size=256, n_objects=10, rang_shape=50, range_height=1.0):
    """
    Generates an image with multiple random elliptical cylinders.

    Parameters:
        size (int): Size of the square image.
        n_objects (int): Number of objects to insert.
        rang_shape (int): Range of possible sizes for the objects.
        range_height (float): Range of possible heights for the objects.
        num_sides: number of sides of the polygon

    Retorna:
        img_final (numpy.ndarray): Image with all the objects.
        masks (list of numpy.ndarray): List of individual masks for each object.
    
    """
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []
  
    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.uniform(1, rang_shape, size=1)[0]
        sp2 = np.random.uniform(sp1 - sp1 * 0.5, sp1, size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.uniform(0, size, size=2)
        rotation = np.random.uniform(0, 180, size=1)[0]

        # Generate rectangle
        img, mask = shapes.generate_elliptical_cylinder_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation)
               
        # Add the rectangle image to the final image
        img_final = np.maximum(img_final, img)

        # Store the mask and the parameters
        masks.append(mask)
   
    return img_final, masks

def rand_ovaloid_image(size=256, n_objects=10, rang_shape=50, range_height=1.0, p_width_max=0.6):
    """
    Generates an image with multiple random ovaloids.

    Parameters:
        size (int): Size of the square image.
        n_objects (int): Number of objects to insert.
        rang_shape (int): Range of possible sizes for the objects.
        range_height (float): Range of possible heights for the objects.
        p_width_max: percentage of the central cylindrical part of the ovaloid

    Retorna:
        img_final (numpy.ndarray): Image with all the objects.
        masks (list of numpy.ndarray): List of individual masks for each object.
    """
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []
  
    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.randint(round(rang_shape/4), rang_shape, size=1)[0]
        pw = np.random.uniform(0, p_width_max, size=1)[0]
        sp2 = np.random.randint(int(max(sp1-sp1 * 0.8,1)), int(max(sp1-sp1*0.5,1)) , size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.randint(25, size-25, size=2)
        cx, cy = int(cx), int(cy)  # Convert to scalar integers
        rotation = np.random.uniform(0, 180, size=1)[0]
        
        # Generate rectangle
        img, mask = shapes.generate_ovaloid_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation,p_width=pw)
                       
        # Add the rectangle image to the final image
        img_final = np.maximum(img_final, img)

        # Store the mask and the parameters
        masks.append(mask)
     

    return img_final, masks

def rand_Lineal_DNA_image(size=256, n_objects=10, rang_shape=50, width=1, range_height=1.0, max_angulo=30, max_overlap = 0.2):
    """
    Generates an image with multiple random linear DNA strands.

    Parameters:
        size (int): Size of the square image.
        n_objects (int): Number of objects to insert.
        rang_shape (int): Range of possible sizes for the objects.
        range_height (float): Range of possible heights for the objects.
        max_angulo: Maximum variation angle of the random walker

    Retorna:
        img_final (numpy.ndarray): Image with all the objects.
        masks (list of numpy.ndarray): List of individual masks for each object.
       
    """
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    full_mask = np.zeros_like(img_final, dtype=np.uint8)
    # Lists for masks 
    masks = []
    images =[]
    count = 0
    n_iter = 0
    while count < n_objects and n_iter < n_objects*4:
        n_iter +=1
        # Generate random parameters
        sp1 = np.random.randint(round(rang_shape/4), rang_shape, size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.1, range_height, size=1)[0]
        cx, cy = np.random.randint(25, size-25, size=2)
        cx, cy = int(cx), int(cy)  # Convert to scalar integers
                
        # Generate object
        img, mask = shapes.generate_Lineal_DNA_image(size=size, size_shape=(sp1, width), height=h, cx=cx, cy=cy, max_angulo=max_angulo)
        if mask_tools.mask_overlap(mask, full_mask) <= max_overlap:
            count +=1               
            masks.append(mask)
            images.append(img)
            full_mask = mask_tools.stack_masks(masks)
   
    return images, masks, full_mask


def rand_circular_DNA_image(size=256,
                            n_objects=10,
                            rang_shape=50,
                            width=1,
                            range_height=1.0,
                            p_points_to_move=0.2,
                            dmax=2,
                            scale_pct=0.5,
                            max_overlap=0.2,
                            min_inner_area=20,
                            min_gap_width=3):
    """
    Generates an image with multiple random closed circular DNA strands.

    Parameters (new):
        min_inner_area : minimum area of the inner hole (in pixels).
        min_gap_width  : minimum radius of the inner hole (in pixels).

    Retorna:
        images (list[np.ndarray]): list of individual images.
        masks  (list[np.ndarray]): list of individual masks.
        full_mask (np.ndarray): combined mask of all objects.
    """
    img_final = np.zeros((size, size), dtype=np.float32)
    full_mask = np.zeros_like(img_final, dtype=np.uint8)

    masks = []
    images = []

    count = 0
    n_iter = 0
    max_iter = n_objects * 4  # for example

    while count < n_objects and n_iter < max_iter:
        n_iter += 1

        # Random parameters of the object
        sp1 = np.random.randint(round(rang_shape * (1.0 - scale_pct)),
                                int(rang_shape * (1.0 + scale_pct)),
                                size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.1,
                              range_height,
                              size=1)[0]
        cx, cy = np.random.randint(25, size - 25, size=2)
        cx, cy = int(cx), int(cy)

        # Generate a "circular" DNA
        img, mask = shapes.generate_circular_DNA_image(
            size=size,
            size_shape=(sp1, width),
            height=h,
            cx=cx,
            cy=cy,
            p_points_to_move=p_points_to_move,
            dmax=dmax
        )

        # 1) Check that it really forms a "decent" closed loop
        if not is_valid_circular_dna(mask,
                                     min_inner_area=min_inner_area,
                                     min_gap_width=min_gap_width):
            continue  # discard and try another

        # 2) Check overlap with the already accepted ones
        if mask_tools.mask_overlap(mask, full_mask) <= max_overlap:
            masks.append(mask)
            images.append(img)
            full_mask = mask_tools.stack_masks(masks)
            count += 1

    return images, masks, full_mask

# def rand_circular_DNA_image(size=256, n_objects=10, rang_shape=50, width=1, 
#                             range_height=1.0, p_points_to_move=0.2, dmax = 2, 
#                             scale_pct= 0.5, max_overlap = 0.2):
#     """
#     Generates an image with multiple random closed circular DNA strands.

#     Parameters:
#         size (int): Size of the square image.
#         n_objects (int): Number of objects to insert.
#         rang_shape (int): Range of possible sizes for the objects.
#         range_height (float): Range of possible heights for the objects.
#         p_points_to_move: Percentage of points that will be moved perpendicularly 
#         dmax: Maximum perpendicular displacement distance of those points

#     Retorna:
#         img_final (numpy.ndarray): Image with all the rectangles.
#         masks (list of numpy.ndarray): List of individual masks for each rectangle.
       
#     """
#     # Imagen base negra
#     img_final = np.zeros((size, size), dtype=np.float32)
#     full_mask = np.zeros_like(img_final, dtype=np.uint8)
#     # Lists for masks 
#     masks = []
#     images =[]
   
#     count = 0
#     n_iter = 0
#     while count < n_objects and n_iter < n_objects*4:
#         n_iter +=1
#         # Generate random parameters
#         sp1 = np.random.randint(round(rang_shape*(1.0-scale_pct)), rang_shape*(1.0+scale_pct), size=1)[0]
#         h = np.random.uniform(range_height - range_height * 0.1, range_height, size=1)[0]
#         cx, cy = np.random.randint(25, size-25, size=2)
#         cx, cy = int(cx), int(cy)  # Convert to scalar integers
                
#         # Generate rectangle
#         img, mask = shapes.generate_circular_DNA_image(size=size, size_shape=(sp1, width), height=h, cx=cx, cy=cy, p_points_to_move=p_points_to_move,dmax = dmax)
#         if mask_tools.mask_overlap(mask, full_mask) <= max_overlap:
#             masks.append(mask)
#             images.append(img)
#             full_mask = mask_tools.stack_masks(masks)
#             count +=1

#     return images, masks,full_mask

def rand_surface_image(size=256, n_objects=10, rang_shape=50, range_height=1.0, n_points_max=14):
    """
    Generates an image with multiple random surfaces.

    Parameters:
        size (int): Size of the square image.
        n_objects (int): Number of objects to insert.
        rang_shape (int): Range of possible sizes for the rectangles.
        range_height (float): Range of possible heights for the rectangles.
        n_points_max: Maximum number of random points to generate inside the shape

    Retorna:
        img_final (numpy.ndarray): Image with all the rectangles.
        masks (list of numpy.ndarray): List of individual masks for each rectangle.
    """
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    
    # Lists for masks and parameters
    masks = []
   
    for _ in range(n_objects):
        # Generate random parameters
        sp1 = np.random.randint(round(rang_shape/4), rang_shape, size=1)[0]
        sp2 = np.random.randint(int(max(sp1-sp1 * 0.2,1)), int(max(sp1+sp1*0.2,1)) , size=1)[0]
        h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
        cx, cy = np.random.randint(0, size, size=2)
        cx, cy = int(cx), int(cy)  # Convert to scalar integers
        num_puntos = np.random.randint(2, max(4,n_points_max), size=1)[0]
        
        # Generate rectangle
        img, mask = shapes.generate_surface_image(size=size, size_shape=(sp1, sp2), height_min=0.1*h, height_max=h, cx=cx, cy=cy, num_puntos=num_puntos)
                       
        # Add the rectangle image to the final image
        img_final = np.maximum(img_final, img)
        img_final = img_final / np.max(img_final) if np.max(img_final) != 0 else img_final
       
        # Store the mask and the parameters
        masks.append(mask)
    
    return range_height* img_final, masks

#full cover
def rand_neighbors_ovaloid_image(size=256, n_objects=10, rang_shape=50, range_height=1.0, p_width_max=0.6):
    cxx=[]
    cyy=[]
    rot=[]
    spp1=[]
    spp2=[]
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    img = np.zeros((size, size), dtype=np.float32)
    # Lists for masks and parameters
    masks = []

    # Generate random parameters
    sp1 = np.random.randint(round(rang_shape/4), rang_shape, size=1)[0]
    pw = np.random.uniform(0, p_width_max, size=1)[0]
    sp2 = np.random.randint(int(max(sp1-sp1 * 0.8,1)), int(max(sp1-sp1*0.4,1)) , size=1)[0]
    h = np.random.uniform(range_height - range_height * 0.8, range_height, size=1)[0]
    cx, cy = np.random.randint(25, size-25, size=2)
    #cx, cy = int(cx), int(cy)  # Convert to scalar integers
    rotation = np.random.uniform(0, 180, size=1)[0]
   
    count=0          #total number of neighbours added
    index = 0       #current neighbour index
    n_iter = 0      #iteration counter
    #print(f'{n_iter}\t{count}\t{index}')
    while count<n_objects and n_iter<n_objects*4:
        n_iter +=1
        img, mask, (sp1,sp2), cx, cy, h, rotation,isnew = add_neighbor_ovaloid(img, size_shape=(sp1,sp2), height=h, cx0 = cx, cy0=cy,p_width=pw, rotation0=rotation)
        if isnew:
            cxx.append(cx)
            cyy.append(cy)
            rot.append(rotation)
            spp1.append(sp1)
            spp2.append(sp2)
            count +=1
            #index +=1

            #print(f'{n_iter}\t{count}\t{index}')
             # Add the rectangle image to the final image
            img_final = np.maximum(img_final, img)

            # Store the mask and the parameters
            masks.append(mask)
        elif count>1:
            index= np.random.randint (0, count-1, size=1)[0]
            sp1,sp2, cx, cy, rotation = spp1[index],spp2[index],cxx[index],cyy[index],rot[index]
        
       
    #Add noise
    full_mask = stack_masks(masks)
    img_final = gnoise.add_perline_noise_inside_mask(img_final,full_mask,scale_px=round(sp1/8), value=0.3)
    return img_final, masks

def rand_neighbors_ellipsoid_image(size=256, n_objects=10, rang_shape=50,p_eccentrisity=0.0, range_height=1.0):
    cxx=[]
    cyy=[]
    rot=[]
    spp1=[]
    spp2=[]
    # Imagen base negra
    img_final = np.zeros((size, size), dtype=np.float32)
    img = np.zeros((size, size), dtype=np.float32)
    # Lists for masks and parameters
    masks = []

    # Generate random parameters
    sp1 = np.random.randint(round(rang_shape/4), rang_shape, size=1)[0]
    if p_eccentrisity>0.0:
        sp2 = np.random.randint(round(sp1-sp1*p_eccentrisity), round(sp1+sp1*p_eccentrisity)+1, size=1)[0]
        sp2 = round(sp1-sp1*p_eccentrisity)
    else:
        sp2=sp1
    h = np.random.uniform(range_height - range_height * 0.5, range_height, size=1)[0]
    cx, cy = np.random.randint(25, size-25, size=2)
    #cx, cy = int(cx), int(cy)  # Convert to scalar integers
    rotation = np.random.uniform(0, 180, size=1)[0]
   
    count=0          #total number of neighbours added
    index = 0       #current neighbour index
    n_iter = 0      #iteration counter
    #print(f'{n_iter}\t{count}\t{index}')
    while count<n_objects and n_iter<n_objects*2:
        n_iter +=1
        img, mask, (sp1,sp2), cx, cy, h, rotation,isnew = add_neighbor_ellipsoid(img, size_shape=(sp1,sp2), height=h, cx0 = cx, cy0=cy, rotation0=rotation)
        if isnew:
            cxx.append(cx)
            cyy.append(cy)
            rot.append(rotation)
            spp1.append(sp1)
            spp2.append(sp2)
            count +=1
            #index +=1

            #print(f'{n_iter}\t{count}\t{index}')
             # Add the rectangle image to the final image
            img_final = np.maximum(img_final, img)

            # Store the mask and the parameters
            masks.append(mask)
        elif count>1:
            index= np.random.randint (0, count-1, size=1)[0]
            sp1,sp2, cx, cy, rotation = spp1[index],spp2[index],cxx[index],cyy[index],rot[index]
        
       
    #Add noise
    full_mask = stack_masks(masks)
    img_final = gnoise.add_perline_noise_inside_mask(img_final,full_mask,scale_px=round(sp1/8), value=0.3)
    return img_final, masks

def add_neighbor_ovaloid(image, size_shape=(25, 12), height=1, cx0 = 128, cy0=128,p_width=0.3, rotation0=0,
                         dc=0.10,ds=0.1, dh=0.05,drot=20, overlap = 0.2):

    size,_ = image.shape
    
    mask1 = np.zeros((size, size), dtype=np.uint8)
    full_mask = np.zeros((size, size), dtype=np.uint8)
    
    full_mask = (image > 0).astype(np.uint8)*255
    isnew=False
    for I in range(4):
        #randomness in shape and position
        type_vecino = np.random.randint(1,4)
        sp1=np.random.uniform(size_shape[0]-size_shape[0]*ds, size_shape[0]+size_shape[0]*ds, size=1)[0]
        sp2=np.random.uniform(size_shape[1]-size_shape[1]*ds, size_shape[1]+size_shape[1]*ds, size=1)[0]
        h=np.random.uniform(height-height*dh, height+height*dh, size=1)[0]
        
        rotation = np.random.uniform(rotation0-drot, rotation0+drot, size=1)[0]
        rotation_rad = np.radians(rotation)
        alfa = 90-rotation0
        alfa_rad = np.radians(alfa)
        ps_re = 0.9
        if type_vecino ==1:
            #vecino derecha
            cx = cx0+ps_re*sp1*np.cos(rotation_rad)
            cy = cy0-ps_re*sp1*np.sin(rotation_rad)
        elif type_vecino ==2:
            #vecino izq
            cx = cx0-ps_re*sp1*np.cos(rotation_rad)
            cy = cy0+ps_re*sp1*np.sin(rotation_rad)
        elif type_vecino ==3:
            #vecino abajo        
            cx = cx0+ps_re*sp2*np.cos(alfa_rad)
            cy = cy0+ps_re*sp2*np.sin(alfa_rad)
        elif type_vecino ==4:
            #vecino arriba
            cx = cx0-ps_re*sp2*np.sin(rotation_rad)
            cy = cy0-ps_re*sp2*np.cos(rotation_rad)
        
        cx = np.random.uniform(cx-sp1*dc, cx+sp1*dc, size=1)[0]
        cy = np.random.uniform(cy-sp1*dc, cy+sp1*dc, size=1)[0]

        if cx<size and cx>0 and cy<size and cy>0:
            img, mask1 = shapes.generate_ovaloid_image(size=size, size_shape=(sp1,sp2), height=h, cx = cx, cy=cy, rotation=rotation,p_width=p_width)
            
            # imgDNA,_ = rand_Lineal_DNA_image(size=256, n_objects=100, rang_shape=10, width=1, range_height=0.3, max_angulo=30)
            # result = np.where(mask, imgDNA+img, matriz)
            if mask_overlap(mask1, full_mask)<overlap:
                image = np.maximum(image, img)
                #image = np.average(image,img)
                isnew = True
                break
    
    return image, mask1, (sp1,sp2), cx, cy, h, rotation, isnew

def add_neighbor_ellipsoid(image, size_shape=(25, 12), height=1, cx0 = 128, cy0=128, rotation0=0,
                         dc=0.10,ds=0.0, dh=0.05,drot=20, overlap = 0.2):

    size,_ = image.shape
    
    mask1 = np.zeros((size, size), dtype=np.uint8)
    full_mask = np.zeros((size, size), dtype=np.uint8)
    
    full_mask = (image > 0).astype(np.uint8)*255
    isnew=False
    for I in range(4):
        #randomness in shape and position
        type_vecino = np.random.randint(1,4)
        sp1=np.random.uniform(size_shape[0]-size_shape[0]*ds, size_shape[0]+size_shape[0]*ds, size=1)[0]
        sp2=np.random.uniform(size_shape[1]-size_shape[1]*ds, size_shape[1]+size_shape[1]*ds, size=1)[0]
        h=np.random.uniform(height-height*dh, height+height*dh, size=1)[0]

        rotation = np.random.uniform(rotation0-drot, rotation0+drot, size=1)[0]
        rotation_rad = np.radians(rotation)
        alfa = 90-rotation0
        alfa_rad = np.radians(alfa)
        ps_re = 0.9
        if type_vecino ==1:
            #vecino derecha
            cx = cx0+ps_re*sp1*np.cos(rotation_rad)
            cy = cy0-ps_re*sp1*np.sin(rotation_rad)
        elif type_vecino ==2:
            #vecino izq
            cx = cx0-ps_re*sp1*np.cos(rotation_rad)
            cy = cy0+ps_re*sp1*np.sin(rotation_rad)
        elif type_vecino ==3:
            #vecino abajo        
            cx = cx0+ps_re*sp2*np.cos(alfa_rad)
            cy = cy0+ps_re*sp2*np.sin(alfa_rad)
        elif type_vecino ==4:
            #vecino arriba
            cx = cx0-ps_re*sp2*np.sin(rotation_rad)
            cy = cy0-ps_re*sp2*np.cos(rotation_rad)
        
        cx = np.random.uniform(cx-sp1*dc, cx+sp1*dc, size=1)[0]
        cy = np.random.uniform(cy-sp1*dc, cy+sp1*dc, size=1)[0]

        if cx<size and cx>0 and cy<size and cy>0:
            # Generate ellipse
            img, mask1 = shapes.generate_ellipsoid_image(size=size, size_shape=(sp1, sp2), height=h, cx=cx, cy=cy, rotation=rotation)
             
            if mask_overlap(mask1, full_mask)<overlap:
                image = np.maximum(image, img)
                #image = np.average(image,img)
                isnew = True
                break
    
    return image, mask1, (sp1,sp2), cx, cy, h, rotation, isnew

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

   
    #(sp1,sp2), cx, cy, h0, rotation = (100, 50), 60, 60, 1, 90
    
    # # img,masks = rand_neighbors_ovaloid_image(size=256, n_objects=10, rang_shape=100, range_height=1.0, p_width_max=0.6)
    img,masks = rand_neighbors_ellipsoid_image(size=256, n_objects=5, rang_shape=12,p_eccentrisity=0.5, range_height=1.0)
    # # Add the rectangle image to the final image
    # full_mask = stack_masks(masks)
    #img, mask =  img, mask = shapes.generate_ellipsoid_image(size=200, size_shape=(200, 100), height=1, cx=100, cy=100, rotation=45)
    # img, masks = rand_polygon_image(size=256, n_objects=1, rang_shape=128, p_eccentrisity=0.0,range_height=1.0, num_sides=5)
    plt.figure(figsize=(12, 12))

    # Original image with the indices
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='viridis')
    plt.title('img')
    plt.colorbar()
    plt.show()

    # # Add the number of each mask   
    # for i, mask in enumerate(masks):
    #     if np.sum(mask) == 0:
    #         continue  # Skip empty masks
    #     cy, cx = center_of_mass(mask)
    #     plt.text(cx, cy, str(i), color='black', fontsize=8, ha='center', va='center')

    # # Accumulated mask
    # plt.subplot(1, 2, 2)
    # plt.imshow(full_mask, cmap='viridis')
    # plt.title('mask')
    # plt.colorbar()

    # plt.show()
    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='viridis')
    # plt.title('img')
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(full_mask, cmap='viridis')
    # plt.title('mask')
    # plt.colorbar()

    # plt.show()
       
    # tip, parm = rand_double_tip(rrang = [35,35],  thetarang = [10,25], seed=None)
   
    # rect, mask= rand_rectangle_image(size = 512, rang_shape=100, range_height = 1)
    # #elipse,mask =rand_surface_image(size=256, n_objects=1, rang_shape=50, range_height=5.0, n_points_max=25)

    # #rect = np.maximum(rect,elipse)
    # topo = shapes.tip_convolution(rect,tip)
    # #tip = shapes.generate_double_elliptical_tip_image(20, 20, 1, 20, 20, 1, 0,0,0,20)
    # print (parm)
    # Visualization
    # plt.figure(figsize=(12, 12))
    # plt.subplot(2, 3, 1)
    # plt.imshow(img, cmap='viridis')
    # plt.title('img')
    # plt.colorbar()

    # plt.subplot(2, 3, 2)
    # plt.imshow(masks[0], cmap='viridis')
    # plt.title('mask0')
    # plt.colorbar()

    # plt.subplot(2, 3, 3)
    # plt.imshow(masks[1], cmap='viridis')
    # plt.title('topo')
    # plt.colorbar()

    # plt.subplot(2, 3, 4)
    # plt.imshow(masks[2], cmap='viridis')
    # plt.title('topo')
    # plt.colorbar()

    # plt.subplot(2, 3, 5)
    # plt.imshow(masks[2], cmap='viridis')
    # plt.title('topo')
    # plt.colorbar()

    # plt.subplot(2, 3, 6)
    # plt.imshow(masks[-1], cmap='viridis')
    # plt.title('topo')
    # plt.colorbar()

    # plt.show()

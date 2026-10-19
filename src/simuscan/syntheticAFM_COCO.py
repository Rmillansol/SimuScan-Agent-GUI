# Generates synthetic AFM data to train an ML model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .COCOBuilder import COCOBuilder
from .prosess_stl import  generate_rand_image_from_objSTL
from .process_sect import generate_rand_image_from_obj_sections, generate_rand_colony_image_from_obj_sections
from .image_shape import generate_image_shape
from .domains_shape import generate_domains_shape, split_domain_instances
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from datetime import datetime
from .SyntheticParams import SyntheticParams, ArtifactsConfig
from . import shapes as shapes
from . import shapes_tips as sh_tips
from . import randshapes as rsh
from . import noisegenerator as gnoise
from . import substratum as subs
from . import mask_tools
from pathlib import Path
from .paths import stl_dir, datasets_root, ensure_dir, user_data_root, ref_objects_dir
import time
import os
import random
from src.bridge.ui_bridge import ui
import cv2
import json
import os

def _make_dataset_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = datasets_root() / ts
    return ensure_dir(out)

def Default_cmap():
    # Function that returns the classic WSxM topography cmap
    blue_points = np.array([[0, 0], [85, 0], [160, 76], [220, 213], [255, 251]])
    green_points = np.array([[0, 0], [17, 0], [47, 21], [106, 96],
                             [182, 208], [232, 248], [255, 249]])
    red_points = np.array([[0, 0], [32, 119], [94, 198],
                           [152, 238], [188, 249], [255, 249]])

    def interpolate_channel(control_points):
        x = control_points[:, 0]
        y = control_points[:, 1]
        return interp1d(x, y, kind='linear',
                        bounds_error=False, fill_value="extrapolate")

    def generate_colormap():
        levels = np.linspace(0, 255, 256)

        blue_interp = interpolate_channel(blue_points)
        green_interp = interpolate_channel(green_points)
        red_interp = interpolate_channel(red_points)

        blue_values = blue_interp(levels) / 255.0
        green_values = green_interp(levels) / 255.0
        red_values = red_interp(levels) / 255.0

        colors = np.stack([red_values, green_values, blue_values], axis=1)
        return ListedColormap(colors)

    custom_cmap = generate_colormap()
    return custom_cmap

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

def _render_domain_layer(objetos, image_size, dconf):
    """
    Render the grain/domain field as a BASE LAYER, before any object is placed.

    A domain field is a scene that fills the frame, not one more object in the
    bag. It is folded into the substrate by the caller, and since the pipeline
    ADDS the substrate to the object topography, ordinary objects end up sitting
    ON TOP of the film at their full height. Max-compositing them (the rule used
    between ordinary objects) would instead sink them into the film and flatten
    the grain texture underneath them.

    Returns (scene_topo, fields) where `fields` is a list of
    (obj, label, meta_by_id) whose instances the caller resolves once it knows
    which objects landed on the film.
    """
    domains = [o for o in objetos if o.shape == "domains"]
    if not domains:
        return None, []
    if len(domains) > 1:
        # Two fields would both fill the frame and their masks would contradict
        # each other on ~97% of the pixels. Fail loudly instead of writing a
        # dataset whose labels disagree with themselves.
        raise ValueError(
            "Only one 'shape: domains' object can be enabled at a time "
            f"(enabled: {[o.label for o in domains]}). Use domains_mode on the "
            "single enabled object to choose tessellation or overlap_interactive."
        )

    obj = domains[0]
    if random.random() >= obj.probability:
        return None, []

    scene_topo, label, meta_by_id = generate_domains_shape(obj, image_size, dconf)
    return scene_topo, [(obj, label, meta_by_id)]


def filter_against_full_mask(topo_t, masks_t, full_mask,
                             max_overlap_frac=0.2):
    """
    Filters instances (topo_t[i], masks_t[i]) by comparing each mask with the
    global full_mask (accumulated occupancy). If the overlap relative to the area of the
    NEW mask exceeds max_overlap_frac, that instance is discarded.

    - Keeps the original order of masks_t.
    - Updates full_mask "on the fly" with the accepted instances.

    Args:
        topo_t: list/array of (H,W) float
        masks_t: list/array of (H,W) uint8/bool
        full_mask: (H,W) uint8/bool (global accumulated)
        max_overlap_frac: float in [0,1]

    Returns:
        topo_keep: list of accepted topographies
        masks_keep: list of accepted masks (uint8 0/1)
        kept_idx: original indices that were accepted
        full_mask_updated: full_mask updated after accepting
    """
    occ = (full_mask > 0).astype(np.uint8).copy()

    topo_keep, masks_keep, kept_idx = [], [], []

    for i, m in enumerate(masks_t):
        m = (m > 0).astype(np.uint8)
        area = int(m.sum())
        if area == 0:
            continue  # empty mask -> no point adding it

        inter = int((m & occ).sum())
        overlap_frac = inter / area

        if overlap_frac > max_overlap_frac:
            continue  # DISCARD

        # ACCEPT
        topo_keep.append(topo_t[i])
        masks_keep.append(m)
        kept_idx.append(i)
        occ = np.maximum(occ, m)

    return topo_keep, masks_keep, kept_idx, occ

def generate_data_for_training(config_yaml_path: str,
    params_json_path: str | None = None,
    base_output_dir: str | Path = "./data/Datasets",
    run_name: str | None = None,
    show_preview: bool = False,
    n_images: int | None = None,
    ):
    # 1) Load params
    if params_json_path is None:
        params = SyntheticParams.from_yaml(config_yaml_path)

        # 2) Create the output folder (NO hardcoded './Datasets')
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S_COCO")

        dataset_path = Path(base_output_dir) / run_name
        dataset_path.mkdir(parents=True, exist_ok=True)

        json_path = dataset_path / "synthetic_params.json"
        params.save_json(str(json_path))

        params.dataset_path = str(dataset_path)
        params.file_name = "synthetic_params.json"
    else:
        params = SyntheticParams.load_json(params_json_path)


    class_info = params.get_class_info()
   
    label_to_id = {label: i for i, (label, _) in enumerate(class_info)}
    categories = [
    {"id": idx, "name": class_name, "supercategory": supercat}
    for idx, (class_name, supercat) in enumerate(class_info)]
    
    coco_builder = COCOBuilder(params.dataset_path, 
                               categories, 
                               split_ratios=params.training_split_ratios,
                               save_instance_map=params.save_instance_map)

    t0 = time.time()
    # Generates different kinds of images depending on the enabled objects
    if n_images is None:
        n_images = params.n_images

    for idx in range(n_images):
        # define tip
        tip, radius_tip,centre_r2,theta,is_double = sh_tips.rand_tip(params.tip, seed=None)
        
        # parameters
        is_new = True
        is_colony = False
        masks = []
        topo = np.zeros((params.image_size,params.image_size), dtype=np.float32)
        full_mask = np.zeros((params.image_size,params.image_size), dtype=np.uint8)
        category_ids = []
        masks_meta   = []          # per-instance extras for COCO (grain_id, ...)
        masks_dilate = []          # per-instance COCO dilation (0 for domains)
        masks_merge  = []          # per-instance: merge contours into one annotation
        objetos = params.get_objects_by_enabled()    # list of enabled objects

        # ── Grain / domain field: a BASE LAYER, folded into the substrate ────
        grain_topo, grain_fields = _render_domain_layer(
            objetos, params.image_size, params.domains
        )
        # `full_mask` stays OBJECTS ONLY: it drives the substrate smoothing and
        # the clipping of grain masks, and the film must appear in neither (it
        # would smooth its own texture away, and clip the grains against
        # themselves). Exclusivity rides on a separate occupancy mask that only
        # the overlap filter sees.
        block_mask = np.zeros((params.image_size,params.image_size), dtype=np.uint8)
        if grain_fields and params.domains.block_other_objects:
            for _o, _lab, _m in grain_fields:
                block_mask = np.maximum(block_mask, (_lab > 0).astype(np.uint8))

        obj_disorder = random.sample(objetos, len(objetos))
        image = np.zeros((params.image_size,params.image_size), dtype=np.float32)
        for obj in obj_disorder:
            if obj.shape == "domains":
                continue           # already rendered as the base layer

            
            if random.random() < obj.probability:  # generates this kind of image with a given probability
                id_classe = label_to_id[obj.label]


                # check whether it is STL, Sections or DNA type
                 # Is it STL type?
                chosen_stl = obj.pick_stl_file()
                
                if obj.shape == "image" and (obj.image_folder or obj.image_file):
                    topo_t, masks_t, full_mask_t = generate_image_shape(
                        obj, params.image_size, ref_objects_dir()
                    )
                    is_new = False
                elif chosen_stl is not None and obj.shape == "STL":
                    # Only reload if the file changes
                    if obj.mesh is None or obj._last_stl_file != chosen_stl:
                        mesh_path = stl_dir()
                        obj.load_mesh(str(mesh_path), chosen_stl)

                    topo_t, masks_t, full_mask_t = generate_rand_image_from_objSTL(obj,params.image_size)
                    is_new = False
                elif obj.category == "DNA":
                    if obj.label == "linearDNA":
                        topo_t, masks_t, full_mask_t = rsh.rand_Lineal_DNA_image(size = params.image_size,
                                                    n_objects= obj.n_objects , 
                                                    rang_shape=obj.size_shape[0], 
                                                    width=obj.size_shape[1], 
                                                    range_height=obj.height[1],
                                                    max_angulo = obj.max_angel_for_DNA,
                                                    max_overlap = obj.max_overlap)
                        is_new = False
                    else:
                        topo_t, masks_t, full_mask_t = rsh.rand_circular_DNA_image(size = params.image_size,
                                                    n_objects= obj.n_objects , 
                                                    rang_shape=obj.size_shape[0], 
                                                    width=obj.size_shape[1], 
                                                    range_height=obj.height[1],
                                                    p_points_to_move=obj.variation_border_pct, 
                                                    dmax = obj.dmax_for_DNA,
                                                    scale_pct = obj.scale_pct,
                                                    max_overlap = obj.max_overlap)
                        is_new = False
                else:
                    # it is via sections
                    if is_new and obj.apply_colony and random.random() < obj.colony_probability:
                        topo_t, masks_t, full_mask_t =  generate_rand_colony_image_from_obj_sections(obj, params.image_size)
                        is_colony = True
                    else:
                        topo_t, masks_t, full_mask_t =  generate_rand_image_from_obj_sections(obj,params.image_size)
                        is_new = False
                
                # FILTER against the accumulated GLOBAL occupancy (objects + the
                # film when block_other_objects is active)
                topo_f, masks_f, kept_idx, _occ = filter_against_full_mask(
                    topo_t, masks_t, np.maximum(full_mask, block_mask),
                    max_overlap_frac=obj.max_overlap,   # threshold per object/class
                )
                for _m in masks_f:
                    full_mask = np.maximum(full_mask, _m)

                obj_id =  [id_classe] * len(masks_f)
                
                # apply convolution and mask for each object
                for i in range(len(topo_f)):
                    topo_s = sh_tips.tip_convolution_roi_fast(topo_f[i],tip)
                    umbral = topo_f[i].max() * 0.15
                    masks_f[i] = (topo_s > umbral).astype(np.uint8)
                    image = np.maximum(image,topo_f[i])

                masks += masks_f
                #full_mask = np.maximum(full_mask,full_mask_t)
                category_ids += obj_id
                masks_meta   += [None]  * len(masks_f)
                masks_dilate += [2]     * len(masks_f)
                masks_merge  += [False] * len(masks_f)
                if is_colony: 
                    is_colony = False
                    break
       
        # `image` is the running max over every object, so it is added ONCE,
        # after the loop. Adding it inside the loop counted the first object's
        # topography once per subsequent object.
        topo += image

        # ── Grain instances, clipped against whatever landed on the film ─────
        # A pixel must never be labelled both 'grain' and 'hexagon', so the
        # objects win and grains that lose too much of themselves are dropped.
        for _obj, _lab, _meta_by_id in grain_fields:
            g_masks, g_meta = split_domain_instances(
                _lab, _meta_by_id, params.domains, occupied=(full_mask > 0)
            )
            masks        += g_masks
            category_ids += [label_to_id[_obj.label]] * len(g_masks)
            masks_meta   += g_meta
            masks_dilate += [int(params.domains.mask_dilate_px)] * len(g_masks)
            masks_merge  += [True] * len(g_masks)

        # define the subset where it will be saved: train, val or test
        subset = coco_builder._choose_subset()

        # Generate the substrate
       
        imgsub = subs.rand_substrate (params.substratum, params.image_size)
        # The grain film IS the surface, so it rides with the substrate. The
        # substrate is ADDED to the object topography, which is exactly what
        # puts the objects on top of the film at their full height.
        if grain_topo is not None:
            imgsub = imgsub + grain_topo
        #subs.show_image(imgsub)
        imgsub = mask_tools.smooth_gaussian_inside_mask(imgsub,full_mask,params.sigma_smooth_inside_mask)
        
        topo += imgsub
        #subs.show_image(topo)
        # save objects plus substrate
        #coco_builder.add_data_training(topo, subset, masks, category_ids, radius_tip,centre_r2,theta,is_double)
        # tip convolution
        topo = sh_tips.tip_convolution_roi_fast(topo,tip)
        # Save without noise
        coco_builder.add_data_training(topo, subset, masks, category_ids, radius_tip,centre_r2,theta,is_double,
                                       mask_meta=masks_meta, dilate_px=masks_dilate,
                                       merge_contours=masks_merge)

        # Save versions at other resolutions
        if random.random() < params.save_low_quality_probability:
            for qu_px in params.simu_size_low_quality:
                low_quality = simulation_reduce_resolution(topo, qu_px)
                coco_builder.add_data_training(low_quality, subset, masks, category_ids, radius_tip,centre_r2,theta,is_double,
                                               mask_meta=masks_meta, dilate_px=masks_dilate,
                                       merge_contours=masks_merge)

        # ── PID simulation — after the tip, before artifacts ─────────────
        pid_topo = topo
        if params.pid.apply_pid and random.random() < params.pid.pid_probability:
            p = params.pid
            pid_topo = gnoise.simulate_pid_topography(
                topo,
                Kp             = float(np.random.uniform(p.Kp[0],             p.Kp[1])),
                Ki             = float(np.random.uniform(p.Ki[0],             p.Ki[1])),
                Kd             = float(np.random.uniform(p.Kd[0],             p.Kd[1])),
                dt             = float(np.random.uniform(p.dt[0],             p.dt[1])),
                max_speed      = float(np.random.uniform(p.max_speed[0],      p.max_speed[1])),
                scan_direction = p.scan_direction,
                stickiness_prob= float(np.random.uniform(p.stickiness_prob[0],p.stickiness_prob[1])),
                jump_prob      = float(np.random.uniform(p.jump_prob[0],      p.jump_prob[1])),
                jump_magnitude = float(np.random.uniform(p.jump_magnitude[0], p.jump_magnitude[1])),
                A0             = p.A0,
                beta           = p.beta,
                z0             = p.z0,
                ASP            = p.ASP,
            )

        # Save tip convolution + PID (without artifacts)
        if params.pid.apply_pid and not np.array_equal(pid_topo, topo):
            coco_builder.add_data_training(pid_topo, subset, masks, category_ids, radius_tip, centre_r2, theta, is_double,
                                           mask_meta=masks_meta, dilate_px=masks_dilate,
                                       merge_contours=masks_merge)

        # Add noise and artifacts
        afm_imagen = add_artefactes(pid_topo, params.artifacts)
        coco_builder.add_data_training(afm_imagen, subset, masks, category_ids, radius_tip,centre_r2,theta,is_double,
                                       mask_meta=masks_meta, dilate_px=masks_dilate,
                                       merge_contours=masks_merge)       # Save flattened
        
        # flatten the image and save
        if params.artifacts.apply_flatten:
            #afm_imagen, mask = FlattenPlus(afm_imagen,perc = params.artifacts.flatten_perc)
            afm_imagen, _offsets = FlattenByMedianDifferences(afm_imagen, smooth_window=5)
            afm_imagen, _k = SubtractGlobalPlane(afm_imagen)
            coco_builder.add_data_training(afm_imagen, subset, masks, category_ids, radius_tip,centre_r2,theta,is_double,
                                       mask_meta=masks_meta, dilate_px=masks_dilate,
                                       merge_contours=masks_merge)       # Save flattened   
        
        # Save versions at other resolutions
        if random.random() < params.save_low_quality_probability:
            for qu_px in params.simu_size_low_quality:
                low_quality = simulation_reduce_resolution(afm_imagen, qu_px)  # 256 128 etc
                coco_builder.add_data_training(low_quality, subset, masks, category_ids, radius_tip,centre_r2,theta,is_double,
                                               mask_meta=masks_meta, dilate_px=masks_dilate,
                                       merge_contours=masks_merge)       # Save low quality
          
        i = idx + 1
        elapsed = time.time() - t0
        rate = elapsed / i
        eta = rate * (params.n_images - i)
        stamp = datetime.now().strftime("%H:%M:%S")

        ui.log("INFO", f"{i}/{params.n_images} images generated | elapsed {elapsed:0.1f}s | ETA {eta:0.1f}s")


    coco_builder.save_jsons()
    params.save_json()

    # Only open matplotlib windows when running interactively
    if show_preview:
        visualize_coco_samples(params.dataset_path, subset="train", num_samples=20)

    return params.dataset_path


def _force_all_effects(params) -> None:
    """
    PREVIEW-ONLY: lift every per-image probability gate to 1.0 so that all ENABLED
    effects and objects are shown at once (otherwise a single fixed-seed image may
    randomly omit an artifact/object and its parameter changes look invisible).

    Mutates ONLY this throwaway `params` instance (loaded fresh for the preview and
    discarded afterwards). It never touches temp.yaml, the in-memory config, or any
    saved file — so dataset generation keeps the user's real probabilities. It also
    respects apply_*/enable flags (disabled things stay off) and does NOT force
    colony_probability (a colony would 'break' the object loop and hide the rest).
    """
    def _set(obj, name, val=1.0):
        if obj is not None and hasattr(obj, name):
            setattr(obj, name, val)

    for name in ("tilt_probability", "stair_probability", "pattern_probability",
                 "shapes_probability", "dirt_probability"):
        _set(params.substratum, name)
    for name in ("gaussian_probability", "loss_contact_probaility",
                 "local_adhesion_probability", "sticky_zones_probability",
                 "line_noise_probability", "periodic_probability",
                 "shift_lines_probability", "delta_probability"):
        _set(params.artifacts, name)
    _set(params.pid, "pid_probability")
    for obj in params.objects:
        _set(obj, "probability")


def generate_preview_image(config_yaml_path: str | None = None,
                           params_json_path: str | None = None,
                           seed: int | None = 0,
                           force_all: bool = True) -> np.ndarray:
    """
    Render a SINGLE synthetic AFM image (full pipeline, full resolution) WITHOUT
    building a dataset: no COCOBuilder, no annotations, no saved variants, no disk
    writes. Returns the final flattened AFM image as a 2-D float32 array — the same
    render path as generate_data_for_training()'s final image, so it looks exactly
    like what the dataset will contain (just not persisted).

    A fixed `seed` (default 0) makes the image fully reproducible, so changing one
    parameter shows the effect of *that* parameter alone. Pass a different seed
    (or None) for a new random layout.

    force_all=True (default): show ALL enabled effects/objects at once by lifting
    the per-image probability gates to 1.0 (preview-only, never persisted). Set
    False to render a single image honouring the real per-image probabilities.
    """
    if params_json_path is not None:
        params = SyntheticParams.load_json(params_json_path)
    else:
        params = SyntheticParams.from_yaml(config_yaml_path)

    # Applied to the throwaway params only — never written back anywhere.
    if force_all:
        _force_all_effects(params)

    if seed is None:
        return _render_preview(params)

    random.seed(seed)
    np.random.seed(seed)
    # Some noise generators build their own np.random.default_rng() (entropy-seeded)
    # which ignores np.random.seed(); make those deterministic for the preview so a
    # fixed seed reproduces the SAME image. Restored in finally, so dataset
    # generation (which must stay random per image) is untouched.
    _orig_default_rng = np.random.default_rng
    _rng_ctr = [0]
    def _det_default_rng(s=None):
        if s is None:
            s = (int(seed) + 9973 + _rng_ctr[0]) & 0x7FFFFFFF
            _rng_ctr[0] += 1
        return _orig_default_rng(s)
    np.random.default_rng = _det_default_rng
    try:
        return _render_preview(params)
    finally:
        np.random.default_rng = _orig_default_rng


def _render_preview(params) -> np.ndarray:
    """Render one final AFM image from an already-loaded SyntheticParams (no I/O).
    Kept in sync with the per-image body of generate_data_for_training()."""
    size = params.image_size
    tip, radius_tip, centre_r2, theta, is_double = sh_tips.rand_tip(params.tip, seed=None)

    is_new    = True
    is_colony = False
    full_mask = np.zeros((size, size), dtype=np.uint8)
    topo      = np.zeros((size, size), dtype=np.float32)
    image     = np.zeros((size, size), dtype=np.float32)

    objetos = params.get_objects_by_enabled()

    # Grain / domain field as a base layer — see generate_data_for_training()
    grain_topo, grain_fields = _render_domain_layer(objetos, size, params.domains)
    block_mask = np.zeros((size, size), dtype=np.uint8)
    if grain_fields and params.domains.block_other_objects:
        for _o, _lab, _m in grain_fields:
            block_mask = np.maximum(block_mask, (_lab > 0).astype(np.uint8))

    obj_disorder = random.sample(objetos, len(objetos))
    for obj in obj_disorder:
        if obj.shape == "domains":
            continue           # already rendered as the base layer
        if random.random() < obj.probability:
            chosen_stl = obj.pick_stl_file()
            if obj.shape == "image" and (obj.image_folder or obj.image_file):
                topo_t, masks_t, full_mask_t = generate_image_shape(obj, size, ref_objects_dir())
                is_new = False
            elif chosen_stl is not None and obj.shape == "STL":
                if obj.mesh is None or obj._last_stl_file != chosen_stl:
                    obj.load_mesh(str(stl_dir()), chosen_stl)
                topo_t, masks_t, full_mask_t = generate_rand_image_from_objSTL(obj, size)
                is_new = False
            elif obj.category == "DNA":
                if obj.label == "linearDNA":
                    topo_t, masks_t, full_mask_t = rsh.rand_Lineal_DNA_image(
                        size=size, n_objects=obj.n_objects, rang_shape=obj.size_shape[0],
                        width=obj.size_shape[1], range_height=obj.height[1],
                        max_angulo=obj.max_angel_for_DNA, max_overlap=obj.max_overlap)
                else:
                    topo_t, masks_t, full_mask_t = rsh.rand_circular_DNA_image(
                        size=size, n_objects=obj.n_objects, rang_shape=obj.size_shape[0],
                        width=obj.size_shape[1], range_height=obj.height[1],
                        p_points_to_move=obj.variation_border_pct, dmax=obj.dmax_for_DNA,
                        scale_pct=obj.scale_pct, max_overlap=obj.max_overlap)
                is_new = False
            else:
                if is_new and obj.apply_colony and random.random() < obj.colony_probability:
                    topo_t, masks_t, full_mask_t = generate_rand_colony_image_from_obj_sections(obj, size)
                    is_colony = True
                else:
                    topo_t, masks_t, full_mask_t = generate_rand_image_from_obj_sections(obj, size)
                    is_new = False

            topo_f, masks_f, kept_idx, _occ = filter_against_full_mask(
                topo_t, masks_t, np.maximum(full_mask, block_mask),
                max_overlap_frac=obj.max_overlap)
            for _m in masks_f:
                full_mask = np.maximum(full_mask, _m)

            for i in range(len(topo_f)):
                image = np.maximum(image, topo_f[i])

            if is_colony:
                is_colony = False
                break

    # `image` is the running max over every object: add it once, after the loop.
    topo += image

    # Substrate (+ the grain film, + smoothing under objects)
    imgsub = subs.rand_substrate(params.substratum, size)
    if grain_topo is not None:
        imgsub = imgsub + grain_topo
    imgsub = mask_tools.smooth_gaussian_inside_mask(imgsub, full_mask, params.sigma_smooth_inside_mask)
    topo += imgsub

    # Tip convolution
    topo = sh_tips.tip_convolution_roi_fast(topo, tip)

    # PID feedback
    pid_topo = topo
    if params.pid.apply_pid and random.random() < params.pid.pid_probability:
        p = params.pid
        pid_topo = gnoise.simulate_pid_topography(
            topo,
            Kp             = float(np.random.uniform(p.Kp[0],             p.Kp[1])),
            Ki             = float(np.random.uniform(p.Ki[0],             p.Ki[1])),
            Kd             = float(np.random.uniform(p.Kd[0],             p.Kd[1])),
            dt             = float(np.random.uniform(p.dt[0],             p.dt[1])),
            max_speed      = float(np.random.uniform(p.max_speed[0],      p.max_speed[1])),
            scan_direction = p.scan_direction,
            stickiness_prob= float(np.random.uniform(p.stickiness_prob[0],p.stickiness_prob[1])),
            jump_prob      = float(np.random.uniform(p.jump_prob[0],      p.jump_prob[1])),
            jump_magnitude = float(np.random.uniform(p.jump_magnitude[0], p.jump_magnitude[1])),
            A0             = p.A0,
            beta           = p.beta,
            z0             = p.z0,
            ASP            = p.ASP,
        )

    # Artifacts + flatten (same final variant the dataset stores)
    afm_imagen = add_artefactes(pid_topo, params.artifacts)
    if params.artifacts.apply_flatten:
        afm_imagen, _offsets = FlattenByMedianDifferences(afm_imagen, smooth_window=5)
        afm_imagen, _k       = SubtractGlobalPlane(afm_imagen)

    return afm_imagen


def visualize_coco_samples(coco_folder, subset="train", num_samples=5):
    # Paths
    images_dir = os.path.join(coco_folder, 'images', subset)
    annotations_path = os.path.join(coco_folder, 'annotations', f'instances_{subset}.json')

    # Load annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Create id -> file_name map
    id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    # Create id -> list of annotations map
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id_to_annotations.setdefault(ann['image_id'], []).append(ann)

    # Show ALL images of the subset, up to a maximum of num_samples (in order)
    all_ids = list(id_to_file.keys())
    selected_ids = all_ids[:num_samples]
    ui.log("INFO", f"Showing {len(selected_ids)} of {len(all_ids)} '{subset}' image(s) "
                   f"(max {num_samples}).")

    for image_id in selected_ids:
        img_path = os.path.join(images_dir, id_to_file[image_id])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annotations = image_id_to_annotations.get(image_id, [])

        for ann in annotations:
            bbox = ann['bbox']
            segmentation = ann['segmentation'][0]
            category_name = ann.get('category_name', str(ann['category_id']))

            # Draw bbox
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)

            # Draw segmentation
            if len(segmentation) >= 4:
                pts = np.array(segmentation, dtype=np.int32).reshape((-1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

            # Draw class name
            cv2.putText(img, category_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Send to GUI plot panel
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Image ID: {image_id}")
        plt.axis('off')
        ui.plot(fig)

def save_coco_afm_samples_all(coco_folder, subset="train"):
    """
    Processes ALL images of the COCO subset.
    For each image it saves:
      - <name>_noann.png  (colormap WSxM)
      - <name>_ann.png    (colormap WSxM + anotaciones)
    """

    # Input paths
    images_dir = os.path.join(coco_folder, "images", subset)
    annotations_path = os.path.join(coco_folder, "annotations", f"instances_{subset}.json")

    # Output folder: same name + "out"
    parent_dir = os.path.dirname(coco_folder)
    coco_name = os.path.basename(coco_folder.rstrip(os.sep))
    out_root = os.path.join(parent_dir, "Out_" + coco_name)
    out_dir = os.path.join(out_root, subset)
    os.makedirs(out_dir, exist_ok=True)

    # Load annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}

    image_id_to_annotations = {}
    for ann in coco_data["annotations"]:
        image_id_to_annotations.setdefault(ann["image_id"], []).append(ann)

    # PROCESS ALL IMAGES
    selected_ids = list(id_to_file.keys())

    # WSxM colormap
    cmap = Default_cmap()

    for image_id in selected_ids:
        file_name = id_to_file[image_id]
        img_path = os.path.join(images_dir, file_name)

        # Read AFM (8/16 bit or RGB)
        raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            ui.log("WARN", f"Could not read: {img_path}")
            continue

        # To gray float32
        if raw.ndim == 3:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = raw.astype(np.float32)

        # Normalization 0-1
        vmin, vmax = np.min(gray), np.max(gray)
        if vmax > vmin:
            norm = (gray - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(gray)

        # Apply WSxM colormap
        img_afm = (cmap(norm)[..., :3] * 255).astype(np.uint8)

        # Save without annotations
        base_name, _ = os.path.splitext(file_name)
        out_noann = os.path.join(out_dir, f"{base_name}_noann.png")
        cv2.imwrite(out_noann, cv2.cvtColor(img_afm, cv2.COLOR_RGB2BGR))

        # Image for annotations
        img_ann = img_afm.copy()
        annotations = image_id_to_annotations.get(image_id, [])

        for ann in annotations:
            bbox = ann["bbox"]
            x, y, w, h = map(int, bbox)
            category_name = ann.get("category_name", str(ann["category_id"]))

            # Draw bbox
            cv2.rectangle(img_ann, (x, y), (x + w, y + h),
                          color=(0, 255, 0), thickness=1)

            # Segmentation (COCO polygon)
            seg = ann.get("segmentation", [])
            if len(seg) > 0 and len(seg[0]) >= 6:
                pts = np.array(seg[0], dtype=np.int32).reshape((-1, 2))
                cv2.polylines(img_ann, [pts], True, (173, 216, 230), 1)

            # Text
            cv2.putText(img_ann, category_name,
                        (x, max(0, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)

        # Save with annotations
        out_ann = os.path.join(out_dir, f"{base_name}_ann.png")
        cv2.imwrite(out_ann, cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR))

    ui.log("OK", f"Processed {len(selected_ids)} image(s). Output: {out_dir}")

def add_artefactes_for_figures(topo,range_height = 0.2,figure_folder = None, index = None):
    # Add noise and artifacts
    # Add electrical noise and other artifacts
    n_delta = np.random.randint(0, 850, size=1)[0]
    h = np.random.uniform(0, range_height)
    afm_noise = gnoise.add_delta_random_noise(topo, num_pixeles=n_delta, valor=h, seed=None)
    
    scale = np.random.uniform(0, 0.5)
    afm_noise = gnoise.add_normal_noise (afm_noise, scale= scale, seed=None)
    SaveMatriz(afm_noise, f"noise{index:3d}.xyz",figure_folder , size_x=None,unixy='nm',uniz='nm')

    afm_lines = gnoise.add_lines_segments_random(topo, max_pendiente =0.02, num_lineas=8, valor_max_offset=2, seed=None)
    SaveMatriz(afm_lines, f"lines{index:3d}.xyz",figure_folder , size_x=None,unixy='nm',uniz='nm')
    
    # adhesion
    afm_adhesion = gnoise.add_local_adhesion_noise(topo, intensity=0.2, probability=0.1)
   
    afm_adhesion = gnoise.add_circular_sticky_zones( afm_adhesion,num_zones = 3, zone_size = 30, intensity = 0.2)
   
    afm_adhesion = gnoise.add_polygonal_sticky_zones( afm_adhesion,num_points=8,intensity= 0.2, rrange=(10,50) )
    SaveMatriz( afm_adhesion, f"ahesion{index:3d}.xyz",figure_folder , size_x=None,unixy='nm',uniz='nm')

   
    afm_lines_art = gnoise.add_line_artifacts(topo, intensity=0.3, probability=0.1)
    SaveMatriz(afm_lines_art, f"lines_artifacts{index:3d}.xyz",figure_folder , size_x=None,unixy='nm',uniz='nm')
   
   # Periodic noise
    fr = 50
    amp = 0.05
    ang = 45
    afm_periodico = gnoise.add_periodic_noise_angled(topo, frequency=fr, amplitude=amp, angle=ang)
    SaveMatriz(afm_periodico, f"periodic{index:3d}.xyz",figure_folder , size_x=None,unixy='nm',uniz='nm')
    
    # shift lines noise
    max_shift = 4
    afm_shift= gnoise.shift_lines_randomly(topo,max_shift )   
    SaveMatriz(afm_shift, f"shift{index:3d}.xyz",figure_folder , size_x=None,unixy='nm',uniz='nm')

    # combination
    afm_imagen = afm_lines + afm_adhesion + afm_lines_art + afm_periodico
    return afm_imagen 

def add_artefactes(topo, arti: ArtifactsConfig):
    # Safe initialization - guarantees afm_imagen always exists
    afm_imagen = topo

    if arti.apply_gaussian:
        if random.random() < arti.gaussian_probability:
            scale = np.random.uniform(0, arti.gaussian_scale)
            afm_imagen = gnoise.add_normal_noise(afm_imagen, scale=scale, seed=None)

    if arti.apply_loss_contact:
        if random.random() < arti.loss_contact_probaility:
            afm_imagen = gnoise.add_lines_segments_random(afm_imagen,
                                                    max_pendiente=arti.loss_contact_max_slope,
                                                    num_lineas=arti.loss_contact_n,
                                                    valor_max_offset=arti.loss_contact_max_offset,
                                                    seed=None)

    if arti.apply_adhesion:
        if random.random() < arti.local_adhesion_probability:
            afm_imagen = gnoise.add_local_adhesion_noise(afm_imagen,
                                                         intensity=arti.local_adhesion_intensity,
                                                         probability=arti.local_adhesion_pixel_probability)

    if arti.apply_sticky_zones:
        if random.random() < arti.sticky_zones_probability:
            size_zone = np.random.uniform(arti.sticky_zones_size_range[0], arti.sticky_zones_size_range[1])
            afm_imagen = gnoise.add_circular_sticky_zones(afm_imagen,
                                                            num_zones=arti.sticky_zones_n // 2,
                                                            zone_size=size_zone,
                                                            intensity=arti.sticky_zones_intensity)
        if random.random() < arti.sticky_zones_probability:
            afm_imagen = gnoise.add_polygonal_sticky_zones(afm_imagen,
                                                           num_points=1 + arti.sticky_zones_n // 2,
                                                           intensity=arti.sticky_zones_intensity,
                                                           rrange=arti.sticky_zones_size_range)

    if arti.apply_line_noise:
        if random.random() < arti.line_noise_probability:
            afm_imagen = gnoise.add_line_artifacts(afm_imagen,
                                                    intensity=arti.line_noise_intesity,
                                                    probability=arti.line_noise_line_probability)

    if arti.apply_periodic_noise:
        if random.random() < arti.periodic_probability:
            fr  = np.random.uniform(arti.periodic_freq / 10, arti.periodic_freq)
            amp = np.random.uniform(0, arti.periodic_amplitud)
            ang = np.random.uniform(arti.periodic_angle_deg_range[0], arti.periodic_angle_deg_range[1])
            afm_imagen = gnoise.add_periodic_noise_angled(afm_imagen, frequency=fr, amplitude=amp, angle=ang)

    if arti.apply_shift_lines_noise:
        if random.random() < arti.shift_lines_probability:
            max_shift = np.random.randint(0, arti.shift_lines_max)
            afm_imagen = gnoise.shift_lines_randomly(afm_imagen, max_shift)

    # Delta at the end - the spikes sit on top of all the other artifacts,
    # without being smoothed by the Gaussian noise. The amplitude is scaled to the
    # real range of the image so it is always visible.
    if arti.apply_delta:
        if random.random() < arti.delta_probability:
            n_delta = np.random.randint(0, arti.delta_n, size=1)[0]
            img_range = float(afm_imagen.max() - afm_imagen.min())
            h = np.random.uniform(0, arti.delta_intensity) * max(img_range, 1e-12)
            afm_imagen = gnoise.add_delta_random_noise(afm_imagen, num_pixeles=n_delta, valor=h)

    return afm_imagen

# Preparation functions for saving training data
# Functions that lower the resolution and then restore the same size to simulate loss of resolution
def simulation_reduce_resolution(img, simu_size):
    h,w = img.shape[:2]
    # lower resolution
    img_low = cv2.resize(img,(simu_size,simu_size))
    # restore resolution
    img_low_quality = cv2.resize(img_low,(w,h),interpolation=cv2.INTER_LINEAR)

    return img_low_quality
# function that normalizes a matrix
def normalize_matrix(matrix):
    """
    Normalizes a matrix to the range [0, 1].
    
    Args:
        matrix (np.ndarray): Input matrix.
    
    Returns:
        np.ndarray: Normalized matrix.
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val == 0:  # Avoid division by zero
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)

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

def FlattenPlus(img,perc = 40):
    # works very well for samples with a lot of substrate
    def linear_fit(x, a, b):
        return a * x + b

    Flatten_img = img.copy()
    mask = np.zeros_like(img, dtype=bool)  # Create the mask with the same size as the image

    # Apply the fit to the lines
    for i in range(0, Flatten_img.shape[0]):
        # Select the points of the current line
        line_current = Flatten_img[i, :]

        # Sort the points of the current line by value
        sorted_indices = np.argsort(line_current)
        n_points = int(len(line_current) * perc)

        # Select the percentage of smallest points
        selected_indices = sorted_indices[:n_points]
         # Mark the selected points in the mask
        mask[i, selected_indices] = True
        # Get the X and Y coordinates of those points
        x_values = np.arange(len(line_current))[selected_indices]
        y_values = line_current[selected_indices]

        # Fit a straight line to the selected points
        popt, _ = curve_fit(linear_fit, x_values, y_values,p0 = (0,0))

        # Compute the fitted line values for the whole row
        fitted_line = linear_fit(np.arange(len(line_current)), *popt)

        # Subtract the fitted line from the current line
        Flatten_img[i, :] = line_current - fitted_line

        # Set the median of the selected points to zero
        median_selected = np.median(Flatten_img[i, selected_indices])
        Flatten_img[i, :] -= median_selected

    return Flatten_img, mask

def FlattenByMedianDifferences(img, mask=None, smooth_window=11, zero_with_mask=True):
    """
    Aligns rows using the median of the differences between adjacent rows,
    but smooths the offsets before applying them to avoid banding.

    Parameters
    ----------
    img : (H, W) ndarray
    mask : (H, W) optional binary ndarray to exclude zones (1 = use, 0 = ignore)
    smooth_window : odd int, size of the smoothing window (1D moving average)
    zero_with_mask : bool, if True centers the level using the mean where mask==1

    Returns
    -------
    img_flat : corrected ndarray
    offsets  : ndarray of size H with the offset applied to each row
    """
    H, W = img.shape
    diffs = np.zeros(H-1, dtype=np.float64)

    for i in range(1, H):
        if mask is None:
            d = img[i, :] - img[i-1, :]
            diffs[i-1] = np.median(d)
        else:
            # use only columns where both rows are inside the mask
            valid = (mask[i, :] == 1) & (mask[i-1, :] == 1)
            if not np.any(valid):
                diffs[i-1] = 0.0
            else:
                d = img[i, valid] - img[i-1, valid]
                diffs[i-1] = np.median(d)

    # Smoothing of offsets between rows to avoid "sawtooth"
    if smooth_window is None or smooth_window < 3:
        diffs_smooth = diffs
    else:
        diffs_smooth = uniform_filter1d(diffs, size=smooth_window, mode='nearest')

    # Integrate to get the offset per row relative to the first one
    offsets = np.zeros(H, dtype=np.float64)
    offsets[1:] = np.cumsum(diffs_smooth)

    # Remove the mean of the offsets so the image is not shifted globally
    offsets -= np.mean(offsets)

    # Apply
    img_flat = img - offsets[:, None]

    # Optional: fix the zero using the mask (substrate)
    if mask is not None and zero_with_mask and np.any(mask == 1):
        img_flat = img_flat - np.mean(img_flat[mask == 1])

    return img_flat, offsets

# Function that subtracts the global plane
def SubtractGlobalPlane(img, show=False):
    # Create the fitted plane
    plane_fitted, correlation = FitPlane(img)

    # Subtract the fitted plane from the original image
    img_flattened = img - plane_fitted

    if show:
        # Show the original image and the flattened image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        cax0 = ax[0].imshow(img, cmap='gray', interpolation='none', origin='upper')
        ax[0].set_title('original image')
        fig.colorbar(cax0, ax=ax[0], orientation='vertical')

        cax1 = ax[1].imshow(img_flattened, cmap='gray', interpolation='none', origin='upper')
        ax[1].set_title('flattened image')
        fig.colorbar(cax1, ax=ax[1], orientation='vertical')

        ui.plot(fig)

    return img_flattened, plane_fitted
# Function that fits a plane according to a mask
def FitPlane(img, mask=None):
    # Create X and Y coordinates
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)

    if mask is None:
        # Flatten the data for the fit
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = img.flatten()
    else:
        # Apply the mask to select only the desired regions
        x_flat = x[mask == 1].flatten()
        y_flat = y[mask == 1].flatten()
        z_flat = img[mask == 1].flatten()

    # Plane fit
    params, _ = curve_fit(plane, (x_flat, y_flat), z_flat)

    # Create the fitted plane
    plane_fitted = plane((x, y), *params).reshape(img.shape)

    # Compute the Pearson correlation coefficient
    z_fitted_masked = plane((x_flat, y_flat), *params)
    correlation, _ = pearsonr(z_flat, z_fitted_masked)

    return plane_fitted, correlation

# Function to define the plane ax + by + c
def plane(coords, a, b, c):
    x, y = coords
    return a * x + b * y + c


if __name__ == '__main__':

    # coco_folder = r'C:\Users\z78\Documents\projects\SimuScan\Datasets\20251209_125831_COCO_shapes'
    # save_coco_afm_samples_all(coco_folder)
    # ruta_absoluta = os.path.abspath(__file__)
    # print("Script path:", ruta_absoluta)
    dataset_path = generate_data_for_training()
    visualize_coco_samples(dataset_path, subset="train", num_samples=3)
    # #generate_data_for_figures()
    # dataset_path = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\20250627_081848_COCO"
    # visualize_coco_samples(dataset_path, subset="train", num_samples=50)
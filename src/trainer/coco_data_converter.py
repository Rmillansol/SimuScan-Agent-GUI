import os
import json
import shutil
import yaml
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import uuid
from copy import deepcopy
from typing import Dict, Optional, List

def convert_COCO_to_UNET_segmentation(coco_folder):
    if not coco_folder.endswith("_COCO"):
        raise ValueError("The input folder name must end with '_COCO'")

    unet_folder = coco_folder.replace("_COCO", "_UNET")
    coco_images_folder = os.path.join(coco_folder, "images")
    coco_annotations_folder = os.path.join(coco_folder, "annotations")

    if not os.path.exists(coco_annotations_folder):
        raise FileNotFoundError(f"Could not find annotations folder at {coco_annotations_folder}")

    unet_images_folder = os.path.join(unet_folder, "images")
    unet_masks_folder = os.path.join(unet_folder, "masks")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(unet_images_folder, split), exist_ok=True)
        os.makedirs(os.path.join(unet_masks_folder, split), exist_ok=True)

    class_list = None

    for split in ["train", "val", "test"]:
        ann_file = os.path.join(coco_annotations_folder, f"instances_{split}.json")
        img_folder = os.path.join(coco_images_folder, split)
        
        if not os.path.exists(ann_file) or not os.path.exists(img_folder):
            print(f"⚠️  {split} not found, skipping...")
            continue

        print(f"✅ Processing {split}...")

        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        images_info = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        category_id_to_index = {cat['id']: idx for idx, cat in enumerate(sorted(coco_data['categories'], key=lambda x: x['id']))}

        if class_list is None:
            class_list = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]

        annotations_per_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_per_image:
                annotations_per_image[img_id] = []
            annotations_per_image[img_id].append(ann)

        for img_id, img_info in tqdm(images_info.items(), desc=f"{split} images"):
            img_w, img_h = img_info['width'], img_info['height']
            img_filename = img_info['file_name']

            src_img_path = os.path.join(img_folder, img_filename)
            dst_img_path = os.path.join(unet_images_folder, split, img_filename)

            if not os.path.exists(src_img_path):
                print(f"⚠️ Image {src_img_path} not found, skipping...")
                continue

            shutil.copyfile(src_img_path, dst_img_path)

            # Initialize mask with background (class 0)
            mask = Image.new("L", (img_w, img_h), 0)
            draw = ImageDraw.Draw(mask)

            anns = annotations_per_image.get(img_id, [])
            for ann in anns:
                segs = ann.get("segmentation", [])
                category_id = ann["category_id"]
                class_idx = category_id_to_index[category_id] + 1  # +1 to reserve 0 for background

                for seg in segs:
                    if len(seg) >= 6:
                        polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                        draw.polygon(polygon, fill=class_idx)

            mask_filename = os.path.splitext(img_filename)[0] + ".png"
            mask.save(os.path.join(unet_masks_folder, split, mask_filename))

    print(f"✅ Conversion complete! UNET dataset saved at {unet_folder}")
    return unet_folder

def convert_single_COCO_to_UNET_segmentation(images_folder: str, annotation_json: str, output_folder: str):
    """
    Converts COCO annotations (JSON format) to UNet masks from a folder of images.

    Parameters:
    - images_folder: Path to the folder with the images.
    - annotation_json: Path to the .json file with COCO annotations.
    - output_folder: Output folder where the images and masks are saved.

    Result:
    - Creates `images` and `masks` subfolders inside `output_folder` with the images and their corresponding masks.
    """
    if not os.path.exists(annotation_json):
        raise FileNotFoundError(f"Annotations file not found: {annotation_json}")
    
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)

    with open(annotation_json, "r") as f:
        coco_data = json.load(f)

    images_info = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    valid_categories = [cat for cat in coco_data['categories'] if cat['name'] in ["bacillus", "cocos"]]
    category_id_to_index = {cat['id']: idx for idx, cat in enumerate(valid_categories)}

    # Group annotations by image
    annotations_per_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_per_image:
            annotations_per_image[img_id] = []
        annotations_per_image[img_id].append(ann)

    # Process each image
    for img_id, img_info in tqdm(images_info.items(), desc="Processing images"):
        img_w, img_h = img_info['width'], img_info['height']
        img_filename = img_info['file_name']

        src_img_path = os.path.join(images_folder, img_filename)
        dst_img_path = os.path.join(output_folder, "images", img_filename)

        if not os.path.exists(src_img_path):
            print(f"⚠️ Image not found: {src_img_path}, skipping.")
            continue

        shutil.copyfile(src_img_path, dst_img_path)

        # Create mask with background = 0
        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)

        anns = annotations_per_image.get(img_id, [])
        for ann in anns:
            segs = ann.get("segmentation", [])
            category_id = ann["category_id"]
            class_idx = category_id_to_index[category_id] + 1  # +1 to reserve 0 for the background

            for seg in segs:
                if len(seg) >= 6:
                    polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    draw.polygon(polygon, fill=class_idx)

        mask_filename = os.path.splitext(img_filename)[0] + ".png"
        mask.save(os.path.join(output_folder, "masks", mask_filename))

    print(f"✅ Conversion completed. UNet data saved in {output_folder}")

def show_unet_segmentation(unet_folder, split='train', num_images=5):
    # Define image and mask paths
    images_dir = os.path.join(unet_folder, "images", split)
    masks_dir = os.path.join(unet_folder, "masks", split)

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Images or masks folder for split '{split}' not found!")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    selected_files = random.sample(image_files, min(num_images, len(image_files)))

    for img_name in selected_files:
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, img_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # assuming mask is grayscale

        if img is None or mask is None:
            print(f"⚠️ Could not read image or mask for {img_name}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display side-by-side
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title(f"Image: {img_name}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='jet')
        plt.title(f"Mask: {img_name}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# function that creates a new annotation file
def convert_coco_by_name_map(
    anno_file: str,
    output_json: Optional[str] = None,
    *,
    model_classes: Dict[int, str],     # e.g. {0:"geometric", 1:"circle", 2:"hexagon", 3:"rectangle"}
    gt_classes: Optional[Dict[int, str]] = None,  # if None, read from the JSON
    name_map: Dict[str, str],          # e.g. {"circle":"ellipse_GT", "hexagon":"hexagon_GT", "rectangle":"square_GT"}
    start_id: int = 0,                 # <-- default 0 so that new_id = start_id + model_index
    drop_unused_images: bool = False,  # True = removes images left without annotations
    strict_exist_check: bool = False   # True = raises an error if any name_map name does not exist in the GT
) -> str:
    """
    Generates a NEW COCO JSON remapping categories and annotations:
      - Only the model classes present in `name_map` (model→GT) are kept.
      - The new categories use **the MODEL NAMES**.
      - The new category IDs are **start_id + model_index**.
      - GT annotations are transferred to the corresponding model class via name_map.
      - Does not modify 'images' unless drop_unused_images=True (then it filters).
    Returns the path to the output JSON.
    """

    # 1) Load the original COCO
    with open(anno_file, "r", encoding="utf-8") as f:
        coco_src = json.load(f)

    # 2) Get GT classes (id->name) if not given
    if gt_classes is None:
        gt_classes = {cat["id"]: cat["name"] for cat in coco_src.get("categories", [])}

    # 3) GT name->id map (to validate existence by name)
    gt_name_to_id = {v: k for k, v in gt_classes.items()}

    # 4) Build the sequence (ordered by model index) of classes to evaluate
    #    Each element: (midx, model_name, gt_name, gt_id)
    eval_seq = []
    seen_model_names = set()
    seen_gt_names = set()

    for midx in sorted(model_classes.keys()):
        mname = model_classes[midx]                 # class name in the model
        tgt_name = name_map.get(mname, None)        # class name in the GT
        if tgt_name is None:
            continue  # the model class does not take part

        # Validate existence in GT
        if tgt_name not in gt_name_to_id:
            msg = f"GT class '{tgt_name}' (from model '{mname}') does not exist in GT."
            if strict_exist_check:
                raise ValueError(msg)
            else:
                print("[WARN]", msg, "→ skipped.")
                continue

        # Prevent 2 model classes from pointing to the same tgt_name (ambiguous)
        if tgt_name in seen_gt_names:
            raise ValueError(
                f"Ambiguity: several model classes map to the same GT name '{tgt_name}'. "
                f"Check name_map."
            )

        if mname in seen_model_names:
            # Almost impossible if model_classes is well formed, but we check it
            raise ValueError(f"Repeated model class name: '{mname}'.")

        seen_model_names.add(mname)
        seen_gt_names.add(tgt_name)
        eval_seq.append((midx, mname, tgt_name, gt_name_to_id[tgt_name]))

    if len(eval_seq) == 0:
        raise ValueError("name_map does not map any model class; nothing to convert.")

    # 5) Build GT_old_id -> NEW_id map (derived from the model index)
    #    NEW_id = start_id + model_index
    gtold_to_new = {}
    for midx, mname, tgt_name, gt_id in eval_seq:
        new_id = start_id + int(midx)
        gtold_to_new[gt_id] = new_id

    # 6) Remap annotations
    new_annotations = []
    for ann in coco_src.get("annotations", []):
        old = ann["category_id"]
        new_id = gtold_to_new.get(old, None)
        if new_id is None:
            continue  # annotation falls in a class that is not mapped → removed
        na = deepcopy(ann)
        na["category_id"] = new_id
        new_annotations.append(na)

    # 7) Build NEW categories using MODEL NAMES and MODEL IDs (+start_id)
    #    Keep the order by model index
    new_categories = []
    for midx, mname, _, _ in eval_seq:
        new_categories.append({
            "id": start_id + int(midx),
            "name": mname
        })

    # 8) Filter images if applicable
    if drop_unused_images:
        keep_img_ids = set(a["image_id"] for a in new_annotations)
        new_images = [im for im in coco_src.get("images", []) if im["id"] in keep_img_ids]
    else:
        new_images = coco_src.get("images", [])

    # 9) Build the final COCO
    coco_new = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }

    # 10) Output
    if output_json is None:
        base = os.path.splitext(os.path.basename(anno_file))[0]
        output_json = os.path.join(
            os.path.dirname(anno_file),
            f"{base}_remapped_{uuid.uuid4().hex[:8]}.json"
        )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_new, f, ensure_ascii=False)

    # 11) Useful information
    print(f"[OK] Converted: {anno_file}")
    print(f"     → {output_json}")
    print("     New classes (id→name):", { (start_id+int(midx)) : mname for midx, mname, _, _ in eval_seq })
    if drop_unused_images:
        print(f"     Images: {len(new_images)}  |  Annotations: {len(new_annotations)}")
    return output_json

#####Functions that convert a data set in COCO format to another format###############################################
def convert_coco_to_yolo_segmentation(coco_folder):
    if not coco_folder.endswith("_COCO"):
        raise ValueError("The input folder name must end with '_COCO'")
    
    yolo_folder = coco_folder.replace("_COCO", "_YOLO")
    coco_images_folder = os.path.join(coco_folder, "images")
    coco_annotations_folder = os.path.join(coco_folder, "annotations")

    if not os.path.exists(coco_annotations_folder):
        raise FileNotFoundError(f"Could not find annotations folder at {coco_annotations_folder}")

    # Create YOLO folders
    yolo_images_folder = os.path.join(yolo_folder, "images")
    yolo_labels_folder = os.path.join(yolo_folder, "labels")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(yolo_images_folder, split), exist_ok=True)
        os.makedirs(os.path.join(yolo_labels_folder, split), exist_ok=True)

    yaml_paths = {}
    class_list = None

    for split in ["train", "val", "test"]:
        ann_file = os.path.join(coco_annotations_folder, f"instances_{split}.json")
        img_folder = os.path.join(coco_images_folder, split)
        
        if not os.path.exists(ann_file) or not os.path.exists(img_folder):
            print(f"⚠️  {split} not found, skipping...")
            continue

        print(f"✅ Processing {split}...")

        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        images_info = {img['id']: img for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        if class_list is None:
            class_list = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]

        annotations_per_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_per_image:
                annotations_per_image[img_id] = []
            annotations_per_image[img_id].append(ann)

        for img_id, img_info in images_info.items():
            img_w = img_info['width']
            img_h = img_info['height']
            img_filename = img_info['file_name']
            label_filename = os.path.splitext(img_filename)[0] + ".txt"

            src_img_path = os.path.join(img_folder, img_filename)
            dst_img_path = os.path.join(yolo_images_folder, split, img_filename)

            if not os.path.exists(src_img_path):
                print(f"⚠️ Image {src_img_path} not found, skipping...")
                continue

            shutil.copyfile(src_img_path, dst_img_path)

            label_path = os.path.join(yolo_labels_folder, split, label_filename)
            with open(label_path, "w") as f:
                anns = annotations_per_image.get(img_id, [])
                for ann in anns:
                    category_id = ann['category_id']
                    bbox = ann['bbox']  # formato COCO: [x_min, y_min, width, height]
                    segmentation = ann.get('segmentation', [])
                    
                    # Convert bbox to normalised YOLO format (x_center, y_center, w, h)
                    x_min, y_min, w, h = bbox
                    x_center = (x_min + w / 2) / img_w
                    y_center = (y_min + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h

                    line = f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

                    # Add segmentation if present
                    if segmentation:
                        for seg in segmentation:
                            if isinstance(seg, list) and len(seg) >= 6:
                                points = [(seg[i] / img_w, seg[i+1] / img_h) for i in range(0, len(seg), 2)]
                                seg_str = " ".join([f"{x:.6f} {y:.6f}" for (x, y) in points])
                                f.write(f"{line} {seg_str}\n")
                            else:
                                continue
                    else:
                        # Box only, no segmentation
                        f.write(f"{line}\n")
                    # if not segmentation:
                    #     continue

                    # for seg in segmentation:
                    #     if len(seg) < 6:
                    #         continue  # skip invalid
                    #     points = [(seg[i] / img_w, seg[i+1] / img_h) for i in range(0, len(seg), 2)]
                    #     points_flat = " ".join([f"{x:.6f} {y:.6f}" for (x, y) in points])
                    #     f.write(f"{category_id} {points_flat}\n")

    # Build dataset.yaml
    yaml_data = {
        "train": os.path.abspath(os.path.join(yolo_images_folder, "train")),
        "val": os.path.abspath(os.path.join(yolo_images_folder, "val")),
        "names": class_list,
        "nc": len(class_list)
    }

    # Only include test if it exists
    if os.path.exists(os.path.join(yolo_images_folder, "test")) and len(os.listdir(os.path.join(yolo_images_folder, "test"))) > 0:
        yaml_data["test"] = os.path.abspath(os.path.join(yolo_images_folder, "test"))

    with open(os.path.join(yolo_folder, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_data, f)

    print(f"✅ Conversion complete! YOLO dataset saved at {yolo_folder}")

    return yolo_folder

import os, json, shutil, glob
from typing import Dict, List, Tuple

def convert_coco_to_yolo_dual(coco_folder: str) -> Tuple[str, str]:
    """
    Converts COCO (with bbox and/or segmentation) to two YOLO datasets:
      - *_YOLO_DET : detection (cls xc yc w h)
      - *_YOLO_SEG : segmentation (cls x1 y1 x2 y2 ... xn yn)
    Structure preserved: images/{train,val,test}, labels/{train,val,test}.
    Returns (yolo_det_folder, yolo_seg_folder).
    """
    if not coco_folder.endswith("_COCO"):
        raise ValueError("The input folder name must end with '_COCO'")

    coco_images_folder = os.path.join(coco_folder, "images")
    coco_annotations_folder = os.path.join(coco_folder, "annotations")
    if not os.path.isdir(coco_annotations_folder):
        raise FileNotFoundError(f"Annotations folder not found: {coco_annotations_folder}")

    # Output folders
    yolo_det_folder = coco_folder.replace("_COCO", "_YOLO_DET")
    yolo_seg_folder = coco_folder.replace("_COCO", "_YOLO_SEG")

    # Create structure
    for out_root in (yolo_det_folder, yolo_seg_folder):
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    class_names: List[str] = []
    catid_to_idx: Dict[int, int] = {}

    def clamp01(v: float, eps: float = 0.0) -> float:
        # To avoid exact borders you can set eps=1e-6
        return max(0.0 + eps, min(1.0 - eps, v))

    def write_yaml(out_root: str, task: str):
        # Use relative paths with a 'path' so it is portable
        data = {
            "path": os.path.abspath(out_root).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": class_names,
            "nc": len(class_names),
            "task": task,
        }
        test_dir = os.path.join(out_root, "images", "test")
        if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
            data["test"] = "images/test"
        with open(os.path.join(out_root, "dataset.yaml"), "w", encoding="utf-8") as f:
            import yaml
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    # Process each split
    for split in ("train", "val", "test"):
        ann_file = os.path.join(coco_annotations_folder, f"instances_{split}.json")
        img_folder = os.path.join(coco_images_folder, split)
        if not (os.path.isfile(ann_file) and os.path.isdir(img_folder)):
            print(f"⚠️  {split} not found, skipping...")
            continue

        print(f"✅ Processing {split}...")
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Classes (ordered by COCO id, mapped to 0..nc-1)
        if not class_names:
            cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
            class_names = [c["name"] for c in cats_sorted]
            catid_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}

        images_info = {im["id"]: im for im in coco["images"]}

        anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # Loop over the split's images
        for img_id, img in images_info.items():
            fname = img["file_name"]
            W, H = img["width"], img["height"]

            # Copy image to both datasets
            src = os.path.join(img_folder, fname)
            if not os.path.isfile(src):
                print(f"⚠️ Missing image {src}, skipping")
                continue

            dst_det = os.path.join(yolo_det_folder, "images", split, fname)
            dst_seg = os.path.join(yolo_seg_folder, "images", split, fname)
            if not os.path.isfile(dst_det):
                shutil.copyfile(src, dst_det)
            if not os.path.isfile(dst_seg):
                shutil.copyfile(src, dst_seg)

            # Label files
            lab_det_path = os.path.join(yolo_det_folder, "labels", split, os.path.splitext(fname)[0] + ".txt")
            lab_seg_path = os.path.join(yolo_seg_folder, "labels", split, os.path.splitext(fname)[0] + ".txt")

            det_lines = []
            seg_lines = []

            for ann in anns_by_img.get(img_id, []):
                cls = catid_to_idx[ann["category_id"]]

                # ---- Detection: bbox -> YOLO (xc,yc,w,h)
                if "bbox" in ann and ann["bbox"]:
                    x, y, w, h = ann["bbox"]  # COCO xywh absolutos
                    if w > 0 and h > 0:
                        xc = clamp01((x + w / 2) / W)
                        yc = clamp01((y + h / 2) / H)
                        wn = clamp01(w / W)
                        hn = clamp01(h / H)
                        det_lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

                # ---- Segmentation: polygon(s) -> YOLO (cls x1 y1 ... xn yn)
                # COCO may provide:
                #  - list of lists (polygons): [[x1,y1,...],[...], ...]
                #  - RLE (dict with 'counts' and 'size'): skipped here if pycocotools is missing
                seg = ann.get("segmentation", [])
                if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
                    # one or several polygons. We write ONE line per polygon
                    for poly in seg:
                        if len(poly) < 6:  # needs >= 3 points
                            continue
                        pts = []
                        for i in range(0, len(poly), 2):
                            px = clamp01(poly[i] / W)
                            py = clamp01(poly[i+1] / H)
                            pts.append(f"{px:.6f} {py:.6f}")
                        seg_lines.append(f"{cls} " + " ".join(pts))
                elif isinstance(seg, dict) and "counts" in seg and "size" in seg:
                    # RLE (optional): with pycocotools you could convert to contours.
                    # Here we warn and ignore it to keep the script dependency-free.
                    # print(f"ℹ️  RLE segmentation found in {fname}; skipped (install pycocotools to convert).")
                    pass

            # Save labels (if there are no annotations, create an empty file)
            with open(lab_det_path, "w", encoding="utf-8") as f:
                f.write("\n".join(det_lines) + ("\n" if det_lines else ""))

            with open(lab_seg_path, "w", encoding="utf-8") as f:
                f.write("\n".join(seg_lines) + ("\n" if seg_lines else ""))

    # dataset.yaml for each output
    write_yaml(yolo_det_folder, task="detect")
    write_yaml(yolo_seg_folder, task="segment")

    print(f"✅ Conversion complete!")
    print(f"   • YOLO (detection):   {yolo_det_folder}")
    print(f"   • YOLO (segmentation): {yolo_seg_folder}")
    return yolo_det_folder, yolo_seg_folder

####Functions to visualize boxing, segmentation, and verify that the conversion is done correctly#####################

def show_yolo_segmentation(yolo_folder, split='train', num_images=5):
    # Read class names
    yaml_path = os.path.join(yolo_folder, "dataset.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"dataset.yaml not found in {yolo_folder}")
    
    with open(yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)
    
    class_names = data_yaml.get("names", [])
    if not class_names:
        raise ValueError("No 'names' field found in dataset.yaml!")

    # Directorios
    images_dir = os.path.join(yolo_folder, "images", split)
    labels_dir = os.path.join(yolo_folder, "labels", split)

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Images or labels folder for split '{split}' not found!")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    selected_files = random.sample(image_files, min(num_images, len(image_files)))

    for img_name in selected_files:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        if not os.path.exists(label_path):
            print(f"⚠️ No label found for {img_name}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # invalid format

            class_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])

            # Convert bbox to absolute coordinates
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw segmentation if present
            if len(parts) > 5:
                seg = list(map(float, parts[5:]))
                if len(seg) % 2 == 0:
                    polygon = []
                    for i in range(0, len(seg), 2):
                        x = int(seg[i] * img_w)
                        y = int(seg[i+1] * img_h)
                        polygon.append((x, y))
                    if len(polygon) >= 3:
                        cv2.polylines(img, [np.array(polygon, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw class name
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            font_scale = max(0.5, img_w / 1000)
            thickness = 2
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_w, text_h = text_size
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10

            cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)


        # Show image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"{split.upper()} - {img_name}")
        plt.show()


if __name__ == '__main__':
    ################ YOLO ###########################

    yolo_det_folder, yolo_seg_folder =convert_coco_to_yolo_dual(r"C:\Users\z78\Documents\projects\SimuScan\Datasets\20251212_111524_COCO")
   

    #yolo_folder = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\DNAOrigami_trinagle_20251008_073353_YOLO"
    #show_yolo_segmentation(yolo_folder, split='train', num_images=20)
    ################ UNET ############################
    # unet_folder = convert_COCO_to_UNET_segmentation(r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\20250922_174657_COCO")
    # show_unet_segmentation(unet_folder, split='train', num_images=5)

    #single folder
    # unet_folder = r'C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\Bacillus_cocos_labeled_UNET'
    # images_folder = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\Bacillus_cocos_labeled\train"
    # annotation_json = r'C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\Bacillus_cocos_labeled\train\_annotations.coco.json'
    # convert_single_COCO_to_UNET_segmentation(images_folder, annotation_json, unet_folder )
    # show_unet_segmentation(unet_folder, split='train', num_images=5)

    # ################ Conversion GT format #############################3
    ####################### Bacillus-cocos
    # image_folder = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\Bacillus_cocos_labeled\train"
    # anno_file    = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\Bacillus_cocos_labeled\train\_annotations.coco.json"
   
    # model_classes = {0:"bacillus", 1:"cocos"}  # ACTUAL model order
    # gt_classes    = {0:"bacillus-cocos", 1:"bacillus", 2:"cocos"}    # what comes in the GT JSON
    # name_map      = {"bacillus": "bacillus", "cocos":"cocos"}  # We do NOT map 'geometric' → it is ignored: name_map: name in the model → name in the GT (synonyms / equivalences)
    ####################################################

    ####################### geometric
     ###############################  Geoemtric #############################################################
     
    # image_folder=r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\Geometric_shapes_labeled_v2\train"
    # anno_file=r'C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Datasets\Geometric_shapes_labeled_v2\train\_annotations.coco.json'
   
    # model_classes = {0:"ellipse", 1:"hexagon", 2:"square", 3:"triangle"}  # ACTUAL model order
    # gt_classes    = {0:"geometric", 1:"circle", 2:"hexagon", 3:"rectangle", 4: "triangle"}    # what comes in the GT JSON
    # name_map      = {"ellipse": "circle", "hexagon":"hexagon", "square": "rectangle", "triangle": "triangle"}  # We do NOT map 'geometric' → it is ignored: name_map: name in the model → name in the GT (synonyms / equivalences)

    # ####################################################


    # out_json = convert_coco_by_name_map(
    #     anno_file=anno_file,
    #     output_json=None,              # saves next to the original with a suffix
    #     model_classes=model_classes,
    #     gt_classes=gt_classes,         # or None to read from the JSON
    #     name_map=name_map,
    #     start_id=1,                    # contiguous ids from 1
    #     drop_unused_images=False,      # set True to remove images without annotations
    #     strict_exist_check=False
    # )

    # print("New JSON:", out_json)

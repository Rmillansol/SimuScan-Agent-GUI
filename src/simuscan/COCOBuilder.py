import os
import json
import cv2
import numpy as np
from . import mask_tools
from .paths import datasets_root, ensure_dir

class COCOBuilder:
    def __init__(self, output_folder, categories, split_ratios=(0.7, 0.2, 0.1),
                 save_instance_map=False):
        self.output_folder = output_folder
        self.images_base_folder = os.path.join(output_folder, 'images')
        self.annotations_folder = os.path.join(output_folder, 'annotations')
        self.instances_base_folder = os.path.join(output_folder, 'instances')

        self.subsets = ['train', 'val', 'test']
        self.split_ratios = split_ratios
        self.images_folders = {subset: os.path.join(self.images_base_folder, subset) for subset in self.subsets}
        self.save_instance_map = save_instance_map
        self.instances_folders = {subset: os.path.join(self.instances_base_folder, subset) for subset in self.subsets}

        for folder in self.images_folders.values():
            os.makedirs(folder, exist_ok=True)
        os.makedirs(self.annotations_folder, exist_ok=True)
        if self.save_instance_map:
            for folder in self.instances_folders.values():
                os.makedirs(folder, exist_ok=True)

        self.coco_dicts = {subset: self._init_coco_dict(categories) for subset in self.subsets}
        self.image_id = 0
        self.annotation_id = 0
        self.image_id_to_subset = {}

        self.id_to_name = {cat['id']: cat['name'] for cat in categories}

    def _init_coco_dict(self, categories):
        return {
            "info": {
                "description": "Automatically generated dataset",
                "version": "1.0",
                "year": 2025
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "CC-BY 4.0",
                    "url": "https://creativecommons.org/licenses/by/4.0/"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": categories
        }

    def _choose_subset(self):
        return np.random.choice(self.subsets, p=self.split_ratios)

    def add_data_training(self, img, subset, masks, category_ids,radius_tip,centre_r2,theta,is_double,
                          mask_meta=None, dilate_px=2, merge_contours=False):
        """
        Save the image and its associated masks and category IDs
        in one single call.

        mask_meta : optional list parallel to `masks`; each entry is a dict of
                    extra per-instance fields merged into the COCO annotation
                    (used by shape: domains to carry the generator's grain_id
                    and its measured size/height into the dataset).
        dilate_px : mask dilation applied before contour extraction, to account
                    for tip convolution broadening. Pass 0 for renderers whose
                    masks are already exact and whose instances share borders
                    (grain/domain fields), where dilating would make every grain
                    bleed into its neighbours. May also be a list parallel to
                    `masks`, so an image mixing isolated objects with a grain
                    field dilates only the former.
        merge_contours : when true for a mask, ALL of its contours become ONE
                    annotation carrying several `segmentation` polygons, instead
                    of one annotation per contour. That is the correct COCO shape
                    for an instance broken into pieces — a grain that an object
                    on top cuts in two is still a single grain, and splitting it
                    would give two annotations sharing one grain_id. May also be
                    a list parallel to `masks`. Defaults to false, which keeps
                    the historical one-annotation-per-contour behaviour.
        """
        image_id = self.save_image(img, subset)
        instance_indices = None
        if self.save_instance_map:
            file_name = self.coco_dicts[subset]["images"][-1]["file_name"]
            instance_indices = self._save_instance_map(masks, subset, file_name)
        self.add_annotations_from_masks_and_categories(masks, category_ids, image_id,radius_tip,centre_r2,theta,is_double,
                                                       mask_meta=mask_meta, dilate_px=dilate_px,
                                                       instance_indices=instance_indices,
                                                       merge_contours=merge_contours)

    def _save_instance_map(self, masks, subset, file_name):
        """
        Write the per-image instance label map: one integer per annotated
        instance, 0 = background, stored as a 16-bit PNG next to the image.

        This is the exact pixel-level ground truth. The COCO `segmentation`
        polygons are a re-derivation of these same masks, and the semantic UNet
        converter collapses every instance of a class into one region — so for a
        grain mosaic this file is the only place individual grains survive.

        Masks are painted in list order, so a later instance wins any overlapping
        pixel. Grain masks are clipped against the objects on top of them before
        they get here, so in practice there is nothing to arbitrate.

        Returns the instance index assigned to each mask (1-based, parallel to
        `masks`), which the annotations record as `instance_index`.
        """
        if not masks:
            return []
        h, w = masks[0].shape[:2]
        inst = np.zeros((h, w), dtype=np.uint16)
        for idx, mask in enumerate(masks, start=1):
            inst[mask > 0] = idx
        path = os.path.join(self.instances_folders[subset],
                            os.path.splitext(file_name)[0] + '.png')
        cv2.imwrite(path, inst)
        return list(range(1, len(masks) + 1))

    def save_image(self, img, subset):
        #subset = self._choose_subset()
        file_name = f"{self.image_id:06d}.png"
        path = os.path.join(self.images_folders[subset], file_name)
        # Save image and normalise it
        image_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        cv2.imwrite(path, image_norm)

        height, width = img.shape[:2]
        self.coco_dicts[subset]["images"].append({
            "id": self.image_id,
            "file_name": file_name,
            "height": height,
            "width": width,
        })

        self.image_id_to_subset[self.image_id] = subset
        self.image_id += 1
        return self.image_id - 1

    def add_annotations_from_masks_and_categories(self, masks, category_ids, image_id,radius_tip,centre_r2,theta,is_double,
                                                  mask_meta=None, dilate_px=2, instance_indices=None,
                                                  merge_contours=False):
        subset = self.image_id_to_subset[image_id]

        if mask_meta is None:
            mask_meta = [None] * len(masks)
        if not isinstance(dilate_px, (list, tuple)):
            dilate_px = [dilate_px] * len(masks)
        if instance_indices is None:
            instance_indices = [None] * len(masks)
        if not isinstance(merge_contours, (list, tuple)):
            merge_contours = [merge_contours] * len(masks)

        for mask, category_id, extra, dil, inst_idx, merge in zip(masks, category_ids, mask_meta, dilate_px, instance_indices, merge_contours):
            # dilate the mask by one pixel to account for the convolution
            # (dil=0 -> the mask is used exactly as the generator gives it)
            if dil > 0:
                mask = mask_tools.dilate_circular_mask(mask, dil)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid = [c for c in contours if cv2.contourArea(c) >= 1]
            if not valid:
                continue
            # merge -> the whole mask is ONE instance, however many pieces it is
            # in; otherwise the historical one annotation per contour.
            groups = [valid] if merge else [[c] for c in valid]

            for group in groups:
                padding = 4     # box padding on each side, in pixels
                boxes = [cv2.boundingRect(c) for c in group]
                x  = min(b[0] for b in boxes)
                y  = min(b[1] for b in boxes)
                x2 = max(b[0] + b[2] for b in boxes)
                y2 = max(b[1] + b[3] for b in boxes)
                x_new = max(0, x - padding)
                y_new = max(0, y - padding)
                w_new = (x2 - x) + 2 * padding
                h_new = (y2 - y) + 2 * padding
                bbox = [float(x_new), float(y_new), float(w_new), float(h_new)]
                segmentation = [list(map(float, c.flatten().tolist())) for c in group]
                area = float(sum(cv2.contourArea(c) for c in group))

                annotation = {
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "category_name": self.id_to_name[category_id],
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "radius_tip": radius_tip,
                    "centre_r2": centre_r2,
                    "theta": theta,
                    "is_double": is_double
                }
                if extra:
                    annotation.update(extra)
                if inst_idx is not None:
                    # Links this annotation to instances/<split>/<image>.png.
                    # A mask that yields several contours shares one index: the
                    # contours are pieces of a single instance.
                    annotation["instance_index"] = int(inst_idx)

                self.coco_dicts[subset]["annotations"].append(annotation)
                self.annotation_id += 1

    def save_jsons(self):
        for subset in self.subsets:
            json_path = os.path.join(self.annotations_folder, f"instances_{subset}.json")
            with open(json_path, 'w') as f:
                json.dump(self.coco_dicts[subset], f, indent=4)
            
            # Compact only the segmentations after saving
            self._fix_segmentation_format(json_path)

    def _fix_segmentation_format(self, json_path):
        import re

        # Read as text
        with open(json_path, 'r') as file:
            text = file.read()

        # Pattern that matches each "segmentation": [[ ... ]]
        pattern = r'"segmentation":\s*\[\s*\[\s*(.*?)\s*\]\s*\]'

        # Function to flatten the points
        def format_segmentation(match):
            points = match.group(1)
            points = re.sub(r'\s+', '', points)  # remove line breaks and spaces
            return f'"segmentation":[[{points}]]'

        # Apply the fix only to the segmentations
        text_fixed = re.sub(pattern, format_segmentation, text, flags=re.DOTALL)

        # Save the fixed text
        with open(json_path, 'w') as file:
            file.write(text_fixed)
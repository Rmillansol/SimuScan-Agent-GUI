import os
import random
import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import csv


def Yolo_process(
        model, image_path,
        conf_threshold=0.5,
        draw_mode="both",     # "box", "centroid", "seg", "both"
        draw_text=True,
        include_cls_ids=None  # e.g., [0, 2]; if [] -> draws nothing
    ):
    # Load as GRAY and convert to RGB (keeping your choice)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_height, img_width = img.shape[:2]

    # Paleta (RGB)
    class_colors_rgb = [
        (0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167),
        (86, 180, 233), (240, 90, 40), (90, 90, 220), (190, 0, 50),
        (127, 127, 127), (163, 100, 30), (0, 170, 170), (150, 120, 220),
        (0, 200, 140), (220, 80, 120), (60, 160, 200), (120, 190, 32),
    ]

    # Class -> colour (RGB) map sized to the model's classes
    try:
        num_classes = len(model.names)
    except Exception:
        num_classes = len(class_colors_rgb)  # fallback
    class_map_rgb = {i: class_colors_rgb[i % len(class_colors_rgb)] for i in range(num_classes)}

    # Run inference (if include_cls_ids is None we do not filter; if [] we return empty)
    classes_arg = include_cls_ids if include_cls_ids is not None else None
    results = model.predict(image_path, conf=conf_threshold, classes=classes_arg)[0]

    def _cls_allowed(c):
        # None = all; [] = none; [ids...] = only those
        return (include_cls_ids is None) or (c in include_cls_ids)

    # Draw segmentations if present
    if draw_mode in ("seg", "both") and hasattr(results, "masks") and results.masks is not None:
        segs = results.masks.data.cpu().numpy()
        for seg, box in zip(segs, results.boxes):
            cls = int(box.cls[0].item())
            if not _cls_allowed(cls):
                continue
            color_bgr = class_map_rgb[cls][::-1]  # RGB -> BGR for OpenCV
            mask = (seg * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color_bgr, 2)

    # Draw boxes or centroids
    count = 0
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0].item())
            if not _cls_allowed(cls):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0].item())
            color_bgr = class_map_rgb[cls][::-1]
            #label = f"{model.names[cls]} {conf:.2f}" if hasattr(model, "names") else f"id{cls} {conf:.2f}"
            label = f"{conf:.2f}"

            if draw_text:
                cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

            if draw_mode in ("box", "both"):
                cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
            elif draw_mode == "centroid":
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # radius proportional to the box size
                r = max(2, max(y2 - y1, x2 - x1) // 100)
                cv2.circle(img, (cx, cy), r, color_bgr, -1)

            count += 1

    if count == 0:
        text = "No objects detected!"
        font_scale = 1.0
        font_thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (img_width - text_size[0]) // 2
        text_y = int(text_size[1] + 20)
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)

    return img, results


def Yolo_process_old(model, image_path,
                    conf_threshold=0.5,
                    draw_mode="both",  # opciones: "box", "centroid", "seg", "both"
                    draw_text = True
                ):
    
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)      # load as grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)             # Convert to RGB so we can paint in green
        # Get the dimensions of the original image
        img_height, img_width = img.shape[:2]
        # Colours in **RGB**
        class_colors_rgb = [
            (0, 114, 178),   # azul
            (213, 94, 0),    # rojo-anaranjado (vermilion)
            (0, 158, 115),   # teal
            (204, 121, 167), # purple
            (86, 180, 233),  # sky blue
            (240, 90, 40),   # coral
            (90, 90, 220),   # bluish indigo
            (190, 0, 50),    # crimson
            (127, 127, 127), # gris
            (163, 100, 30),  # brown
            (0, 170, 170),   # turquesa
            (150, 120, 220), # strong lavender
            (0, 200, 140),   # mint green
            (220, 80, 120),  # muted magenta
            (60, 160, 200),  # acero-azulado
            (120, 190, 32),  # green (avoiding yellow)
        ]

        # Class -> colour (RGB) map
        num_classes = len(class_colors_rgb)  # adjust if you want
        class_map_rgb = {i: class_colors_rgb[i % len(class_colors_rgb)] for i in range(num_classes)}
        # Run inference on the image
        results = model.predict(image_path, conf=conf_threshold)[0]

        # Draw segmentations if present
        if draw_mode in ("seg", "both") and hasattr(results, "masks") and results.masks is not None:
            for i, (seg,box) in enumerate(zip(results.masks.data.cpu().numpy(), results.boxes)):
                cls = int(box.cls[0].item())
                color_bgr = class_map_rgb[cls][::-1]  # flips RGB->BGR
                mask = (seg * 255).astype(np.uint8)
                #color = class_colors[i % len(class_colors)]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, color_bgr, 2)

        # Draw boxes or centroids
        count = 0
        for box in results.boxes:
            count +=1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            color_bgr = class_map_rgb[cls][::-1]  # flips RGB->BGR
            #color = class_colors[cls % len(class_colors)]
            label = f"{model.names[cls]} {conf:.2f}"
            if draw_text:
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
            # if count==1 and draw_text==False:
            #     cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if draw_mode in ("box", "both"):
                cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
            elif draw_mode == "centroid":
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r = max(2, h // 100)
                cv2.circle(img, (cx, cy), r, color_bgr, -1)
        
        if count == 0:
            text = "No objects detected!"
            font_scale = 1.0  # Larger size
            font_thickness = 2  # Text thickness
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = (img_width - text_size[0]) // 2  # Centre in X
            text_y = int(text_size[1] + 20)  # Position at the top
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
        
    
        return img,  results

def train_yolo_model(yolo_folder, model_path="yolov8m-seg.pt", epochs=100, imgsz=512, batch=32):
   
    dataset_yaml_path = os.path.join(yolo_folder, "dataset.yaml")

    print(" PyTorch uses GPU:", torch.cuda.is_available())

    model = YOLO(model_path)

    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,  # 0=GPU, "cpu"=CPU only
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        hsv_v=0.1,
        label_smoothing=0.05,
        #workers=0, cache=False,  # Windows: more deterministic
        project=os.path.join("Out_trainig", 'results_yolo_training'),
        name='run_n'
    )

def show_training_results(results_dir):
    results_csv_path = os.path.join(results_dir, 'results.csv')
    if os.path.exists(results_csv_path):
        df = pd.read_csv(results_csv_path)
        print("\nLatest recorded metrics:\n")
        print(df.tail())

        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
        plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95')
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Metrics over the course of training")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("Metrics file not found.")

    images_to_show = ['confusion_matrix.png', 'F1_curve.png', 'PR_curve.png', 'metrics.png']
    for img_name in images_to_show:
        img_path = os.path.join(results_dir, img_name)
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_name)
            plt.show()

def evaluate_yolo_model(yolo_folder, model_path, conf_threshold=0.5, num_images=5):

    # Load model
    
    model = YOLO(model_path)

    print("🔍 Evaluating model...")

    results = model.val(
        data=os.path.join(yolo_folder, "dataset.yaml"),
        split="val",
        conf=conf_threshold
    )

    mAP50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
    precision = results.results_dict.get('metrics/precision(B)', 0.0)
    recall = results.results_dict.get('metrics/recall(B)', 0.0)

    print(f"📊 Evaluation results:")
    print(f"🔹 mAP@50-95: {mAP50_95:.4f}")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")

    val_images_folder = os.path.join(yolo_folder, "images/val")
    if not os.path.exists(val_images_folder):
        print(f"❌ Validation images folder not found in {val_images_folder}")
        return

    image_files = [f for f in os.listdir(val_images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = random.sample(image_files, min(num_images, len(image_files)))

    for img_name in image_files:
        img_path = os.path.join(val_images_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        pred = model.predict(img_path, conf=conf_threshold)[0]

        # Draw segmentations if present
        if hasattr(pred, "masks") and pred.masks is not None:
            for i, seg in enumerate(pred.masks.data.cpu().numpy()):
                mask = (seg * 255).astype(np.uint8)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, color, 2)

        # Draw boxes and labels
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"
            color = (0, 255, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predictions on {img_name}")
        plt.show()

    return mAP50_95, precision, recall

def evaluate_yolo_model_on_folder(
    model_path,
    images_folder,
    conf_threshold=0.5,
    num_images=5,
    draw_mode="both",  # opciones: "box", "centroid", "seg", "both"
    draw_text = True
):
    
    model = YOLO(model_path)

    if not os.path.exists(images_folder):
        print(f" ERROR: The images folder '{images_folder}' does not exist.")
        return

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        print(f" No images found in '{images_folder}'")
        return

    image_files = random.sample(image_files, min(num_images, len(image_files)))

    np.random.seed(42)
    class_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(100)}

    for img_name in image_files:
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        pred = model.predict(img_path, conf=conf_threshold)[0]

        # Draw segmentations if present
        if draw_mode in ("seg", "both") and hasattr(pred, "masks") and pred.masks is not None:
            for i, seg in enumerate(pred.masks.data.cpu().numpy()):
                mask = (seg * 255).astype(np.uint8)
                color = class_colors[i % len(class_colors)]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, color, 2)

        # Draw boxes or centroids
        count = 0
        for box in pred.boxes:
            count +=1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            color = class_colors[cls % len(class_colors)]
            label = f"{model.names[cls]} {conf:.2f}"
            if draw_text:
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if count==1 and draw_text==False:
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if draw_mode in ("box", "both"):
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            elif draw_mode == "centroid":
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r = max(2, h // 100)
                cv2.circle(img, (cx, cy), r, color, -1)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{img_name}")
        plt.show()

def evaluate_png_folder_old(path_root,
                        path_model,
                        out_folder,
                        conf_threshold=0.5,
                        draw_mode="box",  # opciones: "box", "centroid", "seg", "both"
                        draw_text = True,
                        include_cls_ids=None):              # e.g., [0, 2]; if [] -> draws nothing)
        
    model = YOLO(path_model)

    
    # load all the images
    image_files = [f for f in os.listdir(path_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
    for png_file in image_files:
        img_path = os.path.join(path_root, png_file)
        # Detect objects
        img_boxs, results = Yolo_process( model, img_path, conf_threshold = conf_threshold,
                draw_mode= draw_mode,  # opciones: "box", "centroid", "seg", "both"
                draw_text = draw_text,
                include_cls_ids = include_cls_ids)
        
        image_path =save_numpy_to_pngfile(img_boxs,out_folder,f"prediction_{png_file}")
        print(f" file: {image_path}")



def evaluate_png_folder_oldold(path_root,
                        path_model,
                        out_folder,
                        conf_threshold=0.5,
                        draw_mode="box",  # "box", "centroid", "seg", "both"
                        draw_text=True,
                        include_cls_ids=None,
                        csv_name="detections.csv"):
    """
    Evaluates all the images in a folder with a YOLO model,
    saves the predictions as PNGs and a CSV with the detections.

    The CSV includes, for each detection:
    - filename
    - det_id (index of the detection within that image)
    - class_id
    - class_name
    - score
    - x1, y1, x2, y2 (in pixels, xyxy format)
    - mask_area (number of mask pixels, if there is segmentation; otherwise None)
    """

    os.makedirs(out_folder, exist_ok=True)

    model = YOLO(path_model)

    # CSV path
    csv_path = os.path.join(out_folder, csv_name)
    with open(csv_path, mode="w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        # Cabecera
        writer.writerow([
            "filename",
            "det_id",
            "class_id",
            "class_name",
            "score",
            "x1", "y1", "x2", "y2",
            "mask_area"
        ])

        # Helper to apply exactly the same class logic as in Yolo_process
        def _cls_allowed(c):
            # None = all; [] = none; [ids...] = only those
            return (include_cls_ids is None) or (c in include_cls_ids)

        # Images to process
        image_files = [f for f in os.listdir(path_root)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for png_file in image_files:
            img_path = os.path.join(path_root, png_file)

            # We call your function WITHOUT modifying it
            img_boxs, results = Yolo_process(
                model,
                img_path,
                conf_threshold=conf_threshold,
                draw_mode=draw_mode,
                draw_text=draw_text,
                include_cls_ids=include_cls_ids
            )

            # Save the image with predictions (uses your own function)
            image_path = save_numpy_to_pngfile(
                img_boxs,
                out_folder,
                f"prediction_{png_file}"
            )
            print(f"file: {image_path}")

            # --- Extract info for the CSV ---
            # results is a single Results object (not a list)
            if results is None or getattr(results, "boxes", None) is None:
                continue

            boxes = results.boxes
            if len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()      # (N, 4)
            conf = boxes.conf.cpu().numpy()      # (N,)
            cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

            # Masks if any (segmentation model)
            masks = None
            if hasattr(results, "masks") and results.masks is not None:
                masks = results.masks.data.cpu().numpy()  # (N, H, W)

            for det_id in range(len(boxes)):
                class_id = int(cls_ids[det_id])
                # Apply the same class filter as when drawing
                if not _cls_allowed(class_id):
                    continue

                class_name = model.names.get(class_id, str(class_id))
                score = float(conf[det_id])
                x1, y1, x2, y2 = xyxy[det_id]

                # Mask area if present
                mask_area = None
                if masks is not None and det_id < masks.shape[0]:
                    mask_area = float(masks[det_id].sum())

                writer.writerow([
                    png_file,
                    det_id,
                    class_id,
                    class_name,
                    score,
                    x1, y1, x2, y2,
                    mask_area
                ])

    print(f"Detections CSV saved to: {csv_path}")

import os
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO

def evaluate_png_folder(path_root,
                        path_model,
                        out_folder=None,
                        conf_threshold=0.5,
                        draw_mode="box",
                        draw_text=True,
                        include_cls_ids=None,
                        csv_name="detections.csv"):
    """
    Evaluates all the images in a folder with a YOLO model,
    saves the predictions as PNGs and a CSV with the detections.
    """

    # ------------------------------------------------------
    # Automatically create a folder if out_folder is None or empty
    # ------------------------------------------------------
    if out_folder is None or out_folder == "":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conf_str = f"{conf_threshold:.2f}".replace(".", "p")  # 0.5 -> 0p50
        folder_name = f"out_{draw_mode}_conf{conf_str}_{timestamp}"

        # folder inside path_root
        out_folder = os.path.join(path_root, folder_name)

    os.makedirs(out_folder, exist_ok=True)

    model = YOLO(path_model)

    # CSV path
    csv_path = os.path.join(out_folder, csv_name)
    with open(csv_path, mode="w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "filename",
            "det_id",
            "class_id",
            "class_name",
            "score",
            "x1", "y1", "x2", "y2",
            "mask_area"
        ])

        def _cls_allowed(c):
            return (include_cls_ids is None) or (c in include_cls_ids)

        image_files = [f for f in os.listdir(path_root)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for png_file in image_files:
            img_path = os.path.join(path_root, png_file)

            img_boxs, results = Yolo_process(
                model,
                img_path,
                conf_threshold=conf_threshold,
                draw_mode=draw_mode,
                draw_text=draw_text,
                include_cls_ids=include_cls_ids
            )

            # Save the generated image
            image_path = save_numpy_to_pngfile(
                img_boxs,
                out_folder,
                f"prediction_{png_file}"
            )
            print(f"file: {image_path}")

            if results is None or getattr(results, "boxes", None) is None:
                continue

            boxes = results.boxes
            if len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            masks = None
            if hasattr(results, "masks") and results.masks is not None:
                masks = results.masks.data.cpu().numpy()

            for det_id in range(len(boxes)):
                class_id = int(cls_ids[det_id])
                if not _cls_allowed(class_id):
                    continue

                class_name = model.names.get(class_id, str(class_id))
                score = float(conf[det_id])
                x1, y1, x2, y2 = xyxy[det_id]

                mask_area = None
                if masks is not None and det_id < masks.shape[0]:
                    mask_area = float(masks[det_id].sum())

                writer.writerow([
                    png_file,
                    det_id,
                    class_id,
                    class_name,
                    score,
                    x1, y1, x2, y2,
                    mask_area
                ])

    print(f"Detections CSV saved to: {csv_path}")


def save_numpy_to_pngfile(np_image, out_folder, filename, phys_max = None):
        """
        Saves a NumPy image as PNG, handling both grayscale and colour images.
        
        Args:
            np_image (np.ndarray): NumPy array of the image (grayscale or RGB/BGR).
            out_folder (str): Folder where the image will be saved.
            filename (str): Output file name.

        Returns:
            str: Full path of the saved image.
        """

        # Make sure the image is not completely uniform, to avoid division by zero
        
        if np_image.max() == np_image.min():
            image_norm = np.zeros_like(np_image, dtype=np.uint8)
        else:
            if phys_max is None:
                image_norm = ((np_image - np_image.min()) / (np_image.max() - np_image.min()) * 255).astype(np.uint8)
            else:
                
                image_scaled = (np_image - np_image.min()) / (phys_max) 
                image_scaled_clipped = np.clip(image_scaled, 0, 1)
                image_norm = (image_scaled_clipped * 255).astype(np.uint8)

        # Check whether the image is colour or grayscale
        if len(image_norm.shape) == 2:  # Grayscale image
            pass  # Already properly normalised
        elif len(image_norm.shape) == 3 and image_norm.shape[-1] == 3:  # Colour image (RGB/BGR)
            pass  # Already in the correct format
        elif len(image_norm.shape) == 3 and image_norm.shape[-1] == 4:  # Image with alpha channel (RGBA)
            image_norm = image_norm[:, :, :3]  # Remove alpha channel
        
        # Make sure the output folder exists
        os.makedirs(out_folder, exist_ok=True)

        # Get the file name without extension and append ".png"
        filename_without_ext = os.path.splitext(filename)[0]
        output_filename = os.path.join(out_folder, f"{filename_without_ext}.png")

        # Save the image
        cv2.imwrite(output_filename, image_norm)

        return output_filename

if __name__ == '__main__':

    # modelo_path = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\models\yolo\DNA_bacterium.pt"
    # image_folder = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\DataTest\ADN"
    # out_folder = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\DataTest\ADN\prediction_sep_1"
    # evaluate_png_folder(path_root = image_folder,
    #                         path_model = modelo_path,
    #                         out_folder = out_folder,
    #                         conf_threshold=0.5,
    #                     draw_mode="box",  # opciones: "box", "centroid", "seg", "both"
    #                     draw_text = True,
    #                     include_cls_ids=[6,7])

    dataset_path =r"C:\Users\z78\Documents\projects\SimuScan\Datasets\20251212_111524_YOLO_SEG"
    
    # train_yolo_model(dataset_path, model_path="yolov8m-seg.pt", epochs=50, imgsz=512, batch=32)
    # yolo_folder =r"C:\Users\z78\Documents\projects\SimuScan\Datasets\20251212_111524_YOLO_SEG"
    # model_path = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Out_trainig\results_yolo_training\run_n16\weights\best.pt"
    # evaluate_yolo_model(yolo_folder=yolo_folder,model_path=model_path)
    
    #Show results
    show_training_results("Out_trainig/results_yolo_training/run_n16")

    evaluate_yolo_model(dataset_path, model_path = "Out_trainig/results_yolo_training/run_n16/weights/best.pt" ,conf_threshold=0.8, num_images=5)

    # evaluate_yolo_model_on_folder(
    # model_path=r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Out_trainig\results_yolo_training\run_n13\weights\best.pt",
    # images_folder= r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\DataTest\Bacillus_on_pillars",
    # conf_threshold=0.5,
    # num_images=10,
    # draw_mode="box",       # both box, seg, centroid
    # draw_text = True ) 
    # evaluate_png_folder(
    #     path_root = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\DataTest\Bacillus_on_pillars_2",
    #     path_model = r"C:\Users\z78\Documents\projects\Synthetic4AFM_DeepDetect\Out_trainig\results_yolo_training\run_n13\weights\best.pt",
    #     conf_threshold=0.85,
    #     draw_mode="box",  # opciones: "box", "centroid", "seg", "both"
    #     draw_text = False,
    #     include_cls_ids=None)
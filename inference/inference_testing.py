import argparse
import supervision as sv
from PIL import Image
import os
import cv2
import csv
import re
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import pickle
from scipy.stats import chi2_contingency

# ----- Helper function to compute IoU -----
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

# ——— Modified helper to load YOLO‐format GT boxes, including class_id ———
#def load_gt_boxes(image_path, base_dir, gt_label_dir, image_shape):
#    rel_path = os.path.relpath(image_path, base_dir)
#    rel_folder, image_filename = os.path.split(rel_path)
#    name_wo_ext = os.path.splitext(image_filename)[0]
#    label_filename = name_wo_ext + ".txt"
#    gt_txt_path = os.path.join(gt_label_dir, rel_folder, label_filename)

#    gt_boxes = []
#    img_h, img_w = image_shape[0], image_shape[1]
#
#    if not os.path.isfile(gt_txt_path):
#        return gt_boxes
#
#    with open(gt_txt_path, 'r') as f:
#        for line in f:
#            line = line.strip()
#            if not line:
#                continue
#            parts = line.split()
#            class_id = int(parts[0])
#            x_cn, y_cn, w_n, h_n = map(float, parts[1:])
#            x_center = x_cn * img_w
#            y_center = y_cn * img_h
#            w_box = w_n * img_w
#            h_box = h_n * img_h
#            x1 = x_center - w_box / 2.0
#            y1 = y_center - h_box / 2.0
#            x2 = x_center + w_box / 2.0
#            y2 = y_center + h_box / 2.0
#            gt_boxes.append({
#                "class_id": class_id,
#                "box": [x1, y1, x2, y2]
#            })
#    return gt_boxes

def load_gt_boxes(image_path, base_dir, gt_label_dir, padded_image_shape, max_size=2048):
    """
    Load YOLO-format GT boxes and adjust for resizing + padding.
    The GT box coords are mapped into the padded image size.
    """
    rel_path = os.path.relpath(image_path, base_dir)
    rel_folder, image_filename = os.path.split(rel_path)
    name_wo_ext = os.path.splitext(image_filename)[0]
    label_filename = name_wo_ext + ".txt"
    gt_txt_path = os.path.join(gt_label_dir, rel_folder, label_filename)

    gt_boxes = []

    # Load original image size
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size

    # Compute scale factor and padding
    scale = max_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (max_size - new_w) // 2
    pad_y = (max_size - new_h) // 2

    if not os.path.isfile(gt_txt_path):
        return gt_boxes

    with open(gt_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_cn, y_cn, w_n, h_n = map(float, parts[1:])

            # Convert to original image coordinates
            x_center = x_cn * orig_w
            y_center = y_cn * orig_h
            w_box = w_n * orig_w
            h_box = h_n * orig_h
            x1 = x_center - w_box / 2
            y1 = y_center - h_box / 2
            x2 = x_center + w_box / 2
            y2 = y_center + h_box / 2

            # Scale and apply padding to fit padded image
            x1_p = x1 * scale + pad_x
            y1_p = y1 * scale + pad_y
            x2_p = x2 * scale + pad_x
            y2_p = y2 * scale + pad_y

            gt_boxes.append({
                "class_id": class_id,
                "box": [x1_p, y1_p, x2_p, y2_p]
            })

    return gt_boxes


def is_near_edge(box, image_shape, edge_ratio=0.05):
    """Check if the box is near the edge of the image."""
    x1, y1, x2, y2 = box
    img_h, img_w = image_shape
    edge_x = edge_ratio * img_w
    edge_y = edge_ratio * img_h

    return (
        x1 < edge_x or x2 > (img_w - edge_x) or
        y1 < edge_y or y2 > (img_h - edge_y)
    )

# ----- Load local model -----
def get_local_model(model_path):
    model = YOLO(model_path)
    model.eval()
    return model

# ----- Argument parsing -----
parser = argparse.ArgumentParser(
    description="Run YOLO inference + compute metrics on a folder of images."
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Where to save annotated images and CSV."
)
parser.add_argument(
    "--model_versions",
    type=int,
    nargs="+",
    default=[211],
    help="One or more model version numbers (e.g. 211)."
)
parser.add_argument(
    "--ground_truth",
    action="store_true",
    help="If set, load and evaluate against GT labels."
)
parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.5,
    help="YOLO confidence threshold for predictions."
)
parser.add_argument(
    "--gt_label_dir",
    type=str,
    help="Base directory where ground‐truth label files live."
)
parser.add_argument(
    "--base_dir",
    type=str,
    required=True,
    help="Top‐level directory of images to run inference on."
)
parser.add_argument(
    "--csv_file_name",
    type=str,
    default="results.csv",
    help="Name of the CSV file to write inside output_dir."
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True
)
parser.add_argument(
    "--heatmap_performance",
    action="store_true"
)
args = parser.parse_args()

# Arg parse arguments
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model_versions        = args.model_versions
ground_truth_flag     = args.ground_truth
confidence_threshold  = args.confidence_threshold
gt_label_dir          = args.gt_label_dir
base_dir              = args.base_dir
csv_file              = os.path.join(output_dir, args.csv_file_name)
model_path            = args.model_path


csv_columns = [
    'Model_version',
    'Image_name',
    'Date_subdirectory',
    'DPI',
    'Class',
    'Total_detections',
    'Confidence_threshold',
    'Total_GT',
    'Missed_GT',
    'Average_IoU',
    'Number_objects_in_image',
    'Precision',
    'Recall',
    'Overall_Precision',
    'Overall_Recall'
]

def resize_image(image_path, max_size=2048, output_dir=None):
    """Resize the image so the larger dimension is 'max_size' pixels and save a copy."""
    with Image.open(image_path) as img:
        ratio = max_size / max(img.width, img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        base_name = os.path.basename(image_path)
        resized_image_name = f"resized_{base_name}"
        resized_image_path = os.path.join(output_dir, resized_image_name) if output_dir else resized_image_name
        resized_img.save(resized_image_path)
        return resized_image_path

def resize_with_padding(image_path, max_size=2048, output_dir=None):
    """
    Resize image so the longer side is `max_size`, then pad the shorter side to make it square.
    Returns path to saved image.
    """
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        scale = max_size / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new square image and paste resized image into the center
        new_img = Image.new("RGB", (max_size, max_size), color=(0, 0, 0))  # Black padding
        paste_x = (max_size - new_width) // 2
        paste_y = (max_size - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y))

        # Save
        base_name = os.path.basename(image_path)
        padded_image_name = f"padded_{base_name}"
        padded_image_path = os.path.join(output_dir, padded_image_name) if output_dir else padded_image_name
        new_img.save(padded_image_path)

        return padded_image_path

def get_date_subdirectory(image_path, base_dir):
    """Extract the first subfolder under base_dir, e.g. "7-18-2023_1dpi"."""
    relative_path = os.path.relpath(image_path, base_dir)
    subdir_parts = relative_path.split(os.sep)
    return subdir_parts[0] if len(subdir_parts) > 1 else "Unknown"

def extract_dpi_label(date_subdirectory):
    """Return something like '1dpi' from '7-18-2025_1dpi'."""
    match = re.search(r"(\d+dpi)", date_subdirectory)
    return match.group(1) if match else None

def draw_missed_detections(missed_boxes_by_class, image_shape, output_path):
    # Create blank RGB image
    vis_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # Assign unique color per class
    class_colors = {}
    for cname in missed_boxes_by_class:
        if cname == "Adult_female":
            class_colors[cname] = [255, 0, 0]  # Red
        elif cname == "Adult_male":
            class_colors[cname] = [0, 0, 255]  # Blue
        elif cname == "Dead_mite":
            class_colors[cname] = [0, 255, 255]  # Cyan
        elif cname == "Immature":
            class_colors[cname] = [0, 255, 0]  # Green
        elif cname == "Viable_egg":
            class_colors[cname] = [255, 255, 0]  # Yellow
        #class_colors[cname] = [random.randint(100, 255) for _ in range(3)]

    for cname, box_list in missed_boxes_by_class.items():
        color = class_colors[cname]
        for box in box_list:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color=color, thickness=2)
            cv2.putText(
                vis_image,
                cname,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA
            )

    cv2.imwrite(output_path, vis_image)
    print(f"Missed detection overlay saved to {output_path}")


# ——— Open CSV and write header ———
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

    for model_version in model_versions:
        model_path = model_path
        model = get_local_model(model_path)
        missed_boxes_by_class = defaultdict(list)
        edge_missed = 0
        edge_detected = 0
        center_missed = 0
        center_detected = 0
        misclassified_dead_mites = 0
        missed_dead_mites = 0
        total_gt_dead_mites = 0

        print(f"Running inference with local model version {model_version}…")

        # Initialize here:
        all_predictions_by_class = defaultdict(list)
        all_gt_by_class = defaultdict(list)

        for root, dirs, files in os.walk(base_dir):
            for image_file in files:
                # Only process .jpg/.jpeg/.png (skip hidden files)
                if image_file.lower().endswith((".jpg", ".jpeg", ".png")) and not image_file.startswith("."):
                    image_path = os.path.join(root, image_file)

                    # 1) Resize & load
                    resized_image_path = resize_with_padding(image_path, output_dir=output_dir)
                    image = cv2.imread(resized_image_path)
                    if image is None:
                        print(f"Error reading {resized_image_path}, skipping.")
                        continue

                    date_subdirectory = get_date_subdirectory(image_path, base_dir)
                    print(f"  Inference on {resized_image_path} (conf ≥ {confidence_threshold})…")
                    inference_result = model.predict(image, conf=confidence_threshold)[0]

                    # 2) Extract predictions → list of { "box":[x1,y1,x2,y2], "confidence":float, "class_id":int, "class_name":str }
                    boxes   = inference_result.boxes.xyxy.cpu().numpy()   # (N,4)
                    confs   = inference_result.boxes.conf.cpu().numpy()   # (N,)
                    cls_ids = inference_result.boxes.cls.cpu().numpy()    # (N,)

                    predictions = []
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i]
                        conf = float(confs[i])
                        class_id = int(cls_ids[i])
                        class_name = model.names[class_id] if hasattr(model, "names") else str(class_id)

                        predictions.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": conf,
                            "class_id": class_id,
                            "class_name": class_name
                        })

                    # 3) Convert to Supervision detections (for optional drawing)
                    roboflow_result = {
                        "image": {"width": image.shape[1], "height": image.shape[0]},
                        "predictions": [
                            {
                                "x": (b[0] + b[2]) / 2.0,
                                "y": (b[1] + b[3]) / 2.0,
                                "width": b[2] - b[0],
                                "height": b[3] - b[1],
                                "confidence": pred["confidence"],
                                "class_id": pred["class_id"],
                                "class": pred["class_name"],
                                "class_name": pred["class_name"]
                            }
                            for b, pred in zip(boxes, predictions)
                        ]
                    }
                    detections = sv.Detections.from_inference(roboflow_result)

                    # 4) Load GT boxes (with class_ids) and compute total number of GT in this image
                    if ground_truth_flag:
                        gt_entries = load_gt_boxes(image_path, base_dir, gt_label_dir, image.shape)
                    else:
                        gt_entries = []
                    number_of_gt_boxes = len(gt_entries)

                    # 5) Organize GT by class_name
                    gt_by_class = defaultdict(list)
                    for gt in gt_entries:
                        cid = gt["class_id"]
                        cname = model.names[cid] if hasattr(model, "names") else str(cid)
                        gt_by_class[cname].append(gt["box"])

                    # 6) Annotate the image (predictions in blue, GT in green)
                    annotated_frame = image.copy()

                    # 6a) draw predicted boxes in blue
                    pred_boxes = detections.xyxy.astype(int)
                    for i, box in enumerate(pred_boxes):
                        x1, y1, x2, y2 = box
                        if hasattr(detections, "class_id") and len(detections.class_id) > i:
                            cid = int(detections.class_id[i])
                            lbl = model.names[cid]
                        else:
                            lbl = "Unknown"
                        conf = detections.confidence[i] if hasattr(detections, "confidence") else None
                        txt = f"{lbl} {conf:.2f}" if conf is not None else lbl

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        tx, ty = x1, y2 + th + 4
                        cv2.putText(
                            annotated_frame,
                            txt,
                            (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2
                        )

                    # 6b) draw GT boxes in green
                    if ground_truth_flag:
                        for cname, box_list in gt_by_class.items():
                            for gt_box in box_list:
                                x1_gt, y1_gt, x2_gt, y2_gt = map(int, gt_box)
                                cv2.rectangle(
                                    annotated_frame,
                                    (x1_gt, y1_gt),
                                    (x2_gt, y2_gt),
                                    (0, 255, 0),
                                    2
                                )
                                cv2.putText(
                                    annotated_frame,
                                    "GT",
                                    (x1_gt, y1_gt - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2
                                )

                    # 7) Save the combined image (predictions + GT)
                    dpi_label = extract_dpi_label(date_subdirectory)
                    base_name_no_ext, ext = os.path.splitext(image_file)
                    if dpi_label:
                        out_fn = f"output_v{model_version}_{base_name_no_ext}_{dpi_label}_full{ext}"
                    else:
                        out_fn = f"output_v{model_version}_{base_name_no_ext}_full{ext}"
                    out_path = os.path.join(output_dir, out_fn)
                    cv2.imwrite(out_path, annotated_frame)
                    print(f"    Saved annotated image to {out_path}")

                    # 8) Compute IoU‐based matching per class to get matched_gt_class & missed_gt
                    #    Build a structure of predicted boxes by class_name
                    preds_by_class = defaultdict(list)
                    for pred in predictions:
                        cname = pred["class_name"]
                        preds_by_class[cname].append(pred["box"])

                    # Add predictions to global per-class collection
                    for pred in predictions:
                        all_predictions_by_class[pred["class_name"]].append(pred)

                    # Add GTs to global per-class collection
                    for gt in gt_entries:
                        cname = model.names[gt["class_id"]] if hasattr(model, "names") else str(gt["class_id"])
                        all_gt_by_class[cname].append(gt)

                    #    For each class present in GT, count how many GT boxes match a prediction
                    matched_gt_by_class = {}
                    missed_gt_by_class = {}
                    avg_iou_by_class = {}
                    for cname, gt_list in gt_by_class.items():
                        matched_count = 0
                        iou_sum_for_class = 0.0
                        for gt_box in gt_list:
                            is_edge = is_near_edge(gt_box, (image.shape[0], image.shape[1]))
                            best_iou = 0.0
                            for pred_box in preds_by_class.get(cname, []):
                                iou = compute_iou(gt_box, pred_box)
                                if iou > best_iou:
                                    best_iou = iou
                            iou_sum_for_class += best_iou
                            if best_iou >= 0.5:
                                matched_count += 1
                                if args.heatmap_performance:
                                    if is_edge:
                                        edge_detected += 1
                                    else:
                                        center_detected += 1
                            else:
                                if args.heatmap_performance:
                                    if is_edge:
                                        edge_missed += 1
                                    else:
                                        center_missed += 1
                                    missed_boxes_by_class[cname].append(gt_box)

                        total_gt_cls = len(gt_list)
                        matched_gt_by_class[cname] = matched_count
                        missed_gt_by_class[cname] = total_gt_cls - matched_count
                        avg_iou_by_class[cname] = (iou_sum_for_class / total_gt_cls) if total_gt_cls > 0 else 0.0


                    # 9) Count per‐class detection totals
                    detection_counts = defaultdict(int)
                    for i in range(len(pred_boxes)):
                        if hasattr(detections, "class_id") and len(detections.class_id) > i:
                            class_id = int(detections.class_id[i])
                            class_name = model.names[class_id]
                        else:
                            class_name = "Unknown"
                        detection_counts[class_name] += 1

                    total_matched = sum(matched_gt_by_class.values())
                    total_predicted = sum(detection_counts.values())
                    total_gt_all = number_of_gt_boxes  # sum(len(gt_list) for gt_list in gt_by_class.values())

                    if total_predicted > 0:
                        overall_precision = total_matched / total_predicted
                    else:
                        overall_precision = 0.0

                    if total_gt_all > 0:
                        overall_recall = total_matched / total_gt_all
                    else:
                        overall_recall = 0.0

                    # === Per-image Misclassification analysis for Dead_mite ===
                    #avg_iou_by_class[cname] = (iou_sum_for_class / total_gt_cls) if total_gt_cls > 0 else 0.0

                    if "Dead_mite" in gt_by_class:
                        gt_boxes_dm = gt_by_class["Dead_mite"]
                        pred_boxes_all = [
                            (cname, idx, box)
                            for cname, box_list in preds_by_class.items()
                            for idx, box in enumerate(box_list)
                        ]
                        used_preds = set()
                        for gt_box in gt_boxes_dm:
                            best_iou = 0.0
                            matched_class = None
                            matched_pred_key = None

                            for cname, idx, pred_box in pred_boxes_all:
                                pred_key = (cname, idx)
                                if pred_key in used_preds:
                                    continue
                                iou = compute_iou(gt_box, pred_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    matched_class = cname
                                    matched_pred_key = pred_key

                            if best_iou >= 0.5:
                                if matched_class == "Dead_mite":
                                    used_preds.add(matched_pred_key)  # correct match
                                else:
                                    misclassified_dead_mites += 1
                                    used_preds.add(matched_pred_key)
                            else:
                                missed_dead_mites += 1



                    # 10) Write one CSV row per class that appears either in predictions or in GT
                    all_classes = set(detection_counts.keys()) | set(gt_by_class.keys())
                    for cname in all_classes:
                        pred_count_cls = detection_counts.get(cname, 0)
                        total_gt_cls   = len(gt_by_class.get(cname, []))
                        matched_cls    = matched_gt_by_class.get(cname, 0)
                        missed_cls     = missed_gt_by_class.get(cname, 0)
                        avg_iou_cls    = avg_iou_by_class.get(cname, 0.0)

                        # Precision = TP / (TP + FP) = matched / pred_count_cls
                        if pred_count_cls > 0:
                            precision = matched_cls / pred_count_cls
                        else:
                            precision = 0.0

                        # Recall = TP / (TP + FN) = matched / total_gt_cls
                        if total_gt_cls > 0:
                            recall = matched_cls / total_gt_cls
                        else:
                            recall = 0.0

                        row = {
                            'Model_version': model_version,
                            'Image_name': image_file,
                            'Date_subdirectory': date_subdirectory,
                            'DPI': dpi_label,
                            'Class': cname,
                            'Confidence_threshold': confidence_threshold,
                            'Total_detections': pred_count_cls,
                            'Total_GT': total_gt_cls,
                            'Missed_GT': missed_cls,
                            'Average_IoU': f"{avg_iou_cls:.3f}",
                            'Number_objects_in_image': number_of_gt_boxes,
                            'Precision': f"{precision:.3f}",
                            'Recall': f"{recall:.3f}",
                            'Overall_Precision': f"{overall_precision:.3f}",
                            'Overall_Recall': f"{overall_recall:.3f}"
                        }
                        writer.writerow(row)


if args.heatmap_performance and missed_boxes_by_class:
    shape = (image.shape[0], image.shape[1])
    output_path = os.path.join(output_dir, f"missed_detections_v{model_version}.png")
    draw_missed_detections(missed_boxes_by_class, shape, output_path)
    missed_save_path = os.path.join(output_dir, f"missed_boxes_v{model_version}.pkl")
    with open(missed_save_path, "wb") as f:
        pickle.dump(dict(missed_boxes_by_class), f)
    print(f"Saved missed boxes to {missed_save_path}")

    # Build contingency table
    table = [[edge_missed, edge_detected],
             [center_missed, center_detected]]

    # Check if any row or column in the table has all zeros
    row_sums = [sum(row) for row in table]
    col_sums = [sum(col) for col in zip(*table)]

    if 0 in row_sums or 0 in col_sums:
        print("Chi-squared test skipped due to a row or column with all zeros.")
    else:
        chi2, p_val, dof, expected = chi2_contingency(table)
        print("\n=== Edge Detection Analysis ===")
        print(f"Missed on edge:   {edge_missed}")
        print(f"Detected on edge: {edge_detected}")
        print(f"Missed in center: {center_missed}")
        print(f"Detected in center: {center_detected}")
        print(f"Chi-squared p-value: {p_val:.4f}")

total_gt_dead_mites = len(all_gt_by_class.get("Dead_mite", []))
print(f"Total GT dead mites: {total_gt_dead_mites}")
print(f"Total misclassified dead mites: {misclassified_dead_mites}")
print(f"Total missed dead mites: {missed_dead_mites}")

assert misclassified_dead_mites + missed_dead_mites <= total_gt_dead_mites, "Too many errors counted!"

print("Inference complete.")
print("Done.")

# the three model binary system ended up compounding errors to an unacceptable degree, so this approach was scratched. Retaining code for future reference. 

import torch
import torchvision.transforms as T
from ultralytics import YOLO
import torch.nn as nn
import torchvision.models as models
import os
import glob
from utils.mappings import map_prediction_to_classes, MITE_SUBCLASSES
from utils.evaluation import plot_overall_confusion_matrix, calculate_iou
from PIL import Image
from utils.plotting import (
    plot_overall_confusion_matrix,
    plot_precision_recall_curve,
    plot_tp_fn_breakdown
)
from utils.image_processing import crop_bbox, annotate_image
from utils.transforms import letterbox_resize,  letterbox_transform

# ===========================
# 1. Define Class Mappings
# ===========================
# Updated Class Mappings
DETECTION_CLASS_MAPPING = {
    0: "alive_mature_female_mite",
    1: "alive_mature_male_mite",
    2: "dead_mite",
    3: "alive_immature_mite",
    4: "Viable_egg"
}

# Alive vs. Dead Classification Mapping
ALIVE_DEAD_MAPPING = {
    0: "Alive",
    1: "Alive",
    2: "Dead_mite",
    3: "Alive",
    4: "None"  # Viable_egg doesn't have alive/dead classification
}

# Maturity Classification Mapping
MATURITY_MAPPING = {
    0: "Mature",
    1: "Mature",
    2: "Dead_mite",  # Dead mites don't have maturity
    3: "Immature",
    4: "None"  # Viable_egg doesn't have maturity classification
}

# Sex Classification Mapping
SEX_MAPPING = {
    0: "Adult_female",
    1: "Adult_male",
    2: "None",  # Dead mites don't have sex classification
    3: "None",
    4: "None"  # Viable_egg doesn't have sex classification
}

# Initialize metrics for each class with separate tracking for total FNs and sub-class FNs
detection_metrics = {
    'Mite': {'TP': 0, 'FP': 0, 'FN_total': 0, 'FN_subclasses': {}},
    'Viable_egg': {'TP': 0, 'FP': 0, 'FN_total': 0, 'FN_subclasses': {}}
}

confidences = []
# Define confidence thresholds for each class
confidence_threshold = {
    'Mite': 0.5,
    'Viable_egg': 0.6
}

# Initialize overall detection lists
overall_y_true = []
overall_y_pred = []

# Define Mite sub-classes
MITE_SUBCLASSES = ["alive_mature_female_mite", "alive_mature_male_mite", "dead_mite", "alive_immature_mite"]

# Initialize FN counts for each Mite sub-class
for subclass in MITE_SUBCLASSES:
    detection_metrics['Mite']['FN_subclasses'][subclass] = 0

# For Viable_egg, since there's only one sub-class
detection_metrics['Viable_egg']['FN_subclasses']['Viable_egg'] = 0


# ===========================
# 2. Implement Mapping Functions
# ===========================

def map_detection_class(cls_id):
    """
    Map YOLO detection class IDs to labels.
    """
    return DETECTION_CLASS_MAPPING.get(cls_id, "Unknown")


def map_alive_dead_class(cls_id):
    """
    Map ground truth class IDs to Alive/Dead labels.
    """
    return ALIVE_DEAD_MAPPING.get(cls_id, "Unknown")


def map_maturity_class(cls_id):
    """
    Map ground truth class IDs to Maturity labels.
    """
    return MATURITY_MAPPING.get(cls_id, "Unknown")


def map_sex_class(cls_id):
    """
    Map ground truth class IDs to Sex labels.
    """
    return SEX_MAPPING.get(cls_id, "Unknown")


def map_prediction_to_classes(pred_label):
    """
    Map the hierarchical prediction label to its corresponding classification stages.

    Args:
        pred_label (str): Prediction label from the model.

    Returns:
        dict: Dictionary containing classification stages.
    """
    if pred_label == "Viable_egg":
        return {'detection_label': "Viable_egg"}
    elif pred_label == "dead_mite":
        return {
            'detection_label': "dead_mite",
            'alive_dead_label': "Dead_mite"
        }
    elif pred_label == "alive_immature_mite":
        return {
            'detection_label': "alive_immature_mite",
            'alive_dead_label': "Alive",
            'maturity_label': "Immature"
        }
    elif pred_label == "alive_mature_female_mite":
        return {
            'detection_label': "alive_mature_female_mite",
            'alive_dead_label': "Alive",
            'maturity_label': "Mature",
            'sex_label': "Adult_female"
        }
    elif pred_label == "alive_mature_male_mite":
        return {
            'detection_label': "alive_mature_male_mite",
            'alive_dead_label': "Alive",
            'maturity_label': "Mature",
            'sex_label': "Adult_male"
        }
    else:
        return {
            'detection_label': "Unknown"
        }


# ===========================
# 3. Image Transformation Functions
# ===========================

# ===========================
# 4. Classification Transforms
# ===========================


# Example transforms for classification
classification_transforms = T.Compose([
    letterbox_transform,  # Apply the same letterbox resize
    T.ToTensor(),
    T.Normalize(mean=[0.615, 0.659, 0.567],
                std=[0.229, 0.224, 0.225])
])


# ===========================
# 5. Model Loading Function
# ===========================

def load_models(device):
    """
    Load your trained models (both detection and classification).
    In this example, we're using YOLO for detection and PyTorch models for classification.

    Args:
        device (torch.device): The device to load the models onto (CPU or GPU).

    Returns:
        tuple: A tuple containing the detector model and three classification models.
    """
    # Load YOLO detection model
    try:
        detector_model = YOLO("models/best_v132_11x.pt")
        # print("YOLO detection model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO detection model: {e}")
        raise e  # Re-raise the exception after logging

    # Define a helper function to initialize ResNet50-based models
    def initialize_resnet50(num_classes):
        """
        Initialize a ResNet50 model with a modified final layer for classification.

        Args:
            num_classes (int): Number of output classes for the classification task.

        Returns:
            nn.Module: The modified ResNet50 model.
        """
        model = models.resnet50(weights=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    # Initialize classification models
    classifier_alive_dead = initialize_resnet50(num_classes=2).to(device)
    classifier_maturity = initialize_resnet50(num_classes=2).to(device)
    classifier_sex = initialize_resnet50(num_classes=2).to(device)

    # Define a list of tuples containing models and their corresponding weight files
    classifiers = [
        (classifier_alive_dead, "models/resnet50_alive_final_20250114_080832.pth", "Alive/Dead"),
        (classifier_maturity, "models/resnet50_mature_immature_final_best.pth", "Mature/Immature"),
        (classifier_sex, "models/resnet50_male_female_final_best.pth", "Sex")
    ]

    # Load state dictionaries and set models to eval mode
    for model, weight_path, model_name in classifiers:
        try:
            # Load the state dictionary
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()  # Set model to evaluation mode
            # print(f"{model_name} classification model loaded and set to eval mode successfully.")
        except Exception as e:
            print(f"Error loading state dictionary for {model_name} model from '{weight_path}': {e}")
            raise e  # Re-raise the exception after logging

    return detector_model, classifier_alive_dead, classifier_maturity, classifier_sex


# ===========================
# 7. Mite Classification Function
# ===========================


# ===========================
# 8. Image Annotation Function
# ===========================



# ===========================
# 9. IoU Calculation Function
# ===========================


# ===========================
# 10. Ground Truth Loading Function
# ===========================

def load_ground_truth(label_path, img_width, img_height):
    """
    Load ground truth labels from a YOLO format file.
    Returns a list of dicts with 'class', 'bbox', and hierarchical labels.
    """
    ground_truths = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"  Warning: Invalid line in label file '{label_path}': {line.strip()}")
                continue  # Skip invalid lines
            cls_id = int(float(parts[0]))  # Ensure cls_id is an integer
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert normalized coordinates to absolute pixel values
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height

            # Map to detection class
            detection_label = map_detection_class(cls_id)
            if detection_label == "Unknown":
                print(f"  Warning: Unknown class ID {cls_id} in label file '{label_path}'")

            # Initialize hierarchical labels
            alive_dead_label = ALIVE_DEAD_MAPPING.get(cls_id, "None")
            maturity_label = MATURITY_MAPPING.get(cls_id, "None")
            sex_label = SEX_MAPPING.get(cls_id, "None")

            ground_truths.append({
                'original_class_id': cls_id,
                'detection_label': detection_label,
                'alive_dead_label': alive_dead_label,
                'maturity_label': maturity_label,
                'sex_label': sex_label,
                'bbox': [x1, y1, x2, y2]
            })

    # Debugging: Print total ground truths loaded
    # print(f"  Loaded {len(ground_truths)} ground truths from '{label_path}'")
    return ground_truths


# ===========================
# 11. Mite Classification Function
# ===========================

def classify_mite(cropped_mite, classifier_alive_dead, classifier_maturity, classifier_sex):
    """
    Given a cropped mite image and the classification models:
      1. alive vs. dead
      2. immature vs. mature (if alive)
      3. male vs. female (if mature)
    Return a string describing the final classification.
    """

    # Preprocess image
    input_tensor = classification_transforms(cropped_mite).unsqueeze(0)  # (1, C, H, W)

    # Move tensor to the appropriate device
    device = next(classifier_alive_dead.parameters()).device
    input_tensor = input_tensor.to(device)

    # 1) Alive vs. Dead classification
    with torch.no_grad():
        alive_dead_logits = classifier_alive_dead(input_tensor)
        alive_dead_label = alive_dead_logits.argmax(dim=1).item()

    if alive_dead_label == 1:
        return "dead_mite"
    else:
        # 2) If alive, check maturity
        maturity_logits = classifier_maturity(input_tensor)
        maturity_label = maturity_logits.argmax(dim=1).item()
        # Assume 0 = immature, 1 = mature
        if maturity_label == 0:
            return "alive_immature_mite"
        else:
            # 3) If mature, check sex
            sex_logits = classifier_sex(input_tensor)
            sex_label = sex_logits.argmax(dim=1).item()
            # Assume 0 = female, 1 = male
            if sex_label == 1:
                return "alive_mature_male_mite"
            else:
                return "alive_mature_female_mite"


# ===========================
# 12. Inference Function
# ===========================

def run_inference(image_path, device, global_conf_threshold=0.5, class_conf_thresholds=None):
    """
    1. Run YOLO detection on the image to detect both mites and eggs.
    2. For each object labeled 'mite', crop and run multi-stage classification.
    3. For each object labeled 'egg', label it as 'egg'.
    4. Annotate the image with detections and classifications.
    """
    # Load models with the specified device
    (detector_model,
     classifier_alive_dead,
     classifier_maturity,
     classifier_sex) = load_models(device)

    # Load the image
    image_pil = Image.open(image_path).convert('RGB')

    # Run object detection (Ultralytics YOLO usage)
    try:
        results = detector_model.predict(source=image_path, conf=global_conf_threshold)  # Set global confidence threshold here
        # print("YOLO detection completed successfully.")
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return None, []

    if not results:
        print("No detections found.")
        return image_pil, []

    # The results might contain multiple detections
    # We'll assume results[0] is the relevant one for a single image
    detections = results[0].boxes.data  # [N, 6] typically (x1, y1, x2, y2, score, class)
    class_ids = results[0].boxes.cls  # class IDs for each detection

    # Define class mapping (adjust based on your YOLO training)
    class_mapping = {0: "Mite", 1: "Viable_egg"}

    final_detections = []

    for i, bbox_data in enumerate(detections):
        x1, y1, x2, y2, confidence, cls_id = bbox_data
        cls_id = int(cls_id.item())

        # Map the class ID to a label string
        obj_label = class_mapping.get(cls_id, "Unknown")

        # Apply class-specific confidence threshold if defined
        if class_conf_thresholds and obj_label in class_conf_thresholds:
            required_conf = class_conf_thresholds[obj_label]
            if confidence.item() < required_conf:
                print(f"Skipped detection {i} due to class-specific threshold for {obj_label}: {confidence.item()} < {required_conf}")
                continue  # Skip this detection

        if obj_label == "Mite":
            # Crop the bounding box from the original image
            cropped = crop_bbox(image_pil, (x1, y1, x2, y2))

            # Run the multi-stage classification
            mite_subclass = classify_mite(cropped, classifier_alive_dead, classifier_maturity, classifier_sex)

            # Store final detection info
            final_detections.append({
                "bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
                "confidence": confidence.item(),
                "label": mite_subclass
            })
        elif obj_label == "Viable_egg":
            # If it's an egg, we simply label "Viable_egg"
            final_detections.append({
                "bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
                "confidence": confidence.item(),
                "label": "Viable_egg"
            })
        else:
            # Handle unknown classes if necessary
            final_detections.append({
                "bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
                "confidence": confidence.item(),
                "label": "Unknown"
            })

    # Annotate image
    annotated_image = annotate_image(image_pil.copy(), final_detections)

    return annotated_image, final_detections


# ===========================
# 13. Evaluation Function
# ===========================


def run_evaluation(test_images_dir, test_labels_dir, device, global_conf_threshold=0.5, class_conf_thresholds=None,
                   iou_threshold=0.5):
    """
    Run inference on test images and evaluate against ground truth labels.

    Args:
        test_images_dir (str): Directory containing test images.
        test_labels_dir (str): Directory containing ground truth label files.
        device (torch.device): Device to perform computations on.
        global_conf_threshold (float): Global confidence threshold for detections.
        class_conf_thresholds (dict, optional): Class-specific confidence thresholds.
        iou_threshold (float): IoU threshold to consider a detection as True Positive.

    Returns:
        None
    """

    detection_metrics = {
        'Mite': {
            'TP': 0,
            'FP': 0,
            'FN_total': 0,
            'FN_subclasses': {subclass: 0 for subclass in MITE_SUBCLASSES}
        },
        'Viable_egg': {
            'TP': 0,
            'FP': 0,
            'FN_total': 0,
            'FN_subclasses': {'Viable_egg': 0}
        }
    }


    # Initialize overall detection lists for confusion matrix
    overall_y_true = []
    overall_y_pred = []

    # Define all classes including 'No Detection'
    all_classes = MITE_SUBCLASSES + ["Viable_egg", "No Detection"]

    # Iterate over all test images
    image_paths = glob.glob(os.path.join(test_images_dir, "*.*"))  # Adjust pattern as needed
    for idx_image, image_path in enumerate(image_paths, 1):
        image_name = os.path.basename(image_path)
        label_path = os.path.join(test_labels_dir, os.path.splitext(image_name)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"Label file not found for image '{image_name}'. Skipping.")
            continue

        # Load image to get dimensions
        image_pil = Image.open(image_path).convert('RGB')
        img_width, img_height = image_pil.size

        # Load ground truth with actual image size
        ground_truths = load_ground_truth(label_path, img_width, img_height)

        # Run inference (Assuming run_inference is defined elsewhere and returns annotated_image and predictions)
        annotated_image, predictions = run_inference(image_path, device)

        if annotated_image is None:
            print(f"Inference failed for image '{image_name}'. Skipping.")
            continue

        # Initialize matched ground truths for this image per class
        matched_gt = {
            'Mite': set(),
            'Viable_egg': set()
        }

        # Initialize list to keep track of which ground truths have been detected
        gt_detected = [False] * len(ground_truths)

        # Iterate over predictions and match with ground truths
        for pred in predictions:
            pred_bbox = pred['bbox']
            pred_label = pred['label']
            pred_conf = pred['confidence']

            # Map prediction label to classification stages
            pred_classes = map_prediction_to_classes(pred_label)

            # Determine the class of the prediction
            if pred_classes.get('detection_label') == "Viable_egg":
                pred_class = 'Viable_egg'
                pred_cls_id = 4
            elif pred_classes.get('detection_label') in MITE_SUBCLASSES:
                pred_class = 'Mite'
                # Map detection label to class ID
                detection_label_to_cls_id = {
                    "alive_mature_female_mite": 0,
                    "alive_mature_male_mite": 1,
                    "dead_mite": 2,
                    "alive_immature_mite": 3
                }
                pred_cls_id = detection_label_to_cls_id.get(pred_classes['detection_label'], -1)
            else:
                pred_class = "Unknown"
                pred_cls_id = -1

            if pred_class == "Unknown" or pred_cls_id == -1:
                print(f"  Warning: Unknown or incomplete prediction label '{pred_label}'. Skipping.")
                continue

            # Find the best matching ground truth within the same class
            best_iou = 0
            best_gt_idx = -1
            for idx_gt, gt in enumerate(ground_truths):
                if gt['original_class_id'] != pred_cls_id:
                    continue  # Only match within the same class
                if gt_detected[idx_gt]:
                    continue  # Already matched
                iou = calculate_iou(pred_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx_gt

            if best_iou >= iou_threshold and best_gt_idx != -1:
                # True Positive
                detection_metrics[pred_class]['TP'] += 1
                matched_gt[pred_class].add(best_gt_idx)
                gt_detected[best_gt_idx] = True

                # Add to overall confusion matrix
                actual_label = ground_truths[best_gt_idx]['detection_label']
                overall_y_true.append(actual_label)
                overall_y_pred.append(pred_label)
            else:
                # False Positive
                detection_metrics[pred_class]['FP'] += 1
                print(f"  False Positive: Label '{pred_label}' with Confidence {pred_conf:.2f}, BBox: {pred_bbox}")
                # Add to overall confusion matrix as predicted label vs "No Detection"
                overall_y_true.append("No Detection")
                overall_y_pred.append(pred_label)

        # After processing all predictions, handle False Negatives
        for idx_gt, gt in enumerate(ground_truths):
            if not gt_detected[idx_gt]:
                cls = "Mite" if gt['detection_label'] in MITE_SUBCLASSES else "Viable_egg"
                detection_metrics[cls]['FN_total'] += 1
                detection_metrics[cls]['FN_subclasses'][gt['detection_label']] += 1

                # Add to overall confusion matrix as actual label vs "No Detection"
                overall_y_true.append(gt['detection_label'])
                overall_y_pred.append("No Detection")

        # Debugging: Print counts for this image
        print(f"\nImage {idx_image}/{len(image_paths)}: '{image_name}'")
        print(
            f"Size: {img_width}x{img_height} | Detections: {len(matched_gt['Mite'])} Mites, {len(matched_gt['Viable_egg'])} Viable_eggs")
        print(f"Total Predictions: {len(predictions)}")
        print(f"Processing Time: [Details based on your run_inference implementation]")

        # Print Metrics for this image (optional)
        for cls in ['Mite', 'Viable_egg']:
            total_gt_cls = len(
                [gt for gt in ground_truths if gt['detection_label'] == 'Viable_egg']) if cls == 'Viable_egg' else len(
                [gt for gt in ground_truths if gt['detection_label'] in MITE_SUBCLASSES])
            matched_gt_count = len(matched_gt[cls])
            fn_count = detection_metrics[cls]['FN_total']
            print(f"  Class '{cls}': Total GT: {total_gt_cls}, Matched GT: {matched_gt_count}, FN: {fn_count}")

        # Annotate image with both predictions and ground truths
        annotated_image = annotate_image(image_pil.copy(), predictions, ground_truths)

        # Save the annotated image
        annotated_image_filename = f"annotated_{image_name}"
        output_path = os.path.join("output_images", annotated_image_filename)
        os.makedirs("output_images", exist_ok=True)
        annotated_image.save(output_path)
        print(f"Annotated image saved as '{output_path}'")

    # ===========================
    #  Metrics Calculation
    # ===========================
    # After processing all images, calculate metrics for each class

    for cls in ['Mite', 'Viable_egg']:
        TP = detection_metrics[cls]['TP']
        FP = detection_metrics[cls]['FP']
        FN_total = detection_metrics[cls]['FN_total']

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN_total) if (TP + FN_total) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n=== Detection Metrics for '{cls}' ===")
        print(f"True Positives (TP): {TP}")
        print(f"False Positives (FP): {FP}")
        print(f"False Negatives (FN): {FN_total}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Initialize tp_dict and fn_dict based on the class
        if cls == 'Mite':
            tp_dict = {subclass: detection_metrics[cls]['TP'] for subclass in MITE_SUBCLASSES}
            fn_dict = detection_metrics[cls]['FN_subclasses']
        elif cls == 'Viable_egg':
            tp_dict = {'Viable_egg': detection_metrics[cls]['TP']}
            fn_dict = {'Viable_egg': detection_metrics[cls]['FN_subclasses']['Viable_egg']}
        else:
            # Handle any other classes if present
            tp_dict = {}
            fn_dict = {}

        # Breakdown of FNs by sub-class
        print(f"  False Negatives by Sub-Class:")
        for subclass, fn_count in fn_dict.items():
            print(f"    {subclass}: {fn_count}")

        # Plot TP and FN breakdown bar charts (both dodged and stacked)
        plot_tp_fn_breakdown(tp_dict, fn_dict, cls, plot_type='dodged', save_dir='plots')
        plot_tp_fn_breakdown(tp_dict, fn_dict, cls, plot_type='stacked', save_dir='plots')
        print(f"  TP/FN breakdown bar charts for '{cls}' saved in 'plots/' directory.")

    # Plot Precision-Recall Curve for all classes
    plot_precision_recall_curve(overall_y_true, overall_y_pred, all_classes, save_dir='plots')
    print("Precision-Recall curve saved as 'plots/precision_recall_curve.png'.")

    # ===========================
    #  Overall Confusion Matrix
    # ===========================

    # Plot the overall confusion matrix
    plot_overall_confusion_matrix(overall_y_true, overall_y_pred, all_classes)
    print("Overall confusion matrix saved as 'plots/confusion_matrix_overall.png'.")

    # ===========================
    #  Classification Metrics
    # ===========================
    # [Assuming classification_metrics are handled elsewhere or in a separate part of the function]
    print("\n=== Evaluation Completed ===")


# ===========================
# 14. Main Execution
# ===========================

if __name__ == "__main__":
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths to the test dataset
    #test_images_dir = "/Users/michelewiseman/Downloads/TSSM Detection v2.v142i.yolov11/test/images"
    #test_labels_dir = "/Users/michelewiseman/Downloads/TSSM Detection v2.v142i.yolov11/test/labels"
    test_images_dir = "/Users/michelewiseman/Downloads/small/images"
    test_labels_dir = "/Users/michelewiseman/Downloads/small/labels"

    # Define confidence thresholds
    global_conf_threshold = 0.5
    class_conf_thresholds = {
        "Mite": 0.5,
        "Viable_egg": 0.6  # Higher threshold for 'Viable_egg'
    }

    # Run evaluation
    run_evaluation(test_images_dir, test_labels_dir, device, global_conf_threshold, class_conf_thresholds)

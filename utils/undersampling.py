"""
Creates a class-balanced undersampled YOLO dataset.

This script scans a dataset of images and YOLO-format labels, then creates
an undersampled version where each class has the same number of instances.

Example usage:
    python undersample_dataset.py \
        --images_path /path/to/train/images \
        --labels_path /path/to/train/labels \
        --output_dir undersampled
"""

import os
import random
import argparse
from collections import defaultdict
import shutil

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Undersample dataset to balance class counts.")
parser.add_argument("--images_path", type=str, required=True, help="Path to images directory")
parser.add_argument("--labels_path", type=str, required=True, help="Path to labels directory")
parser.add_argument("--output_dir", type=str, default="undersampled", help="Output folder name for undersampled dataset")
args = parser.parse_args()

images_path = args.images_path
labels_path = args.labels_path

# Output paths
undersampled_images_path = os.path.join(os.path.dirname(images_path), args.output_dir, "images")
undersampled_labels_path = os.path.join(os.path.dirname(labels_path), args.output_dir, "labels")

# Create directories for the undersampled dataset
os.makedirs(undersampled_images_path, exist_ok=True)
os.makedirs(undersampled_labels_path, exist_ok=True)

# ----------------------------
# Functions
# ----------------------------
def get_labels(labels_path):
    labels = defaultdict(list)
    for label_file in os.listdir(labels_path):
        if label_file.endswith(".txt"):
            with open(os.path.join(labels_path, label_file), "r") as file:
                for line in file:
                    class_id = int(line.split()[0])
                    labels[class_id].append(label_file)
    return labels

def print_label_counts(labels, message):
    print(message)
    for class_id, files in labels.items():
        print(f"Class {class_id}: {len(files)} instances")
    print()

def undersample_dataset(labels, target_count):
    for class_id, files in labels.items():
        selected_files = random.sample(files, min(len(files), target_count))
        for file in selected_files:
            # Copy label file
            shutil.copyfile(os.path.join(labels_path, file), os.path.join(undersampled_labels_path, file))
            # Copy corresponding image file (assumes .jpg extension, adjust if needed)
            image_file = file.replace(".txt", ".jpg")
            shutil.copyfile(os.path.join(images_path, image_file), os.path.join(undersampled_images_path, image_file))

# ----------------------------
# Main workflow
# ----------------------------
labels = get_labels(labels_path)
print_label_counts(labels, "Label counts before undersampling:")

min_instances = min([len(files) for files in labels.values()])
undersample_dataset(labels, min_instances)

print_label_counts(get_labels(undersampled_labels_path), "Label counts in the undersampled dataset:")
print(f"Dataset has been undersampled and saved to '{args.output_dir}' directory.")
